function vofi_get_cell_type(impl_func, par, xin, h0, ndim0)
    impl_func = wrap_integrand(impl_func, par, xin)
    T = promote_type(eltype(xin), eltype(h0))
    if ndim0 == 4
        # Handle 4D separately with dynamic arrays
        ws4 = _thread_workspace(T)
        return vofi_cell_type_4D(ws4, impl_func, par, collect(xin), collect(h0))
    end

    x0 = zero(MVector{NDIM, T})
    xin_vec = collect(xin)
    for i in 1:min(length(xin_vec), NDIM)
        x0[i] = T(xin_vec[i])
    end
    hvec = pad_to_ndim(h0, T)
    ws = _thread_workspace(T)
    if ndim0 == 1
        x0[2] = 0.0
        x0[3] = 0.0
        return vofi_cell_type_1D(impl_func, par, x0, hvec)
    elseif ndim0 == 2
        x0[3] = 0.0
        return vofi_cell_type_2D(ws, impl_func, par, x0, hvec)
    elseif ndim0 == 3
        return vofi_cell_type_3D(ws, impl_func, par, x0, hvec)
    else
        throw(ArgumentError("ndim0 must be 1, 2, 3, or 4"))
    end
end

function vofi_cell_type_1D(impl_func, par, x0, h0)
    T = promote_type(eltype(x0), eltype(h0))
    f0 = zero(MVector{NSE, T})
    x1 = zero(MVector{NDIM, T})
    np0 = 0
    nm0 = 0
    icc = -1
    nmax0 = 2
    MIN_GRAD = 1.0e-4
    
    # Set unused dimensions to zero
    x1[2] = x0[2]
    x1[3] = x0[3]
    
    # Evaluate at the two endpoints of the 1D cell
    for i in 0:1
        x1[1] = x0[1] + i * h0[1]
        f = call_integrand(impl_func, par, x1)
        f0[i + 1] = f
        if f > 0.0
            np0 += 1
        elseif f < 0.0
            nm0 += 1
        end
    end
    
    # Simple gradient estimate
    fgrad = (f0[2] - f0[1]) / h0[1]
    fgradmod = max(abs(fgrad), MIN_GRAD)
    hm = 0.5 * h0[1]
    fth = fgradmod * hm
    
    # Check if all endpoints are clearly inside or outside
    if np0 * nm0 == 0
        np0 = 0
        nm0 = 0
        for i in 0:1
            f0mod = abs(f0[i + 1])
            if f0mod > fth
                if f0[i + 1] < 0.0
                    nm0 += 1
                else
                    np0 += 1
                end
            end
        end
        if nm0 == nmax0
            icc = 1  # fully inside
        elseif np0 == nmax0
            icc = 0  # fully outside
        else
            # Near boundary - default to inside if any point is negative
            icc = nm0 > 0 ? 1 : 0
        end
    end
    # else: the interface crosses the cell (different signs)
    
    return icc
end

function vofi_cell_type_2D(ws::VofiWorkspace, impl_func, par, x0, h0)
    T = promote_type(eltype(x0), eltype(h0))
    n0 = @MMatrix zeros(vofi_int, NSE, NSE)
    f0 = zero(MMatrix{NSE, NSE, T})
    x1 = zero(MVector{NDIM, T})
    fgrad = zero(MVector{NSE, T})
    np0 = 0
    nm0 = 0
    icc = -1
    check_dir = -1
    nmax0 = 4
    MIN_GRAD = 1.0e-4
    x1[3] = x0[3]

    for i in 0:1
        for j in 0:1
            x1[1] = x0[1] + i * h0[1]
            x1[2] = x0[2] + j * h0[2]
            f = call_integrand(impl_func, par, x1)
            f0[i + 1, j + 1] = f
            if f > 0.0
                np0 += 1
            elseif f < 0.0
                nm0 += 1
            end
        end
    end

    fgrad[1] = 0.5 * ((f0[2, 2] + f0[2, 1]) - (f0[1, 2] + f0[1, 1])) / h0[1]
    fgrad[2] = 0.5 * ((f0[2, 2] + f0[1, 2]) - (f0[2, 1] + f0[1, 1])) / h0[2]
    fgradsq = Sq2(fgrad)
    fgradmod = max(sqrt(fgradsq), MIN_GRAD)
    hm = 0.5 * max(h0[1], h0[2])
    fth = fgradmod * hm

    if np0 * nm0 == 0
        np0 = 0
        nm0 = 0
        for i in 0:1
            for j in 0:1
                f0mod = abs(f0[i + 1, j + 1])
                if f0mod > fth
                    n0[i + 1, j + 1] = 0
                    if f0[i + 1, j + 1] < 0.0
                        nm0 += 1
                    else
                        np0 += 1
                    end
                else
                    n0[i + 1, j + 1] = 1
                end
            end
        end
        if nm0 == nmax0
            icc = 1
        elseif np0 == nmax0
            icc = 0
        end
        xfsp = MinData{T}()
        if icc < 0
            check_dir = vofi_check_boundary_line(ws, impl_func, par, x0, h0, f0,
                                                 xfsp, n0)
        end
        if check_dir < 0
            icc = nm0 > 0 ? 1 : 0
        end
    end

    return icc
end

function vofi_cell_type_3D(ws::VofiWorkspace, impl_func, par, x0, h0)
    T = promote_type(eltype(x0), eltype(h0))
    n0 = @MArray zeros(vofi_int, NSE, NSE, NSE)
    f0 = zero(MArray{Tuple{NSE, NSE, NSE}, T})
    x1 = zero(MVector{NDIM, T})
    fgrad = zero(MVector{NDIM, T})
    np0 = 0
    nm0 = 0
    icc = -1
    check_dir = -1
    nmax0 = 8
    MIN_GRAD = 1.0e-4

    for i in 0:1
        for j in 0:1
            for k in 0:1
                x1[1] = x0[1] + i * h0[1]
                x1[2] = x0[2] + j * h0[2]
                x1[3] = x0[3] + k * h0[3]
                f = call_integrand(impl_func, par, x1)
                f0[i + 1, j + 1, k + 1] = f
                if f > 0.0
                    np0 += 1
                elseif f < 0.0
                    nm0 += 1
                end
            end
        end
    end

    fgrad[1] = 0.25 * ((f0[2, 2, 2] + f0[2, 1, 2] + f0[2, 2, 1] + f0[2, 1, 1]) -
                       (f0[1, 2, 2] + f0[1, 1, 2] + f0[1, 2, 1] + f0[1, 1, 1])) / h0[1]
    fgrad[2] = 0.25 * ((f0[2, 2, 2] + f0[1, 2, 2] + f0[2, 2, 1] + f0[1, 2, 1]) -
                       (f0[2, 1, 2] + f0[1, 1, 2] + f0[2, 1, 1] + f0[1, 1, 1])) / h0[2]
    fgrad[3] = 0.25 * ((f0[2, 2, 2] + f0[2, 1, 2] + f0[1, 2, 2] + f0[1, 1, 2]) -
                       (f0[2, 2, 1] + f0[2, 1, 1] + f0[1, 2, 1] + f0[1, 1, 1])) / h0[3]
    fgradsq = Sq3(fgrad)
    fgradmod = max(sqrt(fgradsq), MIN_GRAD)
    hm = max(h0[1], h0[2])
    hm = 0.5 * max(h0[3], hm)
    fth = fgradmod * hm / sqrt(2.0)

    if np0 * nm0 == 0
        np0 = 0
        nm0 = 0
        for i in 0:1
            for j in 0:1
                for k in 0:1
                    f0mod = abs(f0[i + 1, j + 1, k + 1])
                    if f0mod > fth
                        n0[i + 1, j + 1, k + 1] = 0
                        if f0[i + 1, j + 1, k + 1] < 0.0
                            nm0 += 1
                        else
                            np0 += 1
                        end
                    else
                        n0[i + 1, j + 1, k + 1] = 1
                    end
                end
            end
        end
        if nm0 == nmax0
            icc = 1
        elseif np0 == nmax0
            icc = 0
        end
        xfsp = [MinData{T}() for _ in 1:3]
        if icc < 0
            check_dir = vofi_check_boundary_surface(ws, impl_func, par, x0, h0, f0,
                                                    xfsp, n0)
        end
        if check_dir < 0
            icc = nm0 > 0 ? 1 : 0
        end
    end

    return icc
end

function vofi_cell_type_4D(ws::VofiWorkspace, impl_func, par, x0::AbstractVector, h0::AbstractVector)
    # Use dynamic arrays for 4D since NDIM is 3
    T = promote_type(eltype(x0), eltype(h0))
    n0 = ws.ct4_n0
    f0 = ws.ct4_f0
    x1 = ws.ct4_x1
    fgrad = ws.ct4_fgrad
    np0 = 0
    nm0 = 0
    icc = -1
    nmax0 = 16
    MIN_GRAD = 1.0e-4

    for i in 0:1
        for j in 0:1
            for k in 0:1
                for l in 0:1
                    x1[1] = x0[1] + i * h0[1]
                    x1[2] = x0[2] + j * h0[2]
                    x1[3] = x0[3] + k * h0[3]
                    x1[4] = x0[4] + l * h0[4]
                    f = call_integrand(impl_func, par, x1)
                    f0[i + 1, j + 1, k + 1, l + 1] = f
                    if f > 0.0
                        np0 += 1
                    elseif f < 0.0
                        nm0 += 1
                    end
                end
            end
        end
    end

    # Gradient estimation (central differences)
    fgrad[1] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[2, 2, 1, 2] + f0[2, 1, 1, 2] +
                         f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[2, 2, 1, 1] + f0[2, 1, 1, 1]) -
                        (f0[1, 2, 2, 2] + f0[1, 1, 2, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2] +
                         f0[1, 2, 2, 1] + f0[1, 1, 2, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / h0[1]
    fgrad[2] = 0.125 * ((f0[2, 2, 2, 2] + f0[1, 2, 2, 2] + f0[2, 2, 1, 2] + f0[1, 2, 1, 2] +
                         f0[2, 2, 2, 1] + f0[1, 2, 2, 1] + f0[2, 2, 1, 1] + f0[1, 2, 1, 1]) -
                        (f0[2, 1, 2, 2] + f0[1, 1, 2, 2] + f0[2, 1, 1, 2] + f0[1, 1, 1, 2] +
                         f0[2, 1, 2, 1] + f0[1, 1, 2, 1] + f0[2, 1, 1, 1] + f0[1, 1, 1, 1])) / h0[2]
    fgrad[3] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[1, 2, 2, 2] + f0[1, 1, 2, 2] +
                         f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[1, 2, 2, 1] + f0[1, 1, 2, 1]) -
                        (f0[2, 2, 1, 2] + f0[2, 1, 1, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2] +
                         f0[2, 2, 1, 1] + f0[2, 1, 1, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / h0[3]
    fgrad[4] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[2, 2, 1, 2] + f0[2, 1, 1, 2] +
                         f0[1, 2, 2, 2] + f0[1, 1, 2, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2]) -
                        (f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[2, 2, 1, 1] + f0[2, 1, 1, 1] +
                         f0[1, 2, 2, 1] + f0[1, 1, 2, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / h0[4]
    fgradsq = fgrad[1]^2 + fgrad[2]^2 + fgrad[3]^2 + fgrad[4]^2
    fgradmod = max(sqrt(fgradsq), MIN_GRAD)
    hm = max(h0[1], h0[2])
    hm = max(h0[3], hm)
    hm = 0.5 * max(h0[4], hm)
    fth = fgradmod * hm / sqrt(3.0)

    if np0 * nm0 == 0
        np0 = 0
        nm0 = 0
        for i in 0:1
            for j in 0:1
                for k in 0:1
                    for l in 0:1
                        f0mod = abs(f0[i + 1, j + 1, k + 1, l + 1])
                        if f0mod > fth
                            n0[i + 1, j + 1, k + 1, l + 1] = 0
                            if f0[i + 1, j + 1, k + 1, l + 1] < 0.0
                                nm0 += 1
                            else
                                np0 += 1
                            end
                        else
                            n0[i + 1, j + 1, k + 1, l + 1] = 1
                        end
                    end
                end
            end
        end
        if nm0 == nmax0
            icc = 1
        elseif np0 == nmax0
            icc = 0
        end
        xfsp = nothing  # check_boundary_hypersurface ignores xfs for cell-type
        check_dir = -1
        if icc < 0
            check_dir = vofi_check_boundary_hypersurface(ws, impl_func, par, x0, h0, f0, xfsp, n0)
        end
        if check_dir < 0
            icc = nm0 > 0 ? 1 : 0
        end
    end

    return icc
end
