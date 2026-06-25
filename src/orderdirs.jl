function vofi_order_dirs_1D(impl_func, par, x0, h0, f0)
    np0 = 0
    nm0 = 0
    icc = -1
    nmax0 = 2
    T = promote_type(eltype(x0), eltype(h0), eltype(f0))
    x1 = zero(MVector{NDIM, T})
    MIN_GRAD = 1.0e-4

    x1[2] = x0[2]
    x1[3] = x0[3]
    
    # Evaluate at the two endpoints
    for i in 0:1
        x1[1] = x0[1] + i * h0[1]
        val = call_integrand(impl_func, par, x1)
        f0[i + 1] = val
        if val > 0
            np0 += 1
        elseif val < 0
            nm0 += 1
        end
    end

    # Compute simple gradient
    fgrad = (f0[2] - f0[1]) / h0[1]
    fgradmod = max(abs(fgrad), MIN_GRAD)
    hm = 0.5 * h0[1]
    fth = fgradmod * hm

    # Check if all endpoints have same sign
    if np0 * nm0 == 0
        np0 = nm0 = 0
        for i in 0:1
            f0mod = abs(f0[i + 1])
            if f0mod > fth
                if f0[i + 1] < 0
                    nm0 += 1
                else
                    np0 += 1
                end
            end
        end
        if nm0 == nmax0
            return 1  # fully inside
        elseif np0 == nmax0
            return 0  # fully outside
        end
        # Near boundary
        return nm0 > 0 ? 1 : 0
    end

    return -1  # interface crosses the cell
end

function vofi_order_dirs_2D(ws::VofiWorkspace, impl_func, par, x0, h0, pdir, sdir, f0, xfs_pt)
    T = promote_type(eltype(x0), eltype(h0))
    # order_dirs runs once per cell. `fc`/`nc` below are built as stack-resident
    # immutable `SArray`s (no heap alloc); `n0` is only needed on the near-boundary
    # branch, so it is allocated lazily there rather than unconditionally here.
    hh = 0.5 .* SVector{NDIM}(h0[1], h0[2], h0[3])
    np0 = 0
    nm0 = 0
    icc = -1
    check_dir = -1
    nmax0 = 4
    x1 = zero(MVector{NDIM, T})
    fgrad = zero(MVector{NSE, T})
    MIN_GRAD = 1.0e-4

    x1[3] = x0[3]
    for i in 0:1
        for j in 0:1
            x1[1] = x0[1] + i * h0[1]
            x1[2] = x0[2] + j * h0[2]
            val = call_integrand(impl_func, par, x1)
            f0[i + 1, j + 1] = val
            if val > 0
                np0 += 1
            elseif val < 0
                nm0 += 1
            end
        end
    end

    fgrad[1] = 0.5 * ((f0[2, 2] + f0[2, 1]) - (f0[1, 2] + f0[1, 1])) / h0[1]
    fgrad[2] = 0.5 * ((f0[2, 2] + f0[1, 2]) - (f0[2, 1] + f0[1, 1])) / h0[2]
    fgradsq = Sq2(fgrad)
    fgradmod = max(sqrt(fgradsq), MIN_GRAD)
    hm = max(hh[1], hh[2])
    fth = fgradmod * hm

    if np0 * nm0 == 0
        n0 = zero(MMatrix{NSE, NSE, vofi_int})
        np0 = nm0 = 0
        for i in 0:1
            for j in 0:1
                f0mod = abs(f0[i + 1, j + 1])
                if f0mod > fth
                    n0[i + 1, j + 1] = 0
                    if f0[i + 1, j + 1] < 0
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
            return 1
        elseif np0 == nmax0
            return 0
        end
        check_dir = vofi_check_boundary_line(ws, impl_func, par, x0, h0, f0, xfs_pt, n0)
        if check_dir < 0
            return nm0 > 0 ? 1 : 0
        end
    end

    have = 0.5 * (h0[1] + h0[2])
    # 3×3 refined stencil as a stack-resident immutable SArray (no heap alloc;
    # GPU-friendly). Entry (a,b) is the level set at x0[d] + (idx_d - 1) * hh[d]
    # (3rd coord fixed at x0[3]); the 4 all-odd corners reuse f0 (already
    # evaluated above). `ntuple(_, Val(9))` unrolls so every index is constant.
    fc = SArray{Tuple{NDIM, NDIM}, T}(ntuple(Val(NDIM * NDIM)) do n
        a = (n - 1) % NDIM + 1
        b = (n - 1) ÷ NDIM + 1
        if isodd(a) && isodd(b)
            @inbounds f0[(a + 1) >> 1, (b + 1) >> 1]
        else
            p = SVector{NDIM, T}(x0[1] + (a - 1) * hh[1],
                                 x0[2] + (b - 1) * hh[2],
                                 x0[3])
            call_integrand(impl_func, par, p)
        end
    end)

    nc = SArray{Tuple{NDIM, NDIM}, vofi_int}(ntuple(Val(NDIM * NDIM)) do n
        v = @inbounds fc[n]
        v > 0 ? vofi_int(1) : v < 0 ? vofi_int(-1) : vofi_int(0)
    end)

    nix = niy = 0
    for i in 0:2:2
        if nc[i + 1, 3] * nc[i + 1, 1] < 0
            niy += 1
        end
    end
    for j in 0:2:2
        if nc[3, j + 1] * nc[1, j + 1] < 0
            nix += 1
        end
    end

    zfill!(pdir, 0)
    zfill!(sdir, 0)
    if niy > nix
        jp, js = 2, 1
    elseif nix > niy
        jp, js = 1, 2
    else
        fgrad .= 0
        rwgt = (100.0, 50.0, 10.0, 2.0, 1.0)
        for i in 1:2
            for j in 1:2
                fx = 0.5 * ((fc[i + 1, j + 1] + fc[i + 1, j]) -
                            (fc[i, j + 1] + fc[i, j])) / hh[1]
                fy = 0.5 * ((fc[i + 1, j + 1] + fc[i, j + 1]) -
                            (fc[i + 1, j] + fc[i, j])) / hh[2]
                tmp = sqrt(fx^2 + fy^2)
                fx /= tmp
                fy /= tmp
                iwgt = abs(nc[i + 1, j + 1] + nc[i + 1, j] + nc[i, j + 1] + nc[i, j])
                w = rwgt[iwgt + 1]
                fgrad[1] += fx * w
                fgrad[2] += fy * w
            end
        end
        @inbounds for ii in 1:NSE; fgrad[ii] = abs(fgrad[ii]); end
        if fgrad[1] >= fgrad[2]
            jp, js = 1, 2
        else
            jp, js = 2, 1
        end
    end
    if check_dir >= 0 && check_dir + 1 != jp
        js, jp = jp, js
    end
    pdir[jp] = 1
    sdir[js] = 1

    if jp == 2
        f0[1, 2], f0[2, 1] = f0[2, 1], f0[1, 2]
    end

    if check_dir < 0
        vofi_check_secondary_side(ws, impl_func, par, x0, h0, pdir, sdir, f0, xfs_pt, fth)
    end

    fx = (fc[3, 2] - fc[1, 2]) * have / (2 * hh[1])
    fy = (fc[2, 3] - fc[2, 1]) * have / (2 * hh[2])
    fxx = (fc[3, 2] + fc[1, 2] - 2 * fc[2, 2]) * have^2 / (hh[1]^2)
    fyy = (fc[2, 3] + fc[2, 1] - 2 * fc[2, 2]) * have^2 / (hh[2]^2)
    fxy = (fc[3, 3] - fc[3, 1] - fc[1, 3] + fc[1, 1]) * have^2 / (4 * hh[1] * hh[2])
    tmp = sqrt((fx^2 + fy^2)^3)
    if !isfinite(tmp) || tmp < EPS_NOT0
        Kappa = zero(T)
    else
        Kappa = abs(fxx * fy^2 - 2 * fx * fy * fxy + fx^2 * fyy) / tmp
    end
    a0, a1, a2, a3 = 2.30477, 28.5312, -46.2729, 56.9179
    est = a0 + Kappa * (a1 + Kappa * (a2 + a3 * Kappa))
    est = isfinite(est) ? est : a0
    est = clamp(est, 4.0, float(NGLM))
    npt = Int(ceil(est))
    xfs_pt.ipt = npt
    return -1
end

function vofi_order_dirs_3D(ws::VofiWorkspace, impl_func, par, x0, h0, pdir, sdir, tdir, f0, xfsp)
    T = promote_type(eltype(x0), eltype(h0))
    fd = zero(MMatrix{NDIM, NDIM, T})
    hh = 0.5 .* SVector{NDIM}(h0[1], h0[2], h0[3])
    fgrad = zero(MVector{NDIM, T})
    np0 = 0
    nm0 = 0
    icc = -1
    check_dir = -1
    MIN_GRAD = 1.0e-4
    nmax0 = 8

    x = zero(MVector{NDIM, T})
    for i in 0:1, j in 0:1, k in 0:1
        x[1] = x0[1] + i * h0[1]
        x[2] = x0[2] + j * h0[2]
        x[3] = x0[3] + k * h0[3]
        val = call_integrand(impl_func, par, x)
        f0[i + 1, j + 1, k + 1] = val
        if val > 0
            np0 += 1
        elseif val < 0
            nm0 += 1
        end
    end

    fgrad[1] = 0.25 * ((f0[2, 2, 2] + f0[2, 1, 2] + f0[2, 2, 1] + f0[2, 1, 1]) -
                       (f0[1, 2, 2] + f0[1, 1, 2] + f0[1, 2, 1] + f0[1, 1, 1])) / h0[1]
    fgrad[2] = 0.25 * ((f0[2, 2, 2] + f0[1, 2, 2] + f0[2, 2, 1] + f0[1, 2, 1]) -
                       (f0[2, 1, 2] + f0[1, 1, 2] + f0[2, 1, 1] + f0[1, 1, 1])) / h0[2]
    fgrad[3] = 0.25 * ((f0[2, 2, 2] + f0[2, 1, 2] + f0[1, 2, 2] + f0[1, 1, 2]) -
                       (f0[2, 2, 1] + f0[2, 1, 1] + f0[1, 2, 1] + f0[1, 1, 1])) / h0[3]
    fgradsq = fgrad[1]^2 + fgrad[2]^2 + fgrad[3]^2
    fgradmod = max(sqrt(fgradsq), MIN_GRAD)
    hm = maximum(hh)
    fth = sqrt(2.0) * fgradmod * hm

    if np0 * nm0 == 0
        # n0[a,b,c] = 1 where the corner is within the gradient band (|f0| ≤ fth),
        # else 0. Built as a stack-resident immutable SArray (no heap MArray); it is
        # only READ downstream (vofi_check_boundary_surface's per-corner write was a
        # dead store — each corner is visited once). ntuple(_, Val(8)) → const indices.
        n0 = SArray{Tuple{NSE, NSE, NSE}, vofi_int}(ntuple(Val(NSE * NSE * NSE)) do n
            a = (n - 1) % NSE
            b = ((n - 1) ÷ NSE) % NSE
            c = (n - 1) ÷ (NSE * NSE)
            @inbounds abs(f0[a + 1, b + 1, c + 1]) > fth ? vofi_int(0) : vofi_int(1)
        end)
        np0 = nm0 = 0
        for i in 0:1, j in 0:1, k in 0:1
            if abs(f0[i + 1, j + 1, k + 1]) > fth
                if f0[i + 1, j + 1, k + 1] < 0
                    nm0 += 1
                else
                    np0 += 1
                end
            end
        end
        if nm0 == nmax0
            return 1
        elseif np0 == nmax0
            return 0
        end
        check_dir = vofi_check_boundary_surface(ws, impl_func, par, x0, h0, f0, xfsp, n0)
        if check_dir < 0
            return nm0 > 0 ? 1 : 0
        end
    end

    # Build the 27-point stencil as a stack-resident immutable SArray (no heap
    # allocation; GPU-friendly). Entry (a,b,c) is the level set at
    # x0[d] + (idx_d - 1) * hh[d]; the 8 all-odd corners reuse f0 (already
    # evaluated above). `ntuple(_, Val(27))` unrolls so every index is constant.
    fc = SArray{Tuple{NDIM, NDIM, NDIM}, T}(ntuple(Val(NDIM * NDIM * NDIM)) do n
        a = (n - 1) % NDIM + 1
        b = ((n - 1) ÷ NDIM) % NDIM + 1
        c = (n - 1) ÷ (NDIM * NDIM) + 1
        if isodd(a) && isodd(b) && isodd(c)
            @inbounds f0[(a + 1) >> 1, (b + 1) >> 1, (c + 1) >> 1]
        else
            p = SVector{NDIM, T}(x0[1] + (a - 1) * hh[1],
                                 x0[2] + (b - 1) * hh[2],
                                 x0[3] + (c - 1) * hh[3])
            call_integrand(impl_func, par, p)
        end
    end)

    fgrad .= 0
    for i in 0:1, j in 0:1, k in 0:1
        fx = 0.25 * ((fc[i + 2, j + 2, k + 2] + fc[i + 2, j + 1, k + 2] +
                      fc[i + 2, j + 2, k + 1] + fc[i + 2, j + 1, k + 1]) -
                     (fc[i + 1, j + 2, k + 2] + fc[i + 1, j + 1, k + 2] +
                      fc[i + 1, j + 2, k + 1] + fc[i + 1, j + 1, k + 1])) / hh[1]
        fy = 0.25 * ((fc[i + 2, j + 2, k + 2] + fc[i + 1, j + 2, k + 2] +
                      fc[i + 2, j + 2, k + 1] + fc[i + 1, j + 2, k + 1]) -
                     (fc[i + 2, j + 1, k + 2] + fc[i + 1, j + 1, k + 2] +
                      fc[i + 2, j + 1, k + 1] + fc[i + 1, j + 1, k + 1])) / hh[2]
        fz = 0.25 * ((fc[i + 2, j + 2, k + 2] + fc[i + 2, j + 1, k + 2] +
                      fc[i + 1, j + 2, k + 2] + fc[i + 1, j + 1, k + 2]) -
                     (fc[i + 2, j + 2, k + 1] + fc[i + 2, j + 1, k + 1] +
                      fc[i + 1, j + 2, k + 1] + fc[i + 1, j + 1, k + 1])) / hh[3]
        tmp = fc[i + 2, j + 2, k + 2] + fc[i + 2, j + 1, k + 2] +
              fc[i + 1, j + 2, k + 2] + fc[i + 1, j + 1, k + 2] +
              fc[i + 2, j + 2, k + 1] + fc[i + 2, j + 1, k + 1] +
              fc[i + 1, j + 2, k + 1] + fc[i + 1, j + 1, k + 1]
        tmp = max(abs(tmp), MIN_GRAD)
        fgrad[1] += fx / tmp
        fgrad[2] += fy / tmp
        fgrad[3] += fz / tmp
    end
    @inbounds for ii in 1:NDIM; fgrad[ii] = abs(fgrad[ii]); end
    jp = 1
    js = 2
    jt = 3
    if fgrad[1] >= fgrad[2]
        jp = 1
        js = 2
    else
        jp = 2
        js = 1
    end
    if fgrad[3] > fgrad[jp]
        jt = js
        js = jp
        jp = 3
    elseif fgrad[3] > fgrad[js]
        jt = js
        js = 3
    end

    if check_dir == 1 && xfsp[jp].isc[1] != 1
        if xfsp[js].isc[1] == 1
            jp, js = js, jp
        else
            jt, js, jp = js, jp, jt
        end
    end

    zfill!(pdir, 0.0)
    zfill!(sdir, 0.0)
    zfill!(tdir, 0.0)
    pdir[jp] = 1.0
    sdir[js] = 1.0
    tdir[jt] = 1.0
    vofi_xyz2pst!(f0, jp, js, jt)

    if check_dir >= 0
        copy!(xfsp[5], xfsp[jp])
    else
        vofi_check_secter_face(ws, impl_func, par, x0, h0, pdir, sdir, tdir, f0, xfsp[5], fth)
    end
    vofi_check_tertiary_side(ws, impl_func, par, x0, h0, pdir, sdir, tdir, f0, xfsp, fth)

    have = 0.5 * (h0[jp] + h0[js])
    pd1, pd2, pd3 = Int(pdir[1]), Int(pdir[2]), Int(pdir[3])
    sd1, sd2, sd3 = Int(sdir[1]), Int(sdir[2]), Int(sdir[3])
    td1, td2, td3 = Int(tdir[1]), Int(tdir[2]), Int(tdir[3])
    # Accumulate the curvature estimate as running scalars; the per-direction
    # weight (sk) and curvature (ck) are consumed immediately, so no MVector
    # scratch is needed (was curv/sumf MVector{3}).
    sumf_curv = zero(T)
    sumf_total = zero(T)
    for k in 1:NDIM
        kz = k - 1
        i0 = kz * td1
        j0 = kz * td2
        k0 = kz * td3
        sk = zero(T)
        for i in 0:NDIM-1, j in 0:NDIM-1
            ii = i0 + i * sd1 + j * pd1
            jj = j0 + i * sd2 + j * pd2
            kk = k0 + i * sd3 + j * pd3
            fd[i + 1, j + 1] = fc[ii + 1, jj + 1, kk + 1]
            sk += fd[i + 1, j + 1]
        end
        sk = one(T) / max(abs(sk), EPS_NOT0)
        fx = (fd[3, 2] - fd[1, 2]) * have / h0[js]
        fy = (fd[2, 3] - fd[2, 1]) * have / h0[jp]
        fxx = (fd[3, 2] + fd[1, 2] - 2 * fd[2, 2]) * have^2 / (hh[js] * hh[js])
        fyy = (fd[2, 3] + fd[2, 1] - 2 * fd[2, 2]) * have^2 / (hh[jp] * hh[jp])
        fxy = (fd[3, 3] - fd[3, 1] - fd[1, 3] + fd[1, 1]) * have^2 / (h0[js] * h0[jp])
        tmp = sqrt((fx^2 + fy^2)^3)
        ck = (!isfinite(tmp) || tmp < EPS_NOT0) ? zero(T) :
             abs(fxx * fy^2 - 2 * fx * fy * fxy + fx^2 * fyy) / tmp
        sumf_curv += sk * ck
        sumf_total += sk
    end
    Kappa = sumf_curv / sumf_total
    a0, a1, a2, a3 = 2.34607, 16.5515, -5.53054, 54.0866
    tmp = a0 + Kappa * (a1 + Kappa * (a2 + a3 * Kappa))
    tmp = isfinite(tmp) ? tmp : a0
    tmp = clamp(tmp, 4.0, float(NGLM))
    npt = Int(ceil(tmp))
    xfsp[5].ipt = max(4, min(npt, NGLM))
    return icc
end
function vofi_xyz2pst!(g0, jp, js, jt)
    if jt == 3 && js == 1
        g0[1, 2, 1], g0[2, 1, 1] = g0[2, 1, 1], g0[1, 2, 1]
        g0[1, 2, 2], g0[2, 1, 2] = g0[2, 1, 2], g0[1, 2, 2]
    elseif jp == 3
        g0[1, 1, 2], g0[2, 1, 1] = g0[2, 1, 1], g0[1, 1, 2]
        g0[1, 2, 2], g0[2, 2, 1] = g0[2, 2, 1], g0[1, 2, 2]
        if jt == 2
            g0[1, 1, 2], g0[1, 2, 1] = g0[1, 2, 1], g0[1, 1, 2]
            g0[2, 1, 2], g0[2, 2, 1] = g0[2, 2, 1], g0[2, 1, 2]
        end
    elseif js == 3
        g0[1, 1, 2], g0[1, 2, 1] = g0[1, 2, 1], g0[1, 1, 2]
        g0[2, 1, 2], g0[2, 2, 1] = g0[2, 2, 1], g0[2, 1, 2]
        if jp == 2
            g0[1, 1, 2], g0[2, 1, 1] = g0[2, 1, 1], g0[1, 1, 2]
            g0[1, 2, 2], g0[2, 2, 1] = g0[2, 2, 1], g0[1, 2, 2]
        end
    end
end

function vofi_permute_hypercube!(f0::Array{T, 4}, perm::NTuple{4, Int}, tmp::Array{T, 4}) where {T}
    permutedims!(tmp, f0, perm)
    f0 .= tmp
    return nothing
end

@inline sign_indicator(val::Real) = val > 0 ? 1 : val < 0 ? -1 : 0

function reset_min_data4d!(md::MinData4D)
    zfill!(md.xval, 0.0)
    md.fval = 0.0
    md.sval = 0.0
    md.span = 0.0
    zfill!(md.isc, 0)
end

function reset_xfsp4d!(xfsp::XFSP4D)
    for md in xfsp.edges
        reset_min_data4d!(md)
    end
    for md in xfsp.sectors
        reset_min_data4d!(md)
    end
    xfsp.ipt = 0
end

@inline edge_index_4d(m, n, o) = 4 * m + 2 * n + o + 1

function fill_base_point!(dest, x0, hvec, pdir, sdir, tdir, m, n, o)
    for i in 1:4
        dest[i] = x0[i] + m * pdir[i] * hvec[i] +
                           n * sdir[i] * hvec[i] +
                           o * tdir[i] * hvec[i]
    end
    return dest
end

function evaluate_point_along!(dest, base, dir, s)
    for i in 1:length(base)
        dest[i] = base[i] + s * dir[i]
    end
    return dest
end

function find_quaternary_bracket(impl_func, par, base, dir, hlen, fstart, fend, x)
    T = eltype(base)
    samples = (0.25, 0.5, 0.75)
    prev_s = zero(T)
    prev_f = fstart
    for frac in samples
        s = frac * hlen
        evaluate_point_along!(x, base, dir, s)
        fval = call_integrand(impl_func, par, x)
        if prev_f * fval <= 0
            return prev_s, s, prev_f, fval
        end
        prev_s = s
        prev_f = fval
    end
    if prev_f * fend <= 0
        return prev_s, hlen, prev_f, fend
    end
    return nothing
end

function bisection_zero_along_dir(impl_func, par, base, dir, sa, sb, fa, fb, x, xroot)
    a = sa
    b = sb
    f_a = fa
    f_b = fb
    iter = 0
    while iter < MAX_ITER_ROOT && abs(b - a) > EPS_ROOT
        iter += 1
        mid = 0.5 * (a + b)
        evaluate_point_along!(x, base, dir, mid)
        f_mid = call_integrand(impl_func, par, x)
        if f_a * f_mid <= 0
            b = mid
            f_b = f_mid
        else
            a = mid
            f_a = f_mid
        end
    end
    s = 0.5 * (a + b)
    evaluate_point_along!(xroot, base, dir, s)
    fval = call_integrand(impl_func, par, xroot)
    return s, fval, xroot
end

function vofi_populate_sector_volume_4D!(ws::VofiWorkspace, xfsp::XFSP4D, impl_func, par, x0, hvec, pdir, sdir, tdir, qdir, f0, fth)
    base = ws.od4_psv_base
    for m in 0:1
        data = xfsp.sectors[m + 1]
        reset_min_data4d!(data)
        np = 0
        nm = 0
        small = false
        for n in 0:1, o in 0:1, q in 0:1
            val = f0[m + 1, n + 1, o + 1, q + 1]
            np += val > 0
            nm += val < 0
            small |= abs(val) <= fth
        end
        if (np == 0 || nm == 0) && !small
            continue
        end
        fill_base_point!(base, x0, hvec, pdir, sdir, tdir, m, 0, 0)
        for i in 1:4
            offset = 0.5 * (sdir[i] * hvec[i] + tdir[i] * hvec[i] + qdir[i] * hvec[i])
            data.xval[i] = base[i] + offset
        end
        data.fval = 0.0
        data.isc[1] = 1
        data.isc[m + 2] = 1
    end
end

function vofi_populate_quaternary_edges_4D!(ws::VofiWorkspace, xfsp::XFSP4D, impl_func, par, x0, hvec,
                                            pdir, sdir, tdir, qdir, f0, fth)
    qlen = axis_length(qdir, hvec)
    dir = ws.od4_pqe_dir
    for i in 1:4
        dir[i] = qdir[i]
    end
    base = ws.od4_pqe_base
    xbuf = ws.od4_pqe_x
    xroot = ws.od4_pqe_xroot
    for m in 0:1
        for n in 0:1
            for o in 0:1
                idx = edge_index_4d(m, n, o)
                data = xfsp.edges[idx]
                reset_min_data4d!(data)
                fill_base_point!(base, x0, hvec, pdir, sdir, tdir, m, n, o)
                data.span = qlen
                data.xval .= base
                fstart = f0[m + 1, n + 1, o + 1, 1]
                fend = f0[m + 1, n + 1, o + 1, 2]
                data.isc[3] = sign_indicator(fstart)
                data.isc[4] = sign_indicator(fend)
                if fstart * fend < 0
                    data.isc[1] = 1
                    data.isc[2] = -1
                    continue
                end
                if abs(fstart) > fth && abs(fend) > fth
                    continue
                end
                bracket = find_quaternary_bracket(impl_func, par, base, dir, qlen, fstart, fend, xbuf)
                if bracket === nothing
                    continue
                end
                sa, sb, fa, fb = bracket
                sroot, froot, xroot = bisection_zero_along_dir(impl_func, par, base, dir, sa, sb, fa, fb, xbuf, xroot)
                data.isc[1] = 1
                data.isc[2] = 1
                data.sval = sroot
                data.fval = froot
                data.xval .= xroot
            end
        end
    end
end

function vofi_order_dirs_4D(ws::VofiWorkspace, impl_func, par, x0::AbstractVector, h0, pdir, sdir, tdir, qdir,
                            f0::AbstractArray{<:Any, 4}, xfsp::XFSP4D)
    length(x0) == 4 || throw(ArgumentError("x0 must have length 4 for 4D cells"))
    length(h0) == 4 || throw(ArgumentError("h0 must have length 4 for 4D cells"))
    T = promote_type(eltype(x0), eltype(h0))
    hvec = ws.od4_hvec
    hh = ws.od4_hh
    for i in 1:4
        hvec[i] = T(h0[i])
        hh[i] = 0.5 * hvec[i]
    end
    MIN_GRAD = 1.0e-4
    nmax0 = 16
    np0 = 0
    nm0 = 0
    x = ws.od4_x
    for i in 0:1, j in 0:1, k in 0:1, l in 0:1
        x[1] = x0[1] + i * hvec[1]
        x[2] = x0[2] + j * hvec[2]
        x[3] = x0[3] + k * hvec[3]
        x[4] = x0[4] + l * hvec[4]
        val = call_integrand(impl_func, par, x)
        f0[i + 1, j + 1, k + 1, l + 1] = val
        if val > 0
            np0 += 1
        elseif val < 0
            nm0 += 1
        end
    end

    fgrad = ws.od4_fgrad
    fgrad[1] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[2, 2, 1, 2] + f0[2, 1, 1, 2] +
                         f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[2, 2, 1, 1] + f0[2, 1, 1, 1]) -
                        (f0[1, 2, 2, 2] + f0[1, 1, 2, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2] +
                         f0[1, 2, 2, 1] + f0[1, 1, 2, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / hvec[1]
    fgrad[2] = 0.125 * ((f0[2, 2, 2, 2] + f0[1, 2, 2, 2] + f0[2, 2, 1, 2] + f0[1, 2, 1, 2] +
                         f0[2, 2, 2, 1] + f0[1, 2, 2, 1] + f0[2, 2, 1, 1] + f0[1, 2, 1, 1]) -
                        (f0[2, 1, 2, 2] + f0[1, 1, 2, 2] + f0[2, 1, 1, 2] + f0[1, 1, 1, 2] +
                         f0[2, 1, 2, 1] + f0[1, 1, 2, 1] + f0[2, 1, 1, 1] + f0[1, 1, 1, 1])) / hvec[2]
    fgrad[3] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[1, 2, 2, 2] + f0[1, 1, 2, 2] +
                         f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[1, 2, 2, 1] + f0[1, 1, 2, 1]) -
                        (f0[2, 2, 1, 2] + f0[2, 1, 1, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2] +
                         f0[2, 2, 1, 1] + f0[2, 1, 1, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / hvec[3]
    fgrad[4] = 0.125 * ((f0[2, 2, 2, 2] + f0[2, 1, 2, 2] + f0[2, 2, 1, 2] + f0[2, 1, 1, 2] +
                         f0[1, 2, 2, 2] + f0[1, 1, 2, 2] + f0[1, 2, 1, 2] + f0[1, 1, 1, 2]) -
                        (f0[2, 2, 2, 1] + f0[2, 1, 2, 1] + f0[2, 2, 1, 1] + f0[2, 1, 1, 1] +
                         f0[1, 2, 2, 1] + f0[1, 1, 2, 1] + f0[1, 2, 1, 1] + f0[1, 1, 1, 1])) / hvec[4]
    grad_sq = zero(T)
    for comp in fgrad
        grad_sq += comp * comp
    end
    fgradmod = max(sqrt(grad_sq), MIN_GRAD)
    fth = sqrt(3.0) * fgradmod * maximum(hh)

    check_dir = -1
    if np0 * nm0 == 0
        n0 = ws.od4_n0; zfill!(n0, 0)
        np0 = 0
        nm0 = 0
        for i in 0:1, j in 0:1, k in 0:1, l in 0:1
            val = abs(f0[i + 1, j + 1, k + 1, l + 1])
            if val > fth
                n0[i + 1, j + 1, k + 1, l + 1] = 0
                if f0[i + 1, j + 1, k + 1, l + 1] < 0
                    nm0 += 1
                else
                    np0 += 1
                end
            else
                n0[i + 1, j + 1, k + 1, l + 1] = 1
            end
        end
        if nm0 == nmax0
            return 1
        elseif np0 == nmax0
            return 0
        end
        check_dir = vofi_check_boundary_hypersurface(ws, impl_func, par, x0, hvec, f0, xfsp, n0)
        if check_dir < 0
            return nm0 > 0 ? 1 : 0
        end
    end

    mags = ws.od4_mags
    order = ws.od4_order
    for i in 1:4
        mags[i] = abs(fgrad[i])
        order[i] = i
    end
    sortperm!(order, mags, rev = true)
    if check_dir > 0
        idx = findfirst(==(check_dir), order)
        if idx !== nothing && idx != 1
            order[1], order[idx] = order[idx], order[1]
        end
    end
    zfill!(pdir, 0.0)
    zfill!(sdir, 0.0)
    zfill!(tdir, 0.0)
    zfill!(qdir, 0.0)
    pdir[order[1]] = 1.0
    sdir[order[2]] = 1.0
    tdir[order[3]] = 1.0
    qdir[order[4]] = 1.0
    vofi_permute_hypercube!(f0, (order[1], order[2], order[3], order[4]), ws.od4_fperm)

    reset_xfsp4d!(xfsp)
    vofi_populate_sector_volume_4D!(ws, xfsp, impl_func, par, x0, hvec, pdir, sdir, tdir, qdir, f0, fth)
    vofi_populate_quaternary_edges_4D!(ws, xfsp, impl_func, par, x0, hvec, pdir, sdir, tdir, qdir, f0, fth)
    have = maximum(hvec)
    Kappa = fgradmod / max(have, EPS_NOT0)
    a0, a1, a2, a3 = 2.34607, 16.5515, -5.53054, 54.0866
    npt_est = a0 + Kappa * (a1 + Kappa * (a2 + a3 * Kappa))
    npt = clamp(Int(ceil(npt_est)), 4, NGLM)
    xfsp.ipt = npt

    return -1
end
