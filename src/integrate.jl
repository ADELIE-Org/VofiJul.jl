function vofi_get_length_1D(impl_func, par, x0, h0, f0, xex, ncen)
    # For 1D, we just need to find the zero crossing
    # f0 has two values: f0[1] at x0[1] and f0[2] at x0[1] + h0[1]
    
    # If both have same sign, something went wrong - should have been caught earlier
    T = promote_type(eltype(x0), eltype(h0), eltype(f0))
    if f0[1] * f0[2] >= 0
        # No zero crossing - return full length if negative, 0 otherwise
        if f0[1] < 0 && f0[2] < 0
            if ncen > 0
                xex[1] = x0[1] + 0.5 * h0[1]
            end
            return T(h0[1])
        else
            return zero(T)
        end
    end
    
    # Find the zero crossing using linear interpolation as initial guess
    # then refine with Newton-Raphson
    denom = abs(f0[1]) + abs(f0[2])
    if denom < EPS_NOT0
        # Both values very close to zero - use midpoint
        frac = 0.5
    else
        frac = abs(f0[1]) / denom
    end
    x_zero = T(x0[1] + frac * h0[1])

    # Refine the zero crossing location
    x1 = zero(MVector{NDIM, T})
    x1[2] = x0[2]
    x1[3] = x0[3]
    
    # Newton-Raphson iteration
    for iter in 1:MAX_ITER_ROOT
        x1[1] = x_zero
        f_val = call_integrand(impl_func, par, x1)
        
        if abs(f_val) < EPS_ROOT
            break
        end
        
        # Compute numerical derivative
        h_eps = EPS_LOC * h0[1]
        x1[1] = x_zero + h_eps
        f_plus = call_integrand(impl_func, par, x1)
        f_deriv = (f_plus - f_val) / h_eps
        
        if abs(f_deriv) < EPS_NOT0
            break
        end
        
        # Newton step
        delta = -f_val / f_deriv
        x_zero += delta
        
        # Keep within bounds
        x_zero = max(x0[1], min(x0[1] + h0[1], x_zero))
        
        if abs(delta) < EPS_ROOT * h0[1]
            break
        end
    end
    
    # The length inside (negative region) is from x0[1] to x_zero if f0[1] < 0
    # or from x_zero to x0[1] + h0[1] if f0[2] < 0
    if f0[1] < 0
        length_inside = x_zero - x0[1]
        if ncen > 0
            # Centroid is at the midpoint of the negative region
            xex[1] = x0[1] + 0.5 * length_inside
        end
    else
        length_inside = (x0[1] + h0[1]) - x_zero
        if ncen > 0
            # Centroid is at the midpoint of the negative region
            xex[1] = x_zero + 0.5 * length_inside
        end
    end
    
    return length_inside
end

function vofi_get_area(ws::VofiWorkspace, impl_func, par, x0, h0, base, pdir, sdir, xhp, centroid, ncen, npt, nsub, nptmp, nsect, ndire)
    T = promote_type(eltype(x0), eltype(h0))
    x1 = ws.ga_x1
    x20 = ws.ga_x20
    x21 = ws.ga_x21
    s0 = ws.ga_s0
    fse = ws.ga_fse
    area = zero(T)
    hp = zero(T)
    hs = zero(T)
    for i in 1:NDIM
        x1[i] = x0[i] + pdir[i] * h0[i]
        hp += pdir[i] * h0[i]
        hs += sdir[i] * h0[i]
    end
    hm = maximum(h0)
    xp = zero(T)
    xs = zero(T)
    it0 = 1
    max_sections = min(length(xhp), length(npt))
    for ns in 1:nsub
        ds = base[ns + 1] - base[ns]
        mdpt = 0.5 * (base[ns + 1] + base[ns])
        if nsect[ns] > 0
            al = ds * hp
            area += al
            if ncen > 0
                xp += 0.5 * hp * al
                xs += mdpt * al
            end
        elseif nsect[ns] < 0
            if it0 > max_sections
                break  # no storage/point-count info for additional sections
            end
            npts = Int(clamp(floor(18 * ds / hm) + 3, 3, 20))
            npts = min(nptmp, npts)
            if 3 <= npt[2] <= 20
                npts = min(npt[2], npts)
            end
            if 3 <= npt[1] <= 20
                npts = max(npt[1], npts)
            end
            xhp[it0].np0 = npts
            f_sign = ndire[ns]
            xhp[it0].f_sign = f_sign
            j = npts - 3
            pts = gauss_legendre_nodes(j + 3)
            # GL tables are Float64 constants; convert into the working type so
            # reduced precision / AD duals propagate through the quadrature.
            # `convert` is an identity (no-op, same object) when T === Float64, so
            # the production hot path is numerically and allocation-wise identical.
            wts = convert(SVector{GL_MAX_ORDER, T}, gauss_legendre_weights(j + 3))

            quada = zero(T)
            quadp = zero(T)
            quads = zero(T)
            a1 = a2 = b1 = b2 = zero(T)
            xhp[it0].ht0[1] = xhp[it0].htp[1] = zero(T)
            xhp[it0].xt0[1] = base[ns]
            xhp[it0].xt0[npts + 2] = base[ns + 1]
            xhp[it0].xt0[2] = mdpt + 0.5 * ds * pts[1]
            for i in 1:NDIM
                tmp = sdir[i] * xhp[it0].xt0[2]
                x20[i] = x0[i] + tmp
                x21[i] = x1[i] + tmp
            end
            fse[1] = call_integrand(impl_func, par, x20)
            fse[2] = call_integrand(impl_func, par, x21)
            s0[1] = hp
            if abs(fse[1]) < abs(fse[2])
                s0[2] = 0.0
                s0[3] = fse[1]
            else
                s0[2] = hp
                s0[3] = fse[2]
            end
            s0[4] = (fse[2] - fse[1]) / hp
            for k in 1:npts
                xhp[it0].ht0[k + 1] = vofi_get_segment_zero(ws, impl_func, par, x20, pdir, s0, f_sign)
                xhp[it0].htp[k + 1] = s0[4]
                quada += wts[k] * xhp[it0].ht0[k + 1]
                if ncen > 0
                    quadp += wts[k] * 0.5 * xhp[it0].ht0[k + 1]^2
                    quads += wts[k] * xhp[it0].ht0[k + 1] * xhp[it0].xt0[k + 1]
                end
                if k < npts
                    xhp[it0].xt0[k + 2] = mdpt + 0.5 * ds * pts[k + 1]
                    s0[2] = xhp[it0].ht0[k + 1]
                    s0[4] = xhp[it0].htp[k + 1]
                    if k > 1
                        dxm1 = xhp[it0].xt0[k + 1] - xhp[it0].xt0[k]
                        dxp1 = xhp[it0].xt0[k + 2] - xhp[it0].xt0[k + 1]
                        a1 = (xhp[it0].ht0[k + 1] - xhp[it0].ht0[k]) / dxm1
                        s0[2] += a1 * dxp1
                        b1 = (xhp[it0].htp[k + 1] - xhp[it0].htp[k]) / dxm1
                        s0[4] += b1 * dxp1
                        if k > 2
                            dxm2 = xhp[it0].xt0[k + 1] - xhp[it0].xt0[k - 1]
                            dxp2 = xhp[it0].xt0[k + 2] - xhp[it0].xt0[k]
                            s0[2] += (a1 - a2) * dxp1 * dxp2 / dxm2
                            s0[4] += (b1 - b2) * dxp1 * dxp2 / dxm2
                        end
                    end
                    if f_sign < 0
                        s0[2] = hp - s0[2]
                    end
                    ratio = s0[2] / hp
                    if ratio < NEAR_EDGE_RATIO
                        s0[2] = 0.0
                    elseif ratio > 1 - NEAR_EDGE_RATIO
                        s0[2] = hp
                    end
                    for i in 1:NDIM
                        x20[i] = x0[i] + sdir[i] * xhp[it0].xt0[k + 2]
                        x21[i] = x20[i] + pdir[i] * s0[2]
                    end
                    s0[3] = call_integrand(impl_func, par, x21)
                end
                a2 = a1
                b2 = b1
            end
            quada *= 0.5 * ds
            area += quada
            if ncen > 0 && quada > 0
                quadp = 0.5 * ds * quadp / quada
                quads = 0.5 * ds * quads / quada
                if f_sign < 0
                    quadp = hp - quadp
                end
                xp += quadp * quada
                xs += quads * quada
            end
            it0 += 1
        end
    end
    centroid[1] = xp
    centroid[2] = xs
    return T(area)
end

function vofi_get_volume(ws::VofiWorkspace, impl_func, par, x0, h0, base_ext, pdir, sdir, tdir,
                         centroid, nex, npt, nsub_ext, nptmp, nvis)
    T = promote_type(eltype(x0), eltype(h0))
    x1 = ws.gv_x1
    base_int = ws.gv_base_int
    xmidt = ws.gv_xmidt
    volume = zero(T)
    surfer = zero(T)
    hp = hs = ht = zero(T)
    for i in 1:NDIM
        hp += pdir[i] * h0[i]
        hs += sdir[i] * h0[i]
        ht += tdir[i] * h0[i]
    end
    hm = maximum(h0)
    xp = xs = xt = zero(T)
    # Scratch structs reused across loop iterations. Each is reset (to the exact
    # state of a fresh constructor) at the point where the original code rebuilt
    # it, preserving the per-iteration cadence and aliasing.
    xhpn1 = ws.gv_xhpn1
    xhpn2 = ws.gv_xhpn2
    xhpo1 = ws.gv_xhpo1
    xhpo2 = ws.gv_xhpo2
    xhpn_edge1 = ws.gv_xhpn_edge1
    xhpn_edge2 = ws.gv_xhpn_edge2
    xfs = ws.gv_xfs
    nsect = ws.gv_nsect
    ndire = ws.gv_ndire

    for nt in 1:nsub_ext
        dt = base_ext[nt + 1] - base_ext[nt]
        mdpt = 0.5 * (base_ext[nt + 1] + base_ext[nt])
        for i in 1:NDIM
            x1[i] = x0[i] + tdir[i] * mdpt
            xfs.isc[i] = 0
        end
        sect_hexa = vofi_check_plane(ws, impl_func, par, x1, h0, xfs, base_int, pdir, sdir,
                                     nsect, ndire)
        if sect_hexa == 0
            if nsect[1] == 1
                vol = dt * hs * hp
                volume += vol
                if nex[1] > 0
                    xp += 0.5 * hp * vol
                    xs += 0.5 * hs * vol
                    xt += mdpt * vol
                end
            end
            continue
        end

        dt_scaled = 18 * dt / hm
        if !isfinite(dt_scaled) || dt_scaled < 0
            dt_scaled = 0.0
        end
        nexpt = min(20, Int(floor(dt_scaled)) + 3)
        if length(npt) >= 4 && 3 <= npt[4] <= 20
            nexpt = min(npt[4], nexpt)
        end
        if length(npt) >= 3 && 3 <= npt[3] <= 20
            nexpt = max(npt[3], nexpt)
        end
        ptx_ext = gauss_legendre_nodes(nexpt)
        ptw_ext = convert(SVector{GL_MAX_ORDER, T}, gauss_legendre_weights(nexpt))

        quadv = quadp = quads = quadt = zero(T)
        reset!(xhpo1)
        reset!(xhpo2)
        reset!(xhpn_edge1)
        reset!(xhpn_edge2)
        xmidt[1] = base_ext[nt]
        xmidt[nexpt + 2] = base_ext[nt + 1]
        for k in 1:nexpt
            xit = mdpt + 0.5 * dt * ptx_ext[k]
            xmidt[k + 1] = xit
            for i in 1:NDIM
                x1[i] = x0[i] + tdir[i] * xit
            end
            nsub_int = vofi_get_limits_inner_2D(ws, impl_func, par, x1, h0, xfs, base_int,
                                                pdir, sdir, nsect, ndire, sect_hexa)
            # Reset xhpn structures for this iteration
            reset!(xhpn1)
            reset!(xhpn2)
            area = vofi_get_area(ws, impl_func, par, x1, h0, base_int, pdir, sdir, (xhpn1, xhpn2),
                                 centroid, nex[1], npt, nsub_int, nptmp, nsect, ndire)
            if nvis[1] > 0
                tecplot_heights(x1, h0, pdir, sdir, (xhpn1, xhpn2))
            end
            if nex[2] > 0
                vofi_end_points(ws, impl_func, par, x1, h0, pdir, sdir, (xhpn1, xhpn2))
                if k == 1
                    xedge = ws.gv_xedge
                    for i in 1:NDIM
                        xedge[i] = x0[i] + tdir[i] * xmidt[1]
                    end
                    nintmp = vofi_get_limits_edge_2D(ws, impl_func, par, xedge, h0, xfs,
                                                     base_int, pdir, sdir, nsub_int)
                    vofi_edge_points(ws, impl_func, par, xedge, h0, base_int, pdir, sdir,
                                     (xhpo1, xhpo2), (xhpn1.np0, xhpn2.np0), nintmp, nsect, ndire)
                    vofi_end_points(ws, impl_func, par, xedge, h0, pdir, sdir, (xhpo1, xhpo2))
                elseif k > 1 && k < nexpt
                    surfer += vofi_interface_surface(ws, impl_func, par, x0, h0, xmidt, pdir,
                                                     sdir, tdir, (xhpn1, xhpn2), (xhpo1, xhpo2), k, nexpt, nvis[2])
                    copy!(xhpo1, xhpn1)
                    copy!(xhpo2, xhpn2)
                else
                    xedge = ws.gv_xedge
                    for i in 1:NDIM
                        xedge[i] = x0[i] + tdir[i] * xmidt[nexpt + 2]
                    end
                    nintmp = vofi_get_limits_edge_2D(ws, impl_func, par, xedge, h0, xfs,
                                                     base_int, pdir, sdir, nsub_int)
                    vofi_edge_points(ws, impl_func, par, xedge, h0, base_int, pdir, sdir,
                                     (xhpn_edge1, xhpn_edge2), (xhpo1.np0, xhpo2.np0), nintmp, nsect, ndire)
                    vofi_end_points(ws, impl_func, par, xedge, h0, pdir, sdir, (xhpn_edge1, xhpn_edge2))
                    surfer += vofi_interface_surface(ws, impl_func, par, x0, h0, xmidt, pdir,
                                                     sdir, tdir, (xhpn_edge1, xhpn_edge2), (xhpo1, xhpo2), k + 1, nexpt, nvis[2])
                end
            end
            quadv += ptw_ext[k] * area
            quadp += ptw_ext[k] * centroid[1]
            quads += ptw_ext[k] * centroid[2]
            quadt += ptw_ext[k] * area * xit
        end
        quadv *= 0.5 * dt
        volume += quadv
        if nex[1] > 0
            xp += 0.5 * dt * quadp
            xs += 0.5 * dt * quads
            xt += 0.5 * dt * quadt
        end
    end

    centroid[1] = xp
    centroid[2] = xs
    centroid[3] = xt
    centroid[4] = surfer
    return T(volume)
end

# Callable wrapper for the 4D→3D slice integrand. Holds its mutable scratch
# (`xbuf`) and the current q-coordinate (`q_current`), so it is allocated once and
# reused instead of building a fresh closure (which heap-allocates) per call.
mutable struct SliceFunc4D{F, P, V, Q}
    impl_func::F
    par::P
    x0::V
    xbuf::V
    ax_p::Int
    ax_s::Int
    ax_t::Int
    ax_q::Int
    q_current::Q
end

@inline function (sf::SliceFunc4D)(coords)
    x0 = sf.x0
    xbuf = sf.xbuf
    for i in 1:length(x0)
        xbuf[i] = x0[i]
    end
    xbuf[sf.ax_p] = coords[1]
    xbuf[sf.ax_s] = coords[2]
    xbuf[sf.ax_t] = coords[3]
    xbuf[sf.ax_q] = sf.q_current
    return call_integrand(sf.impl_func, sf.par, xbuf)
end

function vofi_get_hypervolume(ws::VofiWorkspace, impl_func, par, x0, h0, base, pdir, sdir, tdir, qdir,
                              centroid, nex, npt, nsub, nptmp, nvis)
    T = promote_type(eltype(x0), eltype(h0))
    ax_p = axis_index(pdir)
    ax_s = axis_index(sdir)
    ax_t = axis_index(tdir)
    ax_q = axis_index(qdir)
    hp = axis_length(pdir, h0)
    hs = axis_length(sdir, h0)
    ht = axis_length(tdir, h0)
    hq = axis_length(qdir, h0)
    prod3 = hp * hs * ht
    hm = maximum(h0)

    xin3 = ws.gh_xin3
    xin3[1] = x0[ax_p]
    xin3[2] = x0[ax_s]
    xin3[3] = x0[ax_t]
    h3 = ws.gh_h3
    h3[1] = hp
    h3[2] = hs
    h3[3] = ht
    xex3 = ws.gh_xex3
    nex_slice = ws.gh_nex_slice; zfill!(nex_slice, 0)
    want_centroid = nex[1] > 0
    want_surface = (length(nex) >= 2) && nex[2] > 0
    if want_centroid
        nex_slice[1] = 1
    end
    if want_surface
        nex_slice[2] = 1
    end
    nvis_slice = ws.gh_nvis_slice; zfill!(nvis_slice, 0)

    xbuf = ws.gh_xbuf
    for i in 1:length(x0)
        xbuf[i] = x0[i]
    end
    slice_func = SliceFunc4D(impl_func, par, x0, xbuf, ax_p, ax_s, ax_t, ax_q, x0[ax_q])
    # Resolve the slice integrand's calling convention ONCE (it is 1-arg). Passing
    # the raw callable would make the nested `vofi_get_cc` re-run `applicable`
    # reflection (heap-allocating) on every quadrature node.
    slice_ic = IntegrandCall(slice_func, nothing, true)

    hypervolume = zero(T)
    xp_acc = xs_acc = xt_acc = xq_acc = zero(T)
    surface_acc = zero(T)
    q_origin = x0[ax_q]
    max_nodes = NGLM

    for ns in 1:nsub
        dq = base[ns + 1] - base[ns]
        if dq <= EPS_LOC
            continue
        end
        mdpt = 0.5 * (base[ns + 1] + base[ns])
        nquad = clamp(Int(floor(18 * dq / hm)) + 3, 3, 20)
        if length(npt) >= 4 && 3 <= npt[4] <= 20
            nquad = min(nquad, npt[4])
        end
        if nptmp > 0
            nquad = min(nquad, nptmp)
        end
        nquad = min(nquad, max_nodes)
        nodes = convert(SVector{GL_MAX_ORDER, T}, gauss_legendre_nodes(nquad))
        weights = convert(SVector{GL_MAX_ORDER, T}, gauss_legendre_weights(nquad))
        seg_vol = seg_xp = seg_xs = seg_xt = seg_xq = seg_surface = zero(T)
        for k in 1:nquad
            xi = mdpt + 0.5 * dq * nodes[k]
            xi = clamp(xi, zero(T), hq)
            q_abs = q_origin + xi
            slice_func.q_current = q_abs
            zfill!(xex3, 0.0)
            # Type assertion breaks the vofi_get_cc → vofi_get_hypervolume →
            # vofi_get_cc mutual-recursion inference cycle that otherwise infers
            # `Any` and de-specialises the whole 4D path.
            cc = vofi_get_cc(ws, slice_ic, nothing, xin3, h3, xex3, nex_slice, npt, nvis_slice, 3)::T
            vol3 = cc * prod3
            w = weights[k]
            seg_vol += w * vol3
            if want_centroid && vol3 > 0
                seg_xp += w * vol3 * xex3[1]
                seg_xs += w * vol3 * xex3[2]
                seg_xt += w * vol3 * xex3[3]
                seg_xq += w * vol3 * q_abs
            end
            if want_surface && nex_slice[2] > 0
                seg_surface += w * xex3[end]
            end
        end
        factor = 0.5 * dq
        hypervolume += factor * seg_vol
        if want_centroid
            xp_acc += factor * seg_xp
            xs_acc += factor * seg_xs
            xt_acc += factor * seg_xt
            xq_acc += factor * seg_xq
        end
        if want_surface
            surface_acc += factor * seg_surface
        end
    end

    centroid[1] = xp_acc
    centroid[2] = xs_acc
    centroid[3] = xt_acc
    centroid[4] = xq_acc
    centroid[5] = surface_acc
    return T(hypervolume)
end
