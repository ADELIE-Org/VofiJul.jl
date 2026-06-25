"""
    vofi_get_cc(impl_func, par, xin, h0, xex, nex, npt, nvis, ndim0) -> cc
    vofi_get_cc(ws, impl_func, par, xin, h0, xex, nex, npt, nvis, ndim0) -> cc

Volume fraction of the cut cell. The first form allocates a fresh
[`VofiWorkspace`](@ref) per call; the second reuses a caller-supplied workspace so
that batch drivers can drive per-cell allocation to ~zero. Public signature and
results are unchanged between the two forms.
"""
function vofi_get_cc(impl_func, par, xin, h0, xex, nex, npt, nvis, ndim0)
    T = promote_type(eltype(xin), eltype(h0))
    ws = _thread_workspace(T)
    return vofi_get_cc(ws, impl_func, par, xin, h0, xex, nex, npt, nvis, ndim0)
end

function vofi_get_cc(ws::VofiWorkspace, impl_func, par, xin, h0, xex, nex, npt, nvis, ndim0)
    # Resolve the integrand calling convention once (avoids per-evaluation
    # `applicable` reflection in the inner quadrature loops). Concrete type, so
    # all downstream calls stay type-stable.
    impl_func = wrap_integrand(impl_func, par, xin)
    # Element type for all scratch, derived from the geometry only (no integrand
    # sample, so the Float64 production path pays nothing). A Dual flows when you
    # differentiate w.r.t. the cell geometry (xin/h0) — the natural shape
    # sensitivity — and Float32 flows from Float32 geometry. (Differentiating
    # w.r.t. a level-set *parameter* is the job of the custom adjoint rule.)
    T = promote_type(eltype(xin), eltype(h0))
    # Ensure we have enough slots for barycenter coords plus interface measure
    nex_interface = length(nex) >= 2 ? nex[2] : 0
    required_len = (ndim0 == 4 && nex_interface > 0) ? 5 : max(4, ndim0)
    if length(xex) < required_len
        resize!(xex, required_len)
    end
    zfill!(xex, 0)
    x0 = ws.cc_x0
    # Reuse the workspace buffer instead of allocating a fresh padded vector.
    hvec = ws.cc_hvec
    zfill!(hvec, 0)
    @inbounds for i in 1:min(length(h0), NDIM)
        hvec[i] = T(h0[i])
    end
    if ndim0 == 1
        x0[1] = xin[1]
        x0[2] = 0
        x0[3] = 0
        f0_1D = ws.cc_f0_1D
        icc = vofi_order_dirs_1D(impl_func, par, x0, hvec, f0_1D)
        if icc >= 0
            cc = T(icc)
            if icc > 0 && nex[1] > 0
                xex[1] = x0[1] + 0.5 * hvec[1]
            end
            return cc
        end
        # Interface crosses the cell - compute the fraction
        length_frac = vofi_get_length_1D(impl_func, par, x0, hvec, f0_1D, xex, nex[1])
        cc = length_frac / hvec[1]
        return cc
    elseif ndim0 == 2
        x0[1] = xin[1]
        x0[2] = xin[2]
        xfsp_single = ws.cc_xfsp1; reset!(xfsp_single)  # Only need one for 2D
        pdir = ws.cc_pdir
        sdir = ws.cc_sdir
        f02D = ws.cc_f02D
        icc = vofi_order_dirs_2D(ws, impl_func, par, x0, hvec, pdir, sdir, f02D, xfsp_single)
        if icc >= 0
            cc = T(icc)
            if icc > 0 && nex[1] > 0
                for i in 1:NSE
                    xex[i] = x0[i] + 0.5 * hvec[i]
                end
            end
            return cc
        end
        base = ws.cc_base
        nsect = ws.cc_nsect
        ndire = ws.cc_ndire
        nsub = vofi_get_limits_2D(ws, impl_func, par, x0, hvec, f02D, xfsp_single, base,
                                  pdir, sdir, nsect, ndire)
        centroid = ws.cc_centroid
        xhp1 = ws.cc_xhp1; reset!(xhp1)
        xhp2 = ws.cc_xhp2; reset!(xhp2)
        area = vofi_get_area(ws, impl_func, par, x0, hvec, base, pdir, sdir, (xhp1, xhp2),
                             centroid, nex[1], npt, nsub, xfsp_single.ipt, nsect, ndire)
        cc = area / (hvec[1] * hvec[2])
        if nvis[1] > 0
            tecplot_heights(x0, hvec, pdir, sdir, (xhp1, xhp2))
        end
        if nex[1] > 0 && area > 0
            centroid[1] /= area
            centroid[2] /= area
            centroid[3] = zero(T)
            for i in 1:2
                xex[i] = x0[i] + centroid[1] * pdir[i] + centroid[2] * sdir[i]
            end
        end
        if nex[2] > 0
            xex[4] = vofi_interface_length(ws, impl_func, par, x0, hvec, pdir, sdir, (xhp1, xhp2), nvis[2])
        end
        return cc
    elseif ndim0 == 3
        x0[1] = xin[1]
        x0[2] = xin[2]
        x0[3] = xin[3]
        pdir = ws.cc_pdir
        sdir = ws.cc_sdir
        tdir = ws.cc_tdir
        f03D = ws.cc_f03D
        # Pre-allocate 5 MinData structs without array allocation
        xfsp1 = ws.cc_xfsp1; reset!(xfsp1)
        xfsp2 = ws.cc_xfsp2; reset!(xfsp2)
        xfsp3 = ws.cc_xfsp3; reset!(xfsp3)
        xfsp4 = ws.cc_xfsp4; reset!(xfsp4)
        xfsp5 = ws.cc_xfsp5; reset!(xfsp5)
        xfsp = (xfsp1, xfsp2, xfsp3, xfsp4, xfsp5)

        icc = vofi_order_dirs_3D(ws, impl_func, par, x0, hvec, pdir, sdir, tdir, f03D, xfsp)
        if icc >= 0
            cc = T(icc)
            if icc > 0 && nex[1] > 0
                for i in 1:NDIM
                    xex[i] = x0[i] + 0.5 * hvec[i]
                end
            end
            return cc
        end

        base = ws.cc_base
        nsub = vofi_get_limits_3D(ws, impl_func, par, x0, hvec, f03D, xfsp, base, pdir, sdir, tdir)
        centroid = ws.cc_centroid
        volume = vofi_get_volume(ws, impl_func, par, x0, hvec, base, pdir, sdir, tdir, centroid,
                                 nex, npt, nsub, xfsp[5].ipt, nvis)
        cc = volume / (hvec[1] * hvec[2] * hvec[3])
        if nex[1] > 0 && volume > 0
            centroid[1] /= volume
            centroid[2] /= volume
            centroid[3] /= volume
            for i in 1:NDIM
                xex[i] = x0[i] + centroid[1] * pdir[i] + centroid[2] * sdir[i] + centroid[3] * tdir[i]
            end
        end
        if nex[2] > 0
            xex[4] = centroid[4]
        end
        return cc
    elseif ndim0 == 4
        # Use dynamic arrays for 4D (NDIM is 3; do not change it). All scratch
        # comes from the reusable workspace; per-cell input/output buffers are
        # simply overwritten in full each call.
        length(h0) >= 4 || throw(ArgumentError("h0 must provide 4 entries when ndim0 == 4"))
        x0_4 = ws.cc4_x0
        h4 = ws.cc4_h4
        for i in 1:4
            x0_4[i] = xin[i]
            h4[i] = T(h0[i])
        end
        pdir = ws.cc4_pdir
        sdir = ws.cc4_sdir
        tdir = ws.cc4_tdir
        qdir = ws.cc4_qdir
        f04D = ws.cc4_f0
        xfsp = ws.cc4_xfsp

        icc = vofi_order_dirs_4D(ws, impl_func, par, x0_4, h4, pdir, sdir, tdir, qdir, f04D, xfsp)
        if icc >= 0
            cc = T(icc)
            if icc > 0 && nex[1] > 0
                for i in 1:4
                    xex[i] = x0_4[i] + 0.5 * h4[i]
                end
            end
            return cc
        end

        base = ws.cc4_base
        nsub = vofi_get_limits_4D(ws, impl_func, par, x0_4, h4, f04D, xfsp, base, pdir, sdir, tdir, qdir)
        centroid = ws.cc4_centroid
        hypervolume = vofi_get_hypervolume(ws, impl_func, par, x0_4, h4, base,
                          pdir, sdir, tdir, qdir,
                          centroid, nex, npt, nsub,
                          xfsp.ipt, nvis)
        cc = hypervolume / (h4[1] * h4[2] * h4[3] * h4[4])
        if nex[1] > 0 && hypervolume > 0
            for k in 1:4
                centroid[k] /= hypervolume
            end
            # centroid[1..4] are absolute coordinates in permuted (p,s,t,q) space
            # Map them back to original coordinate indices
            ax_p = axis_index(pdir)
            ax_s = axis_index(sdir)
            ax_t = axis_index(tdir)
            ax_q = axis_index(qdir)
            xex[ax_p] = centroid[1]
            xex[ax_s] = centroid[2]
            xex[ax_t] = centroid[3]
            xex[ax_q] = centroid[4]
        end
        if nex[2] > 0
            xex[5] = centroid[5]
        end
        return cc
    else
        throw(ArgumentError("ndim0 must be 1, 2, 3, or 4"))
    end
end
