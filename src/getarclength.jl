function vofi_interface_length(ws::VofiWorkspace, impl_func, par, x0, h0, pdir, sdir, xhhp, ipf)
    T = promote_type(eltype(x0), eltype(h0))
    hp = zero(T)
    for i in 1:NDIM
        hp += pdir[i] * h0[i]
    end
    arc = zero(T)
    s0 = ws.il_s0
    s0[1] = hp
    nseg = xhhp[min(end, 2)].np0 > 0 ? 2 : (xhhp[1].np0 > 0 ? 1 : 0)
    x20 = ws.il_x20
    x21 = ws.il_x21
    sqrt3 = sqrt(3.0)

    for it0 in 1:nseg
        seg = xhhp[it0]
        npt = seg.np0
        npt > 0 || continue
        f_sign = seg.f_sign
        xt0 = seg.xt0
        ht0 = seg.ht0
        htp = seg.htp
        idx(i) = i + 1

        j0, j1, j2, j3 = 0, 1, 2, 3
        for _ in 1:NSE
            dx1 = xt0[idx(j1)] - xt0[idx(j2)]
            dx2 = xt0[idx(j2)] - xt0[idx(j3)]
            dx12 = xt0[idx(j1)] - xt0[idx(j3)]
            dc1 = xt0[idx(j0)] - xt0[idx(j1)]
            dc2 = xt0[idx(j0)] - xt0[idx(j2)]
            a1 = (ht0[idx(j1)] - ht0[idx(j2)]) / dx1
            a2 = (ht0[idx(j2)] - ht0[idx(j3)]) / dx2
            b1 = (htp[idx(j1)] - htp[idx(j2)]) / dx1
            b2 = (htp[idx(j2)] - htp[idx(j3)]) / dx2
            s0[2] = ht0[idx(j1)] + a1 * dc1 + (a1 - a2) * dc1 * dc2 / dx12
            s0[4] = htp[idx(j1)] + b1 * dc1 + (b1 - b2) * dc1 * dc2 / dx12
            if f_sign < 0
                s0[2] = hp - s0[2]
            end
            ratio = s0[2] / hp
            if ratio < NEAR_EDGE_RATIO
                s0[2] = 0.0
            elseif ratio > 1.0 - NEAR_EDGE_RATIO
                s0[2] = hp
            end
            for i in 1:NDIM
                x20[i] = x0[i] + sdir[i] * xt0[idx(j0)]
                x21[i] = x20[i] + pdir[i] * s0[2]
            end
            s0[3] = call_integrand(impl_func, par, x21)
            ht0[idx(j0)] = vofi_get_segment_zero(ws, impl_func, par, x20, pdir, s0, f_sign)
            htp[idx(j0)] = s0[4]
            j0 = npt + 1
            j1 = npt
            j2 = npt - 1
            j3 = npt - 2
        end

        x0b = xt0[idx(0)]
        h0b = ht0[idx(0)]
        hpb = htp[idx(0)]
        xm = 0.5 * (x0b + xt0[idx(1)])
        xl = x0b
        hl = h0b
        xr = xt0[idx(1)]
        hr = ht0[idx(1)]
        k = 0

        for j in 0:npt
            dx1 = x0b - xt0[idx(k + 1)]
            dx2 = xt0[idx(k + 1)] - xt0[idx(k + 2)]
            dx12 = x0b - xt0[idx(k + 2)]
            dc1 = xm - x0b
            dc2 = xm - xt0[idx(k + 1)]
            a1 = (h0b - ht0[idx(k + 1)]) / dx1
            b1 = (hpb - htp[idx(k + 1)]) / dx1
            a2 = (ht0[idx(k + 1)] - ht0[idx(k + 2)]) / dx2
            b2 = (htp[idx(k + 1)] - htp[idx(k + 2)]) / dx2
            s0[2] = h0b + a1 * dc1 + (a1 - a2) * dc1 * dc2 / dx12
            s0[4] = hpb + b1 * dc1 + (b1 - b2) * dc1 * dc2 / dx12
            if f_sign < 0
                s0[2] = hp - s0[2]
            end
            ratio = s0[2] / hp
            if ratio < NEAR_EDGE_RATIO
                s0[2] = 0.0
            elseif ratio > 1.0 - NEAR_EDGE_RATIO
                s0[2] = hp
            end
            for i in 1:NDIM
                x20[i] = x0[i] + sdir[i] * xm
                x21[i] = x20[i] + pdir[i] * s0[2]
            end
            s0[3] = call_integrand(impl_func, par, x21)
            hm = vofi_get_segment_zero(ws, impl_func, par, x20, pdir, s0, f_sign)
            hpm = s0[4]
            hsum = hl + hr
            hc = 0.5 * hsum + sqrt3 / 3.0 * (2 * hm - hsum)
            xc = xm
            d1 = (xl - xc)^2 + (hl - hc)^2
            d2 = (xr - xc)^2 + (hr - hc)^2
            arc += sqrt(d1) + sqrt(d2)

            x0b = xm
            h0b = hm
            hpb = hpm
            k = min(j, npt - 1)
            xl = xt0[idx(k + 1)]
            hl = ht0[idx(k + 1)]
            xr = xt0[idx(k + 2)]
            hr = ht0[idx(k + 2)]
            xm = 0.5 * (xl + xr)
        end
    end

    return T(arc)
end
