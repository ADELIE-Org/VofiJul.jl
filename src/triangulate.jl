function vofi_interface_surface(ws::VofiWorkspace, impl_func, par, x0, h0, xt, pdir, sdir, tdir,
                                xhpn, xhpo, k, nexpt, ipf)
    T = promote_type(eltype(x0), eltype(h0))
    xa = ws.is_xa
    xb = ws.is_xb
    xc = ws.is_xc
    x1 = ws.is_x1
    x2 = ws.is_x2
    s0 = ws.is_s0
    surfer = zero(T)
    hp = zero(T)
    for i in 1:NDIM
        hp += pdir[i] * h0[i]
    end
    s0[1] = hp
    km = k - 1
    if k == 2 || k > nexpt
        km -= 1
    end
    nsec = xhpn[2].np0 > 0 ? 2 : (xhpn[1].np0 > 0 ? 1 : 0)
    for it0 in 1:nsec
        npn = xhpn[it0].np0
        npo = xhpo[it0].np0
        f_sign = xhpn[it0].f_sign
        djl = djr = djc = -1
        nmin = min(npn, npo)
        nmax = max(npn, npo)
        if nmin == 0
            continue  # no points to connect on one of the sections
        end
        if npn >= npo
            pts1 = xhpn[it0].xt0
            pth1 = xhpn[it0].ht0
            pts2 = xhpo[it0].xt0
            pth2 = xhpo[it0].ht0
            t1 = xt[k + 1]
            t2 = xt[km + 1]
        else
            pts1 = xhpo[it0].xt0
            pth1 = xhpo[it0].ht0
            pts2 = xhpn[it0].xt0
            pth2 = xhpn[it0].ht0
            t2 = xt[k + 1]
            t1 = xt[km + 1]
        end
        if nmin == 1
            djl = nmax - 2
        else
            djc = nmin - 2
            npa = nmax
            psa_idx = 1
            psb_idx = nmax
            dxl = pts2[1] - pts1[psa_idx]
            dxr = pts1[psb_idx] - pts2[nmin]
            while npa > nmin
                if dxr >= dxl
                    djr += 1
                    psb_idx -= 1
                    dxr = pts1[psb_idx] - pts2[nmin]
                else
                    djl += 1
                    psa_idx += 1
                    dxl = pts2[1] - pts1[psa_idx]
                end
                npa -= 1
            end
        end

        pts1_idx = 1
        pth1_idx = 1
        pts2_idx = 1
        pth2_idx = 1
        for _ in 0:djl
            xa .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            xb .= (t2, pts2[pts2_idx], pth2[pth2_idx])
            pts1_idx += 1
            pth1_idx += 1
            xc .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end
        end

        tc = 0.5 * (t1 + t2)
        s0[4] = 0.5 * (xhpo[it0].htp[2] + xhpn[it0].htp[2])
        for _ in 0:djc
            psa_idx = pts1_idx
            psb_idx = pts2_idx
            pha_idx = pth1_idx
            phb_idx = pth2_idx
            pts1_idx += 1
            pts2_idx += 1
            pth1_idx += 1
            pth2_idx += 1
            sc = 0.25 * (pts1[psa_idx] + pts2[psb_idx] + pts1[pts1_idx] + pts2[pts2_idx])
            hsum = pth1[pha_idx] + pth2[phb_idx] + pth1[pth1_idx] + pth2[pth2_idx]
            s0[2] = 0.25 * hsum
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
                x1[i] = x0[i] + tc * tdir[i] + sc * sdir[i]
                x2[i] = x1[i] + s0[2] * pdir[i]
            end
            s0[3] = call_integrand(impl_func, par, x2)
            hc = vofi_get_segment_zero(ws, impl_func, par, x1, pdir, s0, f_sign)
            xa .= (t1, pts1[psa_idx], pth1[pha_idx])
            xb .= (tc, sc, hc)
            xc .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end

            xc .= (t2, pts2[psb_idx], pth2[phb_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end

            xa .= (t2, pts2[pts2_idx], pth2[pth2_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end

            xc .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end
        end

        for _ in 0:djr
            xa .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            xb .= (t2, pts2[pts2_idx], pth2[pth2_idx])
            pts1_idx += 1
            pth1_idx += 1
            xc .= (t1, pts1[pts1_idx], pth1[pth1_idx])
            surfer += vofi_triarea(xa, xb, xc)
            if ipf > 0
                tecplot_triangle(x0, pdir, sdir, tdir, xa, xb, xc, hp, f_sign)
            end
        end
    end
    return T(surfer)
end

function vofi_end_points(ws::VofiWorkspace, impl_func, par, x0, h0, pdir, sdir, xhhp)
    T = promote_type(eltype(x0), eltype(h0))
    x20 = ws.ep_x20
    x21 = ws.ep_x21
    s0 = ws.ep_s0
    hp = zero(T)
    for i in 1:NDIM
        hp += pdir[i] * h0[i]
    end
    s0[1] = hp
    nseg = xhhp[2].np0 > 0 ? 2 : (xhhp[1].np0 > 0 ? 1 : 0)
    for it0 in 1:nseg
        npt = xhhp[it0].np0
        if npt > 1
            f_sign = xhhp[it0].f_sign
            j0, j1, j2, j3 = 1, 2, 3, 4
            for _ in 1:NSE
                dx1 = xhhp[it0].xt0[j1] - xhhp[it0].xt0[j2]
                dx2 = xhhp[it0].xt0[j2] - xhhp[it0].xt0[j3]
                dx12 = xhhp[it0].xt0[j1] - xhhp[it0].xt0[j3]
                dc1 = xhhp[it0].xt0[j0] - xhhp[it0].xt0[j1]
                dc2 = xhhp[it0].xt0[j0] - xhhp[it0].xt0[j2]
                a1 = (xhhp[it0].ht0[j1] - xhhp[it0].ht0[j2]) / dx1
                a2 = (xhhp[it0].ht0[j2] - xhhp[it0].ht0[j3]) / dx2
                b1 = (xhhp[it0].htp[j1] - xhhp[it0].htp[j2]) / dx1
                b2 = (xhhp[it0].htp[j2] - xhhp[it0].htp[j3]) / dx2
                s0[2] = xhhp[it0].ht0[j1] + a1 * dc1 + (a1 - a2) * dc1 * dc2 / dx12
                s0[4] = xhhp[it0].htp[j1] + b1 * dc1 + (b1 - b2) * dc1 * dc2 / dx12
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
                    x20[i] = x0[i] + sdir[i] * xhhp[it0].xt0[j0]
                    x21[i] = x20[i] + pdir[i] * s0[2]
                end
                s0[3] = call_integrand(impl_func, par, x21)
                xhhp[it0].ht0[j0] = vofi_get_segment_zero(ws, impl_func, par, x20, pdir, s0, f_sign)
                xhhp[it0].htp[j0] = s0[4]
                j0 = npt + 2
                j1 = npt + 1
                j2 = npt
                j3 = npt - 1
            end
            for i in 3:npt
                xhhp[it0].xt0[i - 1] = xhhp[it0].xt0[i]
                xhhp[it0].ht0[i - 1] = xhhp[it0].ht0[i]
            end
            xhhp[it0].xt0[npt] = xhhp[it0].xt0[npt + 2]
            xhhp[it0].ht0[npt] = xhhp[it0].ht0[npt + 2]
        else
            xhhp[it0].ht0[1] = xhhp[it0].ht0[2]
        end
    end
    return nothing
end

function vofi_edge_points(ws::VofiWorkspace, impl_func, par, x0, h0, base, pdir, sdir, xhp, npt, nsub, nsect, ndire)
    T = promote_type(eltype(x0), eltype(h0))
    x1 = ws.edp_x1
    x20 = ws.edp_x20
    x21 = ws.edp_x21
    s0 = ws.edp_s0
    fse = ws.edp_fse
    hp = zero(T)
    hs = zero(T)
    for i in 1:NDIM
        hp += pdir[i] * h0[i]
        hs += sdir[i] * h0[i]
    end
    it0 = 1
    for i in 1:NDIM
        x1[i] = x0[i] + pdir[i] * h0[i]
    end
    max_sections = min(length(xhp), length(npt))
    for ns in 1:nsub
        if nsect[ns] < 0
            if it0 > max_sections
                break  # no storage/point-count info for additional sections
            end
            ds = base[ns + 1] - base[ns]
            mdpt = 0.5 * (base[ns + 1] + base[ns])
            npts = ds < 2 * EPS_LOC ? 1 : npt[it0]
            xhp[it0].np0 = npts
            f_sign = ndire[ns]
            xhp[it0].f_sign = f_sign
            j = max(0, npts - 3)
            pts = gauss_legendre_nodes(j + 3)

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
                            dxp2 = xhp[it0].xt0[k + 2] - xhp[it0].xt0[k + 1]
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
            end
            it0 += 1
        end
    end
    return nothing
end

function vofi_triarea(xa, xb, xc)
    u = xb .- xa
    v = xc .- xa
    cross = [u[2] * v[3] - u[3] * v[2],
             u[3] * v[1] - u[1] * v[3],
             u[1] * v[2] - u[2] * v[1]]
    cross_sq = cross[1]^2 + cross[2]^2 + cross[3]^2
    return 0.5 * sqrt(cross_sq)
end
