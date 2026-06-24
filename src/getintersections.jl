function vofi_get_side_intersections(ws::VofiWorkspace, impl_func, par, fse, x0, xfsp, base, dir, hl, nsub, isc)
    T = promote_type(eltype(x0), eltype(base), eltype(fse))
    s0 = ws.gsi_s0
    inters = 0
    if isc < 0
        f2neg = fse[1] < 0 ? 1 : -1
        s0[1] = hl
        if abs(fse[1]) < abs(fse[2])
            s0[2] = 0.0
            s0[3] = fse[1]
        else
            s0[2] = hl
            s0[3] = fse[2]
        end
        s0[4] = (fse[2] - fse[1]) / hl
        dhl = vofi_get_segment_zero(ws, impl_func, par, x0, dir, s0, f2neg)
        if f2neg < 0
            dhl = hl - dhl
        end
        base[nsub + 1] = dhl
        inters += 1
    else
        f2neg = (fse[1] + fse[2]) <= 0 ? 1 : -1
        s0[1] = xfsp.sval
        if abs(fse[1]) < abs(xfsp.fval)
            s0[2] = 0.0
            s0[3] = fse[1]
        else
            s0[2] = xfsp.sval
            s0[3] = xfsp.fval
        end
        s0[4] = (xfsp.fval - fse[1]) / xfsp.sval
        dhl = vofi_get_segment_zero(ws, impl_func, par, x0, dir, s0, f2neg)
        if fse[1] > 0 || xfsp.fval < 0
            dhl = xfsp.sval - dhl
        end
        base[nsub + 1] = dhl
        inters += 1
        f2neg = -f2neg
        s0[1] = hl - xfsp.sval
        if abs(xfsp.fval) < abs(fse[2])
            s0[2] = 0.0
            s0[3] = xfsp.fval
        else
            s0[2] = s0[1]
            s0[3] = fse[2]
        end
        s0[4] = (fse[2] - xfsp.fval) / s0[1]
        dhl = vofi_get_segment_zero(ws, impl_func, par, xfsp.xval, dir, s0, f2neg)
        if xfsp.fval > 0 || fse[2] < 0
            dhl = s0[1] - dhl
        end
        base[nsub + 2] = xfsp.sval + dhl
        inters += 1
    end
    return inters
end

function vofi_get_ext_intersections(ws::VofiWorkspace, impl_func, par, x0, h0, xfsp, base, sdir, tdir, nsub)
    T = promote_type(eltype(x0), eltype(h0), eltype(base))
    pt0 = ws.gei_pt0; pt0 .= xfsp.xval
    pt1 = ws.gei_pt1; pt1 .= pt0
    pt2 = ws.gei_pt2; pt2 .= pt0
    pt = ws.gei_pt
    mp0 = ws.gei_mp0
    mp1 = ws.gei_mp1
    ss = ws.gei_ss
    fse = ws.gei_fse
    s0 = ws.gei_s0
    inters = 0
    js = jt = 1
    f2neg = xfsp.fval < 0 ? 1 : -1
    fse[1] = f2neg * xfsp.fval

    for i in 1:NDIM
        if sdir[i] != 0
            js = i
        elseif tdir[i] != 0
            jt = i
        end
    end

    pt2[js] = x0[js] + h0[js]
    s0[1] = pt2[js] - pt0[js]
    fse[2] = f2neg * call_integrand(impl_func, par, pt2)
    if fse[2] > 0
        if abs(fse[1]) < abs(fse[2])
            s0[2] = 0.0
            s0[3] = fse[1]
        else
            s0[2] = s0[1]
            s0[3] = fse[2]
        end
        s0[4] = (fse[2] - fse[1]) / s0[1]
        dhl = vofi_get_segment_zero(ws, impl_func, par, pt0, sdir, s0, f2neg)
        pt2[js] = pt0[js] + dhl
    end

    pt1[js] = x0[js]
    sdir_js = sdir[js]
    sdir[js] = -1.0
    s0[1] = pt0[js] - pt1[js]
    fse[2] = f2neg * call_integrand(impl_func, par, pt1)
    if fse[2] > 0
        if abs(fse[1]) < abs(fse[2])
            s0[2] = 0.0
            s0[3] = fse[1]
        else
            s0[2] = s0[1]
            s0[3] = fse[2]
        end
        s0[4] = (fse[2] - fse[1]) / s0[1]
        dhl = vofi_get_segment_zero(ws, impl_func, par, pt0, sdir, s0, f2neg)
        pt1[js] = pt0[js] - dhl
    end
    sdir[js] = sdir_js

    mp0 .= (pt1 .+ pt2) ./ 2
    fpt0 = f2neg * call_integrand(impl_func, par, mp0)
    ss0 = pt2[js] - pt1[js]

    for k in (-1, 1)
        iter = 0
        not_conv = true
        ss_val = ss0 / 2
        while not_conv && iter < MAX_ITER_ROOT
            iter += 1
            mp1 .= mp0
            mp1[js] = mp0[js] + k * ss_val
            fse[2] = f2neg * call_integrand(impl_func, par, mp1)
            if fse[2] < 0
                if abs(ss_val) < EPS_ROOT
                    not_conv = false
                else
                    mp0 .= mp1
                    fpt0 = fse[2]
                end
            else
                ss_val *= 0.5
            end
        end
        if not_conv
            continue
        end

        for i in 1:NDIM
            pt[i] = mp0[i]
        end
        s0[1] = 0.0
        for i in 1:NDIM
            if tdir[i] == 1
                s0[1] = pt[i] - x0[i]
            elseif tdir[i] == -1
                s0[1] = x0[i] + h0[i] - pt[i]
            end
        end
        fse[2] = fpt0
        if abs(fse[2]) < EPS_ROOT
            baser_idx = nsub + inters + 1
            base[baser_idx] = s0[1]
            inters += 1
        else
            if abs(fse[1]) < abs(fse[2])
                s0[2] = 0.0
                s0[3] = fse[1]
            else
                s0[2] = s0[1]
                s0[3] = fse[2]
            end
            s0[4] = (fse[2] - fse[1]) / s0[1]
            dhl = vofi_get_segment_zero(ws, impl_func, par, x0, tdir, s0, f2neg)
            if fse[1] > 0 || fse[2] < 0
                dhl = s0[1] - dhl
            end
            baser_idx = nsub + inters + 1
            base[baser_idx] = dhl
            inters += 1
        end
    end

    return inters
end
