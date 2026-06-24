function vofi_get_segment_min(ws::VofiWorkspace, impl_func, par, x0, dir, fse, xfs_pt, s0, ifsign)
    T = promote_type(eltype(x0), eltype(dir))
    xs = ws.gsm_xs  # fully overwritten before each read
    GRIS = 0.5 * (3.0 - sqrt(5.0))
    sign_change = 0
    igold = 1
    not_conv = true

    fa = ifsign * fse[1]
    sa = zero(T)
    fb = ifsign * fse[2]
    sb = T(s0)
    if fa <= fb
        st = zero(T)
        sv = T(s0)
        ft = fa
        fv = fb
    else
        st = T(s0)
        sv = zero(T)
        ft = fb
        fv = fa
    end

    ss = sa + GRIS * (sb - sa)
    for i in 1:NDIM
        xs[i] = x0[i] + ss * dir[i]
    end
    fs = ifsign * call_integrand(impl_func, par, xs)
    fu = fs
    su = ss
    se = st - sv
    sd = ss - st
    if fs > ft
        @SHFT4 fu ft fs fu
        @SHFT4 su st ss su
    end
    if fs < 0
        not_conv = false
    end

    iter = 0
    sm = zero(T)
    fm = zero(T)
    sp = zero(T)
    fp = zero(T)
    sz = zero(T)
    fz = zero(T)
    sc = zero(T)
    se = zero(T)
    sd = zero(T)
    tol = zero(T)
    t2 = zero(T)
    if fs > ft
        fu = ft
        ft = fs
        fs = fu
        su = st
        st = ss
        ss = su
    end
    se = st - sv
    sd = ss - st
    if fs < 0.0
        not_conv = false
    end

    iter = 0
    while not_conv && iter < MAX_ITER_MINI
        sc = 0.5 * (sa + sb)
        tol = EPS_M * abs(ss) + EPS_LOC
        t2 = 2.0 * tol
        if abs(ss - sc) <= t2 - 0.5 * (sb - sa)
            not_conv = false
        else
            iter += 1
            r = zero(T)
            q = zero(T)
            p = zero(T)
            if abs(se) > tol
                r = (ss - st) * (fs - fv)
                q = (ss - sv) * (fs - ft)
                p = (ss - sv) * q - (ss - st) * r
                q = 2.0 * (q - r)
                if q > 0.0
                    p = -p
                else
                    q = abs(q)
                end
                r = se
                se = sd
            end
            if abs(p) < abs(0.5 * q * r) && p > q * (sa - ss) && p < q * (sb - ss)
                sd = p / q
                su = ss + sd
                if (su - sa) < t2 || (sb - su) < t2
                    sd = ss < sc ? tol : -tol
                end
            else
                if ss < sc
                    se = sb - ss
                else
                    se = sa - ss
                end
                sd = GRIS * se
                igold += 1
            end
            if abs(sd) >= tol
                su = ss + sd
            elseif sd > 0.0
                su = ss + tol
            else
                su = ss - tol
            end
            for i in 1:NDIM
                xs[i] = x0[i] + su * dir[i]
            end
            fu = ifsign * call_integrand(impl_func, par, xs)
            if fu < 0.0
                not_conv = false
            end
            if fu <= fs
                if su < ss
                    sb = ss
                    fb = fs
                else
                    sa = ss
                    fa = fs
                end
                @SHFT4 fv ft fs fu
                @SHFT4 sv st ss su
            else
                if su < ss
                    sa = su
                    fa = fu
                else
                    sb = su
                    fb = fu
                end
                if fu <= ft || st == ss
                    @CPSF(sv, st, fv, ft)
                    @CPSF(st, su, ft, fu)
                elseif fu <= fv || sv == ss || sv == st
                    @CPSF(sv, su, fv, fu)
                end
            end
            if igold == 2 && not_conv
                igold = 0
                @CPSF(sm, ss, fm, fs)
                if st < sm && abs(st - sa) > t2
                    @CPSF(sm, st, fm, ft)
                end
                if sv < sm && abs(sv - sa) > t2
                    @CPSF(sm, sv, fm, fv)
                end
                @CPSF(sp, ss, fp, fs)
                if st > sp && abs(st - sb) > t2
                    @CPSF(sp, st, fp, ft)
                end
                if sv > sp && abs(sv - sb) > t2
                    @CPSF(sp, sv, fp, fv)
                end

                p = (sa - sm) * (fp * sb - fb * sp) + (sp - sb) * (fm * sa - fa * sm)
                q = (sa - sm) * (fp - fb) + (sp - sb) * (fm - fa)
                if q < 0.0
                    p = -p
                    q = -q
                end
                sm = min(sa, sm)
                sp = max(sb, sp)
                if p > q * sm && p < q * sp
                    su = p / q
                    for i in 1:NDIM
                        xs[i] = x0[i] + su * dir[i]
                    end
                    fu = ifsign * call_integrand(impl_func, par, xs)
                    iseca = 0
                    if fu < fs
                        if fu < 0.0
                            not_conv = false
                        end
                        tol = EPS_M * abs(su) + EPS_LOC
                        for j in (-1, 1)
                            sz = su + j * tol
                            for i in 1:NDIM
                                xs[i] = x0[i] + sz * dir[i]
                            end
                            fz = ifsign * call_integrand(impl_func, par, xs)
                            if fz > fu
                                iseca += 1
                            end
                        end
                        if iseca == 2
                            @CPSF(ss, su, fs, fu)
                            sb = sz
                            sa = sz - 2.0 * tol
                        end
                    end
                end
            end
        end
    end

    for i in 1:NDIM
        xfs_pt.xval[i] = xs[i]
    end
    xfs_pt.fval = ifsign * fs
    xfs_pt.sval = ss
    if fs < 0.0
        sign_change = 1
    end

    return sign_change
end

function vofi_get_face_min(ws::VofiWorkspace, impl_func, par, x0, h0, dir1, dir2, fve, xfs_pt, ipsc)
    T = promote_type(eltype(x0), eltype(h0))
    xs0 = ws.gfm_xs0
    xs1 = ws.gfm_xs1
    x1f = ws.gfm_x1f
    x1b = ws.gfm_x1b
    x2f = ws.gfm_x2f
    x2b = ws.gfm_x2b
    res = ws.gfm_res
    hes = ws.gfm_hes
    rs0 = ws.gfm_rs0
    hs0 = ws.gfm_hs0
    pcrs = ws.gfm_pcrs
    nmdr = ws.gfm_nmdr
    cndr = ws.gfm_cndr
    ss = ws.gfm_ss
    fse = ws.gfm_fse
    eps2 = EPS_E * EPS_E
    dh = 1.0e-4

    for i in 1:NDIM
        xs0[i] = x0[i] + h0[i] * (ipsc.ind1 * dir1[i] + ipsc.ind2 * dir2[i])
        x1f[i] = xs0[i] + dh * dir1[i]
        x1b[i] = xs0[i] - dh * dir1[i]
        x2f[i] = xs0[i] + dh * dir2[i]
        x2b[i] = xs0[i] - dh * dir2[i]
        rs0[i] = 0.0
        hs0[i] = 1.0 - dir1[i] - dir2[i]
    end

    idx = ipsc.ind1 + 2 * ipsc.ind2 + 1
    fse[1] = fve[idx]
    f2pos = ipsc.consi
    fs0 = f2pos * fse[1]
    fse[2] = 0.0
    f1f = f2pos * call_integrand(impl_func, par, x1f)
    f1b = f2pos * call_integrand(impl_func, par, x1b)
    f2f = f2pos * call_integrand(impl_func, par, x2f)
    f2b = f2pos * call_integrand(impl_func, par, x2b)

    df1 = -0.5 * (f1f - f1b) / dh
    df2 = -0.5 * (f2f - f2b) / dh
    d2f1 = (f1f + f1b - 2.0 * fs0) / (dh * dh)
    d2f2 = (f2f + f2b - 2.0 * fs0) / (dh * dh)
    if d2f1 <= 0.0 || d2f2 <= 0.0
        d2f1 = 1.0
        d2f2 = 1.0
    end

    mcd = zero(T)
    del0 = zero(T)
    for i in 1:NDIM
        res[i] = rs0[i] + df1 * ipsc.swt1 * dir1[i] + df2 * ipsc.swt2 * dir2[i]
        hes[i] = hs0[i] + d2f1 * dir1[i] + d2f2 * dir2[i]
        pcrs[i] = res[i] / hes[i]
        mcd += pcrs[i] * pcrs[i]
        del0 += res[i] * pcrs[i]
    end
    mcd = sqrt(mcd + EPS_NOT0)
    for i in 1:NDIM
        cndr[i] = pcrs[i]
        nmdr[i] = cndr[i] / mcd
        d1 = SGN0P(nmdr[i])
        d2 = abs(nmdr[i]) + EPS_NOT0
        if d2 < EPS_ROOT
            ss[i] = 1000.0 * h0[i]
        else
            a1 = (x0[i] - xs0[i]) / (d1 * d2)
            a2 = (x0[i] + h0[i] - xs0[i]) / (d1 * d2)
            ss[i] = max(a1, a2)
        end
    end
    ss0 = minimum(ss)
    for i in 1:NDIM
        xs1[i] = xs0[i] + ss0 * nmdr[i]
    end
    fse[2] = call_integrand(impl_func, par, xs1)

    delnew = del0
    not_conv = true
    iter = 0
    k = 0
    sign_change = 0
    while not_conv && iter < MAX_ITER_MINI
        sign_change = vofi_get_segment_min(ws, impl_func, par, xs0, nmdr, fse, xfs_pt,
                                           ss0, f2pos)
        xs0 .= xfs_pt.xval
        fse[1] = xfs_pt.fval
        fs0 = f2pos * fse[1]
        if sign_change != 0
            not_conv = false
        else
            for i in 1:NDIM
                x1f[i] = xs0[i] + dh * dir1[i]
                x1b[i] = xs0[i] - dh * dir1[i]
                x2f[i] = xs0[i] + dh * dir2[i]
                x2b[i] = xs0[i] - dh * dir2[i]
            end
            ss0 = xfs_pt.sval
            f1f = f2pos * call_integrand(impl_func, par, x1f)
            f1b = f2pos * call_integrand(impl_func, par, x1b)
            f2f = f2pos * call_integrand(impl_func, par, x2f)
            f2b = f2pos * call_integrand(impl_func, par, x2b)
            df1 = -0.5 * (f1f - f1b) / dh
            df2 = -0.5 * (f2f - f2b) / dh
            d2f1 = (f1f + f1b - 2.0 * fs0) / (dh * dh)
            d2f2 = (f2f + f2b - 2.0 * fs0) / (dh * dh)
            if d2f1 <= 0.0 || d2f2 <= 0.0
                d2f1 = 1.0
                d2f2 = 1.0
            end
            delold = delnew
            delmid = zero(T)
            delnew = zero(T)
            for i in 1:NDIM
                res[i] = rs0[i] + df1 * dir1[i] + df2 * dir2[i]
                delmid += res[i] * pcrs[i]
                hes[i] = hs0[i] + d2f1 * dir1[i] + d2f2 * dir2[i]
                pcrs[i] = res[i] / hes[i]
                delnew += res[i] * pcrs[i]
            end
            beta = (delnew - delmid) / delold
            k += 1
            if k == 2 || beta <= 0.0
                beta = zero(T)
                k = 0
            end
            mcd = zero(T)
            for i in 1:NDIM
                cndr[i] = pcrs[i] + beta * cndr[i]
                mcd += cndr[i] * cndr[i]
            end
            mcd = sqrt(mcd + EPS_NOT0)
            for i in 1:NDIM
                nmdr[i] = cndr[i] / mcd
                d1 = SGN0P(nmdr[i])
                d2 = abs(nmdr[i]) + EPS_NOT0
                a1 = (x0[i] - xs0[i]) / (d1 * d2)
                a2 = (x0[i] + h0[i] - xs0[i]) / (d1 * d2)
                ss[i] = max(a1, a2)
            end
            ss1 = minimum(ss)
            ss0 = min(1.2 * ss0, ss1)
            if delnew < eps2 * del0 || ss0 < EPS_ROOT
                not_conv = false
            else
                for i in 1:NDIM
                    xs1[i] = xs0[i] + ss0 * nmdr[i]
                end
                fse[2] = call_integrand(impl_func, par, xs1)
                iss = 0
                while f2pos * fse[2] < fs0 && iss < 3 && ss0 < ss1
                    ss0 = min(3.0 * ss0, ss1)
                    if iss == 2
                        ss0 = ss1
                    end
                    for i in 1:NDIM
                        xs1[i] = xs0[i] + ss0 * nmdr[i]
                    end
                    fse[2] = call_integrand(impl_func, par, xs1)
                    iss += 1
                end
            end
            iter += 1
        end
    end

    return sign_change
end
