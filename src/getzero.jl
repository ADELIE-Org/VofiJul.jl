function vofi_get_segment_zero(ws::VofiWorkspace, impl_func, par, x0, dir, s0, f_sign)
    # Element type flows from the geometry / segment data so AD duals and reduced
    # precision propagate; defaults to vofi_real for plain Float64 inputs.
    T = promote_type(eltype(x0), eltype(dir), eltype(s0))
    xs = ws.gsz_xs  # fully overwritten before each read
    sl = zero(T)
    sr = T(s0[1])
    not_conv = true
    dsold = T(s0[1])
    dss = dsold
    ss = T(s0[2])
    fs = f_sign * T(s0[3])
    fps = f_sign * T(s0[4])
    # 3-element sliding history windows (were heap Vectors — now scalars to avoid
    # per-iteration allocation in this innermost root-finding loop).
    sv1 = ss; sv2 = ss; sv3 = ss
    fv1 = fs; fv2 = fps; fv3 = zero(T)
    gensec = 0
    fl = T(-EPS_SEGM)
    fr = T(EPS_SEGM)

    if fs < 0.0
        sl = ss
        fl = fs
    elseif fs > 0.0
        sr = ss
        fr = fs
    else
        not_conv = false
    end

    iter = 0
    while not_conv && iter < MAX_ITER_ROOT
        if ((ss - sr) * fps - fs) * ((ss - sl) * fps - fs) > 0.0 ||
           abs(2.0 * fs) > abs(dsold * fps)
            dsold = dss
            dss = 0.5 * (sr - sl)
            ss = sl + dss
            gensec = 0
        else
            dsold = dss
            dss = fs / fps
            ss -= dss
        end
        iter += 1
        if abs(dss) < EPS_ROOT
            not_conv = false
            s0[4] = f_sign * fps
        end

        if not_conv
            for i in 1:NDIM
                xs[i] = x0[i] + ss * dir[i]
            end
            fs = f_sign * call_integrand(impl_func, par, xs)
            fps = (fs - fv1) / (ss - sv3)
            if fs < 0.0
                sl = ss
                fl = fs
            elseif fs > 0.0
                sr = ss
                fr = fs
            else
                not_conv = false
                s0[4] = f_sign * fps
            end
            sv1, sv2, sv3 = sv2, sv3, ss
            ds2 = sv3 - sv1
            if gensec > 0 && abs(ds2) > EPS_ROOT
                fv3 = (fps - fv2) / ds2
            else
                fv3 = zero(T)
            end
            fv1, fv2, fv3 = fs, fps, fv3
            gensec = 1
            fps = fv2 + fv3 * (sv3 - sv2)
        end
    end

    if !not_conv
        sz = f_sign * ss + 0.5 * (1 - f_sign) * s0[1]
    else
        s1 = zero(T)
        f1 = f_sign * call_integrand(impl_func, par, x0)
        s2 = T(s0[1])
        for i in 1:NDIM
            xs[i] = x0[i] + s2 * dir[i]
        end
        f2 = f_sign * call_integrand(impl_func, par, xs)
        if f1 * f2 <= 0.0
            if sl > 0.0
                s1 = sl
                f1 = fl
            end
            if sr < s2
                s2 = sr
                f2 = fr
            end
            ss = s1 - f1 * (s2 - s1) / (f2 - f1)
            sz = f_sign * ss + 0.5 * (1 - f_sign) * s0[1]
            s0[4] = f_sign * fps
        else
            sz = f_sign * s0[1]  # no zero found
            s0[4] = 0.0
        end
    end

    return sz
end
