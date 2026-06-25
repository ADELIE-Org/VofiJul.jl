# Resolution floor for the root-finder's convergence test. `EPS_ROOT` (1e-14) is a
# Float64-scale tolerance: for a reduced-precision `T` (e.g. `Float32`, whose
# `eps` ≈ 1.2e-7) the step `dss` can never shrink below it, so the secant/bisection
# loop stalls — `ss - sv3` underflows to 0 and the slope update `(fs-fv1)/(ss-sv3)`
# becomes 0/0 = NaN, poisoning the cell limits. Scale the tolerance to the working
# precision so the loop terminates cleanly. For `Float64` the `EPS_ROOT` floor still
# dominates (eps(Float64)·len ≪ 1e-14), so the production path is unchanged.
_root_eps(::Type{T}) where {T<:AbstractFloat} = eps(T)
_root_eps(::Type) = eps(Float64)   # Dual / other Reals: assume Float64-backed

function vofi_get_segment_zero(ws::VofiWorkspace, impl_func, par, x0, dir, s0, f_sign)
    # Element type flows from the geometry / segment data so AD duals and reduced
    # precision propagate; defaults to vofi_real for plain Float64 inputs.
    T = promote_type(eltype(x0), eltype(dir), eltype(s0))
    eps_root = max(T(EPS_ROOT), 4 * _root_eps(T) * (abs(T(s0[1])) + one(T)))
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
        if abs(dss) < eps_root
            not_conv = false
            s0[4] = f_sign * fps
        end

        if not_conv
            for i in 1:NDIM
                xs[i] = x0[i] + ss * dir[i]
            end
            fs = f_sign * call_integrand(impl_func, par, xs)
            # Guard the finite-difference slope: when the abscissa step underflows
            # (`ss == sv3` in reduced precision) keep the previous slope rather than
            # forming 0/0. For Float64 the denominator is never exactly zero here, so
            # this never triggers on the production path.
            den = ss - sv3
            fps = iszero(den) ? fps : (fs - fv1) / den
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
