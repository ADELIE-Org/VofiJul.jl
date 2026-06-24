"""
    vofi_interface_centroid(impl_func, par, xin, h0, ndim0; tol=1e-10, max_iter=50)

Best-effort interface centroid inside a single cell. Uses the cell-centre gradient
as a local normal and bisects along that line to locate the zero of `impl_func`.
Returns a vector of length `ndim0`; if no sign change is found inside the cell,
falls back to the cell centre.
"""
function vofi_interface_centroid(impl_func, par, xin, h0, ndim0; tol=1e-10, max_iter=50)
    ndim0 ∈ (1:4) || throw(ArgumentError("ndim0 must be 1,2,3,4"))
    impl_func = wrap_integrand(impl_func, par, xin)
    T = promote_type(eltype(xin), eltype(h0))
    hvec = T.(collect(h0[1:ndim0]))
    x0 = T.(collect(xin[1:ndim0]))
    xcenter = similar(x0)
    for i in 1:ndim0
        xcenter[i] = x0[i] + 0.5 * hvec[i]
    end

    fcenter = call_integrand(impl_func, par, xcenter)
    # Estimate gradient at cell centre using small steps along each axis
    grad = zeros(T, ndim0)
    for i in 1:ndim0
        δ = 0.25 * hvec[i]
        xplus = copy(xcenter); xminus = copy(xcenter)
        xplus[i] += δ; xminus[i] -= δ
        # keep probes inside cell
        xplus[i] = min(xplus[i], x0[i] + hvec[i])
        xminus[i] = max(xminus[i], x0[i])
        fp = call_integrand(impl_func, par, xplus)
        fm = call_integrand(impl_func, par, xminus)
        grad[i] = (fp - fm) / max(2δ, EPS_ROOT)
    end

    gnorm = sqrt(sum(abs2, grad))
    if gnorm < EPS_ROOT
        return xcenter
    end
    n = grad ./ gnorm

    # Search along normal through the cell centre
    radius = 0.5 * sqrt(sum(hvec .^ 2))
    tneg = -radius
    tpos = radius
    fneg = call_integrand(impl_func, par, xcenter .+ tneg .* n)
    fpos = call_integrand(impl_func, par, xcenter .+ tpos .* n)

    if fneg * fpos > 0
        # No clear bracket; fall back to centre
        return xcenter
    end

    for _ in 1:max_iter
        tmid = 0.5 * (tneg + tpos)
        xmid = xcenter .+ tmid .* n
        fmid = call_integrand(impl_func, par, xmid)
        if abs(fmid) < tol || abs(tpos - tneg) < tol
            return xmid
        end
        if fneg * fmid <= 0
            tpos = tmid
            fpos = fmid
        else
            tneg = tmid
            fneg = fmid
        end
    end

    return xcenter .+ 0.5 * (tneg + tpos) .* n
end