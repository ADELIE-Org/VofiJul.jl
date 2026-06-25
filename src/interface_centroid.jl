"""
    vofi_interface_centroid(impl_func, par, xin, h0, ndim0; tol=1e-10, max_iter=50)

Best-effort interface centroid inside a single cell. Uses the cell-centre gradient
as a local normal and bisects along that line to locate the zero of `impl_func`.
Returns a vector of length `ndim0`; if no sign change is found inside the cell,
falls back to the cell centre.
"""
# Allocation-free `SVector`-based variant (stack-resident scratch, no per-iteration
# heap vectors) for the differentiable / GPU paths, where `xin` is a static vector
# and `ndim0 == N`. Returns an `SVector{N}`. Numerically identical to the generic
# method below; ~zero allocation under `ForwardDiff` (the generic one allocated one
# heap vector per bisection step).
# `impl_func::F` forces Julia to specialize on the integrand type — otherwise the
# `Function`-typed argument is not specialized, the inner `call_integrand` falls to
# dynamic dispatch, and every bisection step boxes its `SVector` arg + `Float64`
# result on the heap (was ~120 boxed Float64 + ~37 boxed SVector per call).
function vofi_interface_centroid(impl_func::F, par, xin::SVector{N}, h0, ndim0;
                                 tol=1e-10, max_iter=50) where {F,N}
    ndim0 == N || throw(ArgumentError("ndim0 ($ndim0) must equal length(xin) ($N)"))
    impl = wrap_integrand(impl_func, par, xin)
    T = promote_type(eltype(xin), eltype(h0))
    hvec = SVector{N,T}(ntuple(i -> T(h0[i]), Val(N)))
    x0 = SVector{N,T}(ntuple(i -> T(xin[i]), Val(N)))
    xcenter = x0 .+ T(0.5) .* hvec

    grad = SVector{N,T}(ntuple(Val(N)) do i
        δ = T(0.25) * hvec[i]
        xp = Base.setindex(xcenter, min(xcenter[i] + δ, x0[i] + hvec[i]), i)
        xm = Base.setindex(xcenter, max(xcenter[i] - δ, x0[i]), i)
        (call_integrand(impl, par, xp) - call_integrand(impl, par, xm)) / max(2δ, EPS_ROOT)
    end)

    gnorm = sqrt(sum(abs2, grad))
    gnorm < EPS_ROOT && return xcenter
    n = grad ./ gnorm

    radius = T(0.5) * sqrt(sum(abs2, hvec))
    tneg = -radius; tpos = radius
    fneg = call_integrand(impl, par, xcenter .+ tneg .* n)
    fpos = call_integrand(impl, par, xcenter .+ tpos .* n)
    fneg * fpos > 0 && return xcenter

    for _ in 1:max_iter
        tmid = T(0.5) * (tneg + tpos)
        xmid = xcenter .+ tmid .* n
        fmid = call_integrand(impl, par, xmid)
        (abs(fmid) < tol || abs(tpos - tneg) < tol) && return xmid
        if fneg * fmid <= 0
            tpos = tmid; fpos = fmid
        else
            tneg = tmid; fneg = fmid
        end
    end
    return xcenter .+ T(0.5) * (tneg + tpos) .* n
end

function vofi_interface_centroid(impl_func, par, xin, h0, ndim0; tol=1e-10, max_iter=50)
    ndim0 ∈ (1:4) || throw(ArgumentError("ndim0 must be 1,2,3,4"))
    # Delegate to the allocation-free `SVector` core (stack-resident scratch, no
    # per-iteration heap vectors). Branching on `ndim0` makes the static size a
    # compile-time constant, so each arm builds a concrete `SVector{N}`. Returns an
    # `SVector{ndim0}` (indexable, `norm`-able, `isapprox`-comparable with a Vector).
    if ndim0 == 1
        return vofi_interface_centroid(impl_func, par, SVector{1}(xin[1]), h0, 1;
                                       tol=tol, max_iter=max_iter)
    elseif ndim0 == 2
        return vofi_interface_centroid(impl_func, par, SVector{2}(xin[1], xin[2]), h0, 2;
                                       tol=tol, max_iter=max_iter)
    elseif ndim0 == 3
        return vofi_interface_centroid(impl_func, par, SVector{3}(xin[1], xin[2], xin[3]),
                                       h0, 3; tol=tol, max_iter=max_iter)
    else
        return vofi_interface_centroid(impl_func, par,
                                       SVector{4}(xin[1], xin[2], xin[3], xin[4]),
                                       h0, 4; tol=tol, max_iter=max_iter)
    end
end