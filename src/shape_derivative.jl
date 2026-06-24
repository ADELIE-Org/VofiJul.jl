# ---------------------------------------------------------------------------
# Pass D — shape-derivative / adjoint rule for level-set *parameter* sensitivity
# ---------------------------------------------------------------------------
#
# Forward-mode AD through `vofi_get_cc` gives the *geometry* sensitivity (Dual via
# the cell xin/h0). For the sensitivity of the volume fraction w.r.t. a parameter
# `θ` of the level set `φ(x; θ)` we do NOT differentiate through the root-finder.
# Instead we use the Reynolds-transport / shape-derivative identity. For a moment
# Mf(θ) = ∫_{φ<0} f dx over a fixed cell,
#
#     dMf/dθ = ∫_Γ f · Vn dS,     Vn = -∂_θφ / |∇φ|   (interface normal speed)
#
# For the volume fraction cc = V / V_cell with f ≡ 1:
#
#     d(cc)/dθ = (1/V_cell) ∫_Γ Vn dS ≈ (|Γ_cell| / V_cell) · Vn(x_Γ)
#
# evaluated at the interface centroid x_Γ (exact for an affine φ in the cell, and
# convergent under refinement in general). |Γ_cell| and x_Γ are quantities VofiJul
# already computes (interface measure + `vofi_interface_centroid`). The pointwise
# ∇φ(x_Γ) and ∂_θφ(x_Γ) are obtained with ForwardDiff on the *user's* level set —
# cheap, exact, and independent of the cut-cell algorithm.

using ForwardDiff
import ChainRulesCore

"""
    vofi_cc(θ, φθ, xin, h0, ndim0) -> cc

Volume fraction of the cell `[xin, xin+h0]` for the parametric level set
`x -> φθ(x, θ)` (negative inside). Differentiable w.r.t. the scalar parameter
`θ` via a custom shape-derivative rule (both forward- and reverse-mode through
ChainRules) — it does **not** differentiate through the root-finder. See
[`vofi_cc_and_grad`](@ref) for the explicit value+gradient.
"""
function vofi_cc(θ::Real, φθ, xin, h0, ndim0)
    Tf = float(promote_type(eltype(xin), eltype(h0)))
    xex = zeros(Tf, ndim0 == 4 ? 5 : 4)
    return vofi_get_cc(x -> φθ(x, θ), nothing, xin, h0, xex,
                       (0, 0), (0, 0, 0, 0), (0, 0), ndim0)
end

"""
    vofi_cc_and_grad(θ, φθ, xin, h0, ndim0) -> (cc, dcc_dθ)

Return the volume fraction `cc` and its shape derivative `d(cc)/dθ` for the
parametric level set `x -> φθ(x, θ)`. The derivative uses the Reynolds-transport
identity with the interface measure and centroid; `∇φ` and `∂_θφ` are evaluated
pointwise at the interface centroid with ForwardDiff. Returns a zero derivative
for full/empty cells (no interface).
"""
function vofi_cc_and_grad(θ::Real, φθ, xin, h0, ndim0)
    φ = x -> φθ(x, θ)
    Tf = float(promote_type(eltype(xin), eltype(h0)))
    nexlen = ndim0 == 4 ? 5 : 4
    xex = zeros(Tf, nexlen)
    cc = vofi_get_cc(φ, nothing, xin, h0, xex, (1, 1), (0, 0, 0, 0), (0, 0), ndim0)

    Vcell = one(Tf)
    for i in 1:ndim0
        Vcell *= Tf(h0[i])
    end

    # interface measure |Γ_cell|: a point (=1) in 1D, otherwise the last xex slot
    Γ = ndim0 == 1 ? (zero(cc) < cc < one(cc) ? one(Tf) : zero(Tf)) : Tf(xex[nexlen])
    dz = zero(Tf) * zero(θ)
    Γ > 0 || return cc, dz

    xΓ = vofi_interface_centroid(φ, nothing, xin, h0, ndim0)
    g = ForwardDiff.gradient(x -> φθ(x, θ), xΓ)
    ngrad = sqrt(sum(abs2, g))
    ngrad > 0 || return cc, dz
    ∂θφ = ForwardDiff.derivative(t -> φθ(xΓ, t), θ)

    Vn = -∂θφ / ngrad
    dcc = (Γ / Vcell) * Vn
    return cc, dcc
end

# Reverse-mode (adjoint): the design variable is θ.
function ChainRulesCore.rrule(::typeof(vofi_cc), θ::Real, φθ, xin, h0, ndim0)
    cc, dcc = vofi_cc_and_grad(θ, φθ, xin, h0, ndim0)
    function vofi_cc_pullback(c̄)
        return (ChainRulesCore.NoTangent(), c̄ * dcc,
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return cc, vofi_cc_pullback
end

# Forward-mode.
function ChainRulesCore.frule((_, Δθ, _, _, _, _), ::typeof(vofi_cc),
                              θ::Real, φθ, xin, h0, ndim0)
    cc, dcc = vofi_cc_and_grad(θ, φθ, xin, h0, ndim0)
    return cc, dcc * Δθ
end
