# Pass D: shape-derivative (level-set parameter) sensitivity + ChainRules rules.

using VofiJul
using Test
import ChainRulesCore
const CRC = ChainRulesCore

@testset "shape derivative: affine level sets (exact)" begin
    # φ(x,θ) = x[1] - θ : the wetted fraction is exactly θ in 1D/2D/3D, so
    # cc = θ and d(cc)/dθ = 1 — the per-cell shape derivative is exact for affine φ.
    for (ndim, xin, h) in ((1, [0.0], [1.0]),
                           (2, [0.0, 0.0], [1.0, 1.0]),
                           (3, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
        φθ = (x, θ) -> x[1] - θ
        cc, dcc = vofi_cc_and_grad(0.3, φθ, xin, h, ndim)
        @test cc ≈ 0.3 atol = 1e-6
        @test dcc ≈ 1.0 atol = 1e-6

        # matches a finite difference of the primal
        fd = (vofi_cc(0.3 + 1e-6, φθ, xin, h, ndim) -
              vofi_cc(0.3 - 1e-6, φθ, xin, h, ndim)) / 2e-6
        @test dcc ≈ fd atol = 1e-4

        # reverse-mode rule
        y, pb = CRC.rrule(vofi_cc, 0.3, φθ, xin, h, ndim)
        @test y ≈ cc
        @test pb(1.0)[2] ≈ dcc atol = 1e-6

        # forward-mode rule
        Δ = (CRC.NoTangent(), 1.0, CRC.NoTangent(), CRC.NoTangent(),
             CRC.NoTangent(), CRC.NoTangent())
        _, dyf = CRC.frule(Δ, vofi_cc, 0.3, φθ, xin, h, ndim)
        @test dyf ≈ dcc atol = 1e-6
    end
end

@testset "shape derivative: full/empty ⇒ zero" begin
    φθ = (x, θ) -> x[1] - θ
    # cell [0.6,0.7] with θ=0.3 is fully outside (φ>0) ⇒ cc=0, no interface
    cc, dcc = vofi_cc_and_grad(0.3, φθ, [0.6], [0.1], 1)
    @test cc ≈ 0.0
    @test dcc == 0.0
end

@testset "shape derivative: curved interface (grid-sum identity)" begin
    # circle of radius θ: Σ_cells d(cc_cell)/dθ · V_cell = ∫_Γ Vn dS = circumference = 2πθ.
    θ = 0.3
    φc = (x, θ) -> sqrt(x[1]^2 + x[2]^2) - θ
    n = 40
    h = 1.0 / n
    total = 0.0
    for i in 0:n-1, j in 0:n-1
        xin = [-0.5 + i * h, -0.5 + j * h]
        _, dcc = vofi_cc_and_grad(θ, φc, xin, [h, h], 2)
        total += dcc * h * h
    end
    @test total ≈ 2π * θ rtol = 2e-2
end
