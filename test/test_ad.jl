# Element-type genericity (Pass A) and automatic differentiation.
#
# The 1D path flows a generic element type `T` derived from the cell geometry, so
# ForwardDiff `Dual`s and Float32 propagate end-to-end. Float64 is the default
# and pays nothing extra. The natural AD variable here is the cell geometry
# (origin/width) — the shape sensitivity; differentiating w.r.t. a level-set
# parameter is the planned custom adjoint rule's job, not forward-through.
#
# NOTE: only the 1D path is fully T-generic so far; 2D/3D/4D propagation is the
# remaining A2 work and is intentionally not exercised for AD here.

using VofiJul
using ForwardDiff
using Test

# cc of the cell [x0, x0+h] for the fixed half-space {x < p}.
# For x0 < p < x0+h it equals (p - x0)/h.
function cc_origin(x0; p = 0.5, h = 1.0)
    xex = zeros(eltype(x0), 4)
    vofi_get_cc((x, _) -> x[1] - p, nothing, [x0], [h], xex, [0, 0], [0, 0], [0, 0], 1)
end

function cc_width(h; p = 0.5, x0 = 0.0)
    xex = zeros(eltype(h), 4)
    vofi_get_cc((x, _) -> x[1] - p, nothing, [x0], [h], xex, [0, 0], [0, 0], [0, 0], 1)
end

@testset "1D element-type genericity + AD (shape sensitivity)" begin
    # primal: cc = (0.5 - 0.2)/1.0 = 0.3
    @test cc_origin(0.2) ≈ 0.3

    # d(cc)/d(x0) = -1/h = -1
    @test ForwardDiff.derivative(cc_origin, 0.2) ≈ -1.0

    # d(cc)/d(h): cc = (p - x0)/h ⇒ d/dh = -(p - x0)/h^2.
    # at x0=0, p=0.5, h=0.8: -(0.5)/0.64
    @test ForwardDiff.derivative(cc_width, 0.8) ≈ -0.5 / 0.8^2

    # AD agrees with central finite difference
    fd = (cc_origin(0.2 + 1e-6) - cc_origin(0.2 - 1e-6)) / 2e-6
    @test ForwardDiff.derivative(cc_origin, 0.2) ≈ fd atol = 1e-5

    # Float32 end-to-end: Float32 geometry ⇒ Float32 result (no Float64 upcast)
    cc32 = vofi_get_cc((x, _) -> x[1] - 0.3f0, nothing, Float32[0], Float32[1],
                       zeros(Float32, 4), [0, 0], [0, 0], [0, 0], 1)
    @test cc32 isa Float32
    @test cc32 ≈ 0.3f0
end

# --- 2D / 3D / 4D element-type genericity + shape sensitivity -----------------
#
# Each path uses a sphere/circle SDF centred at the origin; the cell straddles
# the interface (genuinely cut). We differentiate cc w.r.t. the first cell-origin
# coordinate and compare ForwardDiff against a central finite difference.

# 2D circle, radius 0.4, centred at origin. The radius adopts the coordinate
# eltype so a Float32 cell yields a Float32 level-set value (no Float64 upcast).
circle2d(x, _) = sqrt(x[1]^2 + x[2]^2) - oftype(x[1], 0.4)
function cc2d(x0first; x0=[0.2, -0.1], h=[0.3, 0.3])
    T = eltype(x0first)
    xin = T[x0first, x0[2]]
    xex = zeros(T, 4)
    vofi_get_cc(circle2d, nothing, xin, T.(h), xex, [0, 0], [0, 0], [0, 0], 2)
end

# 3D sphere, radius 0.4.
sphere3d(x, _) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - oftype(x[1], 0.4)
function cc3d(x0first; x0=[0.2, -0.1, 0.05], h=[0.3, 0.3, 0.3])
    T = eltype(x0first)
    xin = T[x0first, x0[2], x0[3]]
    xex = zeros(T, 4)
    vofi_get_cc(sphere3d, nothing, xin, T.(h), xex, [0, 0], [0, 0, 0, 0], [0, 0], 3)
end

# 4D hypersphere, radius 0.4.
sphere4d(x, _) = sqrt(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) - oftype(x[1], 0.4)
function cc4d(x0first; x0=[0.2, -0.1, 0.05, 0.0], h=[0.3, 0.3, 0.3, 0.3])
    T = eltype(x0first)
    xin = T[x0first, x0[2], x0[3], x0[4]]
    xex = zeros(T, 8)
    vofi_get_cc(sphere4d, nothing, xin, T.(h), xex, [0, 0], [0, 0, 0, 0], [0, 0], 4)
end

@testset "2D element-type genericity + AD" begin
    # Float32 geometry ⇒ Float32 result.
    cc32 = vofi_get_cc(circle2d, nothing, Float32[0.2, -0.1], Float32[0.3, 0.3],
                       zeros(Float32, 4), [0, 0], [0, 0], [0, 0], 2)
    @test cc32 isa Float32
    @test 0 < cc32 < 1   # genuinely cut

    # primal genuinely cut
    @test 0 < cc2d(0.2) < 1
    fd = (cc2d(0.2 + 1e-5) - cc2d(0.2 - 1e-5)) / 2e-5
    @test ForwardDiff.derivative(cc2d, 0.2) ≈ fd atol = 1e-4
end

@testset "3D element-type genericity + AD" begin
    cc32 = vofi_get_cc(sphere3d, nothing, Float32[0.2, -0.1, 0.05], Float32[0.3, 0.3, 0.3],
                       zeros(Float32, 4), [0, 0], [0, 0, 0, 0], [0, 0], 3)
    @test cc32 isa Float32
    @test 0 < cc32 < 1

    @test 0 < cc3d(0.2) < 1
    fd = (cc3d(0.2 + 1e-5) - cc3d(0.2 - 1e-5)) / 2e-5
    @test ForwardDiff.derivative(cc3d, 0.2) ≈ fd atol = 1e-4
end

@testset "4D element-type genericity + AD" begin
    cc32 = vofi_get_cc(sphere4d, nothing, Float32[0.2, -0.1, 0.05, 0.0],
                       Float32[0.3, 0.3, 0.3, 0.3], zeros(Float32, 8),
                       [0, 0], [0, 0, 0, 0], [0, 0], 4)
    @test cc32 isa Float32
    @test 0 < cc32 < 1

    @test 0 < cc4d(0.2) < 1
    fd = (cc4d(0.2 + 1e-5) - cc4d(0.2 - 1e-5)) / 2e-5
    @test ForwardDiff.derivative(cc4d, 0.2) ≈ fd atol = 1e-4
end
