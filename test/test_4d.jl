using Test
using VofiJul


@testset "4D centroid extraction" begin
    halfspace_sdf(x, _) = x[1]
    xmin = fill(-0.5, 4)
    h = fill(1.0, 4)
    xex_centroid = zeros(Float64, 5)
    cc = vofi_get_cc(halfspace_sdf, nothing, xmin, h, xex_centroid,
                     [1, 0], zeros(Int, 4), [0, 0], 4)
    @test cc ≈ 0.5 atol=1e-8
    @test xex_centroid[1] ≈ -0.25 atol=1e-6
    @test xex_centroid[2] ≈ 0.0 atol=1e-8
    @test xex_centroid[3] ≈ 0.0 atol=1e-8
    @test xex_centroid[4] ≈ 0.0 atol=1e-8
end

function integrate_4d(func; n = 8)
    h = 1.0 / n
    xmin = -0.5
    total = 0.0
    cell_vol = h^4
    xex_tmp = zeros(Float64, 5)
    nex = [0, 0]
    npt = zeros(Int, 4)
    nvis = [0, 0]
    spacing = fill(h, 4)

    for i in 0:n-1, j in 0:n-1, k in 0:n-1, l in 0:n-1
        xin = [xmin + i * h, xmin + j * h, xmin + k * h, xmin + l * h]
        cc = vofi_get_cc(func, nothing, xin, spacing, xex_tmp, nex, npt, nvis, 4)
        total += cc * cell_vol
    end
    return total
end

@testset "4D integration basics" begin
    neg_func(x, _) = -1.0
    pos_func(x, _) = 1.0
    hyperplane_sdf(x, _) = x[1] + x[2] + x[3] + x[4] - 1.0

    xex = zeros(Float64, 5)
    unit_h = fill(1.0, 4)
    cc_full = vofi_get_cc(neg_func, nothing, [0.0, 0.0, 0.0, 0.0], unit_h,
                          xex, [0, 0], zeros(Int, 4), [0, 0], 4)
    @test cc_full ≈ 1.0

    xex_pos = zeros(Float64, 5)
    cc_empty = vofi_get_cc(pos_func, nothing, [0.0, 0.0, 0.0, 0.0], unit_h,
                           xex_pos, [0, 0], zeros(Int, 4), [0, 0], 4)
    @test cc_empty ≈ 0.0

    xex_cut = zeros(Float64, 5)
    cc_cut = vofi_get_cc(hyperplane_sdf, nothing, [0.0, 0.0, 0.0, 0.0], unit_h,
                         xex_cut, [0, 0], zeros(Int, 4), [0, 0], 4)
    @test 0.0 < cc_cut < 1.0
end


@testset "4D hypersphere integration" begin
    function hypersphere_sdf(x, _)
        r = 0.35
        return sqrt(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) - r
    end

    total_hyper = integrate_4d(hypersphere_sdf; n = 8)
    exact = (π^2 / 2) * 0.35^4  # 4D ball volume
    println("4D hypersphere → Numerical = $(total_hyper), Exact = $(exact), Error = $(abs(total_hyper - exact)))")
    @test total_hyper ≈ exact atol=5e-3
end

@testset "4D hypercube integration" begin
    function hypercube_sdf(x, _)
        half = 0.3
        return maximum(abs.(x)) - half
    end

    total_cube = integrate_4d(hypercube_sdf; n = 10)
    exact = (2 * 0.3)^4
    println("4D hypercube → Numerical = $(total_cube), Exact = $(exact), Error = $(abs(total_cube - exact)))")
    @test total_cube ≈ exact atol=2e-3
end

@testset "4D ellipsoid integration" begin
    axes = (0.45, 0.3, 0.25, 0.5)
    function ellipsoid_sdf(x, _)
        return (x[1] / axes[1])^2 + (x[2] / axes[2])^2 + (x[3] / axes[3])^2 + (x[4] / axes[4])^2 - 1.0
    end

    total_ellipsoid = integrate_4d(ellipsoid_sdf; n = 12)
    exact = (π^2 / 2) * prod(axes)
    println("4D ellipsoid → Numerical = $(total_ellipsoid), Exact = $(exact), Error = $(abs(total_ellipsoid - exact)))")
    @test total_ellipsoid ≈ exact atol=3e-3
end

@testset "4D cylinder integration" begin
    radius = 0.25
    height = 0.6
    function cylinder_sdf(x, _)
        radial = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - radius
        axial = abs(x[4]) - height / 2
        return max(radial, axial)
    end

    total_cyl = integrate_4d(cylinder_sdf; n = 14)
    exact = (4 / 3) * π * radius^3 * height
    println("4D cylinder → Numerical = $(total_cyl), Exact = $(exact), Error = $(abs(total_cyl - exact)))")
    @test total_cyl ≈ exact atol=3e-3
end

@testset "4D sinusoidal slab" begin
    offset = 0.1
    amplitude = 0.15
    function sinuslab_sdf(x, _)
        oscillation = sinpi(2 * x[1]) * sinpi(2 * x[2]) * sinpi(2 * x[3])
        target = offset + amplitude * oscillation
        return x[4] - target
    end

    total_sine = integrate_4d(sinuslab_sdf; n = 16)
    exact = 0.5 + offset  # base depth plus constant offset, oscillatory mean is zero
    println("4D sinusoidal slab → Numerical = $(total_sine), Exact = $(exact), Error = $(abs(total_sine - exact)))")
    @test total_sine ≈ exact atol=2e-3
end
