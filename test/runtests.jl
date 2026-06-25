using Test
using VofiJul

@test vofi_real === Float64
@test NDIM == 3 == length(MinData().xval)

nodes3 = gauss_legendre_nodes(3)
weights4 = gauss_legendre_weights(4)
# GL tables are now padded to a fixed size (GL_MAX_ORDER) so the runtime-order
# lookup is type-stable; only the first `order` entries are meaningful, the rest
# are zero padding.
@test length(nodes3) == GL_MAX_ORDER
@test nodes3 isa typeof(gauss_legendre_nodes(20))   # type-stable across orders
@test nodes3[1] ≈ -0.7745966692414833
@test all(iszero, nodes3[4:end])                    # padding
@test sum(weights4) ≈ 2.0                            # padding is zero, sum intact

len = LenData()
@test length(len.xt0) == NGLM + 2

function plane_func(x, _)
    return x[1] + x[2] - 0.5
end

function neg_func(x, _)
    return -1.0
end

function pos_func(x, _)
    return 1.0
end

# Test 1D functionality
@test vofi_get_cell_type(neg_func, nothing, [0.0], [1.0], 1) == 1
@test vofi_get_cell_type(pos_func, nothing, [0.0], [1.0], 1) == 0

function line_func_1d(x, _)
    return x[1] - 0.25
end
@test vofi_get_cell_type(line_func_1d, nothing, [0.0], [1.0], 1) == -1

xex_1d = zeros(Float64, 4)
cc_full_1d = vofi_get_cc(neg_func, nothing, [0.0], [1.0], xex_1d,
                        [0, 0], [0, 0], [0, 0], 1)
@test cc_full_1d ≈ 1.0

xex_1d_cut = zeros(Float64, 4)
cc_cut_1d = vofi_get_cc(line_func_1d, nothing, [0.0], [1.0], xex_1d_cut,
                       [1, 0], [0, 0], [0, 0], 1)
@test cc_cut_1d ≈ 0.25 atol=1e-6
@test xex_1d_cut[1] ≈ 0.125 atol=1e-6

# Test 1D integration over multiple cells (similar to 2D circle test)
function line_sdf(x, _)
    r = 0.4
    return abs(x[1]) - r
end

let
    n = 20
    h = 1.0 / n
    xmin = -0.5
    total_length = 0.0
    cell_length = h
    xex_tmp = zeros(Float64, 4)
    for i in 0:n-1
        xin = [xmin + i * h]
        cc = vofi_get_cc(line_sdf, nothing, xin, [h], xex_tmp,
                         [0, 0], [0, 0], [0, 0], 1)
        total_length += cc * cell_length
    end
    exact = 2 * 0.4  # Length of interval [-0.4, 0.4]
    @test total_length ≈ exact atol=1e-2
end

# Test 2D functionality
@test vofi_get_cell_type(neg_func, nothing, [0.0, 0.0], [1.0, 1.0], 2) == 1
@test vofi_get_cell_type(pos_func, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3) == 0
@test vofi_get_cell_type(plane_func, nothing, [0.0, 0.0], [1.0, 1.0], 2) == -1

xex = zeros(Float64, 4)
cc_full = vofi_get_cc(neg_func, nothing, [0.0, 0.0], [1.0, 1.0], xex,
                      [0, 0], [0, 0], [0, 0], 2)
@test cc_full ≈ 1.0


function slanted_func(x, _)
    return 0.25 - x[1]
end

xex_cut2 = zeros(Float64, 4)
cc_cut2 = vofi_get_cc(slanted_func, nothing, [0.0, 0.0], [1.0, 1.0], xex_cut2,
                      [1, 1], [0, 0], [0, 0], 2)
@test cc_cut2 ≈ 0.75 atol=1e-6
@test xex_cut2[4] > 0.0

xex3 = zeros(Float64, 4)
cc_full3d = vofi_get_cc(neg_func, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], xex3,
                        [1, 0], [0, 0, 0, 0], [0, 0], 3)
@test cc_full3d ≈ 1.0
@test xex3[1:3] ≈ [0.5, 0.5, 0.5]

xex3_empty = zeros(Float64, 4)
cc_empty3d = vofi_get_cc(pos_func, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                         xex3_empty, [0, 0], [0, 0, 0, 0], [0, 0], 3)
@test cc_empty3d ≈ 0.0

function plane_func3d(x, _)
    return x[1] + x[2] + x[3] - 1.5
end

xex3_plane = zeros(Float64, 4)
cc_plane = vofi_get_cc(plane_func3d, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                       xex3_plane, [0, 0], [0, 0, 0, 0], [0, 0], 3)
@test cc_plane ≈ 0.5 atol=1e-3

# simple Cartesian integration of a sphere volume
function sphere_sdf(x, _)
    r = 0.4
    return sqrt((x[1])^2 + (x[2])^2 + (x[3])^2) - r
end

let
    n = 20
    h = 1.0 / n
    xmin = -0.5
    total_vol = 0.0
    cell_vol = h^3
    xex_tmp = zeros(Float64, 4)
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        xin = [xmin + i * h, xmin + j * h, xmin + k * h]
        cc = vofi_get_cc(sphere_sdf, nothing, xin, [h, h, h], xex_tmp,
                         [0, 0], [0, 0, 0, 0], [0, 0], 3)
        total_vol += cc * cell_vol
    end
    exact = 4 / 3 * π * 0.4^3
    @test total_vol ≈ exact atol=2e-2
end

# simple Cartesian integration of a circle area
function circle_sdf(x, _)
    r = 0.4
    return sqrt((x[1])^2 + (x[2])^2) - r
end

let
    n = 20
    h = 1.0 / n
    xmin = -0.5
    total_area = 0.0
    cell_area = h^2
    xex_tmp = zeros(Float64, 4)
    for i in 0:n-1, j in 0:n-1
        xin = [xmin + i * h, xmin + j * h]
        cc = vofi_get_cc(circle_sdf, nothing, xin, [h, h], xex_tmp,
                         [0, 0], [0, 0, 0, 0], [0, 0], 2)
        total_area += cc * cell_area
    end
    exact = π * 0.4^2
    @test total_area ≈ exact atol=2e-2
end

# -----------------------------

@testset "short npt/nex/nvis vectors don't overflow (bounds guard)" begin
    # A genuinely-cut 3D cell. The internal refinement reads npt[3]/npt[4]; passing
    # a short npt must be guarded (length check) rather than throwing a BoundsError.
    sdf(x, _) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 0.3
    args = (sdf, nothing, [0.1, -0.25, -0.25], [0.5, 0.5, 0.5])
    full = vofi_get_cc(args..., zeros(5), [0, 0], [0, 0, 0, 0], [0, 0], 3)
    short = vofi_get_cc(args..., zeros(5), [0, 0], [0, 0], [0, 0], 3)  # npt length 2
    @test 0 < full < 1
    @test short ≈ full
end

@testset "2D Vofi test" begin
    include("test_2d.jl")
end

# -----------------------------
@testset "3D Vofi test" begin
    include("test_3d.jl")
end

# -----------------------------
@testset "4D Vofi test" begin
    include("test_4d.jl")
end

# Test 4D functionality
function neg_func_4d(x, _)
    return -1.0
end

function pos_func_4d(x, _)
    return 1.0
end

function hyperplane_func_4d(x, _)
    return x[1] + x[2] + x[3] + x[4] - 2.0
end

@test vofi_get_cell_type(neg_func_4d, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == 1
@test vofi_get_cell_type(pos_func_4d, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == 0
@test vofi_get_cell_type(hyperplane_func_4d, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == -1

# -----------------------------

@testset "vofi_get_cell_type 1D/2D/3D/4D" begin
    include("test_cell_type.jl")
end

@testset "vofi_interface_centroid tests" begin
    include("test_interface_centroid.jl")
end

@testset "threaded batch driver" begin
    include("test_threading.jl")
end

@testset "AD + element-type genericity" begin
    include("test_ad.jl")
end

@testset "shape derivative (Pass D)" begin
    include("test_shape_derivative.jl")
end

@testset "GPU kernel (Pass E, CPU backend)" begin
    include("test_gpu.jl")
end
