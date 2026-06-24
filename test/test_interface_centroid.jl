using Test
using VofiJul
using LinearAlgebra

"""
Simple signed-distance helper constructors.
"""
sd_line(x) = x[1] - 0.3
sd_plane2D(x) = x[1] + 2x[2] - 0.5
sd_plane3D(x) = x[1] - 0.2 + 0.5x[2] - 0.25x[3]

@testset "vofi_interface_centroid 1D" begin
    xin = [0.0]
    h0 = [1.0]
    c = vofi_interface_centroid(sd_line, nothing, xin, h0, 1)
    @test isapprox(c[1], 0.3; atol=1e-8)
end

@testset "vofi_interface_centroid 2D" begin
    xin = [0.0, 0.0]
    h0 = [1.0, 1.0]
    c = vofi_interface_centroid(sd_plane2D, nothing, xin, h0, 2)
    # The plane x + 2y = 0.5 intersects the unit square near (0.1,0.2)
    @test isapprox(c[1] + 2c[2], 0.5; atol=1e-6)
end

@testset "vofi_interface_centroid 3D" begin
    xin = [0.0, 0.0, 0.0]
    h0 = [1.0, 1.0, 1.0]
    c = vofi_interface_centroid(sd_plane3D, nothing, xin, h0, 3)
    @test isapprox(c[1] - 0.2 + 0.5c[2] - 0.25c[3], 0.0; atol=1e-6)
end

@testset "vofi_interface_centroid 4D" begin
    # 4D test case: plane x + 2y + 3z + 4w = 0.5
    sd_plane4D(x) = x[1] + 2x[2] + 3x[3] + 4x[4] - 0.5
    xin = [0.0, 0.0, 0.0, 0.0]
    h0 = [1.0, 1.0, 1.0, 1.0]
    c = vofi_interface_centroid(sd_plane4D, nothing, xin, h0, 4)
    @test isapprox(c[1] + 2c[2] + 3c[3] + 4c[4], 0.5; atol=1e-6)
end

@testset "vofi_interface_centroid fallback" begin
    # Test fallback to cell centre when no sign change
    sd_constant(x) = 1.0
    xin = [0.0, 0.0]
    h0 = [1.0, 1.0]
    c = vofi_interface_centroid(sd_constant, nothing, xin, h0, 2)
    @test isapprox(c, [0.5, 0.5]; atol=1e-8)
end


@testset "vofi_interface_centroid sphere 3d" begin
    # Test sphere
    sd_sphere(x) = norm(x) - 0.5
    xin = [0.0, 0.0, 0.0]
    h0 = [1.0, 1.0, 1.0]
    c = vofi_interface_centroid(sd_sphere, nothing, xin, h0, 3)
    @test isapprox(norm(c), 0.5; atol=1e-6)
end

@testset "vofi_interface_centroid hypersphere 4d" begin
    # Test hypersphere in 4D
    sd_hypersphere(x) = norm(x) - 0.5
    xin = [0.0, 0.0, 0.0, 0.0]
    h0 = [1.0, 1.0, 1.0, 1.0]
    c = vofi_interface_centroid(sd_hypersphere, nothing, xin, h0, 4)
    @test isapprox(norm(c), 0.5; atol=1e-6)
end

@testset "vofi_interface_centroid 4D hypercube" begin
    # Test hypercube interface in 4D
    sd_hypercube(x) = maximum(abs.(x)) - 0.3
    xin = [0.0, 0.0, 0.0, 0.0]
    h0 = [1.0, 1.0, 1.0, 1.0]
    c = vofi_interface_centroid(sd_hypercube, nothing, xin, h0, 4)
    @test isapprox(norm(c, Inf), 0.3; atol=1e-6)
end