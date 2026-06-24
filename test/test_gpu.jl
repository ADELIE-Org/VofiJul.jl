# Pass E: KernelAbstractions per-cell kernel, validated on the CPU() backend
# (the same kernel runs unchanged on CUDABackend()/ROCBackend() on real hardware).

using VofiJul
using KernelAbstractions
using Test

circle_sdf_gpu(x, _) = sqrt(x[1]^2 + x[2]^2) - 0.4
sphere_sdf_gpu(x, _) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 0.4

@testset "2D kernel matches serial + area" begin
    n = 30
    h = 1.0 / n
    origins = [(-0.5 + i * h, -0.5 + j * h) for i in 0:n-1 for j in 0:n-1]
    serial = [vofi_get_cc(circle_sdf_gpu, nothing, collect(x), [h, h], zeros(4),
                          [0, 0], [0, 0, 0, 0], [0, 0], 2) for x in origins]
    gpu = vofi_get_cc_gpu(circle_sdf_gpu, nothing, origins, (h, h), 2; backend = CPU())
    @test Array(gpu) ≈ serial
    @test sum(gpu) * h * h ≈ π * 0.4^2 atol = 1e-2
end

@testset "3D kernel matches serial + volume" begin
    m = 16
    h = 1.0 / m
    origins = [(-0.5 + i * h, -0.5 + j * h, -0.5 + k * h)
               for i in 0:m-1 for j in 0:m-1 for k in 0:m-1]
    serial = [vofi_get_cc(sphere_sdf_gpu, nothing, collect(x), [h, h, h], zeros(4),
                          [0, 0], [0, 0, 0, 0], [0, 0], 3) for x in origins]
    gpu = vofi_get_cc_gpu(sphere_sdf_gpu, nothing, origins, (h, h, h), 3; backend = CPU())
    @test Array(gpu) ≈ serial
    @test sum(gpu) * h^3 ≈ 4 / 3 * π * 0.4^3 atol = 2e-2
end

@testset "empty input" begin
    out = vofi_get_cc_gpu(circle_sdf_gpu, nothing, NTuple{2,Float64}[], (0.1, 0.1), 2; backend = CPU())
    @test length(out) == 0
end
