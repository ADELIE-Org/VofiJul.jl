# Multithreaded batch driver: results must match the serial path exactly and be
# data-race free. Run with multiple threads (`julia -t auto`) to actually
# exercise concurrency; with one thread it still validates correctness.

using VofiJul
using Test

circle(x, _) = sqrt(x[1]^2 + x[2]^2) - 0.4

@testset "threaded batch matches serial (2D)" begin
    n = 16
    h = 1.0 / n
    xmin = -0.5
    xins = [[xmin + i * h, xmin + j * h] for i in 0:n-1 for j in 0:n-1]

    # serial reference
    cc_serial = map(xins) do xin
        xex = zeros(4)
        vofi_get_cc(circle, nothing, xin, [h, h], xex, [1, 1], [0, 0], [0, 0], 2)
    end

    cc_batch, xex_batch = vofi_get_cc_batch(circle, nothing, xins, [h, h], 2;
                                            nex = (1, 1), npt = (0, 0), nvis = (0, 0))

    @test cc_batch ≈ cc_serial
    @test size(xex_batch, 2) == length(xins)

    # area recovered in parallel equals analytic circle area
    total_area = sum(cc_batch) * h^2
    @test total_area ≈ π * 0.4^2 atol = 2e-2

    # cell types in parallel
    types = vofi_get_cell_type_batch(circle, nothing, xins, [h, h], 2)
    @test all(t -> t in (-1, 0, 1), types)
    @test any(==(-1), types)   # some cut cells exist
end
