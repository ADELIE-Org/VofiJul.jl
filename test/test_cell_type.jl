using Test
using VofiJul

@testset "vofi_get_cell_type 1D/2D/3D/4D" begin
    # 1D cases
    function neg1(x, _)
        return -1.0
    end
    function pos1(x, _)
        return 1.0
    end
    function cross1(x, _)
        return x[1] - 0.25
    end

    @test vofi_get_cell_type(neg1, nothing, [0.0], [1.0], 1) == 1
    @test vofi_get_cell_type(pos1, nothing, [0.0], [1.0], 1) == 0
    @test vofi_get_cell_type(cross1, nothing, [0.0], [1.0], 1) == -1

    # 2D cases
    function neg2(x, _)
        return -1.0
    end
    function pos2(x, _)
        return 1.0
    end
    function cross2(x, _)
        return x[1] + x[2] - 0.5
    end

    @test vofi_get_cell_type(neg2, nothing, [0.0, 0.0], [1.0, 1.0], 2) == 1
    @test vofi_get_cell_type(pos2, nothing, [0.0, 0.0], [1.0, 1.0], 2) == 0
    @test vofi_get_cell_type(cross2, nothing, [0.0, 0.0], [1.0, 1.0], 2) == -1

    # 3D cases
    function neg3(x, _)
        return -1.0
    end
    function pos3(x, _)
        return 1.0
    end
    function cross3(x, _)
        return x[1] - 0.25
    end

    @test vofi_get_cell_type(neg3, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3) == 1
    @test vofi_get_cell_type(pos3, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3) == 0
    @test vofi_get_cell_type(cross3, nothing, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3) == -1

    # 4D case
    function neg4(x, _)
        return -1.0
    end
    function pos4(x, _)
        return 1.0
    end
    function cross4(x, _)
        return x[1] + x[2] + x[3] + x[4] - 1.0
    end

    @test vofi_get_cell_type(neg4, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == 1
    @test vofi_get_cell_type(pos4, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == 0
    @test vofi_get_cell_type(cross4, nothing, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 4) == -1
    
end
