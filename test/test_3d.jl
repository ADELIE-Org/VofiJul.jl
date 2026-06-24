using Test
using VofiJul

# -----------------------------
# Test 1: 3D sphere (parameters at top)
# -----------------------------
const SP_NMX = 2
const SP_NMY = 2
const SP_NMZ = 2
const SP_X0 = 0.0
const SP_Y0 = 0.0
const SP_Z0 = 0.0
const SP_H = 1.0
const SP_CENTER = (0.0, 0.0, 0.0)
const SP_RADIUS = 1.0

function sphere_sdf_param(x, _)
    dx = x[1] - SP_CENTER[1]
    dy = x[2] - SP_CENTER[2]
    dz = x[3] - SP_CENTER[3]
    return sqrt(dx*dx + dy*dy + dz*dz) - SP_RADIUS
end

@testset "VOFI sphere (parameterized)" begin
    h0 = SP_H / SP_NMX
    hvec = [h0, h0, h0]
    xex_tmp = zeros(Float64, 4)
    nex = [0, 0]
    npt = zeros(Int, 4)
    nvis = [0, 0]

    vol_n = 0.0
    for i in 0:SP_NMX-1, j in 0:SP_NMY-1, k in 0:SP_NMZ-1
        xin = [SP_X0 + i * h0, SP_Y0 + j * h0, SP_Z0 + k * h0]
        cc = vofi_get_cc(sphere_sdf_param, nothing, xin, hvec, xex_tmp, nex, npt, nvis, 3)
        vol_n += cc
    end
    vol_n *= h0^3

    # analytical volume: full sphere of radius SP_RADIUS is 4/3*pi*r^3
    # the cube [0,1]^3 contains an 1/8 portion when center at origin (0,0,0)
    vol_a = (4.0 / 3.0) * π * SP_RADIUS^3 / 8.0

    abs_error = abs(vol_n - vol_a)
    rel_error = abs_error / vol_a
    println("Sphere volume test: Numerical = $vol_n, Analytical = $vol_a, Abs error = $abs_error, Rel error = $rel_error")
    @test vol_n ≈ vol_a atol=1e-12
end


# -----------------------------
# Test 2: Sinusoidal surface inside [0,1]^3
# f(x,y,z) = z - a0 - b0*sin(c1*pi*x + pi*d1)*sin(c1*pi*y + pi*e1)
# constants from the request and analytical volume below
# -----------------------------
const SIN_NMX = 5
const SIN_NMY = 5
const SIN_NMZ = 5

const a0 = 0.5
const b0 = 1.0/6.0
const c1 = 1.6
const d1 = 1.0/7.0
const e1 = 1.0/5.0

function sin_surface_func(x, _)
    xv = x[1]
    yv = x[2]
    zv = x[3]
    return zv - (a0 + b0 * sin(c1 * π * xv + π * d1) * sin(c1 * π * yv + π * e1))
end

@testset "Sinusoidal surface volume" begin
    h0 = 1.0 / SIN_NMX
    hvec = [h0, h0, h0]
    xex_tmp = zeros(Float64, 4)
    nex = [0, 0]
    npt = zeros(Int, 4)
    nvis = [0, 0]

    vol_n = 0.0
    for i in 0:SIN_NMX-1, j in 0:SIN_NMY-1, k in 0:SIN_NMZ-1
        xin = [i * h0, j * h0, k * h0]
        cc = vofi_get_cc(sin_surface_func, nothing, xin, hvec, xex_tmp, nex, npt, nvis, 3)
        vol_n += cc
    end
    vol_n *= h0^3

    # analytical volume in [0,1]^3
    termx = (cos(d1 * π) - cos((d1 + c1) * π))
    termy = (cos(e1 * π) - cos((e1 + c1) * π))
    vol_a = a0 + (b0 / (c1 * π * c1 * π)) * termx * termy

    abs_error = abs(vol_n - vol_a)
    rel_error = abs_error / vol_a
    println("Sinusoidal surface volume test: Numerical = $vol_n, Analytical = $vol_a, Abs error = $abs_error, Rel error = $rel_error")
    @test vol_n ≈ vol_a atol=1e-12
end


    # -----------------------------
    # Ellipsoidal cap tests (3 caps)
    # -----------------------------
    function make_ellipsoid_impl(A1,B1,C1,ALPHA,XC,YC,ZC)
        function impl(x, _)
            xx = x[1]; yy = x[2]; zz = x[3]
            A2 = A1*A1; B2 = B1*B1; C2 = C1*C1
            ca = cos(ALPHA); sa = sin(ALPHA)
            c1 = ca*ca/A2 + sa*sa/B2
            c2 = sa*sa/A2 + ca*ca/B2
            c3 = 2.0*ca*sa*(B2 - A2)/(A2*B2)
            c4 = -(2.0*c1*XC + c3*YC)
            c5 = -(2.0*c2*YC + c3*XC)
            c6 = 1.0 - (c1*XC*XC + c2*YC*YC + c3*XC*YC)
            fxy = c1*xx*xx + c2*yy*yy + c3*xx*yy + c4*xx + c5*yy - c6
            return fxy + (zz - ZC)*(zz - ZC)/C2
        end
        return impl
    end

    function run_cap_test(name; NMX=1,NMY=1,NMZ=1, X0=0.0, Y0=0.0, Z0=0.0, H=1.0,
                          A1=4.0,B1=5.0,C1=6.0, ALPHA=π/3, XC=0.5, YC=0.45, ZC=-5.97,
                          atol=1e-6)
        hcell = H / NMX
        hvec = [hcell, hcell, hcell]
        impl = make_ellipsoid_impl(A1,B1,C1,ALPHA,XC,YC,ZC)
        xex = zeros(Float64, 4)
        nex = [0,0]
        npt = zeros(Int,4)
        nvis = [0,0]

        vol_n = 0.0
        for i in 0:NMX-1, j in 0:NMY-1, k in 0:NMZ-1
            xin = [X0 + i*hcell, Y0 + j*hcell, Z0 + k*hcell]
            vol_n += vofi_get_cc(impl, nothing, xin, hvec, xex, nex, npt, nvis, 3)
        end
        vol_n *= hcell^3

        # analytical cap volume: h0 = C1 + ZC
        h0 = C1 + ZC
        vol_a = π * A1 * B1 * h0*h0 * (1.0 - h0/(3.0*C1)) / C1

        @testset "Ellipsoidal cap: $name" begin
            @test isfinite(vol_n)
            @test abs(vol_n - vol_a) ≤ atol
            abs_error = abs(vol_n - vol_a)
            rel_error = abs_error / vol_a
            println("Ellipsoidal cap test ($name): Numerical = $vol_n, Analytical = $vol_a, Abs error = $abs_error, Rel error = $rel_error")
        end
        return vol_n, vol_a
    end


    # Cap 1: domain [0,1]^3, N=1x1x1
    run_cap_test("cap1"; NMX=10, NMY=10, NMZ=10,
                 X0=0.0, Y0=0.0, Z0=0.0, H=1.0,
                 A1=4.0, B1=5.0, C1=6.0, ALPHA=π/3,
                 XC=0.50, YC=0.45, ZC=-5.97,
                 atol=1e-6)

    # Cap 2: domain [-1,1] x [0,2] x [0,2], N=2x1x1
    run_cap_test("cap2"; NMX=10, NMY=10, NMZ=10,
                 X0=-1.0, Y0=0.0, Z0=0.0, H=2.0,
                 A1=4.0, B1=5.0, C1=6.0, ALPHA=π/3,
                 XC=0.30, YC=0.45, ZC=-5.97,
                 atol=1e-4)

    # Cap 3: domain [-1,1] x [-1,1] x [0,2], N=2x2x1
    run_cap_test("cap3"; NMX=10, NMY=10, NMZ=10,
                 X0=-1.0, Y0=-1.0, Z0=0.0, H=2.0,
                 A1=4.0, B1=5.0, C1=6.0, ALPHA=π/3,
                 XC=0.35, YC=0.35, ZC=-5.97,
                 atol=1e-3)
