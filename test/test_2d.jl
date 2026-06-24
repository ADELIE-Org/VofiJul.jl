using Test
using VofiJul
using SpecialFunctions: erf
# -----------------------------
# Test 2D: Sine line
# f(x,y) = y - a0 - b0*sin(c0*pi*x + pi/d0)
# Analytical area: a0 + b0*(-cos((c0 + 1/d0)*pi) + cos(pi/d0))/(c0*pi)
# -----------------------------
@testset "2D Sine line" begin
	a0 = 0.5
	b0 = 0.25
	c0 = 4.0
	d0 = 14.0

	NMX = 10
	NMY = 10
	X0 = 0.0
	Y0 = 0.0
	H = 1.0

	function sine_line(x, _)
		xv = x[1]
		yv = x[2]
		return yv - a0 - b0 * sin(c0 * π * xv + π / d0)
	end

	hx = H / NMX
	hy = H / NMY
	xex = zeros(Float64, 4)
	nex = [0, 0]
	npt = ones(Int, 4) .*20
	nvis = [0, 0]

	area_n = 0.0
	for i in 0:NMX-1, j in 0:NMY-1
		xin = [X0 + i * hx, Y0 + j * hy]
		cc = vofi_get_cc(sine_line, nothing, xin, [hx, hy], xex, nex, npt, nvis, 2)
		area_n += cc
	end
	area_n *= hx * hy

	area_a = a0 + b0 * (-cos((c0 + 1.0/d0) * π) + cos(π / d0)) / (c0 * π)
    abs_error = abs(area_n - area_a)
    rel_error = abs_error / area_a
    println("Sine line area test: Numerical = $area_n, Analytical = $area_a, Abs error = $abs_error, Rel error = $rel_error")
	@test area_n ≈ area_a atol=1e-9
end


# -----------------------------
# Test 2D: Rectangle (possibly rotated)
# Analytical area: 4*a1*b1
# -----------------------------
@testset "2D Rectangle" begin
	a1 = 0.2
	b1 = 0.3
	alpha = 0.0
	xc = 0.52
	yc = 0.44

	NMX = 10
	NMY = 10
	X0 = 0.0
	Y0 = 0.0
	H = 1.0

	ca = cos(alpha)
	sa = sin(alpha)

	function rect_func(x, _)
		xv = x[1]
		yv = x[2]
		xp = (xv - xc) * ca + (yv - yc) * sa
		yp = (yv - yc) * ca - (xv - xc) * sa
		v1 = -a1 - xp
		v2 = xp - a1
		v3 = -b1 - yp
		v4 = yp - b1
		return max(max(v1, v2), max(v3, v4))
	end

	hx = H / NMX
	hy = H / NMY
	xex = zeros(Float64, 4)
	nex = [0, 0]
	npt = zeros(Int, 4)
	nvis = [0, 0]

	area_n = 0.0
	for i in 0:NMX-1, j in 0:NMY-1
		xin = [X0 + i * hx, Y0 + j * hy]
		cc = vofi_get_cc(rect_func, nothing, xin, [hx, hy], xex, nex, npt, nvis, 2)
		area_n += cc
	end
	area_n *= hx * hy

	area_a = 4.0 * a1 * b1
    abs_error = abs(area_n - area_a)
    rel_error = abs_error / area_a
    println("Rectangle area test: Numerical = $area_n, Analytical = $area_a, Abs error = $abs_error, Rel error = $rel_error")
	@test area_n ≈ area_a atol=1e-12
end


# -----------------------------
# Test 2D: Gaussian line
# f(x,y) = y - yy0 - a0*exp(-ga*(x-xx0)^2)
# Analytical area: yy0 + 0.5*a0*sqrt(pi/ga)*(erf(sqrt(ga)*(1-xx0)) - erf(-sqrt(ga)*xx0))
# -----------------------------
@testset "2D Gaussian" begin
	yy0 = 0.22
	a0 = 0.51
	xx0 = 0.541
	ga = 60.3

	NMX = 10
	NMY = 10
	X0 = 0.0
	Y0 = 0.0
	H = 1.0

	function gauss_line(x, _)
		xv = x[1]
		yv = x[2]
		return yv - yy0 - a0 * exp(-ga * (xv - xx0)^2)
	end

	hx = H / NMX
	hy = H / NMY
	xex = zeros(Float64, 4)
	nex = [0, 0]
	npt = ones(Int, 4) .*20
	nvis = [0, 0]

	area_n = 0.0
	for i in 0:NMX-1, j in 0:NMY-1
		xin = [X0 + i * hx, Y0 + j * hy]
		cc = vofi_get_cc(gauss_line, nothing, xin, [hx, hy], xex, nex, npt, nvis, 2)
		area_n += cc
	end
	area_n *= hx * hy

	s = sqrt(ga)
	area_a = yy0 + 0.5 * a0 * sqrt(π / ga) * (erf(s * (1.0 - xx0)) - erf(-s * xx0))
    abs_error = abs(area_n - area_a)
    rel_error = abs_error / area_a
    println("Gaussian line area test: Numerical = $area_n, Analytical = $area_a, Abs error = $abs_error, Rel error = $rel_error")
	@test area_n ≈ area_a atol=1e-12
end


# -----------------------------
# Test 2D: Ellipse
# f(x,y) = c1*x^2 + c2*y^2 + c3*x*y + c4*x + c5*y - c6
# Analytical area: pi*a1*b1
# -----------------------------
@testset "2D Ellipse" begin
	a1 = 0.17
	b1 = 0.21
	alpha = 0.48
	xc = 0.523
	yc = 0.475

	NMX = 10
	NMY = 10
	X0 = 0.0
	Y0 = 0.0
	H = 1.0

	function ellipse_func(x, _)
		xx = x[1]
		yy = x[2]
		a2 = a1 * a1
		b2 = b1 * b1
		ca = cos(alpha)
		sa = sin(alpha)
		c1 = ca*ca / a2 + sa*sa / b2
		c2 = sa*sa / a2 + ca*ca / b2
		c3 = 2.0 * ca * sa * (b2 - a2) / (a2 * b2)
		c4 = -(2.0 * c1 * xc + c3 * yc)
		c5 = -(2.0 * c2 * yc + c3 * xc)
		c6 = 1.0 - (c1 * xc * xc + c2 * yc * yc + c3 * xc * yc)
		return c1*xx*xx + c2*yy*yy + c3*xx*yy + c4*xx + c5*yy - c6
	end

	hx = H / NMX
	hy = H / NMY
	xex = zeros(Float64, 4)
	nex = [0, 0]
	npt = zeros(Int, 4)
	nvis = [0, 0]

	area_n = 0.0
	for i in 0:NMX-1, j in 0:NMY-1
		xin = [X0 + i * hx, Y0 + j * hy]
		cc = vofi_get_cc(ellipse_func, nothing, xin, [hx, hy], xex, nex, npt, nvis, 2)
		area_n += cc
	end
	area_n *= hx * hy

	area_a = π * a1 * b1
    abs_error = abs(area_n - area_a)
    rel_error = abs_error / area_a
    println("Ellipse area test: Numerical = $area_n, Analytical = $area_a, Abs error = $abs_error, Rel error = $rel_error")
	@test area_n ≈ area_a atol=1e-12
end

