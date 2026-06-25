"""
Core constants, helper routines, and lightweight data structures that mirror
`vofi/include/vofi_stddecl.h` from the original C implementation.

`vofi_real` is the **default** element type. The scratch data structures
([`MinData`](@ref), [`MinData4D`](@ref), [`XFSP4D`](@ref), [`LenData`](@ref)) are
parametric on a numeric type `T`, so an AD dual type (e.g. `ForwardDiff.Dual`) or
a reduced precision (`Float32`) can flow through the algorithm. Calling a
constructor with no element type (e.g. `MinData()`) keeps the historical
`vofi_real` default. All scratch lives in per-call locals (no module-global
mutable state), so the routines are reentrant and safe to call concurrently from
different threads on disjoint cells.
"""
const vofi_real = Float64
const vofi_creal = Float64
const vofi_int = Int
const vofi_cint = Int
const vofi_void_cptr = Any
const vofi_int_cpt = Vector{vofi_int}
const Integrand = Function

const EPS_M = 1.5e-7
const EPS_LOC = 1.5e-7
const EPS_E = 5.0e-7
const EPS_SEGM = 1.0e-12
const EPS_ROOT = 1.0e-14
const EPS_NOT0 = 1.0e-90
const NEAR_EDGE_RATIO = 2.0e-2
const MAX_ITER_ROOT = 15
const MAX_ITER_MINI = 50
const NDIM = 3
const NVER = 4
const NSE = 2
const NSEG = 10
const NGLM = 20

@inline MIN(a, b) = min(a, b)
@inline MAX(a, b) = max(a, b)
@inline SGN0P(a) = a < 0 ? -1 : 1
@inline Sq(a) = a * a
@inline Sq2(a) = a[1] * a[1] + a[2] * a[2]
@inline Sq3(a) = a[1] * a[1] + a[2] * a[2] + a[3] * a[3]
@inline Sqd3(a, b) = Sq(a[1] - b[1]) + Sq(a[2] - b[2]) + Sq(a[3] - b[3])

macro SHFT4(a, b, c, d)
    quote
        $(esc(a)) = $(esc(b))
        $(esc(b)) = $(esc(c))
        $(esc(c)) = $(esc(d))
    end
end

macro CPSF(s, t, f, g)
    quote
        $(esc(s)) = $(esc(t))
        $(esc(f)) = $(esc(g))
    end
end

mutable struct MinData{T}
    xval::MVector{NDIM, T}
    fval::T
    sval::T
    isc::MVector{NDIM, vofi_int}
    ipt::vofi_int
end

function MinData{T}(; xval = nothing,
                    fval = zero(T),
                    sval = zero(T),
                    isc = nothing,
                    ipt = zero(vofi_int)) where {T}
    xv = xval === nothing ? zero(MVector{NDIM, T}) :
         MVector{NDIM, T}(xval)
    iscv = isc === nothing ? zero(MVector{NDIM, vofi_int}) :
            MVector{NDIM, vofi_int}(isc)
    return MinData{T}(xv, fval, sval, iscv, ipt)
end

MinData(; kwargs...) = MinData{vofi_real}(; kwargs...)

mutable struct MinData4D{T}
    xval::Vector{T}
    fval::T
    sval::T
    span::T
    isc::Vector{vofi_int}
end

function MinData4D{T}(; xval = nothing,
                      fval = zero(T),
                      sval = zero(T),
                      span = zero(T),
                      isc = nothing) where {T}
    xv = xval === nothing ? zeros(T, 4) : copyto!(Vector{T}(undef, length(xval)), xval)
    iscv = isc === nothing ? zeros(vofi_int, 4) : copy(isc)
    return MinData4D{T}(xv, fval, sval, span, iscv)
end

MinData4D(; kwargs...) = MinData4D{vofi_real}(; kwargs...)

mutable struct XFSP4D{T}
    edges::Vector{MinData4D{T}}
    sectors::Vector{MinData4D{T}}
    ipt::vofi_int
end

function XFSP4D{T}() where {T}
    edges = [MinData4D{T}() for _ in 1:8]
    sectors = [MinData4D{T}() for _ in 1:2]
    return XFSP4D{T}(edges, sectors, 0)
end

XFSP4D() = XFSP4D{vofi_real}()

mutable struct DirData
    ind1::vofi_int
    ind2::vofi_int
    swt1::vofi_int
    swt2::vofi_int
    consi::vofi_int
    function DirData(ind1, ind2, swt1, swt2, consi)
        return new(ind1, ind2, swt1, swt2, consi)
    end
end

DirData(; ind1 = 0, ind2 = 0, swt1 = 0, swt2 = 0, consi = 0) =
    DirData(ind1, ind2, swt1, swt2, consi)

mutable struct LenData{T}
    np0::vofi_int
    f_sign::vofi_int
    xt0::MVector{NGLM + 2, T}
    ht0::MVector{NGLM + 2, T}
    htp::MVector{NGLM + 2, T}
    function LenData{T}(np0, f_sign, xt0, ht0, htp) where {T}
        length(xt0) == NGLM + 2 || throw(ArgumentError("xt0 must have NGLM + 2 entries"))
        length(ht0) == NGLM + 2 || throw(ArgumentError("ht0 must have NGLM + 2 entries"))
        length(htp) == NGLM + 2 || throw(ArgumentError("htp must have NGLM + 2 entries"))
        return new{T}(np0, f_sign, xt0, ht0, htp)
    end
end

function LenData{T}(; np0 = 0, f_sign = 1,
                    xt0 = nothing, ht0 = nothing, htp = nothing) where {T}
    xt = xt0 === nothing ? zero(MVector{NGLM + 2, T}) : MVector{NGLM + 2, T}(xt0)
    ht = ht0 === nothing ? zero(MVector{NGLM + 2, T}) : MVector{NGLM + 2, T}(ht0)
    hp = htp === nothing ? zero(MVector{NGLM + 2, T}) : MVector{NGLM + 2, T}(htp)
    return LenData{T}(np0, f_sign, xt, ht, hp)
end

LenData(; kwargs...) = LenData{vofi_real}(; kwargs...)

# `fill!(::StaticArrays.MArray, x)` boxes (~32 B/call); an explicit indexed loop on
# a local binding does not. `zfill!` is the allocation-free zeroing used for all
# per-cut-cell static workspace/scratch resets. Falls back to `fill!` for ordinary
# `AbstractArray`s (e.g. the 4D dynamic `Vector` scratch), where `fill!` is already
# allocation-free.
@inline function zfill!(a::StaticArrays.StaticArray, x)
    @inbounds @simd for i in eachindex(a)
        a[i] = x
    end
    return a
end
@inline zfill!(a, x) = fill!(a, x)

# Reset a scratch struct to the exact state produced by its no-arg constructor,
# so a hoisted-and-reused instance behaves identically to a freshly allocated one.
function reset!(d::LenData)
    d.np0 = 0
    d.f_sign = 1
    zfill!(d.xt0, 0)
    zfill!(d.ht0, 0)
    zfill!(d.htp, 0)
    return d
end

function reset!(d::DirData)
    d.ind1 = 0
    d.ind2 = 0
    d.swt1 = 0
    d.swt2 = 0
    d.consi = 0
    return d
end

function reset!(d::MinData)
    zfill!(d.xval, 0)
    d.fval = zero(eltype(d.xval))
    d.sval = zero(eltype(d.xval))
    zfill!(d.isc, 0)
    d.ipt = 0
    return d
end

Base.copy!(dest::MinData, src::MinData) = begin
    dest.xval .= src.xval
    dest.fval = src.fval
    dest.sval = src.sval
    dest.isc .= src.isc
    dest.ipt = src.ipt
    dest
end

Base.copy!(dest::LenData, src::LenData) = begin
    dest.np0 = src.np0
    dest.f_sign = src.f_sign
    dest.xt0 .= src.xt0
    dest.ht0 .= src.ht0
    dest.htp .= src.htp
    dest
end

Base.copy!(dest::MinData4D, src::MinData4D) = begin
    dest.xval .= src.xval
    dest.fval = src.fval
    dest.sval = src.sval
    dest.span = src.span
    dest.isc .= src.isc
    dest
end

"""
    VofiWorkspace{T}()

Reusable scratch buffers for the 2D/3D `vofi_get_cc` hot path. Holds one field per
(function, scratch-temporary) so that threading a single workspace through the whole
2D/3D call graph removes per-call/per-quadrature heap allocation. Distinct fields per
function are safe because no 2D/3D function is self-recursive; functions that are
simultaneously on the call stack (e.g. `vofi_get_face_min` and `vofi_get_segment_min`)
use disjoint fields.

Reuse one workspace per thread/task across cells (resetting per cell happens inside the
converted functions exactly where the original code rebuilt its scratch).
"""
mutable struct VofiWorkspace{T}
    # NOTE: vofi_order_dirs_2D/3D run once per cell and use stack-local scratch.
    # A heap-struct MArray field allocates on every dynamic index (read AND write),
    # so the 27-point stencil stays a function-local MArray (one alloc/cell) rather
    # than a workspace field — empirically cheaper.
    # vofi_get_limits_2D
    gl2_basei::MVector{NSEG + 1, vofi_int}
    gl2_sign_sect::MMatrix{NSE, NDIM, vofi_int}
    gl2_nbt::MVector{NSE, vofi_int}
    gl2_fse::MVector{NSE, T}
    gl2_x1::MVector{NDIM, T}
    # vofi_get_limits_3D
    gl3_basei::MVector{NSEG + 1, vofi_int}
    gl3_xs::MVector{NDIM, T}
    gl3_xp::MVector{NDIM, T}
    gl3_xt::MVector{NDIM, T}
    gl3_fse::MVector{NSE, T}
    gl3_xfsl::MinData{T}
    # vofi_check_plane
    cp_basei::MVector{NSEG + 1, vofi_int}
    cp_x1::MVector{NDIM, T}
    cp_x2::MVector{NDIM, T}
    cp_fse::MVector{NSE, T}
    cp_xfsl::MinData{T}
    cp_nbt::MVector{NSE, vofi_int}
    cp_sign_sect::MMatrix{NSE, NDIM, vofi_int}
    # vofi_get_limits_inner_2D
    gli_basei::MVector{NSEG + 1, vofi_int}
    gli_x1::MVector{NDIM, T}
    gli_x2::MVector{NDIM, T}
    gli_fse::MVector{NSE, T}
    gli_xfsl::MinData{T}
    # vofi_get_limits_edge_2D
    gle_basei::MVector{NSEG + 1, vofi_int}
    gle_x1::MVector{NDIM, T}
    gle_x2::MVector{NDIM, T}
    gle_fse::MVector{NSE, T}
    gle_xfsl::MinData{T}
    # vofi_sector_old!
    so_x1::MVector{NDIM, T}
    so_x2::MVector{NDIM, T}
    so_fse::MVector{NSE, T}
    # vofi_get_area
    ga_x1::MVector{NDIM, T}
    ga_x20::MVector{NDIM, T}
    ga_x21::MVector{NDIM, T}
    ga_s0::MVector{4, T}
    ga_fse::MVector{NSE, T}
    # vofi_get_volume
    gv_x1::MVector{NDIM, T}
    gv_base_int::MVector{NSEG + 1, T}
    gv_xmidt::MVector{NGLM + 2, T}
    gv_xhpn1::LenData{T}
    gv_xhpn2::LenData{T}
    gv_xhpo1::LenData{T}
    gv_xhpo2::LenData{T}
    gv_xhpn_edge1::LenData{T}
    gv_xhpn_edge2::LenData{T}
    gv_xfs::MinData{T}
    gv_nsect::MVector{NSEG, vofi_int}
    gv_ndire::MVector{NSEG, vofi_int}
    gv_xedge::MVector{NDIM, T}
    # vofi_get_side_intersections
    gsi_s0::MVector{4, T}
    # vofi_get_ext_intersections
    gei_pt0::MVector{NDIM, T}
    gei_pt1::MVector{NDIM, T}
    gei_pt2::MVector{NDIM, T}
    gei_pt::MVector{NDIM, T}
    gei_mp0::MVector{NDIM, T}
    gei_mp1::MVector{NDIM, T}
    gei_ss::MVector{NDIM, T}
    gei_fse::MVector{NSE, T}
    gei_s0::MVector{4, T}
    # vofi_get_segment_zero
    gsz_xs::MVector{NDIM, T}
    # vofi_get_segment_min
    gsm_xs::MVector{NDIM, T}
    # vofi_get_face_min
    gfm_xs0::MVector{NDIM, T}
    gfm_xs1::MVector{NDIM, T}
    gfm_x1f::MVector{NDIM, T}
    gfm_x1b::MVector{NDIM, T}
    gfm_x2f::MVector{NDIM, T}
    gfm_x2b::MVector{NDIM, T}
    gfm_res::MVector{NDIM, T}
    gfm_hes::MVector{NDIM, T}
    gfm_rs0::MVector{NDIM, T}
    gfm_hs0::MVector{NDIM, T}
    gfm_pcrs::MVector{NDIM, T}
    gfm_nmdr::MVector{NDIM, T}
    gfm_cndr::MVector{NDIM, T}
    gfm_ss::MVector{NDIM, T}
    gfm_fse::MVector{NSE, T}
    # vofi_check_boundary_line
    cbl_nx::MVector{NSE, vofi_int}
    cbl_ny::MVector{NSE, vofi_int}
    cbl_fse::MVector{NSE, T}
    cbl_x1::MVector{NDIM, T}
    cbl_xfsl::MinData{T}
    # vofi_check_boundary_surface
    cbs_nx::MVector{NSE, vofi_int}
    cbs_ny::MVector{NSE, vofi_int}
    cbs_nz::MVector{NSE, vofi_int}
    cbs_fve::MVector{NVER, T}
    cbs_x1::MVector{NDIM, T}
    cbs_xfsl::MinData{T}
    # vofi_check_secondary_side
    css_x1::MVector{NDIM, T}
    css_fse::MVector{NSE, T}
    css_xfsl::MinData{T}
    # vofi_check_secter_face
    csf_x1::MVector{NDIM, T}
    csf_fve::MVector{NVER, T}
    csf_xfsl::MinData{T}
    # vofi_check_tertiary_side
    cts_x1::MVector{NDIM, T}
    cts_fse::MVector{NSE, T}
    cts_xfsl::MinData{T}
    # vofi_check_side_consistency
    csc_xs::MVector{NDIM, T}
    # vofi_check_face_consistency
    cfc_xx::MVector{NDIM, T}
    cfc_x1::MVector{NDIM, T}
    cfc_x2::MVector{NDIM, T}
    cfc_fl::MVector{NVER, T}
    cfc_ipsc::DirData
    # vofi_check_line_consistency
    clc_xs::MVector{NDIM, T}
    # vofi_check_edge_consistency
    cec_xs::MVector{NDIM, T}
    cec_s0::MVector{4, T}
    # vofi_interface_length
    il_s0::MVector{4, T}
    il_x20::MVector{NDIM, T}
    il_x21::MVector{NDIM, T}
    # vofi_interface_surface
    is_xa::MVector{NDIM, T}
    is_xb::MVector{NDIM, T}
    is_xc::MVector{NDIM, T}
    is_x1::MVector{NDIM, T}
    is_x2::MVector{NDIM, T}
    is_s0::MVector{4, T}
    # vofi_end_points
    ep_x20::MVector{NDIM, T}
    ep_x21::MVector{NDIM, T}
    ep_s0::MVector{4, T}
    # vofi_edge_points
    edp_x1::MVector{NDIM, T}
    edp_x20::MVector{NDIM, T}
    edp_x21::MVector{NDIM, T}
    edp_s0::MVector{4, T}
    edp_fse::MVector{NSE, T}
    # vofi_get_cc top-level scratch
    cc_x0::MVector{NDIM, T}
    cc_pdir::MVector{NDIM, T}
    cc_sdir::MVector{NDIM, T}
    cc_tdir::MVector{NDIM, T}
    cc_f02D::MMatrix{NSE, NSE, T}
    cc_f03D::MArray{Tuple{NSE, NSE, NSE}, T}
    cc_base::MVector{NSEG + 1, T}
    cc_nsect::MVector{NSEG, vofi_int}
    cc_ndire::MVector{NSEG, vofi_int}
    cc_centroid::MVector{NDIM + 1, T}
    cc_xhp1::LenData{T}
    cc_xhp2::LenData{T}
    cc_xfsp1::MinData{T}
    cc_xfsp2::MinData{T}
    cc_xfsp3::MinData{T}
    cc_xfsp4::MinData{T}
    cc_xfsp5::MinData{T}
    cc_f0_1D::MVector{NSE, T}
    cc_hvec::MVector{NDIM, T}
    # ---- 4D path (dynamic; NDIM stays 3) ----
    # vofi_order_dirs_4D
    od4_hvec::Vector{T}
    od4_hh::Vector{T}
    od4_x::Vector{T}
    od4_fgrad::Vector{T}
    od4_mags::Vector{T}
    od4_order::Vector{vofi_int}
    od4_n0::Array{vofi_int, 4}
    od4_fperm::Array{T, 4}
    # vofi_populate_sector_volume_4D! / vofi_populate_quaternary_edges_4D!
    od4_psv_base::Vector{T}
    od4_pqe_base::Vector{T}
    od4_pqe_dir::Vector{T}
    od4_pqe_x::Vector{T}      # find_quaternary_bracket / bisection scratch
    od4_pqe_xroot::Vector{T}  # bisection root point
    # vofi_check_boundary_hypersurface
    cbh_fcube::Array{T, 3}
    cbh_x1::Vector{T}
    cbh_nx::Vector{vofi_int}
    cbh_ny::Vector{vofi_int}
    cbh_nz::Vector{vofi_int}
    cbh_nw::Vector{vofi_int}
    # vofi_get_limits_4D
    gl4_cuts::Vector{T}
    # vofi_get_hypervolume
    gh_xin3::Vector{T}
    gh_h3::Vector{T}
    gh_xex3::Vector{T}
    gh_nex_slice::Vector{vofi_int}
    gh_nvis_slice::Vector{vofi_int}
    gh_xbuf::Vector{T}
    # vofi_cell_type_4D
    ct4_n0::Array{vofi_int, 4}
    ct4_f0::Array{T, 4}
    ct4_x1::Vector{T}
    ct4_fgrad::Vector{T}
    # vofi_get_cc 4D top-level scratch
    cc4_x0::Vector{T}
    cc4_h4::Vector{T}
    cc4_pdir::Vector{T}
    cc4_sdir::Vector{T}
    cc4_tdir::Vector{T}
    cc4_qdir::Vector{T}
    cc4_f0::Array{T, 4}
    cc4_xfsp::XFSP4D{T}
    cc4_base::Vector{T}
    cc4_centroid::Vector{T}
end

function VofiWorkspace{T}() where {T}
    z(::Type{S}, dims...) where {S} = zeros(MVector{dims..., S})
    return VofiWorkspace{T}(
        # gl2
        zero(MVector{NSEG + 1, vofi_int}), zero(MMatrix{NSE, NDIM, vofi_int}),
        zero(MVector{NSE, vofi_int}), zero(MVector{NSE, T}), zero(MVector{NDIM, T}),
        # gl3
        zero(MVector{NSEG + 1, vofi_int}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NSE, T}), MinData{T}(),
        # cp
        zero(MVector{NSEG + 1, vofi_int}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NSE, T}), MinData{T}(), zero(MVector{NSE, vofi_int}),
        zero(MMatrix{NSE, NDIM, vofi_int}),
        # gli
        zero(MVector{NSEG + 1, vofi_int}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NSE, T}), MinData{T}(),
        # gle
        zero(MVector{NSEG + 1, vofi_int}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NSE, T}), MinData{T}(),
        # sector_old
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NSE, T}),
        # ga
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{4, T}), zero(MVector{NSE, T}),
        # gv
        zero(MVector{NDIM, T}), zero(MVector{NSEG + 1, T}), zero(MVector{NGLM + 2, T}),
        LenData{T}(), LenData{T}(), LenData{T}(), LenData{T}(), LenData{T}(), LenData{T}(),
        MinData{T}(), zero(MVector{NSEG, vofi_int}), zero(MVector{NSEG, vofi_int}),
        zero(MVector{NDIM, T}),
        # gsi
        zero(MVector{4, T}),
        # gei
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NSE, T}), zero(MVector{4, T}),
        # gsz, gsm
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        # gfm
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        ones(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NSE, T}),
        # cbl
        ones(MVector{NSE, vofi_int}), ones(MVector{NSE, vofi_int}),
        zero(MVector{NSE, T}), zero(MVector{NDIM, T}), MinData{T}(),
        # cbs
        ones(MVector{NSE, vofi_int}), ones(MVector{NSE, vofi_int}), ones(MVector{NSE, vofi_int}),
        zero(MVector{NVER, T}), zero(MVector{NDIM, T}), MinData{T}(),
        # css
        zero(MVector{NDIM, T}), zero(MVector{NSE, T}), MinData{T}(),
        # csf
        zero(MVector{NDIM, T}), zero(MVector{NVER, T}), MinData{T}(),
        # cts
        zero(MVector{NDIM, T}), zero(MVector{NSE, T}), MinData{T}(),
        # csc, cfc, clc, cec
        zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NVER, T}),
        DirData(0, 0, 0, 0, 0),
        zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{4, T}),
        # il
        zero(MVector{4, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        # is
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{4, T}),
        # ep
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{4, T}),
        # edp
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{4, T}), zero(MVector{NSE, T}),
        # cc top-level
        zero(MVector{NDIM, T}), zero(MVector{NDIM, T}), zero(MVector{NDIM, T}),
        zero(MVector{NDIM, T}), zero(MMatrix{NSE, NSE, T}),
        zero(MArray{Tuple{NSE, NSE, NSE}, T}), zero(MVector{NSEG + 1, T}),
        zero(MVector{NSEG, vofi_int}), zero(MVector{NSEG, vofi_int}),
        zero(MVector{NDIM + 1, T}), LenData{T}(), LenData{T}(),
        MinData{T}(), MinData{T}(), MinData{T}(), MinData{T}(), MinData{T}(),
        zero(MVector{NSE, T}), zero(MVector{NDIM, T}),
        # od4 (order_dirs_4D)
        zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4),
        zeros(vofi_int, 4), zeros(vofi_int, 2, 2, 2, 2), zeros(T, NSE, NSE, NSE, NSE),
        # od4 populate helpers
        zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4),
        # cbh (check_boundary_hypersurface)
        zeros(T, 2, 2, 2), zeros(T, 4),
        ones(vofi_int, 2), ones(vofi_int, 2), ones(vofi_int, 2), ones(vofi_int, 2),
        # gl4 (get_limits_4D)
        T[],
        # gh (get_hypervolume)
        zeros(T, NDIM), zeros(T, NDIM), zeros(T, NDIM + 1),
        zeros(vofi_int, 2), zeros(vofi_int, 2), zeros(T, 4),
        # ct4 (cell_type_4D)
        zeros(vofi_int, 2, 2, 2, 2), zeros(T, 2, 2, 2, 2), zeros(T, 4), zeros(T, 4),
        # cc4 (get_cc 4D top-level)
        zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4), zeros(T, 4),
        zeros(T, NSE, NSE, NSE, NSE), XFSP4D{T}(), zeros(T, NSEG + 1), zeros(T, 5),
    )
end

VofiWorkspace() = VofiWorkspace{vofi_real}()

# Per-thread workspace cache for the production `Float64` path. A `VofiWorkspace`
# is reused (its scratch is reset inside the converted functions exactly where
# the original code rebuilt it), so a plain `for cell; vofi_get_cc(...)` loop —
# not just the batch driver — pays ~zero heap allocation per cell, matching the
# C library. SAFETY: the workspace is used *synchronously* within a single
# `vofi_get_cc`/`vofi_get_cell_type` call (pure compute, no yield/I/O points), so
# the running task cannot migrate threads mid-call and a per-thread instance is
# never touched by two concurrently-running tasks. The only nesting (4D→3D) does
# not keep workspace state live in the 4D frame, so the nested 3D call safely
# reuses the same instance. Non-`Float64` element types (AD duals, `Float32`)
# allocate a fresh workspace — those paths are rare and not performance-critical.
const _VOFI_WS_F64 = Vector{VofiWorkspace{Float64}}()
const _VOFI_WS_LOCK = ReentrantLock()

function _thread_workspace(::Type{Float64})
    tid = Threads.threadid()
    if tid > length(_VOFI_WS_F64)
        Base.@lock _VOFI_WS_LOCK begin
            while length(_VOFI_WS_F64) < tid
                push!(_VOFI_WS_F64, VofiWorkspace{Float64}())
            end
        end
    end
    return @inbounds _VOFI_WS_F64[tid]
end

# Non-Float64 (AD / reduced precision): allocate a fresh workspace.
_thread_workspace(::Type{T}) where {T} = VofiWorkspace{T}()

"""
    IntegrandCall(func, par, one_arg::Bool)

Concrete, type-stable wrapper around a level-set function and its parameter.

The level set can be supplied either as `func(coords)` (1-arg) or
`func(coords, par)` (2-arg). Historically this was resolved on *every*
evaluation with `applicable(func, coords)`, whose runtime reflection allocated
~62 bytes per call — multiplied by the thousands of integrand evaluations in a
single cut cell. `IntegrandCall` resolves the arity **once** at the entry point
(storing the result in `one_arg`) and is a concrete type, so the inner loops pay
only a cheap, allocation-free branch.
"""
struct IntegrandCall{F, P}
    f::F
    par::P
    one_arg::Bool
end

# Pass an immutable `SVector` snapshot of the coordinates to the user level set.
# The integrand only reads the point, so handing it a stack-allocated immutable
# (instead of the caller's mutable `MVector`) means the `MVector` no longer
# escapes through this dynamic-dispatch boundary and can stay on the stack.
@inline _snap(coords::MVector{N}) where {N} = SVector{N}(coords)
@inline _snap(coords) = coords
@inline (ic::IntegrandCall)(coords) =
    (c = _snap(coords); ic.one_arg ? ic.f(c) : ic.f(c, ic.par))

"""
    wrap_integrand(func, par, sample_coords) -> IntegrandCall

Resolve, once, whether `func` is called as `func(coords)` or `func(coords, par)`
and return a concrete [`IntegrandCall`](@ref). `sample_coords` is any coordinate
vector of the right dimension used solely for the `applicable` probe.
"""
@inline function wrap_integrand(func, par, sample_coords)
    one_arg = par === nothing && applicable(func, sample_coords)
    return IntegrandCall(func, par, one_arg)
end
# Idempotent: re-wrapping an already-wrapped integrand leaves it unchanged.
@inline wrap_integrand(ic::IntegrandCall, par, sample_coords) = ic

# Fast path: an IntegrandCall already knows its arity — ignore the passed `par`.
@inline call_integrand(ic::IntegrandCall, par, coords) = ic(coords)

# Fallback for raw functions (e.g. direct internal calls / unit tests): preserve
# the original 1-arg-when-par-nothing behaviour.
@inline function call_integrand(func, par, coords)
    if par === nothing && applicable(func, coords)
        return func(coords)
    end
    return func(coords, par)
end
