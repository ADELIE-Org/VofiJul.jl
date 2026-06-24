module VofiJul

using StaticArrays

include("vofi_stddecl.jl")
include("vofi_gl_nodes.jl")
include("vofi_gl_weights.jl")
include("utils.jl")
include("checkconsistency.jl")
include("getmin.jl")
include("checkboundary.jl")
include("gettype.jl")
include("getzero.jl")
include("getintersections.jl")
include("getlimits.jl")
include("triangulate.jl")
include("integrate.jl")
include("orderdirs.jl")
include("getarclength.jl")
include("tecplot.jl")
include("getcc.jl")
include("interface_centroid.jl")
include("batch.jl")
include("shape_derivative.jl")
include("gpu.jl")

export vofi_real, vofi_creal, vofi_int, vofi_cint, vofi_void_cptr,
       vofi_int_cpt, Integrand, MinData, DirData, LenData, VofiWorkspace,
       EPS_M, EPS_LOC, EPS_E, EPS_SEGM, EPS_ROOT, EPS_NOT0,
       NEAR_EDGE_RATIO, MAX_ITER_ROOT, MAX_ITER_MINI,
       NDIM, NVER, NSE, NSEG, NGLM,
       MIN, MAX, SGN0P, Sq, Sq2, Sq3, Sqd3,
       @SHFT4, @CPSF,
       GL_MIN_ORDER, GL_MAX_ORDER,
       gauss_legendre_nodes, gauss_legendre_weights,
       vofi_get_cell_type, vofi_get_cc, vofi_interface_centroid,
       vofi_get_cc_batch, vofi_get_cell_type_batch,
       vofi_cc, vofi_cc_and_grad,
       vofi_get_cc_gpu

end # module
