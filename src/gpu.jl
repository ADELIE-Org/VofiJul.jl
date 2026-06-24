"""
GPU / accelerator support via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl).

The per-cell volume-fraction computation is embarrassingly parallel — each cell is
independent — so it maps directly onto a data-parallel kernel: one work-item per
cell. The kernel is backend-agnostic; pass `KernelAbstractions.CPU()` (default,
multithreaded), or a GPU backend such as `CUDABackend()` / `ROCBackend()` when the
corresponding package is loaded and hardware is available.

Each work-item builds its own [`VofiWorkspace`](@ref) and output scratch, so the
computation is fully reentrant (the same property that makes the threaded
[`vofi_get_cc_batch`](@ref) safe).

!!! note "Inputs must be accelerator-friendly"
    Coordinates are passed as `SVector`s (isbits, register-resident) and the level
    set is pre-wrapped once into a concrete [`IntegrandCall`](@ref) so the kernel
    never runs `applicable` reflection. The level-set function itself must be GPU
    compatible (pure, allocation-free, no captured boxed state) to run on a GPU
    backend; on the `CPU()` backend any Julia function works.
"""

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

@kernel function _vofi_cc_kernel!(out, @Const(xins), h0pad, ic, par,
                                  nex, npt, nvis, ::Val{ndim0}, ::Type{T}) where {ndim0, T}
    i = @index(Global)
    @inbounds begin
        ws = VofiWorkspace{T}()
        xex = zero(MVector{5, T})
        out[i] = vofi_get_cc(ws, ic, par, xins[i], h0pad, xex, nex, npt, nvis, ndim0)
    end
end

"""
    vofi_get_cc_gpu(impl_func, par, xins, h0, ndim0; backend=CPU(), nex, npt, nvis, workgroupsize=64)

Compute the volume fraction of every cell in `xins` in parallel on `backend`,
returning a backend array of length `length(xins)`.

`xins` is a vector of cell origins (each indexable with `ndim0` coordinates) and
`h0` the cell size (length `ndim0`). The returned array lives on `backend`; call
`Array(result)` to bring GPU results back to the host. `nex`/`npt`/`nvis` mirror
the scalar [`vofi_get_cc`](@ref) options (centroid/interface-measure, quadrature
order hints, tecplot flags) and default to "off".

The same call runs on the multithreaded CPU backend or a GPU backend:

```julia
using KernelAbstractions
cc = vofi_get_cc_gpu(sdf, nothing, origins, (h, h), 2)            # CPU()
# using CUDA;  cc = vofi_get_cc_gpu(sdf, nothing, origins, (h,h), 2; backend=CUDABackend())
```
"""
function vofi_get_cc_gpu(impl_func, par, xins, h0, ndim0;
                         backend = KA.CPU(),
                         nex = (0, 0), npt = (0, 0, 0, 0), nvis = (0, 0),
                         workgroupsize = 64)
    n = length(xins)
    n == 0 && return KA.allocate(backend, promote_type(eltype(h0), Float64), 0)
    T = promote_type(eltype(eltype(xins)), eltype(h0))
    XV = SVector{ndim0, T}

    # Host-side, accelerator-friendly inputs.
    xin_h = XV[XV(ntuple(d -> T(x[d]), ndim0)) for x in xins]
    h0p = XV(ntuple(d -> T(h0[d]), ndim0))
    nexv = SVector{2, Int}(nex[1], nex[2])
    nptv = SVector{4, Int}(npt[1], npt[2], npt[3], npt[4])
    nvisv = SVector{2, Int}(nvis[1], nvis[2])
    # Resolve the integrand arity ONCE on the host (no reflection in the kernel).
    ic = wrap_integrand(impl_func, par, xin_h[1])

    # Move inputs / outputs onto the backend (no-op copies on CPU()).
    xin_d = KA.allocate(backend, XV, n)
    copyto!(xin_d, xin_h)
    out = KA.allocate(backend, T, n)

    kernel! = _vofi_cc_kernel!(backend, workgroupsize)
    kernel!(out, xin_d, h0p, ic, par, nexv, nptv, nvisv, Val(ndim0), T; ndrange = n)
    KA.synchronize(backend)
    return out
end
