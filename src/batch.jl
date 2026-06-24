"""
Multithreaded batch driver for cell-wise VOF computations.

`vofi_get_cc` carries no module-global mutable state — every scratch structure
(`MinData`, `LenData`, `XFSP4D`, the `@MVector`/`@MArray` locals) is allocated
per call. It is therefore reentrant and safe to evaluate concurrently on
disjoint cells. These helpers fan a list of cells across `Threads.@threads`,
giving each task its own `xex` output buffer so there is no cross-thread sharing.
"""

"""
    vofi_get_cc_batch(impl_func, par, xins, h0, ndim0; nex, npt, nvis) -> (cc, xex)

Compute the volume fraction for a collection of cells in parallel.

# Arguments
- `xins`: an `AbstractVector` of cell lower corners; each entry is indexable with
  `ndim0` coordinates (e.g. an `SVector`, `NTuple`, or `Vector`).
- `h0`: cell sizes (shared across all cells), length ≥ `ndim0`.
- `ndim0`: spatial dimension (1–4).

# Keywords mirror [`vofi_get_cc`](@ref)
- `nex = (0, 0)`, `npt = (0, 0, 0, 0)`, `nvis = (0, 0)`.

# Returns
- `cc::Vector{T}`: volume fraction per cell.
- `xex::Matrix{T}`: per-cell extra output (centroid coords + interface measure),
  one column per cell, `max(4, ndim0 == 4 && nex[2] > 0 ? 5 : 4)` rows.

Each thread uses a private `xex` column, so the routine is data-race free.
"""
function vofi_get_cc_batch(impl_func, par, xins::AbstractVector, h0, ndim0;
                           nex = (0, 0), npt = (0, 0, 0, 0), nvis = (0, 0))
    ncell = length(xins)
    T = promote_type(eltype(eltype(xins)), eltype(h0), vofi_real)
    xex_len = (ndim0 == 4 && length(nex) >= 2 && nex[2] > 0) ? 5 : max(4, ndim0)
    cc = Vector{T}(undef, ncell)
    xex = zeros(T, xex_len, ncell)
    nexv = collect(Int, nex)
    nptv = collect(Int, npt)
    nvisv = collect(Int, nvis)
    h0v = collect(T, h0)
    # Partition the cells into one contiguous chunk per task and give each chunk a
    # single reusable workspace, reused across every cell in the chunk. Steady-state
    # per-cell allocation is then ~zero (only the small per-cell xex buffer). Chunking
    # (rather than indexing by threadid) is robust to dynamic task scheduling.
    nchunks = min(Threads.nthreads(), ncell)
    nchunks = max(nchunks, 1)
    chunks = collect(Iterators.partition(1:ncell, cld(ncell, nchunks)))
    Threads.@threads for chunk in chunks
        ws = VofiWorkspace{T}()             # one workspace per task, reused below
        xex_c = Vector{T}(undef, xex_len)   # task-private output buffer
        for c in chunk
            cc[c] = vofi_get_cc(ws, impl_func, par, xins[c], h0v, xex_c,
                                nexv, nptv, nvisv, ndim0)
            @inbounds xex[:, c] .= xex_c
        end
    end
    return cc, xex
end

"""
    vofi_get_cell_type_batch(impl_func, par, xins, h0, ndim0) -> Vector{Int}

Classify a collection of cells (full `1`, cut `-1`, empty `0`) in parallel.
See [`vofi_get_cell_type`](@ref). Reentrant; one task per cell.
"""
function vofi_get_cell_type_batch(impl_func, par, xins::AbstractVector, h0, ndim0)
    ncell = length(xins)
    out = Vector{Int}(undef, ncell)
    Threads.@threads for c in 1:ncell
        out[c] = vofi_get_cell_type(impl_func, par, xins[c], h0, ndim0)
    end
    return out
end
