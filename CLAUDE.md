# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Package

**VofiJul.jl** — Julia interface to the VOFI library for computing volume
fractions of implicitly-defined interfaces. Part of the
[ADELIE-org](https://github.com/ADELIE-org) project.

- Package name (module): `VofiJul`
- Repo directory: `VofiJul.jl`
- UUID: `9ebbbdc3-ccde-421a-bbc9-ca81de7cf635`

## Layout

```
src/VofiJul.jl             # the module — all source lives here (or files it includes)
test/runtests.jl           # test entry point (uses Test)
docs/make.jl               # Documenter build script
docs/src/                  # documentation pages (index.md, reference.md)
.github/workflows/ci.yml   # CI: build + test + codecov
.github/workflows/Docs.yml # docs build + deploy
```

`Test` is a test-only dependency: it lives in `[extras]` + `[targets]` of
`Project.toml`, never in `[deps]`. Runtime deps go in `[deps]` with a `[compat]`
entry.

## Common commands

Run from the repo root.

```bash
# Instantiate the project environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the test suite (what CI runs)
julia --project=. -e 'using Pkg; Pkg.test()'

# Build the docs locally
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.resolve(); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Conventions

- Idiomatic Julia: multiple dispatch over type-checking, type-stable functions,
  immutable structs, `StaticArrays` for small fixed-size data. Add a docstring to
  every exported symbol and cover edge cases in tests. See the `julia-language`
  skill.
- Public API symbols are picked up automatically by `docs/src/reference.md`
  (`@autodocs`), so docstrings are the documentation.
- `docs/make.jl` runs with `warnonly = true` and only deploys when `ENV["CI"] == "true"`.
- When adding a docs `@example`/`using` dependency, also add it to `docs/Project.toml`.

## Secrets (CI, set in repo settings)

- `CODECOV_TOKEN` — coverage upload.
- `DOCUMENTER_KEY` — docs deploy. Neither can be tested locally.

## Architecture: performance, AD, threading, GPU

This is a pure-Julia port of the Vofi cut-cell volume-fraction C library (original
C at `/home/libat/github/vofi`, 3D-only; 4D was added here, validated against
analytic volumes only). It has been refactored along four axes. **Keep all of
these properties intact when editing — the test suite is the oracle.**

- **Element-type genericity (`{T}`).** Everything flows a numeric type `T`
  (`Float64` default; `Float32`; `ForwardDiff.Dual`). `vofi_real` is just the
  default. Scratch structs (`MinData`, `LenData`, `XFSP4D`, …) and the workspace
  are parametric. Constants used only for comparison/scaling (`EPS_*`, `MIN_GRAD`)
  stay `Float64`; counters and `DirData` stay `Int`. Don't bake `Float64` into new
  scratch.
- **Allocation: reused workspace.** `mutable struct VofiWorkspace{T}`
  (`vofi_stddecl.jl`) holds one named field per (function, scratch-temporary) and
  is threaded as the **first argument** through the whole 2D/3D (and now 4D) call
  graph. **Invariant: distinct field per function — never share a field between two
  functions** (safe only because no 2D/3D function self-recurses; 4D self-recurses
  4D→3D but keeps live only `cc4_/gh_/od4_` fields, disjoint from the 2D/3D/`cc_`
  fields the nested 3D call clobbers). The public `vofi_get_cc`/`vofi_get_cell_type`
  fetch a **per-thread cached** workspace (`_thread_workspace`), so even a plain
  `for cell; vofi_get_cc(...)` loop allocates ~0/cell. Per-cut-cell allocation:
  2D ~208 B, 3D ~304 B (down from 81 KB / 2.6 MB originally); `vofi_interface_centroid`
  ~0 B. Allocation gotchas / levers learned:
  - **`fill!(::StaticArrays.MArray, x)` BOXES ~32 B per call** (a StaticArrays
    quirk). Use the `zfill!` helper (`vofi_stddecl.jl`) for every per-cell static
    reset — it does an explicit `@inbounds @simd` loop on a local binding (0 B) and
    falls back to plain `fill!` for ordinary `Vector`s. Never write `fill!(ws.field, …)`.
  - Indexing a heap-struct `MArray` *field* by a **runtime** index boxes on both read
    and write (only constant indices are free). But binding it to a local first
    (`s = ws.field; s[i] = …`) is free. Once-per-cell stencils (the `order_dirs`
    27-pt 3D + 9-pt 2D `fc`/`nc`, and the boundary marker `n0`) are built as stack
    `SArray`s via `ntuple(_, Val(N))` (unrolls → constant indices), not workspace fields.
  - A hot loop that calls through a **passed-in function/closure** needs an explicit
    `::F` type parameter to force specialization — otherwise Julia's "don't specialize
    `Function` args" heuristic makes the inner call dynamically dispatch and box every
    result + arg (this was the entire `vofi_interface_centroid` allocation; inference
    via `return_types` looked clean but the *specialized* method still boxed).
  - The integrand is wrapped once into a concrete `IntegrandCall` (no per-eval
    `applicable` reflection) and handed an immutable `SVector` snapshot so the
    caller's `MVector` doesn't escape.
- **AD.** Two paths, **both work in 1D/2D/3D/4D** (FD-verified in `test_ad.jl`):
  (1) forward-through `ForwardDiff` for **geometry/shape sensitivity** — derive `T`
  from the cell geometry (`xin`/`h0`), a `Dual` flows through the root-finder + quadrature
  naturally. Do NOT sample the integrand to derive `T` (it regressed 4D). NB: a
  *degenerate* cell whose interface is entirely interior (all corners same sign) has a
  constant `cc`, so its shape derivative is genuinely 0 — test with a genuinely-cut cell.
  (2) **Level-set parameter sensitivity** = the custom shape-derivative rule in
  `shape_derivative.jl` (`vofi_cc`, `vofi_cc_and_grad`, ChainRulesCore
  `rrule`/`frule`) via the Reynolds-transport identity — never differentiates
  through the root-finder.
- **Threading.** `batch.jl` (`vofi_get_cc_batch`, `vofi_get_cell_type_batch`):
  `Threads.@threads` with a per-task workspace + xex buffer. Reentrant because all
  scratch is per-call/per-task (no module-global mutable state).
- **GPU.** `gpu.jl` (`vofi_get_cc_gpu`): a backend-agnostic KernelAbstractions
  kernel, one work-item per cell, each building its own workspace. Validated on the
  `CPU()` backend only. **It does NOT yet run on a real device** — the kernel builds
  a `VofiWorkspace{T}()` per work-item (~20 KB heap each), which a GC-less GPU can't
  do (see roadmap). The same kernel *will* target `CUDABackend()`/`ROCBackend()` once
  the workspace is made stack-resident.

## Remaining / roadmap

- **Real on-device GPU (the big one, not done).** The `CPU()` backend works today,
  but actual CUDA/ROCm needs the per-work-item scratch to be GC-free on device.
  Measured: the kernel allocates **~20 KB heap per work-item** (`VofiWorkspace{T}()`),
  so it cannot launch on a GC-less device. The mutable heap structs (`VofiWorkspace`,
  `MinData`, `LenData`) and the 4D dynamic `Vector`s/`XFSP4D` are the blockers.
  Completing GPU = a **large functional rewrite** of ~30 root-finder-heavy functions
  to immutable/stack `SArray`-based scratch (mutation → rebuild). High regression risk
  and **its GPU payoff can't be validated without GPU hardware** — do it on a branch,
  oracle-guarded, on a real GPU. Bounded first increment: convert the 2D analytic path
  to an immutable workspace, CPU-oracle-guarded, before extending to 3D/4D. (The
  `order_dirs` 2D/3D stencils + the `n0` marker are already immutable `SArray` as first steps.)
- Smaller: residual ~304 B/3D-cell is dominated by the `xfsp = (xfsp1..xfsp5)`
  `NTuple{5,MinData}` boxing at the `vofi_order_dirs_3D`/`vofi_get_limits_3D` call
  boundary (passing a tuple of mutable refs by value heap-boxes; it's runtime-indexed
  deep so it can't be split into 5 args) plus Base-inlined/kwargs bits — at the floor
  without a multi-function signature refactor. 1D path is not workspaced (it barely allocates).
- Broader ADELIE roadmap: refactor `CartesianGeometry.jl` (router + traits +
  threading), then operators (a colleague's work), then propagate the new
  `CartesianGrids.jl` across the Penguin packages.
