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
  2D ~416 B, 3D ~432 B, 4D ~6 KB (down from 81 KB / 2.6 MB / 48 MB). Gotchas
  learned: indexing a heap-struct `MArray` field by a **runtime** index boxes on
  *both* read and write (only constant indices are free) — so once-per-cell big
  arrays (e.g. the `order_dirs` 27-point stencil) are built as stack `SArray`s via
  `ntuple(_, Val(N))` (unrolls → constant indices), not workspace fields. The
  integrand is wrapped once into a concrete `IntegrandCall` (no per-eval
  `applicable` reflection) and handed an immutable `SVector` snapshot so the
  caller's `MVector` doesn't escape.
- **AD.** Two paths: (1) forward-through `ForwardDiff` for **geometry/shape
  sensitivity** — derive `T` from the cell geometry (`xin`/`h0`), a `Dual` flows
  through naturally. Do NOT sample the integrand to derive `T` (it regressed 4D).
  (2) **Level-set parameter sensitivity** = the custom shape-derivative rule in
  `shape_derivative.jl` (`vofi_cc`, `vofi_cc_and_grad`, ChainRulesCore
  `rrule`/`frule`) via the Reynolds-transport identity — never differentiates
  through the root-finder.
- **Threading.** `batch.jl` (`vofi_get_cc_batch`, `vofi_get_cell_type_batch`):
  `Threads.@threads` with a per-task workspace + xex buffer. Reentrant because all
  scratch is per-call/per-task (no module-global mutable state).
- **GPU.** `gpu.jl` (`vofi_get_cc_gpu`): a backend-agnostic KernelAbstractions
  kernel, one work-item per cell, each building its own workspace. Validated on the
  `CPU()` backend (no GPU hardware in dev env). The same kernel targets
  `CUDABackend()`/`ROCBackend()`.

## Remaining / roadmap

- **Real on-device GPU (the big one, not done).** The `CPU()` backend works today,
  but actual CUDA/ROCm needs the per-work-item scratch to be GC-free on device. The
  mutable heap structs (`VofiWorkspace`, `MinData`, `LenData`) and the 4D dynamic
  `Vector`s/`XFSP4D` are GC-hostile on device. Completing GPU = a **large functional
  rewrite** of ~30 root-finder-heavy functions to immutable/stack `SArray`-based
  scratch (mutation → rebuild). High regression risk and **its GPU payoff can't be
  validated without GPU hardware** — do it on a branch, oracle-guarded, on a real
  GPU. (The `order_dirs` stencil is already immutable `SArray` as a first step.)
- Smaller: residual ~432 B/3D-cell is small `getcc` call-site bits + measurement
  noise — at the floor without the full functional rewrite. 1D path is not
  workspaced (it barely allocates).
- Broader ADELIE roadmap: refactor `CartesianGeometry.jl` (router + traits +
  threading), then operators (a colleague's work), then propagate the new
  `CartesianGrids.jl` across the Penguin packages.
