# VofiJul.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ADELIE-org.github.io/VofiJul.jl/dev)
![CI](https://github.com/ADELIE-org/VofiJul.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/ADELIE-org/VofiJul.jl/graph/badge.svg)](https://codecov.io/gh/ADELIE-org/VofiJul.jl)

A pure-Julia port of the VOFI algorithm for computing the **volume fraction**
(wetted fraction) of a cell cut by an implicitly-defined interface `φ(x) = 0`, plus
the interface centroid and surface measure. Supports **1D, 2D, 3D and 4D** cells.

## Features

- **Volume fractions & moments** for cells cut by any level set: `vofi_get_cc`
  (fraction + optional centroid / interface measure), `vofi_get_cell_type`
  (full / empty / cut), `vofi_interface_centroid`.
- **Dimensions 1–4** (1D length, 2D area, 3D volume, 4D hypervolume).
- **Automatic differentiation** (forward-verified against finite differences in
  **all** of 1D/2D/3D/4D).
  - Geometry/shape sensitivity for free via `ForwardDiff` (feed `Dual`
    coordinates — the whole pipeline, including the root-finder and quadrature, is
    element-type generic).
  - Level-set *parameter* sensitivity (adjoint) via a custom shape-derivative
    rule: `vofi_cc`, `vofi_cc_and_grad` (+ `ChainRulesCore` `rrule`/`frule`),
    using the Reynolds-transport identity rather than differentiating the
    root-finder.
- **Element types:** `Float64` (default), `Float32`, or AD `Dual` flow end-to-end.
- **Low allocation:** scratch is reused through a per-thread workspace, so a tight
  per-cell loop allocates ≈0 — about **0.2 KB per *cut* cell in 2D, 0.3 KB in 3D**;
  `vofi_interface_centroid` is allocation-free; full/empty cells ≈0.
- **Multithreading:** `vofi_get_cc_batch`, `vofi_get_cell_type_batch`.
- **GPU / accelerators:** `vofi_get_cc_gpu` — a backend-agnostic
  [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernel
  (one cell per work-item). Runs on the multithreaded `CPU()` backend today; running
  on-device (`CUDABackend()` / `ROCBackend()`) still needs the per-work-item scratch
  made GC-free (see **Status**).

## Installation

Part of the [ADELIE](https://github.com/ADELIE-org) project. After cloning:

```julia
using Pkg
Pkg.dev("VofiJul")
```

## Quick Example

```julia
using VofiJul

# Level set for a circle of radius 0.4 (2-arg form f(x, par); 1-arg f(x) also works)
circle(x, _) = sqrt(x[1]^2 + x[2]^2) - 0.4

# Volume fraction of the cell [0.3,0.4] × [0.0,0.1]
xex = zeros(4)                                   # output scratch (centroid / measure)
cc = vofi_get_cc(circle, nothing,
                 [0.3, 0.0],                     # cell origin
                 [0.1, 0.1],                     # cell size h
                 xex,
                 [0, 0], [0, 0, 0, 0], [0, 0],   # nex (moments), npt (order hints), nvis
                 2)                              # ndim
# cc ∈ [0,1]: fraction of the cell inside φ < 0

# Integrate the circle area over a grid  (≈ π·0.4²)
n = 64; h = 1/n
area = sum(vofi_get_cc(circle, nothing, [-0.5 + i*h, -0.5 + j*h], [h, h],
                       zeros(4), [0,0], [0,0,0,0], [0,0], 2) * h^2
           for i in 0:n-1, j in 0:n-1)
```

### Threaded batch

```julia
origins = [[-0.5 + i*h, -0.5 + j*h] for i in 0:n-1 for j in 0:n-1]
ccs = vofi_get_cc_batch(circle, nothing, origins, [h, h], 2)   # uses Threads.@threads
```

### GPU / KernelAbstractions

```julia
using KernelAbstractions
ccs = vofi_get_cc_gpu(circle, nothing, origins, (h, h), 2; backend = CPU())
# using CUDA;  ccs = vofi_get_cc_gpu(circle, nothing, origins, (h, h), 2; backend = CUDABackend())
```

### Differentiating w.r.t. a level-set parameter

```julia
φθ = (x, θ) -> sqrt(x[1]^2 + x[2]^2) - θ        # circle of radius θ
cc, dcc = vofi_cc_and_grad(0.3, φθ, [0.0, 0.0], [1.0, 1.0], 2)  # cc and d(cc)/dθ
```

## Status

CPU (serial, threaded) and the `CPU()` KernelAbstractions backend are complete and
tested. On-device GPU execution (CUDA/ROCm) needs the per-work-item scratch to be
GC-free; that immutable rewrite is the remaining GPU step (see `CLAUDE.md`).

## License

This package is part of the ADELIE project. See [LICENSE](LICENSE).
