```@meta
CurrentModule = VofiJul
```

# VofiJul.jl

## Installation

```julia
using Pkg
Pkg.add("VofiJul")
```

## Overview

A pure-Julia port of the VOFI algorithm for computing the **volume fraction**
(wetted fraction) of a cell cut by an implicitly-defined interface `φ(x) = 0`, plus
the interface centroid and surface measure, in **1D, 2D, 3D and 4D**.

Highlights:

- Element-type generic: `Float64` (default), `Float32`, or `ForwardDiff.Dual`
  flow end-to-end, so **geometry/shape sensitivity** comes for free via
  forward-mode AD — FD-verified in all of 1D/2D/3D/4D.
- **Level-set parameter sensitivity (adjoint)** via a custom shape-derivative rule
  (`vofi_cc_and_grad`, with `ChainRulesCore` `rrule`/`frule`) — based on the
  Reynolds-transport identity, not differentiation through the root-finder.
- **Low allocation** (scratch reused through a per-thread workspace; ≈0.2–0.3 KB
  per cut cell in 2D/3D, allocation-free `vofi_interface_centroid`),
  **multithreaded batch** drivers, and a backend-agnostic **KernelAbstractions**
  kernel (`vofi_get_cc_gpu`, `CPU()` backend today — on-device GPU pending a
  GC-free workspace).

## Quick Example

```julia
using VofiJul

circle(x, _) = sqrt(x[1]^2 + x[2]^2) - 0.4     # circle of radius 0.4

xex = zeros(4)
cc = vofi_get_cc(circle, nothing, [0.3, 0.0], [0.1, 0.1], xex,
                 [0, 0], [0, 0, 0, 0], [0, 0], 2)   # fraction of the cell inside φ < 0
```

Integrate over a grid, run a threaded batch, or differentiate w.r.t. a level-set
parameter:

```julia
# Threaded batch over many cells
n = 64; h = 1/n
origins = [[-0.5 + i*h, -0.5 + j*h] for i in 0:n-1 for j in 0:n-1]
ccs = vofi_get_cc_batch(circle, nothing, origins, [h, h], 2)

# GPU / accelerator kernel (CPU() backend today; on-device CUDA/ROCm pending)
using KernelAbstractions
ccs_gpu = vofi_get_cc_gpu(circle, nothing, origins, (h, h), 2; backend = CPU())

# d(volume fraction)/d(radius θ)
φθ = (x, θ) -> sqrt(x[1]^2 + x[2]^2) - θ
cc, dcc = vofi_cc_and_grad(0.3, φθ, [0.0, 0.0], [1.0, 1.0], 2)
```

## Main Sections

- [API Reference](reference.md)
