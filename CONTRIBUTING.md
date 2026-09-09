# Contributing to xchplot2

Thanks for taking the time. A few notes to keep review loops short.

## Building + running the tests

Build and run the parity tests following the
[Build](https://github.com/Jsewill/xchplot2#build) section of the
README. The parity binaries under `tools/parity/` are the correctness
gate:

- `aes_parity`, `xs_parity`, `t1_parity`, `t2_parity`, `t3_parity` —
  bit-exact CPU vs GPU per-phase agreement with pos2-chip's reference.
- `sycl_sort_parity`, `sycl_g_x_parity`, `sycl_bucket_offsets_parity` —
  portable kernel parity on the selected SYCL backend. Vendor-specific
  compilation and driver behavior require a GPU from that vendor.
- `plot_file_parity` — writer + reader round-trip on the final
  `.plot2`.

Any change that touches a kernel, the sort path, or the plot file
format **must** keep the parity tests passing at k=22 (quick) and at
k=28 (slow — the realistic production k). Output bytes are specified
to be identical to the pos2-chip CPU reference; this is the hard
invariant.

After a functional change, spot-check one real batch end-to-end with
`xchplot2 verify <plot>` — zero proofs over 100 random challenges is
a regression even if all parity tests pass.

## GPU CI

Hosted PR jobs lint and build without GPU hardware. `GPU hardware` runs only
on trusted `main` / `cuda-only` pushes, schedules, and manual dispatches:

| Suite | When | Checks |
| --- | --- | --- |
| `quick` | Branch pushes; manual | All CTest tests, then k=18 CPU byte parity and 100 full-proof challenges for every tier and disk-spill variant |
| `vram` | Daily, 04:17 UTC; manual | Quick CTest tests, three k=28 plots at each tier's budget, rejection 1 MiB below it, then k=28 CPU byte parity and full proofs for every tier and spill variant |
| `physical` | Sunday, 07:47 UTC; manual | Actual 2/4/6/8 GiB capacity, every k=28 tier that fits, plus three uncapped auto-tier plots with full-proof verification |

The default branch schedules both `main` and `cuda-only`, because
[GitHub schedules run only on the default branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
Pushes to `cuda-only` also run that branch's quick suite directly.
Both branches must contain the GPU CI changes before enabling the schedule.

The SYCL branch tests five tiers and four disk variants. Native CUDA tests
four tiers, Compact's spill engine, and Minimal's file mappings; native Tiny
cannot spill its GPU-visible host tables.

Provide Linux x64 self-hosted runners with these additional labels:

| Label | Hardware / toolchain |
| --- | --- |
| `gpu-cuda` | NVIDIA, CUDA toolkit, AdaptiveCpp with its CUDA backend; enough free VRAM for every k=28 tier |
| `gpu-hip` | AMD, ROCm, AdaptiveCpp with its HIP backend; enough free VRAM for every k=28 tier |
| `gpu-level-zero` | Intel, Level Zero runtime and Sysman, AdaptiveCpp with its Level Zero backend and LLVM SPIR-V translator; enough free VRAM for every k=28 tier |
| `gpu-cuda-2gb`, `gpu-cuda-4gb`, `gpu-cuda-6gb`, `gpu-cuda-8gb` | Physical NVIDIA cards with those capacities; use CUDA 12.9 for Pascal cards |
| `gpu-hip-4gb`, `gpu-level-zero-8gb` | Physical 4 GiB AMD and 8 GiB Intel cards with the same vendor toolchains as above |

The weekly physical matrix covers these four NVIDIA capacities plus AMD 4 GiB
and Intel 8 GiB on `main`, and all four NVIDIA capacities on `cuda-only`. Set the Actions variable
`GPU_PHYSICAL_MATRIX` to match a different fleet, including AMD and Intel:

```json
{"include":[{"backend":"cuda","ref":"main","runner":"my-2gb-nvidia","vram_mib":2048},{"backend":"hip","ref":"main","runner":"my-4gb-amd","vram_mib":4096},{"backend":"level_zero","ref":"main","runner":"my-8gb-intel","vram_mib":8192},{"backend":"cuda","ref":"cuda-only","runner":"my-2gb-nvidia","vram_mib":2048}]}
```

Keep entries for the capacities and vendors you intend to certify. Missing
runner labels leave jobs queued; no GPU or a wrong backend fails the job.
Physical capacity comes from the device inventory, so a software-capped 24 GiB
GPU cannot pass a 2 GiB lane. A physical card with no fitting k=28 tier fails.

Install the project's build dependencies before registering runners: CMake
3.24+, a supported C++ compiler, Rust, Python 3.11+, and the matching GPU
toolchain. `scripts/install-deps.sh --gpu nvidia|amd|intel` covers the SYCL
build; follow the README for vendor driver setup. Pin runner images and
toolchain versions, and update them deliberately. Put custom toolchain paths
in the runner service environment (`PATH`, `CMAKE_PREFIX_PATH`, `CUDACXX`,
`CXX`, `CUDAHOSTCXX`, `LD_LIBRARY_PATH` as needed).

Use dedicated ephemeral runners with one visible GPU per job and sufficient
host RAM and fast scratch storage; 64 GiB host RAM and 40 GiB scratch are a
useful starting point for the k=28 suites. Configure device visibility in the
runner environment (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, or
`ZE_AFFINITY_MASK`). Jobs set the AdaptiveCpp backend mask themselves and fail
if zero or multiple GPUs remain visible. No PR workflow targets these hosts.
The harness serializes GPU tests per host; other GPU workloads, including a
desktop, can still inflate driver memory measurements.

`gpu_ci_info` obtains each k=28 floor from the production backend, including
sort scratch, and records physical/free VRAM and the safety margin. The
boundary script requires a positive driver-reported peak within the allowed
budget, independently of allocator bookkeeping. `POS2GPU_VRAM_MARGIN_MB`
remains available for driver/hardware calibration; record why a runner needs
a different value. Other plotting overrides and self-test bypasses are
removed by the harness.

Run the same checks locally after building all CMake targets:

Temporary plots and spill files use the checkout's filesystem by default.
Use `--scratch /path/to/disk` to select another existing directory. RAM-backed
filesystems such as tmpfs cannot validate disk spilling and are rejected by
the plotter.

```bash
python3 scripts/test/gpu-ci-test.py
python3 scripts/test/gpu-ci.py build --backend cuda --suite quick --logs /tmp/gpu-quick
python3 scripts/test/gpu-ci.py build --backend hip --suite vram --logs /tmp/gpu-vram
python3 scripts/test/gpu-ci.py build --backend level_zero --suite physical --physical-vram-mib 8192 --logs /tmp/gpu-physical
```

Actions retain build logs, device inventory, CTest JUnit results, per-tier
boundary/rejection logs, proof logs, and a JSON summary for 14 days. Generated
plots are temporary; byte comparisons and reference hashes are recorded.

## Commit style

Short imperative subjects, lowercase scope prefix, no trailing period:

```
gpu: split xs-sort keys_a to d_storage tail — drops pool VRAM min ~1.3 GB
docs: tighten streaming peak (~7.3 GB measured), add AMD row
CMakeLists: re-enable -O3 for SYCL TUs
```

Body paragraphs explain *why* (what invariant was wrong, what the
measurement was, what alternative was considered and why it was
rejected). The *what* is in the diff.

## Scope of changes

- Keep unrelated refactors out of correctness or performance commits.
- Performance changes should cite before/after numbers on a named GPU
  at a specified `k`.
- New runtime knobs go in `README.md`'s
  [Environment variables](https://github.com/Jsewill/xchplot2#environment-variables)
  table so users can discover them.

## PRs

The `main` branch carries the SYCL/AdaptiveCpp port; the
[`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only)
branch is the original CUDA-only path, preserved as the most-tested
NVIDIA configuration. A PR that only helps NVIDIA may still land on
`main`, but don't regress parity on AMD (`gfx1031`) along the way.

## Reporting bugs

Open an issue with:

- Exact command line and the full stderr output.
- GPU vendor + model + VRAM (`nvidia-smi -L` / `rocminfo | grep gfx`).
- Build flavor: container (service name + `ACPP_GFX` / `CUDA_ARCH`),
  native `scripts/install-deps.sh`, or `cargo install`.
- Whether parity tests pass on your build.
