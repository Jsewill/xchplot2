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
  the SYCL/AdaptiveCpp backends vs the CUDA reference, so AMD/Intel
  breakage is caught on NVIDIA hardware too.
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
