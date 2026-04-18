# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces farmable
`.plot2` files at the GPU compute floor — **~2.68 s/plot at k=28
strength=2 on an RTX 4090**, byte-identical to the
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference and
to `chia plots create --v2`.

## Performance

k=28, strength=2, on an RTX 4090, sm_89:

| Mode | Per-plot wall |
|------|---------------|
| pos2-chip CPU baseline | ~50 s |
| `xchplot2 plot --num 1` | ~3.6 s |
| `xchplot2 plot --num 10` (steady-state batch) | **2.68 s** |
| Theoretical floor (kernel-time only) | 2.69 s |

The pipeline sits at the GPU compute floor — further gains require
kernel algorithm work, not orchestration.

## Build

Requires:

- CUDA Toolkit 12+ with C++20 nvcc (tested on 13.x)
- C++20 host compiler (g++ ≥ 11 or clang++ ≥ 14)
- CMake ≥ 3.24
- Rust toolchain (`rustc` / `cargo`) — only at build time, for `keygen-rs`

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = sm_89 / RTX 4090; adjust per GPU
cmake --build build -j
```

`pos2-chip` is auto-fetched into `third_party/pos2-chip` at configure
time (pinned to a specific commit). Override with
`-DPOS2_CHIP_DIR=/abs/path/to/pos2-chip` if you want to point at a local
checkout.

Produces:

- `build/tools/xchplot2/xchplot2`
- `build/tools/parity/{aes,xs,t1,t2,t3}_parity` — CPU↔GPU bit-exactness tests

## Use

### Standalone (farmable plots)

```bash
xchplot2 plot --k 28 --num 10 \
    --farmer-pk    <96 hex chars> \
    --pool-contract-address xch1...      \
    --out          /mnt/plots
```

Other pool flavours: `--pool-pk <hex>` (96 hex), `--pool-ph <hex>`
(64 hex). Optional: `--strength S`, `--testnet`, `--seed <64 hex>` for
reproducible runs, `--plot-index N`, `--meta-group N`.

### Lower-level subcommands

```bash
xchplot2 test  <k> <plot_id_hex> [strength] ...   # single plot, raw inputs
xchplot2 batch <manifest.tsv> [-v]                # batched, raw inputs
```

## Architecture

```
xchplot2 (this repo)
├── src/gpu/                  CUDA kernels — AES, Xs, T1, T2, T3
├── src/host/
│   ├── GpuPipeline.{hpp,cu}  end-to-end Xs → T1 → T2 → T3 orchestration
│   ├── GpuBufferPool         persistent device + 2× pinned host pool
│   ├── BatchPlotter          producer / consumer batch driver
│   └── PlotFileWriterParallel sole TU touching pos2-chip headers
│                              (parallel FSE compress + write)
├── tools/xchplot2/main.cpp   CLI: test / batch / plot
├── tools/parity/             CPU↔GPU bit-exactness tests
└── keygen-rs/                Rust staticlib over chia-rs:
                                 plot_id_v2, BLS HD derivation,
                                 bech32m address decode
```

## VRAM requirements

PoS2 plots are k=28 by spec. The persistent buffer pool needs **~15 GB
of device VRAM**, so a 16 GB or larger card is required (RTX 4080 /
4090 / 5080 / 5090, A6000, etc.). `xchplot2` queries `cudaMemGetInfo`
at startup and refuses with an actionable error if the pool won't fit.

## Acknowledgments

Built collaboratively with [Claude](https://claude.ai/code) (Anthropic),
acting as a pair-programmer under human direction. Co-author lines on
the relevant commits make the split explicit.

## License

[MIT](LICENSE).

Depends on (built/fetched separately, not vendored):

- `pos2-chip` — Apache-2.0, fetched via CMake `FetchContent`.
- `chia` 0.42 (chia-bls, chia-protocol, chia-sha2) — Apache-2.0, via
  cargo.
- `bech32` 0.11 — MIT, via cargo.
