# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces farmable
`.plot2` files byte-identical to the
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

## Performance

k=28, strength=2, RTX 4090 (sm_89), PCIe Gen4 x16:

| Mode | Per plot |
|---|---|
| pos2-chip CPU baseline | ~50 s |
| `xchplot2 batch` steady-state wall (pool path) | **2.06 s** |
| `xchplot2 batch` steady-state wall (streaming path, ≤8 GB cards) | ~3.7 s |
| Producer GPU time, steady-state | 1.96 s |
| Device-kernel floor (single-plot nsys) | 1.91 s |

A physically narrower PCIe slot (e.g. Gen4 x4) adds ~240 ms per plot to
the final fragment D2H copy. Check `cat /sys/bus/pci/devices/*/current_link_width`
under load if numbers look off by that much.

## Build

Requires CUDA Toolkit 12+ (tested on 13.x), C++20 host compiler, CMake
≥ 3.24, and a Rust toolchain (for `keygen-rs`).

### `cargo install`

```bash
cargo install --git https://github.com/Chia-Network/xchplot2
```

`build.rs` auto-detects the local GPU's compute capability by querying
`nvidia-smi --query-gpu=compute_cap` and builds for only that
architecture. That keeps the binary small and the build fast when the
install and the target GPU are the same machine.

If auto-detection fails (no `nvidia-smi` in `PATH`, or
`nvidia-smi` can't see a GPU — common when building inside a container
or on a headless build host that lacks the CUDA driver), the build
falls back to `sm_89`.

If you need to target a GPU that isn't the one doing the build — or if
you want a single "fat build" binary that covers multiple
architectures — override with `$CUDA_ARCHITECTURES`:

```bash
# Fat build for Ada (4090) and Blackwell (5090):
CUDA_ARCHITECTURES="89;120" cargo install --git https://github.com/Chia-Network/xchplot2

# Single target (e.g. Turing 2080 Ti):
CUDA_ARCHITECTURES=75 cargo install --git https://github.com/Chia-Network/xchplot2
```

Common values: `61` GTX 10-series, `70` Volta, `75` Turing, `80` A100,
`86` RTX 30-series, `89` RTX 40-series, `90` H100, `120` RTX 50-series.

### CMake (also builds the parity tests)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`pos2-chip` is auto-fetched via `FetchContent`; override with
`-DPOS2_CHIP_DIR=/abs/path/to/pos2-chip` to point at a local checkout.

Outputs:

- `build/tools/xchplot2/xchplot2`
- `build/tools/parity/{aes,xs,t1,t2,t3}_parity` — bit-exact CPU/GPU tests

## Use

### Standalone (farmable plots)

```bash
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk> \
    -c <pool-contract-address> \
    -o <output-dir>
```

Pool variants: `-p <pool-pk>` or `--pool-ph <pool-ph>`. Other common
flags: `-s <strength>`, `-T` testnet, `-S <seed>` for reproducible runs,
`-v` verbose. Full help: `xchplot2 -h`.

#### Grouping plots: `-i <plot-index>` and `-g <meta-group>`

Both are v2 PoS fields and default to 0.
`<plot-index>` (u16) is the within-group identifier; `plot -n N`
uses it as the base and increments per plot (so `-i 0 -n 1000`
produces plots with `plot_index` 0..999).
`<meta-group>` (u8) is a challenge-isolation boundary — plots with
different meta_group values are guaranteed never to pass the same
challenge.

The PoS2 spec defines a grouped-plot file layout (multiple plots
interleaved into one container per storage device, for harvester
seek amortization), but the on-disk format is not yet defined
upstream in `pos2-chip` / `chia-rs`. xchplot2 currently produces one
`.plot2` file per plot — this is in lieu of those upstream
decisions. When the grouped layout lands, the auto-incrementing
`<plot-index>` above is the per-plot within-group identifier it
will expect.

### Lower-level subcommands

```bash
xchplot2 test  <k> <plot-id-hex> [strength] ...   # single plot, raw inputs
xchplot2 batch <manifest.tsv> [-v]                # batched, raw inputs
```

## Testing farming on a testnet

v2 (CHIP-48) farming in stock chia-blockchain is presently unfinished
upstream — services aren't wired into the farmer group, a message
handler's signature doesn't match its decorator, `ProofOfSpace.
challenge` is computed from the wrong input, and the dependency pin
on `chia_rs` excludes the 0.42 release where `compute_plot_id_v2`
lives. `contrib/testnet-farming.patch` is a minimal self-contained
fix-up that gets a private testnet running end-to-end:

```bash
git clone https://github.com/Chia-Network/chia-blockchain
cd chia-blockchain
git checkout 39f8bec88   # 2.7.0 Checkpoint Merge
git apply /path/to/xchplot2/contrib/testnet-farming.patch
```

The patch's header comment describes each hunk. None of the changes
are xchplot2-specific — they're the farmer / harvester / daemon
pieces any v2 plot needs for farming, regardless of who produced it.

## Architecture

```
src/gpu/                 CUDA kernels — AES, Xs, T1, T2, T3
src/host/
├── GpuPipeline          Xs → T1 → T2 → T3 device orchestration;
│                          pool + streaming (low-VRAM) variants
├── GpuBufferPool        persistent device + 2× pinned host pool
├── BatchPlotter         producer / consumer batch driver
└── PlotFileWriterParallel  sole TU touching pos2-chip headers
tools/xchplot2/          CLI: plot / test / batch
tools/parity/            CPU↔GPU bit-exactness tests
keygen-rs/               Rust staticlib: plot_id_v2, BLS HD, bech32m
```

## VRAM

PoS2 plots are k=28 by spec. Two code paths, dispatched automatically
based on available VRAM:

- **Pool path (~15 GB, 16 GB+ cards).** The persistent buffer pool is
  sized worst-case and reused across plots in `batch` mode for
  amortised allocator cost and double-buffered D2H. Targets for
  steady-state: RTX 4080 / 4090 / 5080 / 5090, A6000, etc.
- **Streaming path (~8 GB).** Allocates per-phase and frees between
  phases; T1/T2 sorts are tiled (N=2 and N=4 respectively) and the
  merge-with-gather is split into three passes so the live set stays
  under 8 GB. Targets 8 GB cards (GTX 1070 class and up). Slower per
  plot (~3.7 s vs ~2.1 s at k=28 on a 4090) because it pays per-phase
  `cudaMalloc`/`cudaFree` instead of amortising.

`xchplot2` queries `cudaMemGetInfo` at pool construction; if the
pool doesn't fit, it transparently falls back to the streaming
pipeline with no flag needed. Force streaming on any card with
`XCHPLOT2_STREAMING=1`, useful for testing or for users who want the
smaller peak regardless.

Plot output is bit-identical between the two paths — the streaming
code reorganises memory, not algorithms.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).
