# pos2-gpu

GPU plotter for Chia PoS2 (CHIP-48), built on top of the
[pos2-chip](../pos2-chip) reference implementation.

## Status — first-cut scaffold

| Component | Status | Notes |
|-----------|--------|-------|
| AES core (`AesHashGpu`) | Implemented | T-table AES round on device; mirrors `_mm_aesenc_si128` semantics. **Must pass `aes_parity` before anything else is trusted.** |
| T1 GPU kernel | Skeleton | F1 / `g_x` parallelised; matching deferred to CPU helper until parity confirmed. |
| T2 GPU kernel | Skeleton | CUB radix sort wired; matching predicate parked at TODO. |
| T3 GPU kernel | Skeleton | Same shape as T2; second AES pass + bit-drop TODO. |
| `gpu_plotter` CLI | Builds | Mirrors `plotter test <k> <plot_id> ...` arg surface; falls back to CPU per phase via flags. |
| Parity tests | Stubs | `aes_parity`, `t1_parity`, `t2_parity`, `t3_parity` — start with `aes_parity`. |

**Correctness gate:** every GPU phase must be byte-for-byte equal to the
CPU reference for a fixed `(plot_id, k, strength)` before it ships. PoS2
is consensus-critical — there are no acceptable shortcuts.

## Build

Requires:

- CUDA Toolkit 12+ (tested 13.x at `/opt/cuda`)
- C++20 compiler (g++ ≥ 11 or clang++ ≥ 14)
- CMake ≥ 3.24
- pos2-chip checked out at `../pos2-chip` relative to this directory

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = sm_89 = RTX 4090; adjust per GPU
cmake --build build -j
```

Produces:

- `build/tools/gpu_plotter/gpu_plotter`
- `build/tools/parity/aes_parity`
- `build/tools/parity/t1_parity`
- `build/tools/parity/t2_parity`
- `build/tools/parity/t3_parity`

## Run order (testing)

1. `./build/tools/parity/aes_parity` — confirms GPU AES round produces the same
   state as CPU `_mm_aesenc_si128` for a deterministic input set. **Must pass.**
2. `./build/tools/parity/t1_parity` — confirms GPU T1 output matches CPU T1
   for `(plot_id = 0xab × 32, k = 18, strength = 2)`.
3. `./build/tools/parity/t2_parity`
4. `./build/tools/parity/t3_parity`
5. `./build/tools/gpu_plotter/gpu_plotter test 18 <plot_id_hex> 2` —
   produces a `.plot2` file. Cross-check with `pos2-chip/build/.../prover check`.

## Architecture

```
                pos2-chip (reference, header-only)
                ├── src/pos/aes/AesHash.hpp      <-- mirrored on GPU
                ├── src/pos/ProofParams.hpp      <-- shared
                ├── src/plot/Plotter.hpp         <-- CPU pipeline (fallback)
                └── src/plot/TableConstructor*   <-- T1/T2/T3 reference

                pos2-gpu (this repo)
                ├── src/gpu/AesGpu.cuh           AES T-table device functions
                ├── src/gpu/AesHashGpu.cuh       AesHash mirror (g_x, pairing, chain)
                ├── src/gpu/T1Kernel.cu          T1 generation kernel
                ├── src/gpu/T2Kernel.cu          T2 sort + match
                ├── src/gpu/T3Kernel.cu          T3 sort + AES + bit-drop
                ├── src/host/GpuPlotter.hpp      Host orchestration
                ├── tools/gpu_plotter/main.cpp   CLI
                └── tools/parity/                CPU↔GPU bit-exactness tests
```

The GPU phases are designed to be **drop-in replaceable** with the CPU
implementation. `GpuPlotter` exposes per-phase flags so you can run e.g.
T1-on-GPU + T2/T3-on-CPU while debugging.

## What stays on CPU

- FSE entropy compression (`ChunkCompressor.hpp`) — serial.
- `PlotFile::writeData` — disk I/O.
- `ProofParams` construction, `ProofValidator`, all framing.

## What's not done yet

- Full bit-exact T1/T2/T3 kernels. The match/pairing/bit-drop logic in
  `TableConstructor*Generic.hpp` is hundreds of lines and needs to be
  ported with care. This scaffold establishes the build, the AES core,
  and the harness — the actual phase ports are clearly-marked TODOs.
- Multi-GPU sharding.
- Direct-IO plot file writer.
- Strength > 2 hot-paths (current plan is to validate at strength=2 first).

## License

Apache-2.0 (matching pos2-chip).
