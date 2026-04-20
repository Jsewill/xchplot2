# xchplot2 performance optimization plan

## Current state (2026-04-19, post-PCIe fix)

After the software commits and the GPU slot swap that let PCIe train at
Gen4 × 16 instead of x4, single-plot device breakdown (5-plot avg, k=28,
strength=2, RTX 4090 with `chia_recompute_server` present but idle during
measurement):

| Phase | Time | vs original 2227 ms |
|---|---:|---:|
| T1 match | 591 ms | neutral |
| T2 match | 534 ms | neutral |
| T3 match + Feistel | 539 ms | **−8.0 %** (fk-const) |
| D2H copy (T3 frags) | **88 ms** | **−73 %** (PCIe x16) |
| Sort + permute + misc | ~160 ms | neutral |
| **TOTAL device** | **~1925 ms** | **−13.6 %** |

Commits that landed in this round:
- `56fd580` GPU T3: FeistelKey → `__constant__` memory (−9.2 % T3 match)
- `71d0f80` GPU T3: SoA split sorted_t2 (neutral perf, pipeline consistency)
- (next) GpuPipeline: drop 5 redundant `cudaStreamSynchronize` calls that
  were already covered by the synchronous `cudaMemcpy(&count)` drains.
  Neutral single-plot, correctness-preserving, helps host-side batch
  overlap.

Plus hardware: GPU slot swap so PCIe trains at Gen4 × 16. Responsible for
~240 ms of the 300 ms total per-plot savings.

### Evaluated and did not ship

- **Tezcan bank-replicated T0 + `__byte_perm`** (commit `f60d1e4`, files
  `AesTezcan.cuh` + `aes_tezcan_bench.cu`). Wins 1.24× in a pure-AES
  bench with 16× T0 replication; regresses the match kernel by 14.7 %
  because 16 KB smem/block busts Ada's default carveout and the match
  kernel is already L1/TEX-bound. 8× replication fits the carveout but
  still regresses by 6.5 %. Don't reintegrate without a new throughput
  regime (e.g. fewer LDGs per thread, bigger per-SM smem budget).
- **CUDA Graphs.** Not attempted. Single-plot launch-overhead budget is
  only ~100-400 μs/plot (< 0.02 %) given the kernel density; would
  require phase-level sub-graphs because the mid-pipeline count syncs
  break capture. Not worth the refactor at current kernel sizes.

## Historical context

`match_all_buckets` dominates (89 % of device time). Inside it:

| Component | Share |
|---|---|
| matching_target AES | 20.99 % |
| pairing AES | 9.63 % |
| **AES total** | **30.6 %** |
| Non-AES (global loads on sorted_t2, binary search, r-walk LDG, atomicAdd, feistel, loop control) | **69.4 %** |

BS-AES is off the table on Ada (measured 0.61× vs T-table smem; see
`feedback_bs_aes_evaluated`). Perf headroom is in the non-AES 70 %.

## Instrumented breakdown (2026-04-18, T3 k=28, RTX 4090)

clock64 was wrapped around every region in T3 `match_all_buckets`.
Behind compile flag `-DXCHPLOT2_INSTRUMENT_MATCH=ON`. Two back-to-back
runs agree to <0.1 % — ratios are stable under external GPU contention.

| Region | % of instr. total | per-thread cycles |
|---|---:|---:|
| pre (l-side load) | 0.50 | 4,993 |
| **aes_matching_target** | **16.34** | 163,505 |
| **bsearch on sorted_mi** | **40.21** 🔥 | 402,385 |
| r_loop_total | 42.95 | 429,764 |
| &nbsp;&nbsp;└─ ldg_mi (target_r) | 3.15 | — |
| &nbsp;&nbsp;└─ ldg_meta (meta_r/x_bits) | 0.60 | — |
| &nbsp;&nbsp;└─ aes_pairing | 9.57 | — |
| &nbsp;&nbsp;└─ feistel | 2.60 | — |
| &nbsp;&nbsp;└─ atomic | **0.33** | — |
| &nbsp;&nbsp;└─ misc (loop ctrl + LDG latency) | 26.69 | — |

**Counts at k=28:** 1.074 B active threads, 2.147 B r-walk iterations
(exactly **2.00 per thread** — structural), 50 % target-match rate,
25 % pass pairing test. Final output: 268.5 M T3 pairings.

### Reshuffled priorities

Data killed several hypotheses from the pre-instrumentation plan:

- ❌ **Warp-aggregated atomic** — 0.33 %, not worth the code.
- ❌ **Software prefetch of r-walk LDG** — r-walk inner LDG is 3.75 %
  combined, and only 2 iterations per thread. No headroom.
- ❌ **Candidate early-reject before AES chain** — the existing target
  check already rejects 50 % cheaply; pairing AES only runs on actual
  target hits. Moving the reject earlier has no room.

**New #1 (was "last resort"): reduce bsearch cost.** Each thread does
~24 LDG iterations on sorted_mi, concentrated in the 40 % bsearch
bucket. sorted_mi's low 24 bits are effectively uniform (AES output),
so interpolation search converges in O(log log N) ≈ 5 iterations.

Concrete plan — **3-step interpolation + binary fallback**:

```
uint64_t lo = r_start, hi = r_end;
uint32_t v_lo = 0;
uint32_t v_hi = 1u << num_target_bits;
for (int i = 0; i < 3 && hi - lo > 16 && v_lo < v_hi; ++i) {
    uint64_t est = lo + uint64_t(target_l - v_lo) * (hi - lo)
                      / (v_hi - v_lo);
    if (est >= hi) est = hi - 1;
    uint32_t v_est = sorted_mi[est] & target_mask;
    if (v_est < target_l) { lo = est + 1; v_lo = v_est; }
    else                  { hi = est;     v_hi = v_est; }
}
// Classic lower_bound bsearch on the narrowed [lo, hi).
while (lo < hi) { … }
```

- Expected LDGs: ~3 interp + ~3 bsearch = **6, down from 24 (~75 %
  reduction on the 40 % bucket → ~30 % kernel speedup)**.
- Risk: low. Bit-identical output; parity tests gate.
- Same fix applies to T2 match_all_buckets (identical structure).

### Still valid (in order)

1. **Interpolation search for T3 + T2 bsearch** — see above. Primary.
2. **L2 persistent cache window on sorted_mi** — synergistic; cached
   residency for the remaining ~6 LDGs/thread. 3-6 % expected.
3. **CUDA Graphs** — 1-3 % wall-clock, orthogonal.
4. **`__launch_bounds__` re-tune after (1)+(2)** — kernel's register /
   occupancy sweet spot will move after the bsearch collapse.

### Definitively off the table

- BS-AES on Ada (0.61× measured).
- Warp-aggregated atomic (0.33 % of kernel).
- R-walk prefetch (3.75 % combined).
- Candidate early-reject (structurally no headroom).

## Implementation results (2026-04-19)

**ncu throughput regime:**

| Metric | T1 | T2 | T3 |
|---|---:|---:|---:|
| Compute (SM) Throughput | 81.9 % | 90.5 % | 87.6 % |
| L1/TEX Cache Throughput | 83.6 % | 92.2 % | 87.6 % |
| L2 Cache Throughput | 40.0 % | 43.3 % | 45.6 % |
| DRAM Throughput | 18.2 % | 16.1 % | 19.4 % |
| Achieved Occupancy | 88.1 % | 86.2 % | 58.6 % |
| Registers / thread | 36 | 38 | **55** |

All three kernels are **simultaneously SM-compute-saturated and L1/TEX
throughput-bound**, with L2 and DRAM well below ceiling. Bsearch-shrink
ideas (interpolation, arithmetic seek) trade LDGs for ALU and regress
because the SM is already pegged.

**What worked: FeistelKey → `__constant__` memory (T3 only).**

`FeistelKey` is 40 bytes (32-B plot_id + 2 ints). Passed by value, it
spilled to per-thread LMEM (T3 `STACK:40`), making every
`fk.plot_id[i]` access inside `feistel_encrypt` a scattered LMEM LDG —
catastrophic for an L1-bound kernel. Hoisted to file-scope
`__constant__ FeistelKey g_t3_fk` with `cudaMemcpyToSymbolAsync`
before launch.

| | Before | After |
|---|---:|---:|
| T3 REG / STACK | 55 / 40 | **39 / 0** |
| T3 match | 587 ms | **533 ms** (−9.2 %) |
| Total device | 2227 ms | **2143 ms** (−3.8 %) |

Parity bit-identical across all three tables.

**What didn't work** (experiments retained in git stash / memory):

| Attempt | Outcome | Notes |
|---|---|---|
| 3-step interpolation bsearch | T1 +89 %, T2 +2 %, T3 +22 % | 64-bit divides + register pressure |
| 1-step arithmetic seek on T3 | −34 % | Saturated SM, LMEM spill re-triggered |
| 1-step seek on T2 (no spill) | +38 % | Same — SM saturated, any added ALU regresses |
| `__launch_bounds__(256, 3)` on T3 | neutral | compiler didn't use relaxed budget |
| `__launch_bounds__(256, 5)` on T3 | neutral | occupancy doesn't help when L1-bound |
| SoA split of sorted_t2 (T3) | neutral | kept in stash for future reference |

Key lesson (saved to session memory): clock64-per-region ratios measure
SM-residence time, not wall-time optimisation potential. Always check
throughput regime (ncu `--set detailed`) before betting on cycle-shrink
ideas. And check `cuobjdump --dump-resource-usage` for stack-spilled
structs — that's where cheap wins hide.

## Next candidates (not yet attempted)

- **CUDA Graphs** — still orthogonal, ~1–3 % wall-clock.
- **Move other large-struct args** to `__constant__` — `AesHashKeys`
  (32 B) in T1/T2/T3 might have similar (smaller) wins even though they
  don't spill currently. Would free ~8 regs/kernel.
- **Phases not yet touched**: Xs gen_kernel (44 ms), sort phases
  (~210 ms combined), D2H copy (346 ms).

## Ranked opportunities

### High value (direct attack on the non-AES 70 %)

#### 1. L2 persistent cache windows on sorted_t2

Use `cudaAccessPolicyWindow` on the match stream to pin the hot sorted_t2
range in Ada's 72 MB L2. The r-walk LDG latency is the named hotspot, and
binary-search access is irregular enough that hardware prefetch misses.

- **Expected payoff:** 5–10 % on match_all_buckets.
- **Risk:** low. Isolated to stream setup in `GpuPipeline.cu`.
- **Validation:** nsys section on L2 hit rate before/after; clock64
  instrumentation on the r-walk LDG block.

#### 2. Warp-aggregated atomicAdd for bucket-offset writes

Collapse N per-lane `atomicAdd`s per warp into 1 using
`__ballot_sync` + `__popc` (leader-writes-sum, broadcast base). Classic
pattern; any kernel that atomically appends to per-bucket counters benefits.

- **Expected payoff:** 3–8 % on match kernels if atomics are a meaningful
  slice of the 69.4 %. Need to instrument first to confirm share.
- **Risk:** zero algorithmic risk; output bit-identical.
- **Touch points:** T1/T2/T3 match kernels' output append.

#### 3. Software prefetch of next r-iteration

`__ldg` the next sorted_t2 stripe into registers while the current AES
chain runs. Overlaps LDG with ALU — directly attacks the cited LDG stall.

- **Expected payoff:** 5–12 % on match_all_buckets if LDG really is the
  bottleneck.
- **Risk:** register pressure interacts with existing
  `__launch_bounds__(256, 4)`. May spill and regress. Re-tune launch
  bounds alongside.
- **Validation:** nsys stall-reason histogram (long scoreboard → short
  scoreboard is the signal); occupancy before/after.

### Medium value

#### 4. CUDA Graphs across Xs → T1 → T2 → T3

Launch overhead at 2 s/plot is small, but graphs also eliminate
stream-ordering fences and let the driver schedule ahead. Cheap A/B —
build the graph once per plot, replay per batch entry.

- **Expected payoff:** 1–3 % wall-clock.
- **Risk:** low. Graph capture of dynamic kernel params requires care;
  CUB SortPairs allocations need to be pool-sourced (already are).

#### 5. Candidate early-reject before AES chain

If any cheap predicate (top bits of meta, bucket parity, small hash of
meta) can kill a fraction of candidates before the 32-round AES chain,
that's a direct cut of both AES (30.6 %) and the LDG chain following it.

- **Expected payoff:** potentially the largest single win — scales with
  rejection rate.
- **Risk:** highest — requires algorithmic analysis to prove correctness
  against pos2-chip CPU reference. Parity tests in `tools/parity/` are
  the gate.
- **Prereq:** characterise the candidate→match acceptance rate. If it's
  already ~100 %, this is a dead end.

#### 6. Fused permute_t{1,2} into next match

Memory already flagged this as 2–3 %, marginal. Worth bundling only if
the surrounding code is being touched for another reason.

### Worth measuring, unclear payoff

#### 7. Re-tune `__launch_bounds__`

(256, 4) was chosen before the SoA meta change and any prefetch work.
Sweet spot likely moved. Cheap to sweep (128/256/384 × 2/3/4).

- **Expected payoff:** 0–5 %, unpredictable.
- **Risk:** zero — pure config.

#### 8. Binary search → cuckoo / perfect hash

Binary search on sorted_t2 is part of the LDG-bound 69 %. A cuckoo hash
is O(1) expected with fewer dependent loads, but:

- Big change, big surface area.
- Memory overhead; VRAM budget is already tight (~15 GB).
- Likely only worthwhile if (1)–(3) don't move the needle.

### Off the table

- **BS-AES on Ada.** Already measured 0.61× vs T-table smem. Revisit
  only on new hardware or a hybrid that sidesteps shuffle cost.

## Suggested execution order

1. **Instrument first.** Split the 69.4 % into atomics / LDG / binary
   search / feistel with clock64. This decides whether (1)/(2)/(3) or (5)
   is the right starting point.
2. **(1) L2 persistent windows** — self-contained, low-risk, informative.
3. **(2) Warp-aggregated atomics** — if step 1's instrumentation shows
   atomics are > 5 % of kernel time.
4. **(3) sw-prefetch + launch_bounds re-tune together** — these interact.
5. **(5) candidate early-reject** — only after (1)–(3) are measured, and
   only if the candidate acceptance rate leaves room.
6. **(4) CUDA Graphs** — easy win to bank once the kernel-internal work
   settles.
7. **(8) hash-table match** — last resort if the above don't close the
   gap to the next round number (~1.5 s device).

## Validation gates

Every change must:

- Pass `tools/parity/` (aes, xs, t1, t2, t3) — bit-exact vs pos2-chip.
- Produce an `xchplot2` binary whose canonical test plot matches the
  expected SHA.
- Be benchmarked with `nvidia-smi --query-compute-apps` verifying no
  contending GPU process (`chia_recompute_server` in particular).
- Report both single-plot nsys device time and 10-plot batch wall time
  — the two can move in opposite directions.
