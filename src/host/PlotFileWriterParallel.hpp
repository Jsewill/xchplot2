// PlotFileWriterParallel.hpp — parallel PoS2 1.0 group/raw writer and
// CPU reference boundary. Single-member group bytes match upstream
// PlotGroupFile::writeData; raw members match PlotFile::writeData.
//
// That .cpp is the SOLE TU in pos2-gpu that includes pos2-chip's plot/* and
// pos/ProofParams.hpp headers — keeping it that way avoids the multiple-
// definition link errors caused by non-inline soft_aesenc / soft_aesdec
// in pos2-chip's pos/aes/soft_aes.hpp. Other TUs talk to us via raw bytes
// and never see those types directly.
//
// Takes a span<uint64_t const> instead of PlotData so callers can pass
// pinned-host memory directly, avoiding a ~1 s pinned→heap memcpy per plot
// in batch mode.

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace pos2gpu {

struct BatchEntry;
std::array<uint8_t, 32> plot_id_for_group(
    std::array<uint8_t, 32> const& group_id, uint16_t index, uint8_t meta_group);
// Bounded header/chunk-index validation; checks identity when resuming a batch.
bool plot_file_matches(std::string const& filename, BatchEntry const& expected);

// Writes a single-member .gplot, or a raw-v2 member for assembly. Returns bytes written.
//
// `t3_fragments` must already be sorted by proof_fragment (low 2k bits) —
// matching what GpuPipeline / pos2-chip's CPU plotter produce.
//
// `entry` contains the group identity, derived member ID, parameters, and memo.
// No upstream types cross this boundary.
//
// `thread_count == 0` uses std::thread::hardware_concurrency().
size_t write_plot_file_parallel(
    std::string const& filename,
    std::span<uint64_t const> t3_fragments,
    BatchEntry const& entry,
    unsigned thread_count = 0);

// Construct the shared compression pool NOW, on the calling thread.
//
// The pool is a function-local static, so its worker threads are created by
// whichever thread first reaches write_plot_file_parallel() — and on Linux they
// inherit that thread's nice value, because nice is a per-thread attribute that
// clone() copies into children.
//
// That is a live hazard: BatchPlotter nices its CPU worker down so it stops
// starving the GPU workers (see nice_current_thread there), and the CPU worker
// now writes through this pool too. If it won the race to construct the pool,
// all 32 compression threads would be born niced and EVERY GPU worker's FSE
// would inherit the penalty — the exact opposite of the intent, silently, and
// unfixable at runtime since an unprivileged process cannot lower nice again.
//
// So run_batch calls this from the main thread before it spawns any worker.
// Idempotent; cheap after the first call.
void warm_writer_pool();

// Run pos2-chip's CPU `Plotter` end-to-end and return the sorted T3
// proof_fragment vector. Encapsulated here so other TUs don't need to
// include plot/Plotter.hpp (and through it pos/aes/soft_aes.hpp).
std::vector<uint64_t> run_cpu_plotter_to_fragments(
    uint8_t const* plot_id_32,
    uint8_t k,
    uint8_t strength,
    uint8_t testnet,
    bool    verbose);

// Reads a raw member or single-member group written by this writer (or
// pos2-chip's CPU writer) and returns the concatenated decompressed
// T3 proof fragments in on-disk order (groups deduplicate fragments). Used to
// verify write + read round-trip without exposing pos2-chip's
// plot/PlotFile.hpp to other TUs.
std::vector<uint64_t> read_plot_file_fragments(std::string const& filename);

// Result of a `verify_plot_file` call.
//   trials                 — how many random challenges were tried
//   challenges_with_proof  — challenges that produced ≥ 1 proof
//   proofs_found           — total proofs summed across all trials
struct VerifyResult {
    size_t trials                = 0;
    size_t challenges_with_proof = 0;
    size_t proofs_found          = 0;
    size_t full_proofs_validated  = 0;
};

// Samples quality chains. With full=true, solve and validate a full proof for
// every returned chain; a chain without a valid full proof throws.
VerifyResult verify_plot_file(std::string const& filename, size_t n_trials, bool full = false);

} // namespace pos2gpu
