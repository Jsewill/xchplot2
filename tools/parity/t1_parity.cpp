// t1_parity — placeholder until launch_t1_match is implemented.
//
// Once the GPU T1 matching kernel exists, this should:
//   1. Run pos2-chip's Table1Constructor to produce reference T1Pairings.
//   2. Run launch_t1_match for the same plot_id / k / strength.
//   3. Sort both buffers (CPU may emit in a different order) and memcmp.
//   4. Print first divergence (xL, xR, meta, hash).

#include <cstdio>

int main()
{
    std::printf("t1_parity: not implemented yet — waiting on launch_t1_match.\n"
                "Run aes_parity first; once it passes, fill in launch_t1_match\n"
                "in src/gpu/T1Kernel.cu and wire this harness to the CPU\n"
                "reference in pos2-chip/src/plot/TableConstructorGeneric.hpp.\n");
    return 77; // POSIX 'skipped'
}
