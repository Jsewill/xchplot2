// SortDispatch.cpp — runtime backend dispatch for the radix sort wrappers.
//
// Two implementations can coexist in the same binary on dual-toolchain
// builds:
//
//   launch_sort_*_cub   — CUB-backed (SortSyclCub.cpp + SortCuda.cu);
//                          present only when XCHPLOT2_HAVE_CUB defined.
//   launch_sort_*_sycl  — pure-SYCL hand-rolled radix (SortSycl.cpp);
//                          always present.
//
// The dispatcher picks based on the queue's device backend, so a hybrid
// host (NVIDIA + AMD on the same box) runs CUB on the NVIDIA worker and
// SYCL radix on the AMD worker without rebuilding. Single-vendor builds
// (BUILD_CUDA=OFF) compile out the CUB branch entirely; the dispatcher
// reduces to a single tail call.

#include "gpu/Sort.cuh"

namespace pos2gpu {

#if defined(XCHPLOT2_HAVE_CUB)
void launch_sort_pairs_u32_u32_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

void launch_sort_keys_u64_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

void launch_segmented_sort_pairs_u32_u32_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit,
    sycl::queue& q);
#endif

void launch_sort_pairs_u32_u32_sycl(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

void launch_sort_keys_u64_sycl(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

void launch_segmented_sort_pairs_u32_u32_sycl(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit,
    sycl::queue& q);

void launch_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
#if defined(XCHPLOT2_HAVE_CUB)
    if (q.get_device().get_backend() == sycl::backend::cuda) {
        launch_sort_pairs_u32_u32_cub(
            d_temp_storage, temp_bytes,
            keys_in, keys_out, vals_in, vals_out,
            count, begin_bit, end_bit, q);
        return;
    }
#endif
    launch_sort_pairs_u32_u32_sycl(
        d_temp_storage, temp_bytes,
        keys_in, keys_out, vals_in, vals_out,
        count, begin_bit, end_bit, q);
}

void launch_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
#if defined(XCHPLOT2_HAVE_CUB)
    if (q.get_device().get_backend() == sycl::backend::cuda) {
        launch_sort_keys_u64_cub(
            d_temp_storage, temp_bytes,
            keys_in, keys_out,
            count, begin_bit, end_bit, q);
        return;
    }
#endif
    launch_sort_keys_u64_sycl(
        d_temp_storage, temp_bytes,
        keys_in, keys_out,
        count, begin_bit, end_bit, q);
}

void launch_segmented_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
#if defined(XCHPLOT2_HAVE_CUB)
    if (q.get_device().get_backend() == sycl::backend::cuda) {
        launch_segmented_sort_pairs_u32_u32_cub(
            d_temp_storage, temp_bytes,
            keys_in, keys_out, vals_in, vals_out,
            num_items, num_segments,
            d_begin_offsets, d_end_offsets,
            begin_bit, end_bit, q);
        return;
    }
#endif
    launch_segmented_sort_pairs_u32_u32_sycl(
        d_temp_storage, temp_bytes,
        keys_in, keys_out, vals_in, vals_out,
        num_items, num_segments,
        d_begin_offsets, d_end_offsets,
        begin_bit, end_bit, q);
}

} // namespace pos2gpu
