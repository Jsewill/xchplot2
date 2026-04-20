// AesStub.cpp — provides the symbols defined by AesGpu.cu when the build
// excludes the CUDA AOT path (XCHPLOT2_BUILD_CUDA=OFF). The CUDA path
// uploads AES T-tables into __constant__ memory; the SYCL path keeps them
// in a USM device buffer (SyclBackend.hpp's aes_tables_device(q)) which
// is initialised lazily on first kernel call. So this stub simply makes
// initialize_aes_tables a no-op — the SYCL kernels don't depend on it.

namespace pos2gpu {

void initialize_aes_tables() {
    // No-op on non-CUDA builds. AES T-tables are uploaded by
    // SyclBackend.hpp's aes_tables_device(q) on first use.
}

} // namespace pos2gpu
