// AesSmemConfig.cuh — compile-time selection between the 4-table shared-
// memory AES round (ttable4, 4 KB/block) and Tezcan's bank-replicated
// single-T0 + __byte_perm variant (tezcan16, 16 KB/block).
//
// CMake option XCHPLOT2_AES_ROUND={auto,ttable4,tezcan16}:
//   ttable4   → XCHPLOT2_AES_FORCE_TTABLE4
//   tezcan16  → XCHPLOT2_AES_FORCE_TEZCAN16
//   auto      → tezcan16 on __CUDA_ARCH__ == 890 only, else ttable4
//
// Per-TU opt-in for incremental family landing: define
// XCHPLOT2_AES_TEZCAN_KERNEL before including this header (or AesHashGpu)
// so only that translation unit takes the Tezcan path under `auto`.
// Once every hot kernel family has landed, the opt-in can be dropped and
// `auto` applies globally.
//
// Macros for kernel smem sites:
//   XCHPLOT2_AES_SMEM_DECL(name)  — __shared__ table storage
//   XCHPLOT2_AES_SMEM_LOAD(name)  — populate from constant memory

#pragma once

#if defined(XCHPLOT2_AES_FORCE_TEZCAN16)
#  define XCHPLOT2_AES_SMEM_TEZCAN 1
#elif defined(XCHPLOT2_AES_FORCE_TTABLE4)
#  undef XCHPLOT2_AES_SMEM_TEZCAN
#elif defined(XCHPLOT2_AES_TEZCAN_KERNEL)
#  if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 890)
#    define XCHPLOT2_AES_SMEM_TEZCAN 1
#  endif
#endif

#ifndef XCHPLOT2_AES_TEZCAN_BANK
#  define XCHPLOT2_AES_TEZCAN_BANK 16
#endif

#if defined(XCHPLOT2_AES_SMEM_TEZCAN)
#  define XCHPLOT2_AES_SMEM_DECL(name) \
      __shared__ uint32_t name[256 * XCHPLOT2_AES_TEZCAN_BANK]
#  define XCHPLOT2_AES_SMEM_LOAD(name) \
      ::pos2gpu::load_aes_t0_smem_rep<XCHPLOT2_AES_TEZCAN_BANK>(name)
#else
#  define XCHPLOT2_AES_SMEM_DECL(name) \
      __shared__ uint32_t name[4 * 256]
#  define XCHPLOT2_AES_SMEM_LOAD(name) \
      ::pos2gpu::load_aes_tables_smem(name)
#endif
