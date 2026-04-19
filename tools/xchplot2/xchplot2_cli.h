// xchplot2_cli.h — C-ABI entry point for the xchplot2 CLI.
//
// The CLI parsing + dispatch logic lives in cli.cpp as
// `extern "C" int xchplot2_main(int argc, char* argv[])`. Two consumers:
//
//   - tools/xchplot2/main.cpp: a one-line shim that becomes the
//     standalone `xchplot2` executable produced by the CMake build.
//   - The top-level Cargo crate's src/main.rs: marshals std::env::args
//     into a NUL-terminated argv array and calls this function so
//     `cargo install --git ...` produces an equivalent binary.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int xchplot2_main(int argc, char* argv[]);

#ifdef __cplusplus
} // extern "C"
#endif
