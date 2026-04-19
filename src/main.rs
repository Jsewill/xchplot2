// xchplot2 — Rust entry that marshals argv into a NUL-terminated C-style
// array and delegates to `xchplot2_main` (defined in tools/xchplot2/cli.cpp,
// linked statically via build.rs). This indirection only exists so that
// `cargo install --git ...` produces a working binary; everything of substance
// lives in the C++ / CUDA half of the project.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::process::ExitCode;

unsafe extern "C" {
    fn xchplot2_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

fn main() -> ExitCode {
    // Convert each &str arg to an owned CString. Args containing interior
    // NULs (impossible for shell-supplied args on POSIX) get replaced with
    // a placeholder so we don't crash.
    let owned: Vec<CString> = std::env::args()
        .map(|s| CString::new(s).unwrap_or_else(|_| CString::new("?").unwrap()))
        .collect();
    // Build a parallel Vec<*const c_char> that lives as long as `owned`.
    let raw: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();

    let rc = unsafe { xchplot2_main(raw.len() as c_int, raw.as_ptr()) };

    // Keep `owned` alive across the FFI call (raw pointers borrow from it).
    drop(owned);

    // Pass the C exit code through. ExitCode wraps a u8; clamp negatives.
    let code = if (0..=255).contains(&rc) { rc as u8 } else { 1 };
    ExitCode::from(code)
}
