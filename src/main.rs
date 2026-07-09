// xchplot2 — Rust entry that marshals argv into a NUL-terminated C-style
// array and delegates to `xchplot2_main` (defined in tools/xchplot2/cli.cpp,
// linked statically via build.rs). This indirection only exists so that
// `cargo install --git ...` produces a working binary; everything of substance
// lives in the C++ / CUDA half of the project.

use std::ffi::{CString, OsString};
use std::os::raw::{c_char, c_int};
use std::process::ExitCode;

unsafe extern "C" {
    fn xchplot2_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

// Convert one OS argument to a CString without going through &str:
// std::env::args() panics on non-UTF-8 arguments, but Linux paths are
// arbitrary bytes — `xchplot2 batch <non-utf8-path>` must plot, not abort.
fn arg_to_cstring(arg: OsString) -> CString {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStringExt;
        arg.into_vec()
    };
    #[cfg(not(unix))]
    let bytes = arg.to_string_lossy().into_owned().into_bytes();
    // Interior NULs are impossible for shell-supplied args on POSIX;
    // substitute a placeholder rather than crash if one appears anyway.
    CString::new(bytes).unwrap_or_else(|_| CString::new("?").unwrap())
}

fn main() -> ExitCode {
    let owned: Vec<CString> = std::env::args_os().map(arg_to_cstring).collect();
    // Parallel Vec<*const c_char> that lives as long as `owned`, plus the
    // terminating null pointer the C standard promises (argv[argc] == NULL).
    let mut raw: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
    raw.push(std::ptr::null());

    let rc = unsafe { xchplot2_main(owned.len() as c_int, raw.as_ptr()) };

    // Keep `owned` alive across the FFI call (raw pointers borrow from it).
    drop(owned);

    // Pass the C exit code through. ExitCode wraps a u8; clamp negatives.
    let code = if (0..=255).contains(&rc) { rc as u8 } else { 1 };
    ExitCode::from(code)
}
