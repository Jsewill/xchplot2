// main.cpp — one-line shim that turns the cli.cpp library entrypoint into
// a real `int main` for the CMake-built xchplot2 binary. The Rust top-level
// crate skips this file entirely and calls xchplot2_main directly.

#include "xchplot2_cli.h"

int main(int argc, char* argv[])
{
    return xchplot2_main(argc, argv);
}
