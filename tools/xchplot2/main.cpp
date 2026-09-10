// main.cpp — one-line shim that turns the cli.cpp library entrypoint into
// a real `int main` for the CMake-built xchplot2 binary. The Rust top-level
// crate skips this file entirely and calls xchplot2_main directly.

#include "xchplot2_cli.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[])
{
#ifdef _WIN32
    ::SetConsoleOutputCP(CP_UTF8);
#endif
    return xchplot2_main(argc, argv);
}
