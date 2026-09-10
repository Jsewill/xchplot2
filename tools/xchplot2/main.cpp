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
    auto const code_page = ::GetConsoleOutputCP();
    ::SetConsoleOutputCP(CP_UTF8);
#endif
    int const result = xchplot2_main(argc, argv);
#ifdef _WIN32
    if (code_page) ::SetConsoleOutputCP(code_page);
#endif
    return result;
}
