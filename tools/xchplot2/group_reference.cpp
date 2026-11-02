// Standalone fixture interface for low-level construction and recovery tests.
#include "group_commands.hpp"
#include <iostream>

int main(int argc, char** argv) try
{
    return pos2gpu::run_group(pos2gpu::parse_group_options(argc, argv));
}
catch (std::exception const& error)
{
    std::cerr << "same-file group growth failed: " << error.what() << '\n';
    return 1;
}
