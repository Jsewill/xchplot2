#pragma once
#include <stdexcept>
#include <string_view>

namespace pos2gpu
{
struct ComputeOutputOptions
{
    bool verbose = false, quiet = false;
    bool parse(std::string_view option)
    {
        if (option == "--verbose" || option == "-v") verbose = true;
        else if (option == "--no-verbose") verbose = false;
        else if (option == "--quiet" || option == "-q") quiet = true;
        else if (option == "--no-quiet") quiet = false;
        else return false;
        return true;
    }
    bool details() const { return verbose && !quiet; }
    void validate() const
    {
        if (verbose && quiet)
            throw std::invalid_argument("-q/--quiet and -v/--verbose are mutually exclusive");
    }
};
inline constexpr char compute_output_help[] =
    "  -v, --verbose       show detailed status and settings\n"
    "  -q, --quiet         suppress informational output\n";
}
