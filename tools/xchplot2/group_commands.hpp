#pragma once

#include "host/GroupPlotter.hpp"

namespace pos2gpu {
inline unsigned group_option_values(std::string_view option)
{
    if (option == "--relocate") return 2;
    if (option == "--" || !option.starts_with('-') || option == "--resume" || option == "-r" ||
        option == "--no-resume" || option == "--plan-only" || option == "--no-plan-only" ||
        option == "--metadata-only" || option == "--no-metadata-only" ||
        option == "--assess-jobs" || option == "--no-assess-jobs" ||
        option == "--calibrate" || option == "--no-calibrate" || option == "--calibration-worker" ||
        option == "--json" || option == "--no-json" || option == "--quiet" || option == "--no-quiet" ||
        option == "-q" || option == "--verbose" || option == "--no-verbose" || option == "-v" ||
        option == "--progress" || option == "--no-progress" || option == "--help" || option == "--help-all" || option == "-h" ||
        option == "--extend" || option == "--no-extend" || option == "--allow-missing" || option == "--no-allow-missing") return 0;
    return 1;
}
GroupOptions parse_group_options(int argc, char** argv, std::string_view command = {});
GroupOptions group_batch_options(BatchOptions const&, std::vector<std::string> const&);
int run_group_command(std::string_view command, std::vector<std::string> const&);
int run_compute_broker(int argc, char** argv);
int run_compute_worker(int argc, char** argv);
int run_compute_setup(int argc, char** argv);
int run_compute_membership(int argc, char** argv);
int run_compute_status(int argc, char** argv);
int run_compute_service(int argc, char** argv);
} // namespace pos2gpu
