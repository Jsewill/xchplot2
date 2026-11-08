#pragma once

#include "host/GroupPlotter.hpp"

namespace pos2gpu {
inline bool is_group_command(std::string_view command)
{
    return command == "plan" || command == "fill" || command == "grow" || command == "status" ||
        command == "pause" || command == "drain" || command == "resume" || command == "retire";
}
inline unsigned group_option_values(std::string_view option)
{
    if (option == "--relocate") return 2;
    if (option == "--" || !option.starts_with('-') || option == "--resume" || option == "-r" ||
        option == "--no-resume" || option == "--plan-only" || option == "--no-plan-only" ||
        option == "--metadata-only" || option == "--no-metadata-only" ||
        option == "--assess-jobs" || option == "--no-assess-jobs" ||
        option == "--inventory" || option == "--no-inventory" ||
        option == "--status" || option == "--no-status" || option == "--revalidate" || option == "--no-revalidate" ||
        option == "--watch" || option == "--no-watch" || option == "--inspect" || option == "--no-inspect" ||
        option == "--live" || option == "--no-live" ||
        option == "--calibrate" || option == "--no-calibrate" || option == "--calibration-worker" || option == "--replot-proof-worker" || option == "--replot-structure-worker" ||
        option == "--json" || option == "--no-json" || option == "--quiet" || option == "--no-quiet" ||
        option == "-q" || option == "--verbose" || option == "--no-verbose" || option == "-v" ||
        option == "--progress" || option == "--no-progress" || option == "--help" || option == "--help-all" || option == "-h" ||
        option == "--extend" || option == "--no-extend" || option == "--update" || option == "--no-update" ||
        option == "--cancel-update" || option == "--no-cancel-update" ||
        option == "--policy-only" || option == "--no-policy-only" || option == "--upgrade-job" || option == "--no-upgrade-job" ||
        option == "--allow-missing" || option == "--no-allow-missing") return 0;
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
