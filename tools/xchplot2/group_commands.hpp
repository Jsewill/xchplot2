#pragma once

#include "host/GroupPlotter.hpp"

namespace pos2gpu {
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
