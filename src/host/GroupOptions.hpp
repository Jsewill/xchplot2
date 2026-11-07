#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace group_grow { struct FarmingProfile; struct WorkerConfig; struct WorkerTaskAdmission; }

namespace pos2gpu {
class ComputeSession;

struct GroupOptions
{
    std::string command;
    bool json = false, quiet = false, verbose = false, progress = false, report_metrics = true;
    std::function<bool()> cancelled;
    std::function<void(std::filesystem::path const&, uint16_t, uint64_t, bool)> on_output;
    std::filesystem::path target, fixtures, manifest, save_plan, from_plan, save_profile;
    std::filesystem::path replot_inventory;
    std::string farm_adapter = "chia";
    uint64_t farm_adapter_timeout_ms = 30000;
    bool farm_adapter_set = false, farm_adapter_timeout_set = false;
    bool chia_root_set = false, chia_executable_set = false, farming_profile_set = false;
    uint64_t reclaim_before_bytes = 0;
    std::vector<std::filesystem::path> replot_source_roots, replot_exclude_roots;
    uint64_t replot_max_source_files = 0, replot_max_source_bytes = 0;
    bool replot_status = false, replot_revalidate = false;
    std::optional<std::string> replot_verification;
    std::optional<uint64_t> replot_proof_first_groups, replot_proof_every_groups;
    std::vector<std::filesystem::path> assessment_plans;
    std::vector<std::filesystem::path> calibration_plots;
    uint64_t calibration_samples = 16, calibration_timeout_ms = 30000, calibration_memory_bytes = 512 * 1024 * 1024;
    std::vector<std::filesystem::path> destinations, temporary;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> relocations;
    std::optional<std::string> group, memo;
    std::optional<std::string> farmer, pool_pk, pool_ph, pool_address, seed;
    std::optional<std::string> seed_owner;
    std::optional<uint64_t> k, strength, meta;
    uint64_t max_ram = 14000000000ull;
    uint64_t capacity = 0, headroom = 64 * 1024 * 1024, temporary_capacity = 0;
    uint64_t window = 16 * 1024 * 1024, workers = 0, batch = 65535, max_members = 65535;
    uint64_t group_members = 0, max_groups = UINT64_MAX, max_total_members = UINT64_MAX;
    uint64_t checkpoint_bytes = 256 * 1024 * 1024, device = 0;
    std::string backend = "cpu";
    std::string tier = "auto";
    std::string member_storage = "compact";
    std::string construction = "auto";
    std::string destination_temp = "off";
    bool destination_temp_set = false;
    bool io_overlap = true, generation_overlap = true, cache_release = true;
    bool window_set = false, checkpoint_set = false, generation_overlap_auto = true;
    bool backend_set = false;
    bool resume = false, group_auto = false, plan_only = false, directory_farm = false, metadata_only = false, assess_jobs = false;
    bool calibrate = false, calibration_worker = false;
    bool replot_proof_worker = false, replot_structure_worker = false;
    uint64_t replot_proof_cpus = 0, replot_proof_solves = 1;
    bool protected_resources = false;
    uint64_t protected_gpu_bytes = 0;
    bool inventory = false;
    bool extend = false, allow_missing = false;
    bool capacity_set = false, headroom_set = false, max_groups_set = false, max_total_members_set = false;
    std::optional<std::string> farming_reader, farming_objective;
    std::optional<uint64_t> farming_cores, farming_lookup_workers, farming_memory, farming_index_cache, farming_filter_bits, farming_future_filter_bits, farming_deadline_ms, farming_chain_us, farming_index_ns;
    std::optional<uint64_t> farming_cpu_work_weight, farming_io_work_weight, farming_max_eligible_groups;
    std::filesystem::path farming_profile_path, chia_root, chia_executable;
    bool farming_detect_local = false;
    std::shared_ptr<group_grow::FarmingProfile const> farming_profile;
    std::string pool_broker, pool_owner, pool_ca, pool_cert, pool_key, pool_listen, pool_advertise;
    std::string compute_mode = "auto", pool_directory, pool_server_name, pool_session_owner;
    bool automatic_compute = false, automatic_local_compute = true, pool_owner_explicit = false;
    std::shared_ptr<ComputeSession> compute_session;
    std::function<void()> compute_admit_session;
    std::function<void(group_grow::WorkerConfig const&)> compute_admit_worker;
    std::function<void(group_grow::WorkerTaskAdmission const&)> compute_admit_task;
    std::string pool_offer_backend = "cpu";
    uint64_t pool_offer_ram = 0, pool_offer_cpus = 0, pool_offer_vram = 0, pool_offer_slots = 1, pool_inflight = 64;
    uint64_t pool_timeout_ms = 3600000, pool_socket_timeout_ms = 30000, pool_frame_bytes = 1024 * 1024;
    uint64_t pool_codec_ram = 0;
    bool pool_codec_auto = true, pool_offer_backend_set = false;
};

} // namespace pos2gpu
