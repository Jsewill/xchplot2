// Local connection preferences, separate from portable resource declarations.
#pragma once
#include "host/GroupOptions.hpp"
#include "host/GroupHarvesterProfile.hpp"

namespace pos2gpu
{
inline bool set_group_compute_setting(GroupOptions& args, std::string key, std::string value)
{
    using group_grow::profile_detail::profile_require;
    if (key.starts_with("--pool-")) key.replace(2, 5, "compute-");
    auto integer = [&]
    {
        uint64_t result = 0;
        auto const parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        profile_require(!value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
            "compute setting requires a nonnegative integer");
        return result;
    };
    auto path = [&]
    {
        profile_require(!value.empty() && value.size() <= 4096 && value.find('\0') == value.npos, "invalid compute settings path");
        auto const native = std::filesystem::absolute(std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal().u8string();
        value.assign(reinterpret_cast<char const*>(native.data()), native.size());
        return value;
    };
    profile_require(value.size() <= 4096 && value.find('\0') == value.npos, "invalid compute setting value");
    if (key == "--compute")
    { profile_require(value == "auto" || value == "off", "--compute requires auto or off"); args.compute_mode = value; }
    else if (key == "--compute-local")
    { profile_require(value == "on" || value == "off", "--compute-local requires on or off"); args.automatic_local_compute = value == "on"; }
    else if (key == "--compute-directory") args.pool_directory = path();
    else if (key == "--compute-broker") args.pool_broker = value;
    else if (key == "--compute-owner-id") { args.pool_owner = value; args.pool_owner_explicit = true; }
    else if (key == "--compute-ca") args.pool_ca = path();
    else if (key == "--compute-cert") args.pool_cert = path();
    else if (key == "--compute-key") args.pool_key = path();
    else if (key == "--compute-listen") args.pool_listen = value;
    else if (key == "--compute-advertise") args.pool_advertise = value;
    else if (key == "--compute-offer-backend")
    {
        profile_require(value == "cpu" || value == "gpu" || value == "auto", "--compute-offer-backend requires cpu, gpu or auto");
        args.pool_offer_backend = value; args.pool_offer_backend_set = true;
    }
    else if (key == "--compute-offer-ram") args.pool_offer_ram = args.pool_offer_ram_limit = integer();
    else if (key == "--compute-offer-cpus") args.pool_offer_cpus = args.pool_offer_cpu_limit = integer();
    else if (key == "--compute-offer-vram") args.pool_offer_vram = args.pool_offer_vram_limit = integer();
    else if (key == "--compute-offer-slots") args.pool_offer_slots = integer();
    else if (key == "--compute-output")
    {
        profile_require(value == "auto" || value == "packed" || value == "encoded", "--compute-output requires auto, packed or encoded");
        args.pool_output = value;
    }
    else if (key == "--compute-writers")
    {
        auto const count = value == "auto" ? 0 : integer();
        profile_require(value == "auto" || (count && count <= 1024), "--compute-writers requires auto or 1 to 1024");
        args.pool_writers = unsigned(count);
    }
    else if (key == "--compute-inflight") args.pool_inflight = integer();
    else if (key == "--compute-timeout-ms") args.pool_timeout_ms = integer();
    else if (key == "--compute-socket-timeout-ms") args.pool_socket_timeout_ms = integer();
    else if (key == "--compute-frame-bytes") args.pool_frame_bytes = integer();
    else if (key == "--compute-codec-ram")
    { args.pool_codec_auto = value == "auto"; args.pool_codec_ram = args.pool_codec_auto ? 0 : integer(); }
    else return false;
    args.compute_settings[key] = std::move(value); return true;
}
inline void write_group_compute_settings(std::ostream& out, GroupOptions const& args)
{
    out << "{\"schema\":1,\"options\":{"; bool comma = false;
    for (auto const& [key, value] : args.compute_settings)
    {
        if (comma) out << ','; comma = true;
        write_json_string(out, key); out << ':'; write_json_string(out, value);
    }
    out << "}}";
}
inline bool apply_group_compute_settings(GroupOptions& args, group_grow::profile_detail::Json const& document)
{
    using namespace group_grow::profile_detail;
    auto const* saved = field(document, "compute_settings"); if (!saved) return false;
    auto const* schema = field(*saved, "schema"), *options = field(*saved, "options");
    profile_require(saved->kind == Json::object && saved->fields.size() == 2 && schema && integer(*schema) == 1 &&
        options && options->kind == Json::object && options->fields.size() <= 21, "unsupported local compute settings");
    GroupOptions selected;
    for (auto const& [key, value] : options->fields)
        profile_require(set_group_compute_setting(selected, key, text(value)), "unknown local compute setting");
    auto const mode = args.compute_settings.find("--compute"), local = args.compute_settings.find("--compute-local");
    bool const disabled = mode != args.compute_settings.end() && mode->second == "off";
    bool const local_disabled = local != args.compute_settings.end() && local->second == "off";
    if (disabled || local_disabled)
    {
        auto inherited = std::move(selected.compute_settings); selected = GroupOptions{};
        for (auto const& [key, value] : inherited)
        {
            bool const offer = key.starts_with("--compute-offer-") || key == "--compute-listen" || key == "--compute-advertise";
            if (disabled ? key != "--compute" && key != "--compute-directory" : offer) continue;
            (void)set_group_compute_setting(selected, key, value);
        }
    }
    // A profile is a default. Explicit CLI choices (including auto/on and zero)
    // override it; resolved enrollment defaults are never saved as explicit.
    for (auto const& [key, value] : args.compute_settings)
        profile_require(set_group_compute_setting(selected, key, value), "unknown explicit compute setting");
    if (selected.compute_settings == args.compute_settings) return false;
    profile_require(!args.compute_session && !args.compute_worker_reservation, "cannot change compute settings in an active session");
    if (args.compute_requested)
    {
        static_cast<GroupComputeOptions&>(args) = *args.compute_requested;
        profile_require(args.compute_reserved_ram <= UINT64_MAX - args.max_ram, "compute RAM partition overflow");
        args.max_ram += args.compute_reserved_ram; args.compute_reserved_ram = 0;
        args.backend = args.compute_requested_backend; args.compute_requested.reset();
    }
    // Clear inherited fields suppressed by an explicit off selection too.
    static_cast<GroupComputeOptions&>(args) = static_cast<GroupComputeOptions const&>(selected);
    args.compute_settings.clear();
    for (auto const& [key, value] : selected.compute_settings)
        (void)set_group_compute_setting(args, key, value);
    return true;
}
}
