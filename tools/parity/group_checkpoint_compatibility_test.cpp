#include "host/GroupStaging.hpp"
#include "host/GroupMerge.hpp"
#include "host/GroupExternal.hpp"
#include "host/GroupCheckpoint.hpp"

#include <iostream>
#include <map>
#include <sys/wait.h>

using namespace group_grow;
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024;

std::vector<uint8_t> contents(fs::path const& path)
{ auto in = group_tools::input(path.string()); return {std::istreambuf_iterator<char>(in), {}}; }
void replace(fs::path const& path, std::vector<uint8_t> const& bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<char const*>(bytes.data()), bytes.size()); require(bool(out), "cannot write checkpoint fixture");
}
auto snapshot(fs::path const& root)
{
    std::map<std::string, std::vector<uint8_t>> result;
    for (auto const& entry : fs::recursive_directory_iterator(root))
        result.emplace(entry.path().string(), entry.is_regular_file() ? contents(entry.path()) : std::vector<uint8_t>{});
    return result;
}
Identity identity()
{
    Identity result; result.header.magic = PlotGroupFile::MAGIC; result.header.version = PlotGroupFile::FORMAT_VERSION;
    result.header.group_id = PlotGroupId(std::string(64, 'a')); result.header.k = 18; result.header.strength = 2;
    result.memo = {0, 17, 255}; result.header.memo_length = uint8_t(result.memo.size());
    result.header.chunk_index_offset = sizeof(Header) + result.memo.size(); return result;
}
std::vector<std::vector<uint64_t>> members()
{
    std::vector<std::vector<uint64_t>> result(1);
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(18); ++chunk)
        for (uint64_t i = 0; i < 64; ++i)
        {
            result.front().push_back(chunk * range + i * (range / 64) + 7);
            if (i == 3) result.front().push_back(result.front().back());
        }
    return result;
}
Config config()
{
    Config cfg; cfg.capacity_bytes = 64 * mib; cfg.memory_limit_bytes = 256 * mib;
    cfg.window_bytes = 32768; cfg.codec_bytes = 8 * mib; cfg.checkpoint_bytes = 65536;
    cfg.io_overlap = false; return cfg;
}
ExternalConfig external(fs::path const& path)
{
    auto const scratch = path.parent_path() / "scratch"; fs::create_directories(scratch);
    ExternalConfig ext; ext.segment_bytes = 65536; ext.fan_in = 2;
    ext.scratch.push_back({scratch.string(), 16 * mib}); return ext;
}
ConstructionInspection inspect(std::string const& engine, fs::path const& path, Config cfg = config())
{
    ConstructionInspectionLimits limits{64 * mib, std::chrono::steady_clock::now() + std::chrono::seconds(30)};
    if (engine == "grow") return Engine::inspect(path.string(), cfg, limits, identity());
    if (engine == "stage") return StagingEngine::inspect(path.string(), cfg, limits, identity());
    if (engine == "merge") return MergeEngine::inspect(path.string(), cfg, limits, identity());
    return ExternalEngine::inspect_construction(path.string(), 32 * mib, limits, cfg.path_bindings).construction;
}
void complete(std::string const& engine, fs::path const& path, bool resume, bool tight = false, Config cfg = config())
{
    auto input = members();
    if (tight)
    {
        if (engine == "external")
        {
            auto const saved = ExternalEngine::inspect(path.string(), cfg.memory_limit_bytes, cfg.path_bindings);
            cfg.capacity_bytes = saved.peak_bytes;
        }
        else cfg.capacity_bytes = fs::file_size(path);
    }
    if (engine == "grow")
    {
        Engine output(path.string(), cfg, resume, identity()); output.confirm_identity();
        auto const saved = output.identity();
        if (saved.pending) output.resume(saved.target_members > saved.old_members ? Members(input) : Members{});
        if (!output.identity().header.group_size) require(output.append(input).accepted, "grow fixture rejected input");
    }
    else if (engine == "stage")
    {
        StagingEngine output(path.string(), cfg, resume, identity()); output.confirm_identity();
        if (!output.committed_member_count()) require(output.append(input).accepted, "staging fixture rejected input");
        output.finish();
    }
    else if (engine == "merge")
    {
        MergeEngine output(path.string(), cfg, resume, identity()); output.confirm_identity();
        if (!output.committed_member_count()) require(output.append(input).accepted, "merge fixture rejected input");
        output.finish();
    }
    else
    {
        auto ext = resume ? ExternalEngine::inspect(path.string(), cfg.memory_limit_bytes, cfg.path_bindings).external : external(path);
        ExternalEngine output(path.string(), cfg, ext, resume, identity()); output.confirm_identity();
        if (!output.committed_member_count()) require(output.append(input).accepted, "external fixture rejected input");
        output.finish();
    }
}
int child(fs::path const& binary, std::vector<std::string> arguments, std::string const& failpoint = {})
{
    auto const pid = ::fork(); require(pid >= 0, "cannot fork cross-build fixture");
    if (!pid)
    {
        if (failpoint.empty()) ::unsetenv("XCHPLOT2_GROW_FAILPOINT");
        else ::setenv("XCHPLOT2_GROW_FAILPOINT", failpoint.c_str(), 1);
        if (failpoint.empty()) ::unsetenv("XCHPLOT2_STREAM_FAILPOINT");
        else ::setenv("XCHPLOT2_STREAM_FAILPOINT", failpoint.c_str(), 1);
        std::vector<char*> command{const_cast<char*>(binary.c_str())};
        for (auto& argument : arguments) command.push_back(argument.data()); command.push_back(nullptr);
        ::execv(binary.c_str(), command.data()); std::_Exit(127);
    }
    int status = 0; require(::waitpid(pid, &status, 0) == pid && WIFEXITED(status), "cross-build fixture did not exit normally");
    return WEXITSTATUS(status);
}
struct RecordLocation { size_t offset, bytes, extension; };
std::vector<RecordLocation> records(std::string const& engine, std::vector<uint8_t> const& bytes)
{
    std::string const magic = engine == "grow" ? "XCHPLOT2GROWv1" : engine == "stage" ? "XCHPLOT2STAGEv2" : "XCHPLOT2MERGEv2";
    std::string const algorithm = construction_algorithm_identity;
    std::vector<RecordLocation> result;
    for (auto found = std::search(bytes.begin(), bytes.end(), magic.begin(), magic.end()); found != bytes.end();
        found = std::search(found + 1, bytes.end(), magic.begin(), magic.end()))
    {
        size_t const offset = size_t(found - bytes.begin());
        size_t length = 1024;
        if (engine == "merge")
        {
            if (sizeof(MergeState::Record) > bytes.size() - offset) continue;
            MergeState::Record saved{}; std::memcpy(&saved, bytes.data() + offset, sizeof(saved));
            if (saved.slot_bytes > bytes.size() - offset) continue; length = size_t(saved.slot_bytes);
        }
        if (length > bytes.size() - offset || length < 32) continue;
        auto const body = std::span(bytes).subspan(offset, length - 32); auto const checksum = grow_digest(body);
        if (!std::equal(checksum.begin(), checksum.end(), bytes.begin() + offset + length - 32)) continue;
        auto extension = std::search(body.begin(), body.end(), algorithm.begin(), algorithm.end());
        require(extension != body.end() && sizeof(ConstructionCompatibility) <= size_t(body.end() - extension), "checkpoint fixture has no compatibility extension");
        result.push_back({offset, length, offset + size_t(extension - body.begin())});
    }
    require(!result.empty(), "cross-build fixture has no durable checkpoint"); return result;
}
void resign(std::vector<uint8_t>& bytes, RecordLocation const& record)
{
    auto const digest = grow_digest(std::span(bytes).subspan(record.offset, record.bytes - 32));
    std::copy(digest.begin(), digest.end(), bytes.begin() + record.offset + record.bytes - 32);
}
fs::path checkpoint_path(std::string const& engine, fs::path const& path)
{ return engine == "external" ? fs::path(path.string() + ".external") / "checkpoint" : path; }
std::vector<uint8_t> alter(std::string const& engine, std::vector<uint8_t> bytes, std::string const& change)
{
    if (engine == "external")
    {
        using namespace external_detail;
        CheckpointInput in(std::string_view(reinterpret_cast<char const*>(bytes.data() + 65), bytes.size() - 65));
        auto version = get_string(in); auto const producer = get_string(in); auto compatibility = get<ConstructionCompatibility>(in);
        if (change == "algorithm") compatibility.algorithm.front() ^= 1;
        if (change == "representation") compatibility.representation.front() ^= 1;
        if (change == "format") version = "xchplot2-external-unsupported";
        bool const legacy = change == "legacy";
        if (legacy) version = version == "xchplot2-external-6" ? "xchplot2-external-4" : "xchplot2-external-3";
        std::ostringstream out(std::ios::binary); put_string(out, version); put_string(out, producer);
        if (!legacy) put(out, compatibility);
        auto const tail = in.rdbuf()->in_avail(); std::string rest(size_t(tail), '\0'); in.read(rest.data(), rest.size()); out.write(rest.data(), rest.size());
        auto const body = out.str(); auto const hash = stream_state::digest(body);
        bytes.assign(hash.begin(), hash.end()); bytes.push_back('\n'); bytes.insert(bytes.end(), body.begin(), body.end());
        if (change == "corrupt") bytes.back() ^= 1;
        return bytes;
    }
    auto const locations = records(engine, bytes);
    for (auto const& record : locations)
    {
        if (change == "algorithm") bytes[record.extension] ^= 1;
        if (change == "representation") bytes[record.extension + offsetof(ConstructionCompatibility, representation)] ^= 1;
        if (change == "legacy")
        {
            std::fill_n(bytes.begin() + record.extension, sizeof(ConstructionCompatibility), 0);
            if (engine == "grow") { uint32_t const version = 1; std::memcpy(bytes.data() + record.offset + 16 + sizeof(uint64_t), &version, sizeof(version)); }
            else
            {
                auto const marker = engine == "stage" ? "XCHPLOT2STAGEv1\0\0" : "XCHPLOT2MERGEv1\0";
                std::memcpy(bytes.data() + record.offset, marker, 16);
            }
        }
        if (change == "format")
        {
            if (engine == "grow") { uint32_t const version = UINT32_MAX; std::memcpy(bytes.data() + record.offset + 16 + sizeof(uint64_t), &version, sizeof(version)); }
            else bytes[record.offset + std::string(engine == "stage" ? "XCHPLOT2STAGEv" : "XCHPLOT2MERGEv").size()] = '9';
        }
        resign(bytes, record);
    }
    if (change == "corrupt") bytes[locations.back().extension] ^= 1;
    return bytes;
}
void refuses_unchanged(std::string const& engine, fs::path const& path, fs::path const& root, std::string const& expected)
{
    auto const before = snapshot(root); bool refused = false;
    try { (void)inspect(engine, path); }
    catch (LegacyConstructionBuildError const& error)
    {
        refused = true;
        require(error.producer_build == "checkpoint-source-a" && error.engine == (engine == "grow" ? "growth" :
            engine == "stage" ? "staging" : engine) && std::string(error.what()).find(expected) != std::string::npos,
            "legacy checkpoint diagnostic lost its original build or engine");
    }
    catch (std::exception const& error) { refused = true; require(std::string(error.what()).find(expected) != std::string::npos, error.what()); }
    require(refused && snapshot(root) == before, "incompatible checkpoint inspection changed saved bytes");
    refused = false;
    try { complete(engine, path, true); }
    catch (std::exception const& error) { refused = true; require(std::string(error.what()).find(expected) != std::string::npos, error.what()); }
    require(refused && snapshot(root) == before, "incompatible checkpoint recovery changed saved bytes");
}
void stream_compatibility(fs::path const& root, fs::path const& producer, std::vector<uint8_t> const& expected)
{
    auto const stream_producer = producer.parent_path() / "pos2_pr118_stream_checkpoint_fixture";
    auto const stream_recovery = producer.parent_path() / "pos2_pr118_stream_checkpoint_recovery";
    auto input = members(); PlotData data; data.t3_proof_fragments = input.front(); auto const id = identity();
    PlotGroupParams const params(id.header.group_id, 18, 2, 0);
    auto const raw = root / "stream-raw.plot", manifest = root / "stream-inputs";
    PlotFile::writeData(raw.string(), data, params, 0, id.memo);
    { std::ofstream out(manifest); out << "18 2 0 " << id.header.group_id.to_string() << ' ' << group_tools::hex(id.memo) << '\n' << std::quoted(raw.string()) << '\n'; }
    for (auto const* boundary : {"stage-after-checkpoint", "final-after-checkpoint"})
    {
        auto const stream_work = root / (std::string("stream-") + boundary);
        std::vector<std::string> args{manifest.string(), stream_work.string(), "64", "1", "2", "4", "1"};
        require(child(stream_producer, args, boundary) == 86, "stream source build did not reach interrupted checkpoint");
        args.push_back("--resume");
        require(child(stream_recovery, args) == 0 && contents(stream_work / "group.gplot") == expected,
            "cross-build stream recovery differs from pinned CPU reference");
    }
    auto const work = root / "stream"; fs::create_directory(work);
    require(child(producer, {"--stream", work.string()}) == 0, "source build did not create stream checkpoint");
    stream_state::Config cfg{1, 2, 4096, 4096, false, false}; auto const path = work / "checkpoint";
    auto const original = contents(path);
    auto saved = stream_state::load_state(work, cfg, "cross-build-stream", 1, mib, 1);
    require(saved.producer_build == "checkpoint-source-a" && saved.version == 2, "stream checkpoint lost original producer provenance");
    stream_state::save_state(work, saved, "cross-build-stream", "cross-build-stream");
    require(contents(path) == original, "compatible stream resave changed canonical checkpoint bytes");
    auto changed = cfg; ++changed.fan; bool refused = false;
    try { (void)stream_state::load_state(work, changed, "cross-build-stream", 1, mib, 1); } catch (std::runtime_error const&) { refused = true; }
    require(refused && contents(path) == original, "stream configuration mismatch changed checkpoint");
    for (auto const* field : {"format", "algorithm", "representation", "checksum"})
    {
        std::string body(original.begin() + 65, original.end());
        if (std::string(field) == "format") body.replace(0, std::string("xchplot2-stream-2").size(), "xchplot2-stream-unsupported");
        else if (std::string(field) == "algorithm") body.at(body.find(construction_algorithm_identity)) ^= 1;
        else if (std::string(field) == "representation")
        {
            auto const representation = group_tools::hex(construction_compatibility({}).representation);
            auto& digit = body.at(body.find(representation)); digit = digit == '0' ? '1' : '0';
        }
        std::string record = stream_state::digest(body) + '\n' + body;
        if (std::string(field) == "checksum") record.back() ^= 1;
        replace(path, {record.begin(), record.end()}); auto const unchanged = contents(path); refused = false;
        try { (void)stream_state::load_state(work, cfg, "cross-build-stream", 1, mib, 1); }
        catch (std::runtime_error const& error) { refused = std::string(error.what()).find(field) != std::string::npos; }
        require(refused && contents(path) == unchanged, "incompatible stream contract was accepted or changed saved bytes");
    }
    replace(path, original);
    saved.version = 1; stream_state::save_state(work, saved, "cross-build-stream", "cross-build-stream");
    auto const legacy = contents(path); refused = false;
    try { (void)stream_state::load_state(work, cfg, "cross-build-stream", 1, mib, 1); }
    catch (std::runtime_error const& error) { refused = std::string(error.what()).find("original exact build") != std::string::npos; }
    require(refused && contents(path) == legacy, "foreign legacy stream checkpoint lost its exact-build guard");
}
void retired_compatibility(fs::path const& root, fs::path const& producer, std::vector<uint8_t> const& expected)
{
#ifdef __linux__
    std::vector<std::string> boundaries{"external-handoff-retired"};
    if (!::setxattr(root.c_str(), "user.xchplot2.compatibility-test", "x", 1, 0))
    {
        require(!::removexattr(root.c_str(), "user.xchplot2.compatibility-test"), "cannot remove xattr fixture probe");
        boundaries.push_back("external-handoff-checkpoint-removed");
    }
    for (auto const& boundary : boundaries)
    {
        auto const path = root / (boundary + ".gplot");
        require(child(producer, {"--retire", "external", path.string()}, boundary) == 86, "source build did not interrupt terminal retirement");
        fs::path holder;
        for (auto const& entry : fs::directory_iterator(root))
            if (entry.path().filename().string().starts_with(path.filename().string() + ".external.retired-")) holder = entry.path();
        require(!holder.empty(), "retired construction fixture has no owner directory");
        auto const before = snapshot(root); auto const saved = ExternalEngine::inspect_retired(path.string(), holder.string());
        require(saved.complete && saved.construction.checkpoint_identity.producer_build == "checkpoint-source-a" &&
            saved.construction.checkpoint_identity.algorithm == construction_algorithm_identity &&
            !saved.construction.checkpoint_identity.legacy_exact_build && contents(path) == expected && snapshot(root) == before,
            "cross-build terminal inspection changed retired ownership or canonical output");
    }
#endif
}
void external_rebinding(fs::path const& root, fs::path const& producer, std::vector<uint8_t> const& expected)
{
    auto at_boundary = [](std::string const& boundary, auto operation)
    {
        auto const pid = ::fork(); require(pid >= 0, "cannot fork rebound checkpoint fixture");
        if (!pid) { ::setenv("XCHPLOT2_GROW_FAILPOINT", boundary.c_str(), 1); operation(); std::_Exit(0); }
        int status = 0; require(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "rebound checkpoint fixture missed its durable boundary: " + boundary);
    };
    for (auto const* boundary : {"external-initialized-before-checkpoint", "external-run-after-checkpoint", "external-pack-after-checkpoint"})
        for (bool outside : {false, true})
        {
            auto const name = std::string(boundary) + (outside ? "-outside" : "-local");
            auto const original = root / (name + "-original"), current = root / (name + "-current"); fs::create_directory(original);
            auto const native = farming_directory_identity(original); auto const path = original / "group.gplot";
            auto const scratch = outside ? root / (name + "-scratch") : original / "scratch"; fs::create_directory(scratch);
            require(child(producer, {"--external-scratch", "external", path.string(), scratch.string()}, boundary) == 86,
                "external alias fixture did not reach its durable boundary");
            fs::rename(original, current); fs::create_directory(original); auto const target = current / path.filename();
            auto const before = snapshot(root); bool refused = false;
            try { (void)inspect("external", target); } catch (std::exception const&) { refused = true; }
            require(refused && snapshot(root) == before, "external checkpoint inferred path authority from a renamed directory");
            auto cfg = config(); cfg.path_bindings.push_back({original.string(), current.string(), native});
            auto invalid = cfg; invalid.path_bindings.front().directory_identity = "foreign directory"; refused = false;
            try { complete("external", target, true, false, invalid); } catch (std::exception const&) { refused = true; }
            require(refused && snapshot(root) == before, "external alias adopted a different persistent directory identity or changed saved bytes");
            auto const reviewed = inspect("external", target, cfg);
            require(reviewed.checkpoint && snapshot(root) == before, "external alias inspection changed its owned checkpoint");
            auto const held = current / "held-target"; fs::rename(target, held); fs::copy_file(held, target); auto const substituted = snapshot(root); refused = false;
            try { complete("external", target, true, false, cfg); } catch (std::exception const&) { refused = true; }
            require(refused && snapshot(root) == substituted, "external alias accepted a copied target object");
            fs::remove(target); fs::rename(held, target);
            at_boundary("external-ready-after-checkpoint", [&] { complete("external", target, true, false, cfg); });
            auto const checkpoint = contents(checkpoint_path("external", target));
            external_detail::CheckpointInput saved(std::string_view(reinterpret_cast<char const*>(checkpoint.data() + 65), checkpoint.size() - 65));
            (void)external_detail::get_string(saved);
            require(external_detail::get_string(saved) == "checkpoint-source-a", "rebound checkpoint replaced its producer provenance");
            (void)external_detail::get<ConstructionCompatibility>(saved);
            require(external_detail::get_string(saved) == path.string(), "rebound checkpoint replaced its original path authority");
            auto const returned = root / (name + "-returned"); fs::rename(current, returned); fs::create_directory(current);
            auto const returned_target = returned / path.filename(); auto latest = config();
            latest.path_bindings.push_back({original.string(), returned.string(), native});
            if (outside)
            {
                auto const displaced = root / (name + "-displaced-scratch"); fs::rename(scratch, displaced); fs::create_directory(scratch);
                auto const missing_scratch = snapshot(root); refused = false;
                try { complete("external", returned_target, true, false, latest); } catch (std::exception const&) { refused = true; }
                require(refused && snapshot(root) == missing_scratch,
                    "destination rediscovery authorized an independently moved outside scratch root or changed files");
                fs::remove(scratch); fs::rename(displaced, scratch);
            }
            complete("external", returned_target, true, false, latest);
            require(contents(returned_target) == expected && fs::is_empty(current) && fs::is_empty(original) &&
                ExternalEngine::inspect(returned_target.string(), latest.memory_limit_bytes, latest.path_bindings).complete,
                "fresh recovery after a second native root move changed canonical bytes or required its former alias");
        }
    {
        auto const target_root = root / "stationary-target", original = root / "mapped-scratch-original";
        auto const current = root / "mapped-scratch-current", returned = root / "mapped-scratch-returned";
        fs::create_directory(target_root); fs::create_directory(original);
        auto const native = farming_directory_identity(original), path = (target_root / "group.gplot").string();
        require(child(producer, {"--external-scratch", "external", path, original.string()}, "external-run-after-checkpoint") == 86,
            "stationary target fixture did not reach its durable boundary");
        fs::rename(original, current); auto cfg = config();
        cfg.path_bindings.push_back({original.string(), current.string(), native});
        at_boundary("external-ready-after-checkpoint", [&] { complete("external", path, true, false, cfg); });
        fs::rename(current, returned); fs::create_directory(current);
        auto latest = config(); latest.path_bindings.push_back({original.string(), returned.string(), native});
        complete("external", path, true, false, latest);
        require(contents(path) == expected && fs::is_empty(current) &&
            ExternalEngine::inspect(path, latest.memory_limit_bytes, latest.path_bindings).complete,
            "stationary target lost a mapped scratch drive after repeated native root moves");
    }
#ifdef __linux__
    for (auto const* boundary : {"external-handoff-retired", "external-handoff-checkpoint-removed"})
    {
        auto const original = root / (std::string(boundary) + "-alias-original"), current = root / (std::string(boundary) + "-alias-current");
        fs::create_directory(original); auto const native = farming_directory_identity(original);
        if (std::string(boundary) == "external-handoff-checkpoint-removed")
        {
            if (::setxattr(original.c_str(), "user.xchplot2.alias-test", "x", 1, 0)) continue;
            require(!::removexattr(original.c_str(), "user.xchplot2.alias-test"), "cannot remove external alias xattr probe");
        }
        auto const path = original / "group.gplot";
        require(child(producer, {"--produce", "external", path.string()}, "external-run-after-checkpoint") == 86,
            "retired alias source boundary was not reached");
        fs::rename(original, current); auto const target = current / path.filename(); auto cfg = config();
        cfg.path_bindings.push_back({original.string(), current.string(), native});
        complete("external", target, true, false, cfg);
        at_boundary(boundary, [&]
        {
            auto const ext = ExternalEngine::inspect(target.string(), cfg.memory_limit_bytes, cfg.path_bindings).external;
            ExternalEngine output(target.string(), cfg, ext, true); output.confirm_identity(); output.release_completed();
        });
        fs::path holder;
        for (auto const& entry : fs::directory_iterator(current))
            if (entry.path().filename().string().starts_with("group.gplot.external.retired-")) holder = entry.path();
        auto const before = snapshot(root);
        std::vector<ConstructionPathBinding> bindings{{original.string(), current.string(), native}};
        auto const saved = ExternalEngine::inspect_retired(target.string(), holder.string(), 64 * mib, bindings);
        require(saved.complete && contents(target) == expected && snapshot(root) == before, "retired external alias changed native ownership or output bytes");
        auto const returned = root / (std::string(boundary) + "-alias-returned"); fs::rename(current, returned);
        auto const returned_target = returned / path.filename(), returned_holder = returned / holder.filename();
        bindings.front().current_root = returned.string(); auto const second_move = snapshot(root);
        auto const repeated = ExternalEngine::inspect_retired(returned_target.string(), returned_holder.string(), 64 * mib, bindings);
        require(repeated.complete && repeated.construction.checkpoint_identity.producer_build == "checkpoint-source-a" &&
            contents(returned_target) == expected && snapshot(root) == second_move,
            "fresh retired inspection after a second native root move required a former alias or changed ownership");
    }
#endif
}

int main(int argc, char** argv)
try
{
    if (argc == 3 && std::string(argv[1]) == "--stream")
    {
        stream_state::State state; state.config = {1, 2, 4096, 4096, false, false}; state.raw.resize(1);
        stream_state::save_state(argv[2], state, "cross-build-stream", "cross-build-stream"); return 0;
    }
    if (argc >= 4)
    {
        if (std::string(argv[1]) == "--external-scratch")
        {
            fs::path const path(argv[3]); auto ext = external(path); ext.scratch.front().directory = argv[4];
            ExternalEngine output(path.string(), config(), ext, false, identity()); output.confirm_identity();
            require(output.append(members()).accepted, "external alias source rejected input"); output.finish(); return 0;
        }
        if (std::string(argv[1]) == "--retire")
        {
            fs::path const path(argv[3]); complete("external", path, false);
            ExternalEngine output(path.string(), config(), external(path), true); output.confirm_identity(); output.release_completed(); return 0;
        }
        complete(argv[2], argv[3], std::string(argv[1]) == "--recover", argc == 5); return 0;
    }
    require(argc == 1, "usage: grow-checkpoint-compatibility-test");
    auto const producer = fs::absolute(argv[0]).parent_path() / "pos2_pr118_grow_checkpoint_fixture";
    require(fs::is_regular_file(producer), "cross-build fixture executable is missing");
    auto name = (producer.parent_path() / "checkpoint-compatibility-XXXXXX").string();
    require(::mkdtemp(name.data()), "cannot create cross-build checkpoint fixture directory");
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{name}; fs::path const root(name);
    auto input = members(); PlotData data; data.t3_proof_fragments = input.front(); auto const id = identity();
    auto const reference = root / "reference.gplot";
    PlotGroupFile::writeData(reference.string(), data, PlotGroupParams(id.header.group_id, 18, 2, 0), PlotGroupFile::PROOFS_PER_CHUNK_BITS, id.memo);
    auto const expected = contents(reference);
    for (auto const* engine : {"grow", "stage", "merge", "external"})
    {
        std::string const point = std::string(engine) == "grow" ? "direct-committed:2" : std::string(engine) == "stage" ? "stage-run-committed" :
            std::string(engine) == "merge" ? "merge-run-committed" : "external-run-after-checkpoint";
        std::string const resave_point = std::string(engine) == "grow" ? "direct-committed" : std::string(engine) == "stage" ? "stage-pack-initialized" :
            std::string(engine) == "merge" ? "merge-finishing-committed" : "external-pack-initialized-after-checkpoint";
        auto const path = root / (std::string(engine) + ".gplot");
        require(child(producer, {"--produce", engine, path.string()}, point) == 86, "source build did not reach interrupted checkpoint: " + point);
        auto const before = snapshot(root); auto const inspected = inspect(engine, path);
        require(inspected.checkpoint && inspected.checkpoint_identity.producer_build == "checkpoint-source-a" &&
            inspected.checkpoint_identity.algorithm == construction_algorithm_identity && !inspected.checkpoint_identity.legacy_exact_build &&
            !inspected.checkpoint_identity.representation.empty() && snapshot(root) == before,
            "compatible cross-build inspection lost identity or changed checkpoint bytes");
        auto const checkpoint = checkpoint_path(engine, path); auto const original = contents(checkpoint);
        for (auto const* change : {"format", "algorithm", "representation"})
        {
            replace(checkpoint, alter(engine, original, change)); refuses_unchanged(engine, path, root, change); replace(checkpoint, original);
        }
        bool refused = false;
        if (std::string(engine) != "external")
        {
            auto denied = config(); denied.capacity_bytes = fs::file_size(path) - 1;
            try { (void)inspect(engine, path, denied); } catch (std::exception const&) { refused = true; }
            require(refused && contents(checkpoint) == original, "incompatible saved capacity was accepted or changed bytes");
        }
        if (std::string(engine) == "external")
        {
            auto incompatible = external(path); incompatible.segment_bytes *= 2; auto const unchanged = snapshot(root); refused = false;
            try { ExternalEngine output(path.string(), config(), incompatible, true); } catch (std::runtime_error const&) { refused = true; }
            require(refused && snapshot(root) == unchanged, "incompatible external settings were accepted or changed saved files");
        }
        replace(checkpoint, alter(engine, original, "legacy")); refuses_unchanged(engine, path, root, "original exact build");
        require(child(producer, {"--recover", engine, path.string()}, resave_point) == 86, "original build did not resave legacy checkpoint");
        refuses_unchanged(engine, path, root, "original exact build");
        require(child(producer, {"--recover", engine, path.string()}) == 0 && contents(path) == expected, "original build failed legacy checkpoint recovery");
        auto const torn = root / (std::string(engine) + "-torn.gplot");
        auto const torn_point = std::string(engine) == "grow" ? "checkpoint-partial:2" : std::string(engine) == "stage" ? "stage-checkpoint-partial:2" :
            std::string(engine) == "merge" ? "merge-checkpoint-partial:2" : "external-initialized-before-checkpoint";
        require(child(producer, {"--produce", engine, torn.string()}, torn_point) == 86, "source build did not reach torn checkpoint boundary");
        complete(engine, torn, true);
        require(contents(torn) == expected, "cross-build torn checkpoint recovery differs from pinned CPU reference");
        auto const compatible = root / (std::string(engine) + "-compatible.gplot");
        require(child(producer, {"--produce", engine, compatible.string()}, point) == 86, "source build did not create compatible fixture");
        require(child(fs::absolute(argv[0]), {"--recover", engine, compatible.string()}, resave_point) == 86, "compatible build did not resave checkpoint");
        require(inspect(engine, compatible).checkpoint_identity.producer_build == "checkpoint-source-a", "compatible resave replaced producer provenance");
        complete(engine, compatible, true, true);
        require(contents(compatible) == expected, "cross-build recovery at exact saved capacity differs from pinned CPU reference");
        auto const corrupt = root / (std::string(engine) + "-corrupt.gplot");
        require(child(producer, {"--produce", engine, corrupt.string()}, point) == 86, "source build did not create checksum fixture");
        auto const corrupt_checkpoint = checkpoint_path(engine, corrupt);
        replace(corrupt_checkpoint, alter(engine, contents(corrupt_checkpoint), "corrupt"));
        if (std::string(engine) == "external") refuses_unchanged(engine, corrupt, root, "checksum");
        else { complete(engine, corrupt, true); require(contents(corrupt) == expected, "corrupt checkpoint generation bypassed checksum recovery"); }
    }
    stream_compatibility(root, producer, expected);
    retired_compatibility(root, producer, expected);
    external_rebinding(root, producer, expected);
    std::cout << "cross-build construction checkpoint compatibility and canonical recovery passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << "checkpoint compatibility: " << error.what() << '\n'; return 1; }
