#include "host/GroupFarmAdapters.hpp"
#include <iostream>

using namespace group_grow;
using namespace group_grow::farm;
namespace fs = std::filesystem;
template<class F> void rejects(F operation, char const* message)
{ bool failed = false; try { operation(); } catch (std::exception const&) { failed = true; } require(failed, message); }
void append(std::vector<uint8_t>& out, uint64_t value, size_t bytes, bool little = false)
{ for (size_t i = 0; i < bytes; ++i) out.push_back(uint8_t(value >> (8 * (little ? i : bytes - i - 1)))); }
std::vector<uint8_t> legacy(bool compressed)
{
    std::vector<uint8_t> out;
    std::string magic = compressed ? "PLOT" : std::string("Proof of Space Plot\0", 19);
    out.insert(out.end(), magic.begin(), magic.end());
    if (compressed) append(out, 2, 4, true);
    out.insert(out.end(), 32, 0x12); out.push_back(32);
    if (!compressed) { append(out, 4, 2); for (char c : std::string("v1.0")) out.push_back(uint8_t(c)); }
    append(out, 112, 2); out.insert(out.end(), 32, 0x34); out.insert(out.end(), 48, 0x56); out.insert(out.end(), 32, 0x9a);
    if (compressed) { append(out, 1, 4, true); out.push_back(5); }
    size_t const first = out.size() + 80;
    for (unsigned i = 0; i < 10; ++i) append(out, first + 8 * i, 8);
    out.resize(first + 128); return out;
}
void write(fs::path const& file, std::vector<uint8_t> const& bytes)
{ std::ofstream out(file, std::ios::binary); out.write(reinterpret_cast<char const*>(bytes.data()), bytes.size()); require(bool(out), "fixture write failed"); }
void inventory(fs::path const& directory)
{
    fs::create_directories(directory / "nested"); write(directory / "og.plot", legacy(false)); write(directory / "nested" / "bb.plot", legacy(true));
    write(directory / "unknown.drplot", {1, 2, 3});
    PlotGroupFile::Header h{}; h.magic = PlotGroupFile::MAGIC; h.version = PlotGroupFile::FORMAT_VERSION;
    h.group_id = PlotGroupId(std::string(64, 'a')); h.k = 18; h.strength = 2; h.group_size = 4; h.memo_length = 112; h.chunk_index_offset = 200;
    std::vector<uint8_t> group(256); std::memcpy(group.data(), &h, sizeof(h));
    std::fill(group.begin() + sizeof(h), group.begin() + sizeof(h) + 112, uint8_t(0x7b)); write(directory / "group.gplot", group);
    std::array<fs::path, 1> roots{directory}; auto observed = scan_farm_inventory(roots);
    require(observed.complete && observed.plots.size() == 4, "bounded inventory omitted files");
    auto find = [&](std::string const& name) -> InventoryPlot const&
    { auto at = std::find_if(observed.plots.begin(), observed.plots.end(), [&](auto const& p) { return p.relative_path == name; }); require(at != observed.plots.end(), "missing fixture path"); return *at; };
    auto const og = find("og.plot"), bb = find("nested/bb.plot"), native = find("group.gplot");
    require(og.algorithm == PlotAlgorithm::pos1 && og.header_validated && og.compression_family == "uncompressed" &&
        bb.container == PlotContainer::chiapos_v2 && bb.compression_family == "bladebit" && bb.compression_level == 5 &&
        native.algorithm == PlotAlgorithm::pos2 && native.members == 4 && find("unknown.drplot").status == "protected", "format observations were conflated");
    auto text = farm_inventory_text(observed); auto restored = parse_farm_inventory(text);
    require(farm_inventory_text(restored) == text && text.find(std::string(64, '9')) == std::string::npos &&
        text.find("9a9a9a9a9a9a9a9a9a9a9a9a9a9a9a9a") == std::string::npos, "inventory lost determinism or exposed local memo secret");
    Resource storage; storage.id = "fixture-storage"; storage.host = observed.host; storage.kind = ResourceKind::storage;
    observed.resource_template = make_profile_template({storage}, {{"known-header", 3}, {"protected-unknown", 1}}, observed.captured_at, observed.producer_build);
    text = farm_inventory_text(observed); restored = parse_farm_inventory(text);
    require(restored.resource_template && farm_inventory_text(restored) == text &&
        !restored.resource_template->resources[0].capacity_known && restored.resource_template->workloads[0].cost == CostKind::unknown,
        "inventory template became a resource/cost guarantee or failed roundtrip");
    (void)recheck_inventory_source(observed.roots[0], og);
    InventoryLimits tiny; tiny.max_files = 1; rejects([&] { (void)scan_farm_inventory(roots, tiny); }, "entry bound was ignored");
    auto empty_tree = directory / "empty-tree"; fs::create_directories(empty_tree / "one" / "two" / "three");
    std::array<fs::path, 1> empty_roots{empty_tree}; tiny.max_files = 2;
    rejects([&] { (void)scan_farm_inventory(empty_roots, tiny); }, "already visited empty directories escaped the total entry bound");
    fs::create_directories(directory / "other-empty-tree" / "child");
    std::array<fs::path, 2> split_roots{empty_tree / "one" / "two", directory / "other-empty-tree"}; tiny.max_files = 1;
    rejects([&] { (void)scan_farm_inventory(split_roots, tiny); }, "total entry bound restarted at each root");
    fs::remove_all(empty_tree); fs::remove_all(directory / "other-empty-tree");
    tiny = {}; tiny.cancelled = [] { return true; }; rejects([&] { (void)scan_farm_inventory(roots, tiny); }, "cancellation was ignored");
    rejects([&] { (void)inventory_detail::catalog_deadline({}, 100001); }, "catalog allowance ignored the entry bound");
    auto malicious = observed; malicious.plots[0].relative_path = "../outside";
    rejects([&] { (void)farm_inventory_text(malicious); }, "parent path escaped inventory root");
    malicious = observed; malicious.plots[1].alias_of = 0;
    rejects([&] { (void)farm_inventory_text(malicious); }, "forged alias hid another native file");
    std::array<fs::path, 2> overlap{directory, directory / "nested"}; auto overlapped = scan_farm_inventory(overlap);
    require(std::count_if(overlapped.plots.begin(), overlapped.plots.end(), [](auto const& p) { return p.status == "protected" && p.reason == "duplicate native file identity"; }) == 2,
        "overlapping roots counted the same native file as reclaimable twice");
    (void)farm_inventory_text(overlapped);
    write(directory / "og.plot", {0}); rejects([&] { (void)recheck_inventory_source(observed.roots[0], og); }, "changed source accepted");
#ifndef _WIN32
    fs::create_directory_symlink(directory, directory / "loop"); fs::create_symlink(directory / "group.gplot", directory / "linked.plot");
    auto links = scan_farm_inventory(roots);
    require(links.plots.size() == 6 && std::count_if(links.plots.begin(), links.plots.end(), [](auto const& p) { return p.status == "protected"; }) >= 3,
        "links were followed or concealed");
    auto current = observe_inventory_file(observed.roots[0], "group.gplot"); InventorySourcePin pin(observed.roots[0], current);
    fs::create_directory(directory / ".claim"); fs::rename(directory / "group.gplot", directory / ".claim" / "one");
    require(pin.matches_claim(".claim/one"), "rename changed a pinned source's content identity");
    InventorySourcePin recovered(observed.roots[0], current, {}, ".claim/one");
    require(recovered.matches_claim(".claim/one"), "crash after claim could not reopen the exact source");
    write(directory / ".claim" / "one", {1}); require(!pin.matches_claim(".claim/one"), "claimed source mutation was ignored");
    fs::create_hard_link(directory / "nested" / "bb.plot", directory / "alias.plot");
    links = scan_farm_inventory(roots);
    require(std::count_if(links.plots.begin(), links.plots.end(), [](auto const& p) { return p.identity.links > 1 && p.status == "protected"; }) == 2,
        "hardlinks were independently reclaimable");
#endif
}
int fake_adapter(int argc, char** argv)
{
    std::string mode = argc > 2 ? fs::path(argv[2]).filename().string() : "";
    if (mode == "timeout") { std::this_thread::sleep_for(std::chrono::seconds(5)); return 0; }
    if (mode == "flood") { for (unsigned i = 0; i < 65536; ++i) std::cout << "private-memo-must-never-be-logged\n"; return 1; }
    if (mode == "failed") { std::cout << "private-memo-must-never-be-logged"; return 1; }
    if (argc == 4 && std::string_view(argv[3]) == "version") { std::cout << "fixture-native-1\n"; return 0; }
    std::string endpoint = argv[argc - 1];
    auto const root = fs::path(argv[2]); std::string const service = argv[4];
    auto flag = [&](char const* name) { return fs::exists(root / name); };
    if (endpoint == "get_version")
    {
        if (service == "farmer" && flag("farmer-unavailable")) return 1;
        std::string version = "fixture-native-1"; std::ifstream in(fs::path(argv[2]) / "version"); if (in) in >> version;
        if ((service == "farmer" && flag("farmer-version")) || (service == "solver" && flag("solver-version"))) version = "fixture-native-2";
        std::cout << "{\"success\":true,\"version\":\"" << version << "\"}";
    }
    else if (endpoint == "get_network_info") std::cout << "{\"success\":true,\"network_name\":\"simulator0\",\"genesis_challenge\":\"" << std::string(64, service == "solver" && flag("solver-network") ? 'b' : 'a') << "\"}";
    else if (endpoint == "get_connections")
    {
        struct Peer { unsigned type, local, server, port; char id; }; std::vector<Peer> peers;
        unsigned const harvester_port = flag("harvester-advertised") ? 8448u : 0u;
        if (service == "harvester" && !flag("farmer-disconnected")) peers = {{3, harvester_port, 8447, 8447, '1'}};
        if (service == "farmer")
        {
            if (!flag("farmer-disconnected")) peers.push_back({2, flag("farmer-zero-port") ? 0u : 8447u,
                flag("harvester-port-mismatch") ? 8449u : harvester_port, 61934, '3'});
            peers.push_back({1, 8447, flag("node-mismatch") ? 8445u : 8444u, 8444, '4'});
            if (!flag("solver-disconnected")) peers.push_back({8, 8447, 8666, 8666, '5'});
        }
        if (service == "full_node") peers = {{3, 8444, 8447, 61935, '2'}};
        if (service == "solver" && !flag("solver-disconnected")) peers = {{3, 8666, flag("solver-mismatch") ? 8449u : 8447u, 61936, '2'}};
        std::cout << "{\"success\":true,\"connections\":[";
        for (size_t i = 0; i < peers.size(); ++i)
        { auto const p = peers[i]; if (i) std::cout << ','; std::cout << "{\"type\":" << p.type << ",\"local_port\":" << p.local << ",\"peer_server_port\":" << p.server
            << ",\"peer_port\":" << p.port << ",\"peer_host\":\"" << (flag("remote-peers") ? (service != "farmer" ? "10.0.0.2" : p.type == 2 ? "10.0.0.1" : p.type == 1 ? "10.0.0.3" : "10.0.0.4") : "127.0.0.1")
            << "\",\"node_id\":\"" << std::string(64, p.id)
            << "\",\"creation_time\":" << (flag("connection-churn") ? "2.25" : "1.25") << ",\"last_message_time\":0}"; }
        std::cout << "]}";
    }
    else if (endpoint == "get_state") std::cout << "{\"success\":true,\"started\":" << (flag("solver-stopped") ? "false" : "true") << '}';
    else if (endpoint == "get_pool_state")
    {
        std::string const pool(64, flag("pool-mismatch") ? '8' : '4');
        std::cout << "{\"success\":true,\"pool_state\":[{\"p2_singleton_puzzle_hash\":\"" << pool << "\",\"pool_config\":{\"p2_singleton_puzzle_hash\":\"" << pool
            << "\",\"pool_url\":\"" << ((flag("pool-external") || flag("pool-uninitialized")) ? "https://pool.example" : "")
            << "\",\"launcher_id\":\"" << std::string(64, '1') << "\",\"owner_public_key\":\"" << std::string(96, '6')
            << "\",\"target_puzzle_hash\":\"" << std::string(64, '2') << "\",\"version\":" << (flag("pool-version-invalid") ? 3 : flag("pool-v2") ? 2 : 1)
            << "},\"current_difficulty\":" << (flag("pool-uninitialized") ? "null" : "17") << ",\"authentication_token_timeout\":5}]}";
    }
    else if (endpoint == "get_harvesters" || endpoint == "get_harvesters_summary")
    {
        std::ifstream in(root / (flag("farmer-override") ? "farmer-plots.json" : "plots.json"));
        std::string raw = in ? std::string(std::istreambuf_iterator<char>(in), {}) : "{\"plots\":[]}";
        auto parsed = profile_detail::JsonReader(raw, 16 * 1024 * 1024).read(); auto const count = inventory_detail::required_field(parsed, "plots").items.size();
        auto const start = raw.find('['), end = raw.rfind(']'); require(start != raw.npos && end != raw.npos, "bad fake farmer registry");
        std::cout << "{\"success\":true,\"harvesters\":[{\"connection\":{\"node_id\":\"" << std::string(64, flag("farmer-mismatch") ? '6' : '3')
            << "\",\"host\":\"" << (flag("remote-peers") ? "10.0.0.1" : "127.0.0.1") << "\",\"port\":61934},\"syncing\":" << (flag("farmer-syncing") ? "{}" : "null")
            << ",\"last_sync_time\":" << (flag("farmer-zero-sync") ? "0" : flag("farmer-future-sync") ? "99999999999" : flag("sync-churn") ? "2.25" : "1.25") << ",\"plots\":";
        if (endpoint == "get_harvesters_summary") std::cout << count; else std::cout << raw.substr(start, end - start + 1);
        std::cout << '}';
        if (flag("other-harvester"))
        {
            std::cout << ",{\"connection\":{\"node_id\":\"" << std::string(64, '9') << "\",\"host\":\"10.0.0.9\",\"port\":61939},\"syncing\":" << (flag("other-harvester-syncing") ? "{}" : "null") << ",\"last_sync_time\":"
                << (flag("other-harvester-growth") ? "2.25" : "1.25") << ",\"plots\":";
            if (endpoint == "get_harvesters_summary") std::cout << (flag("other-harvester-empty") ? 0 : flag("other-harvester-growth") ? 2 : 1);
            else std::cout << (flag("other-harvester-empty") ? "[]" : "[{\"size\":130,\"group_size\":4}]");
            std::cout << '}';
        }
        std::cout << "]}";
    }
    else if (endpoint == "get_constants") std::cout << "{\"success\":true,\"constants\":{\"GENESIS_CHALLENGE\":\"" << std::string(64, 'a') << "\",\"HARD_FORK2_HEIGHT\":100,\"MIN_PLOT_STRENGTH\":2,\"MAX_PLOT_STRENGTH\":17,\"PLOT_SIZE_V2\":28}}";
    else if (endpoint == "get_plots")
    {
        if (auto const* delay = std::getenv("XCHPLOT2_INVENTORY_TEST_RPC_DELAY_MS"))
            std::this_thread::sleep_for(std::chrono::milliseconds(std::stoul(delay)));
        auto counter = fs::path(argv[2]) / "registry-count"; unsigned value = 0; { std::ifstream in(counter); in >> value; }
        { std::ofstream out(counter); out << value + 1; }
        std::ifstream in(fs::path(argv[2]) / "plots.json");
        if (in) std::cout << in.rdbuf(); else std::cout << "{\"success\":true,\"plots\":[]}";
    }
    else if (endpoint == "get_blockchain_state")
    {
        bool const inactive = fs::exists(fs::path(argv[2]) / "inactive");
        std::cout << "{\"success\":true,\"blockchain_state\":{\"sync\":{\"synced\":" << (flag("unsynced") ? "false" : "true") << ",\"sync_mode\":false},\"peak\":{\"prev_transaction_block_height\":"
            << (inactive ? 99 : 101) << ",\"timestamp\":" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "}}}";
    }
    else return 2;
    return 0;
}
void target_prefix_snapshot(HarvesterAdapterOptions const& options, NetworkProjection const& projection, InventoryRoot const& root,
    ReplacementExpectation const& prototype, std::vector<uint8_t> const& bytes, std::string const& loaded_key)
{
    constexpr unsigned files = 12;
    TargetServiceExpectation service{prototype.farmer_public_key, prototype.pool_public_key, prototype.pool_contract_puzzle_hash};
    auto expectation = [&](unsigned index)
    {
        auto value = prototype; value.path = prototype.path.parent_path() / ("prefix-" + std::to_string(index) + ".plot2");
        value.identity = observe_inventory_file(root, file_io::path_text(value.path.filename()), options.limits).identity; return value;
    };
    for (unsigned i = 0; i < files; ++i) write(prototype.path.parent_path() / ("prefix-" + std::to_string(i) + ".plot2"), bytes);
    auto registry = [&](bool wrong = false, bool duplicate = false, bool wrong_meta = false)
    {
        std::ofstream out(options.root / "plots.json"); out << "{\"success\":true,\"plots\":[";
        for (unsigned i = 0; i < files + unsigned(duplicate); ++i)
        {
            auto expected = expectation(i % files); if (i) out << ','; out << "{\"filename\":"; profile_detail::quote(out, file_io::path_text(expected.path));
            out << ",\"size\":130,\"plot_id\":\"" << expected.group_id << "\",\"file_size\":" << expected.identity.stamp.bytes << ",\"time_modified\":" << expected.identity.stamp.mtime
                << ",\"pool_contract_puzzle_hash\":\"" << expected.pool_contract_puzzle_hash << "\",\"plot_public_key\":\"" << (wrong && i == files - 1 ? std::string(96, '7') : loaded_key)
                << "\",\"meta_group\":" << unsigned(expected.meta_group + uint8_t(wrong_meta && i == files - 1)) << ",\"group_size\":" << expected.members << '}';
        }
        out << "]}";
    };
    auto count = [&] { unsigned value = 0; std::ifstream in(options.root / "registry-count"); in >> value; return value; };
    auto publication = [&](ReplacementExpectation const& expected, unsigned index)
    { return ReplacementPublication{expected.path, expected.identity, std::string(64, 'b'), index}; };
    registry();
    FarmInventory complete_scope; complete_scope.roots.push_back(root);
    rejects([&] { TargetReadinessSnapshot outside(options, projection, prototype.k, prototype.strength, service, &complete_scope); },
        "target reader accepted loaded files absent from the complete catalog");
    auto published = [&](std::string_view path, std::string_view id)
    {
        if (id != prototype.group_id) return false;
        for (unsigned i = 0; i < files; ++i)
            if (path == file_io::path_text(expectation(i).path)) return true;
        return false;
    };
    auto future = [&](std::string_view path, std::string_view id)
    { return path != file_io::path_text(expectation(files - 1).path) && published(path, id); };
    rejects([&] { TargetReadinessSnapshot unpublished(options, projection, prototype.k, prototype.strength, service, &complete_scope, future); },
        "target registry accepted an unpublished future replacement name");
    auto before = count();
    {
        TargetReadinessSnapshot snapshot(options, projection, prototype.k, prototype.strength, service, &complete_scope, published);
        for (unsigned i = 0; i < files; ++i)
        { auto expected = expectation(i); require(snapshot.check(expected, publication(expected, i)).ready, "prefix snapshot rejected an exact sealed publication"); }
        snapshot.finish();
    }
    require(count() == before + 1, "prefix target readiness repeated get_plots per completed file");
    registry(true);
    {
        TargetReadinessSnapshot mismatch(options, projection, prototype.k, prototype.strength, service); auto expected = expectation(files - 1);
        require(!mismatch.check(expected, publication(expected, files - 1)).ready, "prefix snapshot accepted a mismatched loaded public key");
        rejects([&] { mismatch.finish(); }, "failed prefix check retained a healthy snapshot");
    }
    registry(false, false, true);
    {
        TargetReadinessSnapshot mismatch(options, projection, prototype.k, prototype.strength, service); auto expected = expectation(files - 1);
        require(!mismatch.check(expected, publication(expected, files - 1)).ready, "prefix snapshot ignored an explicitly reported wrong meta group");
    }
    registry(false, true);
    rejects([&] { TargetReadinessSnapshot duplicate(options, projection, prototype.k, prototype.strength, service); }, "duplicate target registry paths accepted");
    registry();
    auto bounded = options; bounded.limits.max_files = files - 1;
    rejects([&] { TargetReadinessSnapshot too_many(bounded, projection, prototype.k, prototype.strength, service); }, "target snapshot ignored its parsed index count bound");
    {
        TargetReadinessSnapshot changed(options, projection, prototype.k, prototype.strength, service); auto expected = expectation(0);
        require(changed.check(expected, publication(expected, 0)).ready, "prefix configuration fixture did not start ready");
#ifndef _WIN32
        { std::ofstream config(options.root / "config" / "config.yaml"); config << "self_hostname: localhost\nharvester:\n  num_threads: 3\n"; }
        rejects([&] { changed.finish(); }, "prefix audit ignored a configuration change after its RPC snapshot");
        { std::ofstream config(options.root / "config" / "config.yaml"); config << "self_hostname: localhost\nharvester:\n  num_threads: 2\n"; }
#else
        // The existing native Windows pin denies both writes and replacement.
        std::ofstream config(options.root / "config" / "config.yaml"); require(!config, "snapshot config pin allowed a concurrent write"); changed.finish();
#endif
    }
    {
        auto short_options = options; short_options.limits.timeout_ms = 3000;
        TargetReadinessSnapshot expired(short_options, projection, prototype.k, prototype.strength, service);
        // The constructor has completed all RPC work. Waiting its full grant
        // from here deterministically exceeds the original shared deadline.
        std::this_thread::sleep_for(std::chrono::milliseconds(short_options.limits.timeout_ms + 1));
        auto expected = expectation(0);
        require(!expired.check(expected, publication(expected, 0)).ready, "prefix audit silently reset its deadline for the next file");
        rejects([&] { expired.finish(); }, "expired target snapshot remained usable");
    }
    { std::ofstream absent(options.root / "plots.json"); absent << "{\"success\":true,\"plots\":[]}"; }
    TargetReadinessSnapshot next_boundary(options, projection, prototype.k, prototype.strength, service); auto expected = expectation(0);
    require(!next_boundary.check(expected, publication(expected, 0)).ready, "next retirement boundary reused a cached loaded registry");
}
#ifdef __linux__
void source_metadata_snapshot(fs::path const& directory, HarvesterAdapterOptions options, std::string const& loaded_key)
{
    auto root = directory / "metadata-native-files"; fs::create_directory(root);
    auto bytes = legacy(false); auto const id = adapter_detail::legacy_id_from_loaded_key(loaded_key, std::string(64, '4'));
    auto public_id = group_tools::hex(id); std::copy(public_id.begin(), public_id.end(), bytes.begin() + 19); bytes[54] = 'x';
    write(root / "first.plot", bytes); write(root / "second.plot", bytes);
    std::array<fs::path, 1> roots{root}; auto catalog = scan_farm_inventory(roots);
    auto registry = [&](bool missing = false)
    {
        std::ofstream out(options.root / "plots.json"); out << "{\"success\":true,\"plots\":[";
        for (size_t i = 0; i < catalog.plots.size() - size_t(missing); ++i)
        {
            auto const& plot = catalog.plots[i]; if (i) out << ','; out << "{\"filename\":";
            profile_detail::quote(out, file_io::path_text(root / plot.relative_path));
            out << ",\"size\":32,\"plot_id\":\"" << id << "\",\"file_size\":" << plot.identity.stamp.bytes << ",\"time_modified\":" << plot.identity.stamp.mtime
                << ",\"pool_contract_puzzle_hash\":\"" << std::string(64, '4') << "\",\"plot_public_key\":\"" << loaded_key << "\",\"compression_level\":33}";
        }
        out << "]}";
    };
    registry();
    observe_harvester_inventory(catalog, options);
    require(std::all_of(catalog.plots.begin(), catalog.plots.end(), [](auto const& p) { return p.native_identified && p.status == "native-observed"; }),
        "reader metadata retirement required an unavailable compressed-plot GPU checker");
    std::array<uint64_t, 2> indices{0, 1};
    auto count = [&] { unsigned value = 0; std::ifstream in(options.root / "registry-count"); in >> value; return value; };
    auto const before = count();
    {
        SourceReadinessSnapshot snapshot(options, catalog, indices);
        for (auto const& plot : catalog.plots) (void)snapshot.check(catalog.roots[plot.root], plot);
        snapshot.finish();
    }
    require(count() == before + 2, "source snapshot rescanned the complete registry per source instead of bounded capture/final fences");
    {
        SourceReadinessSnapshot snapshot(options, catalog, indices); registry(true);
        rejects([&] { snapshot.finish(); }, "source loaded-population loss survived an unchanged reader version"); registry();
    }
    {
        SourceReadinessSnapshot snapshot(options, catalog, indices);
        auto registry_file = options.root / "plots.json"; std::ifstream in(registry_file); std::string raw(std::istreambuf_iterator<char>(in), {});
        auto row = raw.substr(raw.find('{', raw.find('[')), raw.find('}', raw.find('[')) - raw.find('{', raw.find('[')) + 1);
        auto const path = file_io::path_text(root / "first.plot"); auto where = row.find(path);
        require(where != row.npos, "source arrival fixture cannot locate its original row"); row.replace(where, path.size(), file_io::path_text(root / "arrival.plot"));
        { std::ofstream changed(registry_file); changed << raw.substr(0, raw.rfind(']')) << ',' << row << "]}"; }
        rejects([&] { snapshot.finish(); }, "new source arrival survived the coherent route scope fence"); registry();
    }
    registry(true);
    rejects([&] { SourceReadinessSnapshot snapshot(options, catalog, indices); (void)snapshot.check(catalog.roots[0], catalog.plots[1]); },
        "missing source reader acknowledgement retained native identification");
    registry();
    {
        SourceReadinessSnapshot snapshot(options, catalog, indices);
        { std::ofstream changed(options.root / "version"); changed << "fixture-native-2"; }
        rejects([&] { snapshot.finish(); }, "source snapshot lost its version fence");
        fs::remove(options.root / "version");
    }
    auto drchia = options; drchia.kind = HarvesterAdapterKind::drchia;
    auto dr_catalog = scan_farm_inventory(roots); observe_harvester_inventory(dr_catalog, drchia);
    require(dr_catalog.plots[0].native_identified && dr_catalog.plots[0].reader == "drchia" && dr_catalog.plots[0].reader_version == "cli:fixture-native-1" &&
        dr_catalog.plots[0].container == PlotContainer::unknown, "compatible DrChia public metadata was either rejected or given a guessed grammar");
    { std::ofstream empty(options.root / "plots.json"); empty << "{\"success\":true,\"plots\":[]}"; }
    rejects([&] { SourceReadinessSnapshot initial(options, catalog, indices); (void)initial.check(catalog.roots[0], catalog.plots[0]); },
        "initial source phase accepted files never acknowledged by its reader");
    {
        SourceReadinessSnapshot resumed(options, catalog, indices, {}, true);
        for (auto const& plot : catalog.plots) (void)resumed.check(catalog.roots[plot.root], plot);
        resumed.finish();
    }
    auto unreviewed = catalog; unreviewed.plots[0].native_identified = false;
    rejects([&] { SourceReadinessSnapshot resumed(options, unreviewed, indices, {}, true); (void)resumed.check(catalog.roots[0], unreviewed.plots[0]); },
        "unloaded recovery invented previously unobserved reader metadata");
    write(root / "known.plot", legacy(false));
    auto known = observe_inventory_file(catalog.roots[0], "known.plot");
    FarmInventory known_catalog; known_catalog.roots = catalog.roots; known_catalog.plots = {known};
    std::array<uint64_t, 1> known_index{0};
    auto legacy_known_catalog = known_catalog;
    observe_harvester_inventory(known_catalog, options);
    require(!known_catalog.plots[0].native_identified && known_catalog.plots[0].reader == "gigahorse" &&
        !known_catalog.plots[0].reader_identity.empty() && known_catalog.plots[0].plot_public_key.empty(),
        "absent ordinary header lost route context or acquired invented loaded-file metadata");
    {
        SourceReadinessSnapshot snapshot(options, known_catalog, known_index);
        auto header_only = snapshot.check(known_catalog.roots[0], known);
        require(header_only.header_validated && !header_only.native_identified, "unloaded ordinary public-header identity required invented reader acknowledgement");
        snapshot.finish();
    }
    write(root / "known.plot", {1});
    rejects([&] { SourceReadinessSnapshot snapshot(options, known_catalog, known_index); (void)snapshot.check(known_catalog.roots[0], known); },
        "unloaded known-header source mutation survived the native object fence");
    fs::remove(root / "known.plot");
    require(observe_sources_retired(options, known_catalog, known_index).no_loaded_record &&
        observe_sources_retired(options, legacy_known_catalog, known_index).no_loaded_record,
        "retired unloaded ordinary header could not release old arrivals under a fresh complete route registry");
    {
        std::ofstream loaded(options.root / "plots.json"); loaded << "{\"success\":true,\"plots\":[{\"filename\":";
        profile_detail::quote(loaded, file_io::path_text(root / "known.plot")); loaded << ",\"plot_id\":\"" << known.plot_id << "\"}]}";
    }
    require(!observe_sources_retired(options, known_catalog, known_index).no_loaded_record,
        "still-registered ordinary source released old arrivals");
    { std::ofstream empty(options.root / "plots.json"); empty << "{\"success\":true,\"plots\":[]}"; }
    auto stale_opaque = catalog; observe_harvester_inventory(stale_opaque, options);
    require(std::none_of(stale_opaque.plots.begin(), stale_opaque.plots.end(), [](auto const& plot) { return plot.native_identified; }) &&
        std::all_of(stale_opaque.plots.begin(), stale_opaque.plots.end(), [](auto const& plot) { return plot.status == "protected"; }),
        "refresh retained stale opaque reader identification while its registry was empty");
    auto malformed = catalog; malformed.plots[0].plot_id = std::string(64, '9');
    rejects([&] { SourceReadinessSnapshot resumed(options, malformed, indices, {}, true); (void)resumed.check(catalog.roots[0], malformed.plots[0]); },
        "unloaded recovery accepted malformed public plot-key/route identity metadata");
    write(root / catalog.plots[0].relative_path, {1});
    rejects([&] { SourceReadinessSnapshot resumed(options, catalog, indices, {}, true); (void)resumed.check(catalog.roots[0], catalog.plots[0]); },
        "unloaded recovery ignored mutation of the exact reviewed native object");
}

#endif
void adapters(fs::path const& directory)
{
    auto self = calibration_process::executable(); auto root = directory / "native"; fs::create_directories(root / "config");
    write(directory / "native-source.plot", legacy(false));
    { std::ofstream config(root / "config" / "config.yaml"); config << "self_hostname: localhost\nharvester:\n  num_threads: 2\n"; }
    HarvesterAdapterOptions options; options.executable = self; options.root = root;
    auto projection = observe_network_projection(options);
    require(projection.available && projection.network == "simulator0" && projection.genesis_challenge == std::string(64, 'a') && projection.plot_size == 28,
        "actual typed RPC projection was not observed");
    require(observe_target_network(options, projection, 28, 2).ready, "active target-only network gate incorrectly required a replacement publication");
    ReplacementExpectation expected; expected.path = directory / "published.plot2"; expected.network = "simulator0"; expected.genesis_challenge = std::string(64, 'a');
    expected.reader_version = projection.reader_version; expected.reader_identity = projection.reader_identity;
    expected.network_version = projection.version;
    expected.farmer_version = projection.farmer_version; expected.solver_version = projection.solver_version;
    expected.k = 28; expected.strength = 2; expected.members = 4; expected.farmer_public_key = std::string(96, '6'); expected.pool_contract_puzzle_hash = std::string(64, '4');
    std::string const loaded_key(96, '5'); expected.group_id = adapter_detail::group_id_from_loaded_key(loaded_key, expected.pool_contract_puzzle_hash, expected.strength);
    PlotGroupFile::Header header{}; header.magic = PlotGroupFile::MAGIC; header.version = PlotGroupFile::FORMAT_VERSION; header.k = expected.k;
    header.strength = expected.strength; header.group_size = expected.members; header.group_id = PlotGroupId(expected.group_id); header.memo_length = 112; header.chunk_index_offset = 200;
    std::vector<uint8_t> bytes(256); std::memcpy(bytes.data(), &header, sizeof(header));
    std::fill(bytes.begin() + sizeof(header), bytes.begin() + sizeof(header) + 32, uint8_t(0x44));
    std::fill(bytes.begin() + sizeof(header) + 32, bytes.begin() + sizeof(header) + 80, uint8_t(0x66));
    std::fill(bytes.begin() + sizeof(header) + 80, bytes.begin() + sizeof(header) + 112, uint8_t(0x99)); write(expected.path, bytes);
    std::array<fs::path, 1> roots{directory}; auto observed = scan_farm_inventory(roots);
    auto found = std::find_if(observed.plots.begin(), observed.plots.end(), [](auto const& p) { return p.relative_path == "published.plot2"; });
    require(found != observed.plots.end(), "replacement fixture not inventoried"); expected.identity = found->identity;
    auto loaded = [&](std::string const& key)
    {
        std::ofstream out(root / "plots.json"); out << "{\"success\":true,\"plots\":[{\"filename\":"; profile_detail::quote(out, file_io::path_text(expected.path));
        out << ",\"size\":130,\"plot_id\":\"" << expected.group_id << "\",\"file_size\":" << expected.identity.stamp.bytes << ",\"time_modified\":" << expected.identity.stamp.mtime
            << ",\"pool_contract_puzzle_hash\":\"" << expected.pool_contract_puzzle_hash << "\",\"plot_public_key\":\"" << key << "\",\"group_size\":" << expected.members << "}]}";
    };
    TargetServiceExpectation service{expected.farmer_public_key, expected.pool_public_key, expected.pool_contract_puzzle_hash};
    require(observe_target_prerequisites(options, projection, 28, 2, service).ready,
        "empty target registry incorrectly required a fabricated new-plot/key acknowledgement");
    { std::ofstream flag(root / "harvester-advertised"); flag << '1'; }
    require(observe_target_prerequisites(options, projection, 28, 2, service).ready,
        "matching advertised harvester ports were refused");
    fs::remove(root / "harvester-advertised");
    for (char const* failure : {"farmer-unavailable", "farmer-disconnected", "farmer-mismatch", "node-mismatch", "farmer-syncing",
        "farmer-zero-sync", "farmer-future-sync", "solver-stopped", "solver-disconnected", "solver-mismatch", "solver-network",
        "pool-uninitialized", "pool-version-invalid", "pool-mismatch", "farmer-version", "solver-version", "farmer-zero-port", "harvester-port-mismatch"})
    {
        { std::ofstream flag(root / failure); flag << '1'; }
        require(!observe_target_prerequisites(options, projection, 28, 2, service).ready, "missing genuine target service prerequisite was accepted");
        fs::remove(root / failure);
    }
    auto og = service; og.pool_public_key = std::string(96, '4'); og.pool_contract_puzzle_hash.clear();
    require(observe_target_prerequisites(options, projection, 28, 2, og).ready, "OG public route was unnecessarily refused");
    { std::ofstream external(root / "pool-external"); external << '1'; }
    auto pooled = observe_target_prerequisites(options, projection, 28, 2, service);
    require(pooled.ready && pooled.pool.external && pooled.pool.difficulty == 17 && pooled.pool.protocol_version == 1,
        "ordinary initialized pool startup required a partial or invented authentication evidence");
    { std::ofstream modern(root / "pool-v2"); modern << '1'; }
    pooled = observe_target_prerequisites(options, projection, 28, 2, service);
    require(pooled.ready && pooled.pool.protocol_version == 2, "RewardPuzzle v2 route was refused");
    fs::remove(root / "pool-v2"); fs::remove(root / "pool-external");
    { std::ofstream other(root / "other-harvester"); other << '1'; }
    auto other_receivers = observe_target_prerequisites(options, projection, 28, 2, service);
    require(other_receivers.ready && other_receivers.other_harvesters.size() == 1 && other_receivers.other_harvesters[0].files == 1 && other_receivers.other_harvesters[0].members == 4,
        "other receiver population was hidden or conflated with the selected exact file scope");
    {
        TargetReadinessSnapshot snapshot(options, projection, expected.k, expected.strength, service);
        { std::ofstream growth(root / "other-harvester-growth"); growth << '1'; }
        rejects([&] { snapshot.finish(); }, "new other-harvester workload escaped the live phase lease");
        fs::remove(root / "other-harvester-growth");
    }
    { std::ofstream empty(root / "other-harvester-empty"); empty << '1'; }
    { std::ofstream syncing(root / "other-harvester-syncing"); syncing << '1'; }
    require(!observe_target_prerequisites(options, projection, 28, 2, service).ready,
        "empty partial sync was mistaken for a known empty other-harvester workload");
    fs::remove(root / "other-harvester-syncing");
    {
        TargetReadinessSnapshot snapshot(options, projection, expected.k, expected.strength, service);
        snapshot.finish();
        { std::ofstream syncing(root / "other-harvester-syncing"); syncing << '1'; }
        rejects([&] { snapshot.finish(); }, "empty other receiver began synchronization without invalidating readiness");
        fs::remove(root / "other-harvester-syncing");
    }
    fs::remove(root / "other-harvester-empty"); fs::remove(root / "other-harvester");
    auto remote = options; remote.host = "fixture-harvester";
    for (auto const& [role, host, node, farmer] : {std::tuple{"harvester", "10.0.0.1", '3', '1'},
        std::tuple{"farmer", "10.0.0.2", '1', '1'}, std::tuple{"full_node", "10.0.0.3", '4', '2'}, std::tuple{"solver", "10.0.0.4", '5', '2'}})
    {
        FarmServiceBinding binding; binding.root = directory / (std::string("remote-") + role); binding.rpc_host = host;
        binding.host = std::string("fixture-") + role; binding.node_id = std::string(64, node); binding.farmer_node_id = std::string(64, farmer);
        fs::create_directories(binding.root / "config");
        { std::ofstream config(binding.root / "config" / "config.yaml"); config << "self_hostname: " << host << "\nharvester:\n  num_threads: 2\n"; }
        { std::ofstream flags(binding.root / "remote-peers"); flags << '1'; }
        if (std::string_view(role) == "harvester") { remote.root = binding.root; remote.rpc_host = binding.rpc_host; }
        remote.services.emplace(role, std::move(binding));
    }
    auto remote_projection = observe_network_projection(remote);
    require(remote_projection.available && remote_projection.reader_identity != projection.reader_identity &&
        observe_target_prerequisites(remote, remote_projection, 28, 2, service).ready,
        "explicit authenticated remote contexts and reciprocal TLS node IDs were refused");
    auto misbound = remote; misbound.services.at("solver").node_id = std::string(64, '8');
    auto misbound_projection = observe_network_projection(misbound);
    require(!observe_target_prerequisites(misbound, misbound_projection, 28, 2, service).ready,
        "remote solver service was matched by host/port instead of its explicit TLS identity");
    require(target_service_hosts(remote).at("solver") == "fixture-solver", "remote service lost its physical resource host binding");
    auto unreviewed = projection; unreviewed.farmer_version.clear();
    require(!observe_target_prerequisites(options, unreviewed, 28, 2, service).ready, "unreviewed service version silently authorized execution");
    { std::ofstream flag(root / "farmer-unavailable"); flag << '1'; }
    auto forecast = observe_network_projection(options); fs::remove(root / "farmer-unavailable");
    require(forecast.available && forecast.farmer_version.empty() && !forecast.service_status.empty(), "missing service removed network-only forecast capability");
    loaded(loaded_key);
    observe_harvester_inventory(observed, options);
    auto enriched = std::find_if(observed.plots.begin(), observed.plots.end(), [](auto const& p) { return p.relative_path == "published.plot2"; });
    require(enriched != observed.plots.end() && enriched->reader == "chia" && enriched->reader_version == "fixture-native-1", "installed reader inventory provenance was lost");
    require(!observe_replacement_readiness(options, expected).ready, "public metadata substituted for private first-publication evidence");
    ReplacementPublication publication{expected.path, expected.identity, std::string(64, 'b'), 0};
    require(observe_replacement_readiness(options, expected, publication).ready, "actual native API fields could not attest a sealed first publication");
    for (char const* failure : {"connection-churn", "sync-churn", "farmer-syncing", "solver-stopped", "pool-external", "farmer-version", "solver-version", "inactive", "unsynced"})
    {
        TargetReadinessSnapshot snapshot(options, projection, expected.k, expected.strength, service);
        require(snapshot.check(expected, publication).ready, "service drift fixture did not begin with exact sync");
        { std::ofstream flag(root / failure); flag << '1'; }
        rejects([&] { snapshot.finish(); }, "service connection/sync/pool/version drift survived snapshot finish"); fs::remove(root / failure);
    }
    {
        TargetReadinessSnapshot snapshot(options, projection, expected.k, expected.strength, service);
        require(snapshot.check(expected, publication).ready, "replacement path fence fixture did not begin ready");
        snapshot.finish();
        auto hidden = expected.path; hidden += ".hidden"; fs::rename(expected.path, hidden);
        rejects([&] { snapshot.finish(); }, "local replacement deletion escaped unchanged cached RPC metadata");
        fs::rename(hidden, expected.path);
        expected.identity = observe_inventory_file(observed.roots[0], "published.plot2").identity; publication.identity = expected.identity; loaded(loaded_key);
    }
    {
        auto short_options = options; short_options.limits.timeout_ms = 1000;
        TargetReadinessSnapshot snapshot(short_options, projection, expected.k, expected.strength, service);
        require(snapshot.check(expected, publication).ready, "expiry fixture did not begin ready");
        snapshot.finish(); std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        rejects([&] { snapshot.finish(); }, "expired readiness phase lease authorized another retirement action");
    }
    auto farmer_override = [&](bool empty, bool group)
    {
        std::ifstream in(root / "plots.json"); std::string raw((std::istreambuf_iterator<char>(in)), {});
        if (empty) raw = "{\"plots\":[]}";
        else if (!group)
        { auto const field = ",\"group_size\":" + std::to_string(expected.members); auto at = raw.find(field); require(at != raw.npos, "missing group fixture field"); raw.erase(at, field.size()); }
        { std::ofstream out(root / "farmer-plots.json"); out << raw; }
        { std::ofstream flag(root / "farmer-override"); flag << '1'; }
    };
    farmer_override(true, true);
    require(!observe_replacement_readiness(options, expected, publication).ready, "local loaded plot substituted for missing farmer-side exact publication sync");
    farmer_override(false, false);
    require(!observe_replacement_readiness(options, expected, publication).ready, "farmer without multi-member metadata attested an entire group");
    fs::remove(root / "farmer-override");
    auto const multiple = expected; auto single_bytes = bytes; auto single_header = header; single_header.group_size = 1;
    std::memcpy(single_bytes.data(), &single_header, sizeof(single_header)); expected.path = directory / "single-member.plot2"; expected.members = 1;
    write(expected.path, single_bytes); expected.identity = observe_inventory_file(observed.roots[0], "single-member.plot2").identity;
    loaded(loaded_key); farmer_override(false, false);
    require(observe_replacement_readiness(options, expected, ReplacementPublication{expected.path, expected.identity, std::string(64, 'b'), 0}).ready,
        "single-member farmer compatibility incorrectly required new group metadata");
    fs::remove(root / "farmer-override"); expected = multiple; loaded(loaded_key);
    target_prefix_snapshot(options, projection, observed.roots[0], expected, bytes, loaded_key); loaded(loaded_key);
    loaded(std::string(96, '7')); require(!observe_replacement_readiness(options, expected, publication).ready, "wrong loaded public key route accepted"); loaded(loaded_key);
    { std::ofstream inactive(root / "inactive"); inactive << '1'; }
    require(!observe_replacement_readiness(options, expected, publication).ready, "preactivation last transaction height accepted"); fs::remove(root / "inactive");
    { std::ofstream inactive(root / "inactive"); inactive << '1'; }
    require(!observe_target_network(options, projection, 28, 2).ready, "target-only predelete gate accepted inactive consensus"); fs::remove(root / "inactive");
    require(!observe_source_retired(options, observed.roots[0], *enriched).no_loaded_record, "loaded source prematurely released arrivals");
    { std::ofstream absent(root / "plots.json"); absent << "{\"success\":true,\"plots\":[]}"; }
    require(observe_source_retired(options, observed.roots[0], *enriched).no_loaded_record, "fresh native source absence was not observed");
    unsigned count = 0, after = 0; { std::ifstream in(root / "registry-count"); in >> count; }
    std::vector<uint64_t> indices{uint64_t(enriched - observed.plots.begin())};
    require(observe_sources_retired(options, observed, indices).no_loaded_record, "batch source absence was unavailable");
    { std::ifstream in(root / "registry-count"); in >> after; }
    require(after == count + 1, "batch source absence repeated per-file registry queries");
    {
        std::ofstream outside(root / "plots.json"); outside << "{\"success\":true,\"plots\":[{\"filename\":";
        profile_detail::quote(outside, file_io::path_text(directory / "omitted-root" / "legacy.plot"));
        outside << ",\"plot_id\":\"" << std::string(64, '7') << "\"}]}";
    }
    rejects([&] { observe_harvester_inventory(observed, options); }, "inventory silently omitted a loaded outside-root plot");
    rejects([&] { observe_registry_scope(options, observed); }, "runtime scope silently omitted a loaded outside-root plot");
    require(!observe_sources_retired(options, observed, indices).no_loaded_record,
        "uncovered outside registry scope released selected cohort arrivals");
    loaded(loaded_key);
    auto nonplot_catalog = observed; auto& classified = nonplot_catalog.plots[size_t(enriched - observed.plots.begin())];
    classified.header_validated = false; classified.status = "non-plot"; classified.reason = "explicit reviewed classification";
    rejects([&] { observe_registry_scope(options, nonplot_catalog); }, "loaded path retained a zero-workload non-plot classification");
    { std::ofstream absent(root / "plots.json"); absent << "{\"success\":true,\"plots\":[]}"; }
#ifdef __linux__
    auto proprietary = std::find_if(observed.plots.begin(), observed.plots.end(), [](auto const& p) { return p.relative_path == "unknown.drplot"; });
    require(proprietary != observed.plots.end(), "missing native failure fixture");
    options.kind = HarvesterAdapterKind::gigahorse;
    auto proprietary_bytes = legacy(false); auto native_id = adapter_detail::legacy_id_from_loaded_key(loaded_key, std::string(64, '4'));
    auto native_id_bytes = group_tools::hex(native_id); std::copy(native_id_bytes.begin(), native_id_bytes.end(), proprietary_bytes.begin() + 19);
    proprietary_bytes[54] = 'x'; // A genuinely unknown header grammar to our parser.
    write(directory / "native-opaque.plot", proprietary_bytes);
    auto native_plot = observe_inventory_file(observed.roots[0], "native-opaque.plot");
    require(!native_plot.header_validated && native_plot.status == "protected", "unknown native container was guessed");
    auto native_registry = [&]
    {
        std::ofstream out(root / "plots.json"); out << "{\"success\":true,\"plots\":[{\"filename\":"; profile_detail::quote(out, file_io::path_text(directory / "native-opaque.plot"));
        out << ",\"size\":32,\"plot_id\":\"" << native_id << "\",\"file_size\":" << native_plot.identity.stamp.bytes << ",\"time_modified\":" << native_plot.identity.stamp.mtime
            << ",\"pool_contract_puzzle_hash\":\"" << std::string(64, '4') << "\",\"plot_public_key\":\"" << loaded_key << "\",\"compression_level\":7}]}";
    };
    native_registry(); FarmInventory native_catalog; native_catalog.roots = observed.roots; native_catalog.plots = {native_plot}; native_catalog.complete = true;
    observe_harvester_inventory(native_catalog, options); native_plot = native_catalog.plots[0];
    require(native_plot.native_identified && native_plot.members == 1 && native_plot.status == "native-observed" && native_plot.farmer_public_key.empty() && native_plot.container == PlotContainer::unknown,
        "native observation invented undisclosed proprietary metadata");
    std::array<uint64_t, 1> native_indices{0};
    {
        SourceReadinessSnapshot snapshot(options, native_catalog, native_indices);
        (void)snapshot.check(native_catalog.roots[0], native_plot); snapshot.finish();
    }
    auto stale_catalog = native_catalog; stale_catalog.plots[0].reader_version = "older-reader";
    rejects([&] { SourceReadinessSnapshot snapshot(options, stale_catalog, native_indices);
        (void)snapshot.check(stale_catalog.roots[0], stale_catalog.plots[0]); snapshot.finish(); },
        "changed reader version silently refreshed reviewed costs");
    stale_catalog = native_catalog; stale_catalog.plots[0].checker_identity = std::string(64, '0');
    {
        SourceReadinessSnapshot snapshot(options, stale_catalog, native_indices);
        (void)snapshot.check(stale_catalog.roots[0], stale_catalog.plots[0]); snapshot.finish();
    } // Historical optional diagnostics are not retirement authority.
    { std::ofstream config(root / "config" / "config.yaml"); config << "self_hostname: localhost\nharvester:\n  num_threads: 3\n"; }
    rejects([&] { SourceReadinessSnapshot snapshot(options, native_catalog, native_indices);
        (void)snapshot.check(native_catalog.roots[0], native_plot); snapshot.finish(); },
        "changed reader configuration silently refreshed reviewed costs");
    { std::ofstream config(root / "config" / "config.yaml"); config << "self_hostname: localhost\nharvester:\n  num_threads: 2\n"; }
    InventorySourcePin pin(native_catalog.roots[0], native_plot, options.limits);
    fs::rename(directory / "native-opaque.plot", directory / ".claim" / "native");
    { std::ofstream absent(root / "plots.json"); absent << "{\"success\":true,\"plots\":[]}"; }
    require(pin.matches_claim(".claim/native", options.limits), "native claim rename lost the exact held object");
    {
        InventorySourcePin recovered(native_catalog.roots[0], native_plot, options.limits, ".claim/native");
        require(recovered.matches_claim(".claim/native", options.limits),
            "claimed recovery incorrectly required original-path loaded registry state");
    }
    fs::rename(directory / ".claim" / "native", directory / ".claim" / "native-original");
    write(directory / ".claim" / "native", proprietary_bytes);
    require(!pin.matches_claim(".claim/native", options.limits), "replacement claim object matched the held native identity");
    rejects([&] { InventorySourcePin recovered(native_catalog.roots[0], native_plot, options.limits, ".claim/native"); },
        "substituted claim object accepted for recovered native identity");
    fs::remove(directory / ".claim" / "native");
    fs::rename(directory / ".claim" / "native-original", directory / ".claim" / "native");
    write(directory / ".claim" / "native", {1});
    require(!pin.matches_claim(".claim/native", options.limits), "claimed native object mutation was ignored");
    rejects([&] { InventorySourcePin recovered(native_catalog.roots[0], native_plot, options.limits, ".claim/native"); },
        "changed native object accepted for claimed recovery");
    source_metadata_snapshot(directory, options, loaded_key);
#endif
    AdapterCommand command; command.executable = self; command.root = directory / "flood";
    calibration_process::Limits limits{64 * 1024 * 1024, 2000, 4096, {}};
    rejects([&] { (void)run_farm_adapter(command, limits); }, "native output bound ignored");
    command.root = directory / "timeout"; limits.timeout_ms = 50;
    rejects([&] { (void)run_farm_adapter(command, limits); }, "native adapter deadline ignored");
    command.root = root; limits.timeout_ms = 2000; limits.cancelled = [] { return true; };
    rejects([&] { (void)run_farm_adapter(command, limits); }, "native adapter cancellation ignored");
    { std::ofstream config(root / "config" / "config.yaml"); config << "self_hostname: farm.example.com\nharvester:\n  num_threads: 2\n"; }
    require(!observe_network_projection(options).available, "nonloopback native adapter accepted");
}
void routes(fs::path const& directory)
{
    auto self = calibration_process::executable();
    std::ostringstream out;
    auto quote = [&](fs::path const& path) { profile_detail::quote(out, file_io::path_text(path)); };
    out << "{\"farm_adapters\":{\"schema\":1,\"target_route\":\"new-chia\",\"routes\":[{\"id\":\"old-gh\",\"kind\":\"gigahorse\",\"root\":";
    quote(directory / "gh-config"); out << ",\"executable\":"; quote(self); out << ",\"source_roots\":["; quote(directory);
    out << "]},{\"id\":\"new-chia\",\"kind\":\"chia\",\"root\":"; quote(directory / "chia-config"); out << ",\"executable\":"; quote(self); out << ",\"source_roots\":[]}]}}";
    std::istringstream input(out.str()); auto table = read_farm_adapters(input);
    InventoryRoot root; root.path = file_io::path_text(directory); InventoryPlot plot; plot.harvester = "old-gh";
    require(resolve_source(table, root, plot).kind == HarvesterAdapterKind::gigahorse && resolve_target(table, "new-chia").kind == HarvesterAdapterKind::chia,
        "mixed old/new native readers were conflated");
    auto swapped = table; swapped.target_route = "old-gh";
    rejects([&] { (void)resolve_target(swapped, "new-chia"); }, "swapped target route escaped review binding");
    swapped = table; swapped.routes[0].source_roots.clear(); swapped.routes[1].source_roots.push_back(root.path);
    rejects([&] { (void)resolve_source(swapped, root, plot); }, "swapped source route escaped review binding");
    swapped.routes[0].source_roots.push_back(root.path);
    rejects([&] { (void)resolve_source(swapped, root, plot); }, "ambiguous source route accepted");
    root.path += "/missing"; rejects([&] { (void)resolve_source(table, root, plot); }, "missing source route inferred from a parent prefix");
}
void large_catalog(fs::path const& directory)
{
    std::array<fs::path, 1> roots{directory}; auto catalog = scan_farm_inventory(roots);
    auto prototype = catalog.plots.front(); prototype.alias_of.reset(); prototype.native_identified = false; prototype.identity.links = 1;
    catalog.plots.clear(); catalog.resource_template.reset(); catalog.plots.reserve(7000);
    for (uint64_t i = 0; i < 7000; ++i)
    {
        auto plot = prototype; plot.relative_path = "large-" + std::to_string(i) + ".plot"; plot.identity.stamp.inode = i + 1;
        catalog.plots.push_back(std::move(plot));
    }
    InventoryLimits limits; limits.memory_bytes = 1024 * 1024 * 1024; limits.max_output_bytes = 64 * 1024 * 1024;
    auto text = farm_inventory_text(catalog, limits);
    bool structure = false;
    try { (void)profile_detail::JsonReader(text, limits.memory_bytes / 2).read(); }
    catch (std::runtime_error const& error) { structure = std::string_view(error.what()).find("structure limit") != std::string_view::npos; }
    require(structure, "large-catalog fixture did not exceed the unchanged default node bound");
    require(parse_farm_inventory(text, limits).plots.size() == catalog.plots.size(), "valid large catalog failed its explicit bounded roundtrip");
    auto tight = limits; tight.memory_bytes = 128 * 1024 * 1024; tight.max_output_bytes = 16 * 1024 * 1024;
    rejects([&] { (void)parse_farm_inventory(text, tight); }, "large catalog escaped a tighter parser memory budget");
    tight = limits; tight.max_output_bytes = text.size() - 1;
    rejects([&] { (void)parse_farm_inventory(text, tight); }, "large catalog escaped its byte bound");
}
int main(int argc, char** argv)
{
    if (argc > 1 && std::string_view(argv[1]) == "--root-path") return fake_adapter(argc, argv);
    auto root = fs::temp_directory_path() / ("xchplot2-inventory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try { fs::create_directory(root); inventory(root); large_catalog(root); adapters(root); routes(root); fs::remove_all(root); std::cout << "farm inventory tests passed\n"; return 0; }
    catch (std::exception const& error) { fs::remove_all(root); std::cerr << error.what() << '\n'; return 1; }
}
