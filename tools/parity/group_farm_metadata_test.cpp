#include "host/GroupFarmMetadata.hpp"
#include <fstream>
#include <iostream>
#include <random>

using namespace group_grow;
namespace fs = std::filesystem;
void check(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function action, char const* message)
{
    bool rejected = false;
    try { action(); } catch (std::runtime_error const&) { rejected = true; }
    check(rejected, message);
}
struct Directory
{
    fs::path path;
    Directory()
    {
        std::random_device random;
        for (unsigned attempt = 0; attempt < 32; ++attempt)
        {
            path = fs::temp_directory_path() / ("group-farm-metadata-test-" + std::to_string(random()) + '-' + std::to_string(random()));
            if (!file_io::make_directory(path)) return;
            check(errno == EEXIST, "cannot create metadata test directory");
        }
        throw std::runtime_error("cannot create unique metadata test directory");
    }
    ~Directory() { std::error_code error; fs::remove_all(path, error); }
};
void record(fs::path const& path)
{ std::ofstream file(path); file << "unchanged legacy bytes\n"; check(bool(file), "cannot write metadata test record"); }
int main() try
{
    Directory fixture;
    auto const fresh = fixture.path / "fresh", legacy = fixture.path / "legacy", missing = fixture.path / "missing";
    fs::create_directory(fresh); fs::create_directory(legacy); fs::create_directory(missing);
    std::string const job(64, 'a');
    check(farming_metadata_namespace(fresh / ".xchplot2") &&
        farming_metadata_namespace(fresh / ".xchplot2" / "replot-absent" / "nested") &&
        !farming_metadata_namespace(fresh / ".xchplot2-other"), "reserved destination namespace classification differs");
    check(!fs::exists(fresh / ".xchplot2"), "namespace classification created an absent directory");
#ifdef _WIN32
    check(farming_metadata_namespace(fresh / ".XCHPLOT2" / "nested"), "reserved namespace ignored Windows case equivalence");
#endif
    {
        FarmingMetadataDirectory metadata(fresh);
        check(!metadata.legacy && !metadata.recorded && metadata.path == fresh / ".xchplot2",
            "fresh metadata did not select the private namespace");
        metadata.check();
        FarmingReplotDirectory migration(fresh, job, false);
        check(!migration.pin && !fs::exists(fresh / ".xchplot2"), "read-only metadata discovery created a directory");
    }
    {
        FarmingMetadataDirectory metadata(fresh); metadata.create();
        check(fs::is_directory(metadata.path) && !metadata.recorded, "private metadata directory was not created");
        FarmingReplotDirectory migration(fresh, job, true);
        check(migration.path == fresh / ".xchplot2" / ("replot-" + job) && migration.pin,
            "migration bookkeeping escaped its private destination namespace");
        (void)metadata.allocation(); (void)migration.allocation();
        metadata.check(); migration.check();
        rejects([&] { require_farming_review_path(fresh, metadata.path / "review.plan"); },
            "portable review claimed the private recovery namespace");
        rejects([&] { require_farming_review_path(fresh, migration.path / "successor.plan"); },
            "portable successor claimed the private job namespace");
        require_farming_review_path(fresh, fixture.path / "review.plan");
        check(!fs::exists(metadata.path / "review.plan") && !fs::exists(migration.path / "successor.plan"),
            "reserved review path validation wrote a file");
    }
    auto const orphan = fixture.path / "orphan"; fs::create_directory(orphan);
    {
        FarmingMetadataDirectory metadata(orphan); metadata.create();
        record(metadata.path / "unrelated");
        rejects([&] { metadata.require_fresh(); }, "fresh job adopted populated bookkeeping without its saved record");
        check(fs::exists(metadata.path / "unrelated") && !fs::exists(metadata.path / "grow-farm.plan"),
            "fresh namespace rejection changed orphan bookkeeping");
    }
    record(legacy / "grow-farm.plan.next");
    {
        // An unrelated private directory cannot relocate an established job.
        FarmingMetadataDirectory empty(missing); empty.create();
        record(missing / ".xchplot2" / "unrelated");
        std::vector<fs::path> roots{missing, legacy};
        check(farming_metadata_legacy(roots), "pending legacy record did not bind missing replicas");
        FarmingMetadataDirectory copy(missing, farming_metadata_legacy(roots));
        check(copy.legacy && copy.path == missing, "missing legacy replica moved to a different metadata layout");
        copy.create(); copy.check();
        check(!fs::exists(missing / ".xchplot2" / "grow-farm.plan"), "legacy recovery created a replacement private record");
    }
    {
        FarmingMetadataDirectory saved(legacy);
        check(saved.legacy && saved.recorded && saved.path == legacy && !saved.allocation(), "legacy record changed its path or allocation");
        saved.create(); saved.check();
        check(!fs::exists(legacy / ".xchplot2"), "legacy metadata discovery or execution created a new directory");
        FarmingReplotDirectory migration(legacy, job, true);
        check(migration.path == legacy / (".xchplot2-replot-" + job), "legacy migration bookkeeping changed location");
    }
    record(fresh / ".xchplot2" / "grow-replot.plan");
    rejects([&] { std::vector<fs::path> roots{legacy, fresh}; (void)farming_metadata_legacy(roots); },
        "conflicting replica layouts were accepted");
    record(fresh / "grow-farm.plan");
    rejects([&] { FarmingMetadataDirectory metadata(fresh); }, "simultaneous legacy/private fixed records were accepted");
    fs::remove(fresh / "grow-farm.plan");
    auto const unavailable = fixture.path / "offline";
    {
        FarmingMetadataDirectory metadata(unavailable, true);
        FarmingReplotDirectory migration(unavailable, job, false, true);
        check(!metadata.root_pin && !migration.pin && !fs::exists(unavailable), "read-only missing destination was recreated");
    }
    auto const invalid = fixture.path / "invalid"; fs::create_directory(invalid);
    record(invalid / ".xchplot2");
    rejects([&] { FarmingMetadataDirectory metadata(invalid); }, "regular file was accepted as a metadata directory");
    fs::remove(invalid / ".xchplot2");
#ifndef _WIN32
    fs::create_directory_symlink(fresh / ".xchplot2", fixture.path / "metadata-alias");
    check(farming_metadata_namespace(fixture.path / "metadata-alias" / "nested"), "canonical alias escaped reserved namespace");
    fs::remove(fixture.path / "metadata-alias");
    fs::create_directory_symlink(fresh / ".xchplot2", invalid / ".xchplot2");
    rejects([&] { FarmingMetadataDirectory metadata(invalid); }, "metadata directory followed a symbolic link");
    fs::remove(invalid / ".xchplot2");
    fs::create_directory(invalid / ".xchplot2");
    fs::permissions(invalid / ".xchplot2", fs::perms::owner_all | fs::perms::group_read);
    rejects([&] { FarmingMetadataDirectory metadata(invalid); }, "non-private metadata directory permissions were accepted");
    fs::permissions(invalid / ".xchplot2", fs::perms::owner_all);
    {
        FarmingMetadataDirectory metadata(invalid);
        fs::rename(metadata.path, invalid / "retained"); fs::create_directory(metadata.path);
        fs::permissions(metadata.path, fs::perms::owner_all);
        rejects([&] { metadata.check(); }, "metadata directory native substitution bypassed its pin");
    }
    fs::remove(invalid / ".xchplot2");
    fs::rename(invalid / "retained", invalid / ".xchplot2");
    {
        FarmingReplotDirectory metadata(invalid, job, true);
        fs::rename(metadata.path, invalid / ".xchplot2" / "retained"); fs::create_directory(metadata.path);
        fs::permissions(metadata.path, fs::perms::owner_all);
        rejects([&] { metadata.check(); }, "migration directory native substitution bypassed its pin");
    }
#endif
    auto long_path = fixture.path;
    for (unsigned depth = 0; depth < 8; ++depth) long_path /= std::string(80, 'p');
    fs::create_directories(long_path);
    {
        FarmingReplotDirectory metadata(long_path, job, false);
        check(farming_metadata_memory_bound(long_path, job) > farming_metadata_memory_bound(fixture.path, job) &&
            !fs::exists(long_path / ".xchplot2"), "long-path metadata memory or read-only behavior changed");
    }
    std::cout << "private farm metadata, legacy layouts and native directory pins passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
