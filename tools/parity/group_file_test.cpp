#include "host/GroupFile.hpp"
#include <array>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>

namespace io = group_grow::file_io;
namespace fs = std::filesystem;
void require(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
struct File
{
    int fd;
    explicit File(int value) : fd(value) { require(fd >= 0, "cannot open group file fixture"); }
    ~File() { if (fd >= 0) io::close_file(fd); }
    void close() { int const opened = fd; fd = -1; require(!io::close_file(opened), "cannot close group file fixture"); }
};
void same_metadata(io::Stat const& before, io::Stat const& after)
{
    bool same = true;
    auto compare = [&](char const* name, auto a, auto b)
    {
        if (a == b) return;
        std::cerr << name << " changed across close: " << a << " -> " << b << '\n'; same = false;
    };
    compare("device", before.st_dev, after.st_dev); compare("inode", before.st_ino, after.st_ino);
    compare("links", before.st_nlink, after.st_nlink); compare("mode", before.st_mode, after.st_mode);
    compare("size", before.st_size, after.st_size); compare("blocks", before.st_blocks, after.st_blocks);
    compare("mtime seconds", before.st_mtim.tv_sec, after.st_mtim.tv_sec);
    compare("mtime nanoseconds", before.st_mtim.tv_nsec, after.st_mtim.tv_nsec);
    compare("ctime seconds", before.st_ctim.tv_sec, after.st_ctim.tv_sec);
    compare("ctime nanoseconds", before.st_ctim.tv_nsec, after.st_ctim.tv_nsec);
    require(same, "synced construction metadata changed when its writer closed");
}
struct Directory
{
    fs::path path;
    Directory()
    {
        std::random_device random;
        for (unsigned attempt = 0; attempt < 32; ++attempt)
        {
            path = fs::temp_directory_path() / ("group-file-test-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (!io::make_directory(path)) return;
            require(errno == EEXIST, "cannot create group file test directory");
        }
        throw std::runtime_error("cannot create unique group file test directory");
    }
    ~Directory() { std::error_code error; fs::remove_all(path, error); }
};
int main() try
{
    Directory directory; auto const target = directory.path / fs::path(u8"positioned-\u00e9.dat");
    auto const flags = io::read_write | io::create | io::exclusive | io::no_follow | io::close_on_exec;
    File file(io::open_file(target, flags));
    io::Stat initial{}; require(!io::inspect_fd(file.fd, &initial) && io::is_regular(initial.st_mode) && initial.st_nlink == 1,
        "native file identity/type is unavailable");
    require(!io::lock_file(file.fd, io::lock_exclusive | io::lock_nonblocking), "cannot lock group file");
    {
        File competitor(io::open_file(target, io::read_write | io::no_follow));
        require(io::lock_file(competitor.fd, io::lock_exclusive | io::lock_nonblocking) != 0,
            "independent construction handles obtained the same file lock");
    }
    constexpr uint64_t reserved = uint64_t(1) << 20;
    require(!io::allocate_file(file.fd, io::keep_size, 0, reserved), "native preallocation failed");
    io::Stat allocated{}; require(!io::inspect_fd(file.fd, &allocated) && !allocated.st_size && uint64_t(allocated.st_blocks) * 512 >= reserved,
        "physical preallocation changed EOF or failed to reserve allocation");
    require(!io::allocate_file(file.fd, io::keep_size, 0, reserved / 2), "smaller physical reservation failed");
    io::Stat retained{}; require(!io::inspect_fd(file.fd, &retained) && !retained.st_size && retained.st_blocks >= allocated.st_blocks,
        "smaller reservation released previously admitted physical storage");
    require(!io::resize_file(file.fd, 4096), "native file extension failed");
    io::Stat extended{}; require(!io::inspect_fd(file.fd, &extended) && extended.st_size == 4096 && extended.st_blocks >= allocated.st_blocks,
        "extending EOF released preflight allocation");
    std::array<uint8_t, 4096> first{}, second{}; first.fill(0x51); second.fill(0xb2);
    auto a = std::async(std::launch::async, [&] { require(io::write_at(file.fd, first.data(), first.size(), 0) == first.size(), "first positioned write failed"); });
    auto b = std::async(std::launch::async, [&] { require(io::write_at(file.fd, second.data(), second.size(), first.size()) == second.size(), "second positioned write failed"); });
    a.get(); b.get(); require(!io::flush(file.fd), "native file barrier failed");
    {
        File reader(io::open_file(target, io::read_only | io::no_follow));
        std::array<uint8_t, 4096> value{};
        require(io::read_at(reader.fd, value.data(), value.size(), 0) == value.size() && value == first,
            "construction lock prevented ordinary reader access or changed first region");
        require(io::read_at(reader.fd, value.data(), value.size(), first.size()) == value.size() && value == second,
            "concurrent positioned writes shared a file offset");
    }
    io::Stat written{}; require(!io::inspect_link(target, &written) && written.st_dev == initial.st_dev && written.st_ino == initial.st_ino &&
        written.st_size == first.size() + second.size(), "file identity/size changed across native I/O");
    std::array<io::IoVec, 2> vectors{{{first.data(), first.size()}, {second.data(), second.size()}}};
    require(io::writev_at(file.fd, vectors.data(), int(vectors.size()), 0) == first.size() + second.size(), "native vector write changed content length");
    require(!io::resize_file(file.fd, first.size()) && !io::flush(file.fd), "native file resize/barrier failed");
    io::Stat before_close{}, after_close{};
    require(!io::inspect_link(target, &before_close), "cannot inspect synced construction metadata");
    require(before_close.st_size == first.size() && before_close.st_blocks < allocated.st_blocks,
        "truncation retained excess preflight allocation or changed EOF");
    file.close();
    require(!io::inspect_link(target, &after_close), "cannot inspect closed construction metadata");
    same_metadata(before_close, after_close);
    {
        File reopened(io::open_file(target, io::read_write | io::no_follow));
        require(!io::allocate_file(reopened.fd, io::keep_size, 0, reserved) &&
            !io::resize_file(reopened.fd, first.size()) && !io::flush(reopened.fd),
            "cannot trim a reservation without changing EOF");
        std::array<uint8_t, 4096> value{};
        require(io::read_at(reopened.fd, value.data(), value.size(), 0) == value.size() && value == first,
            "releasing excess allocation changed retained file contents");
        require(!io::inspect_link(target, &before_close) && before_close.st_blocks < allocated.st_blocks,
            "same-size truncation retained excess allocation");
        reopened.close();
        require(!io::inspect_link(target, &after_close), "cannot inspect reopened construction metadata");
        same_metadata(before_close, after_close);
    }
    File publication(io::open_file(target, io::read_write | io::no_follow));
    require(!io::lock_file(publication.fd, io::lock_exclusive | io::lock_nonblocking), "cannot relock group file for publication");
    auto const occupied = directory.path / "occupied";
    { File other(io::open_file(occupied, flags)); }
    require(io::rename_file(target, occupied, false) != 0 && !io::inspect_link(target, &written) && written.st_ino == initial.st_ino,
        "no-replace publication overwrote an existing destination");
    auto const published = directory.path / "published";
    require(!io::rename_file(target, published, false) && !io::inspect_link(published, &written) && written.st_ino == initial.st_ino,
        "write-through publication changed file identity");
    {
        File parent(io::open_file(directory.path, io::read_only | io::directory | io::no_follow));
        require(!io::lock_file(parent.fd, io::lock_exclusive | io::lock_nonblocking), "cannot lock group directory");
        File competitor(io::open_file(directory.path, io::read_only | io::directory | io::no_follow));
        require(io::lock_file(competitor.fd, io::lock_exclusive | io::lock_nonblocking) != 0,
            "directory aliases obtained independent construction locks");
        require(!io::flush(parent.fd), "native namespace publication barrier failed");
    }
    auto temporary = (directory.path / "private-XXXXXX").u8string();
    std::string pattern(reinterpret_cast<char const*>(temporary.data()), temporary.size());
    { File temp(io::temp_file(pattern.data())); require(pattern.size() == temporary.size(), "temporary creation changed its bounded template length");
      require(io::write_file(temp.fd, first.data(), first.size()) == first.size() && !io::flush(temp.fd), "private temporary file I/O failed"); }
    require(!io::unlink_file(pattern) && !io::unlink_file(occupied) && !io::unlink_file(published), "native namespace removal failed");
    require(io::inspect_link(published, &written) != 0 && errno == ENOENT, "deleted file remained in native namespace");
    std::cout << "group native identity, allocation, positioned I/O, locks and publication passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
