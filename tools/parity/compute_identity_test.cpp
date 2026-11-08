#include "host/ComputeIdentity.hpp"
#include "compute_pool_fixture.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <sys/stat.h>
#endif

using namespace group_grow::pool;
using pool_test::require;
template <typename F> void rejects(F&& operation)
{
    try { operation(); } catch (std::exception const&) { return; }
    throw std::runtime_error("invalid pool setup was accepted");
}
int main()
{
    try
    {
        pool_test::Directory directory;
        auto root = directory.path / std::filesystem::path(u8"enrollment-\u03bb-\u5b58"); std::filesystem::create_directory(root);
        require(!load_pool(root / "missing"), "read-only pool lookup created a profile");
        auto authority = create_pool(root / "authority", "storage machine");
        require(authority.can_invite && !authority.node_id.empty() && !authority.authority_expired && !authority.certificate_expired &&
            !authority.certificate_expires.empty() && !authority.authority_expires.empty(), "pool initialization omitted authority, identity or validity");
        rejects([&] { (void)create_pool(root / "authority", "replacement"); });
        auto invitation = root / "worker.invite";
        invite_pool(authority, invitation, "shared worker");
        rejects([&] { invite_pool(authority, invitation, "replacement"); });
        auto worker = join_pool(invitation, root / "worker");
        require(load_worker_intent(worker) == WorkerIntent::Active && !std::filesystem::exists(worker.directory / "worker-control"),
            "default worker intent was not read-only or accepting");
        save_worker_intent(worker, WorkerIntent::Paused);
        require(load_worker_intent(*load_pool(worker.directory)) == WorkerIntent::Paused,
            "private worker control did not survive enrollment reload");
        rejects([&] { save_worker_intent(worker, WorkerIntent(0)); });
        require(load_worker_intent(worker) == WorkerIntent::Paused, "invalid control changed saved intent");
        std::filesystem::copy_file(worker.directory / "worker-control", authority.directory / "worker-control");
        rejects([&] { (void)load_worker_intent(authority); });
        std::filesystem::remove(authority.directory / "worker-control");
        require(worker.pool_id == authority.pool_id && worker.server_name == authority.server_name &&
            worker.node_id != authority.node_id && worker.name == "shared worker" && !worker.can_invite &&
            certificate_identity(worker.tls.cert_file) != certificate_identity(authority.tls.cert_file),
            "invitation did not create an independent member of the same pool");
        require(!std::filesystem::exists(root / "worker/authority.key"), "invitation copied the signing authority");
        rejects([&] { invite_pool(worker, root / "unauthorized.invite", "another worker"); });
        rejects([&] { (void)join_pool(invitation, root / "worker"); });
        auto defaults = worker.defaults; defaults.broker = "127.0.0.1:12345";
        defaults.cpus = 2; defaults.host_bytes = 1ULL << 30; defaults.interfaces = {"eth0", "wlan0"}; defaults.max_brokers = 256;
        configure_pool(worker, defaults); auto reloaded = *load_pool(worker.directory);
        require(reloaded.defaults.broker == defaults.broker && reloaded.defaults.cpus == 2 &&
            reloaded.defaults.host_bytes == defaults.host_bytes && reloaded.defaults.interfaces == defaults.interfaces &&
            reloaded.defaults.max_brokers == defaults.max_brokers && reloaded.node_id == worker.node_id,
            "saved defaults changed membership or did not persist");
        defaults.backend = "invalid"; rejects([&] { configure_pool(worker, defaults); });
        require(load_pool(worker.directory)->defaults.backend == "auto", "rejected defaults changed the profile");
        defaults.backend = "auto"; defaults.max_brokers = 0; rejects([&] { configure_pool(worker, defaults); });
        defaults.max_brokers = 4097; rejects([&] { configure_pool(worker, defaults); });
        std::ostringstream original;
        original << "xchplot2-compute-1\n" << std::quoted(worker.name) << ' ' << worker.node_id << '\n'
            << std::quoted(defaults.broker) << ' ' << defaults.cpus << ' ' << defaults.host_bytes << ' '
            << defaults.gpu_bytes << ' ' << defaults.backend << ' ' << defaults.discovery << '\n';
        for (auto const& interface : defaults.interfaces) original << std::quoted(interface) << '\n';
        { std::ofstream output(worker.directory / "profile"); output << original.str(); }
        require(load_pool(worker.directory)->defaults.max_brokers == 128, "original profile did not receive the default broker limit");
#ifdef _WIN32
        auto key_path = std::filesystem::path(reinterpret_cast<char8_t const*>(worker.tls.key_file.c_str()));
        auto key = pos2gpu::create_private_file_handle(key_path, GENERIC_READ, OPEN_EXISTING); ::CloseHandle(key);
        PSECURITY_DESCRIPTOR public_access = nullptr;
        require(::ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;FA;;;WD)", SDDL_REVISION_1, &public_access, nullptr) &&
            ::SetFileSecurityW(invitation.c_str(), DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, public_access),
            "cannot make public invitation ACL fixture");
        ::LocalFree(public_access);
#else
        struct stat status{};
        require(!::stat(worker.tls.key_file.c_str(), &status) && (status.st_mode & 0777) == 0600 &&
            !::stat(worker.directory.c_str(), &status) && (status.st_mode & 0777) == 0700,
            "pool credentials were published with public permissions");
        ::chmod(invitation.c_str(), 0644);
#endif
        rejects([&] { (void)join_pool(invitation, root / "public-invitation"); });
#ifdef _WIN32
        pos2gpu::PrivateFileSecurity private_access;
        require(::SetFileSecurityW(invitation.c_str(), DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, private_access.descriptor),
            "cannot restore private invitation ACL");
#else
        ::chmod(invitation.c_str(), 0600);
#endif
        auto bytes = std::filesystem::file_size(invitation);
        std::filesystem::resize_file(invitation, bytes - 1);
        rejects([&] { (void)join_pool(invitation, root / "truncated"); });
        require(!std::filesystem::exists(root / "truncated"), "invalid invitation left a partial setup");
        auto credential_path = std::filesystem::path(reinterpret_cast<char8_t const*>(worker.tls.key_file.c_str()));
        std::filesystem::remove(credential_path);
#ifdef _WIN32
        pos2gpu::create_private_directory(credential_path);
#else
        std::filesystem::create_symlink(authority.tls.key_file, credential_path);
#endif
        rejects([&] { (void)load_pool(worker.directory); });
        std::cout << "pool enrollment, private publication, validation and saved defaults passed\n";
        return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
