#include "host/ComputeIdentity.hpp"
#include "compute_pool_fixture.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <thread>
#include <future>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <sys/stat.h>
#endif

using namespace group_grow::pool;
using pool_test::require;
namespace
{
template <typename F> void rejects(F&& operation)
{
    try { operation(); } catch (std::exception const&) { return; }
    throw std::runtime_error("invalid membership operation was accepted");
}
void write(std::filesystem::path const& path, std::string const& text)
{
#ifdef _WIN32
    if (!std::filesystem::exists(path))
    { auto file = pos2gpu::create_private_file_handle(path); ::CloseHandle(file); }
#endif
    std::ofstream output(path, std::ios::binary); output << text; output.close();
#ifdef _WIN32
    require(bool(output), "cannot write private membership fixture");
#else
    require(bool(output) && !::chmod(path.c_str(), 0600), "cannot write private membership fixture");
#endif
}
std::string read(std::filesystem::path const& path)
{
    std::ifstream input(path, std::ios::binary); return {std::istreambuf_iterator<char>(input), {}};
}
std::string read(std::string const& path) { return read(std::filesystem::path(reinterpret_cast<char8_t const*>(path.c_str()))); }
void write(std::string const& path, std::string const& text) { write(std::filesystem::path(reinterpret_cast<char8_t const*>(path.c_str())), text); }
PoolIdentity legacy(PoolIdentity const& pool, bool expired = false, PoolIdentity const* authority = nullptr)
{
    auto encoded = read(pool.tls.cert_file); auto marker = encoded.find("\n# xchplot2-compute-policy-1 ");
    if (marker != encoded.npos) encoded.resize(marker);
    if (expired)
    {
        auto text = read(authority->directory / "authority.key");
        std::unique_ptr<BIO, decltype(&BIO_free)> key_input(BIO_new_mem_buf(text.data(), int(text.size())), BIO_free);
        pool_test::Key key(PEM_read_bio_PrivateKey(key_input.get(), nullptr, nullptr, nullptr), EVP_PKEY_free);
        std::unique_ptr<BIO, decltype(&BIO_free)> cert_input(BIO_new_mem_buf(encoded.data(), int(encoded.size())), BIO_free);
        pool_test::Certificate cert(PEM_read_bio_X509(cert_input.get(), nullptr, nullptr, nullptr), X509_free);
        require(cert && key && X509_gmtime_adj(X509_getm_notBefore(cert.get()), -7200) &&
            X509_gmtime_adj(X509_getm_notAfter(cert.get()), -3600) && X509_sign(cert.get(), key.get(), EVP_sha256()) > 0,
            "cannot expire legacy certificate fixture");
        std::unique_ptr<BIO, decltype(&BIO_free)> output(BIO_new(BIO_s_mem()), BIO_free);
        require(output && PEM_write_bio_X509(output.get(), cert.get()) == 1, "cannot encode expired fixture");
        char* data = nullptr; auto size = BIO_get_mem_data(output.get(), &data);
        auto key_position = encoded.find("-----BEGIN PRIVATE KEY-----"); require(key_position != encoded.npos, "missing fixture key");
        encoded = std::string(data, size_t(size)) + encoded.substr(key_position);
    }
    auto key_position = encoded.find("-----BEGIN PRIVATE KEY-----"); require(key_position != encoded.npos, "missing fixture key");
    write(pool.directory / "cert.pem", encoded.substr(0, key_position)); write(pool.directory / "key.pem", encoded.substr(key_position));
    std::filesystem::remove(pool.directory / "credentials.pem"); return *load_pool(pool.directory);
}
struct Connection
{
    Channel client, server;
};
Connection connect_members(PoolIdentity const& server, PoolIdentity const& client, bool saved_server = true, bool saved_client = true,
    bool read_only_client = false)
{
    auto server_tls = server.tls, client_tls = client.tls; server_tls.timeout_ms = client_tls.timeout_ms = 1000;
    client_tls.read_only_membership = read_only_client;
    if (!saved_server) server_tls.membership_directory.clear();
    if (!saved_client) client_tls.membership_directory.clear();
    server_tls.listen_interfaces = {"lo"};
    Listener listener({"127.0.0.1", 0, server.server_name}, server_tls);
    auto endpoint = listener.endpoint(); endpoint.server_name = server.server_name;
    auto accepted = std::async(std::launch::async, [&] { return listener.accept(1000); });
    try { auto channel = Channel::connect(endpoint, client_tls); return {std::move(channel), accepted.get()}; }
    catch (...) { try { (void)accepted.get(); } catch (...) {} throw; }
}
void lifecycle()
{
    pool_test::Directory directory; auto root = directory.path / std::filesystem::path(u8"membership-\u03bb-\u5b58"); std::filesystem::create_directory(root);
    auto authority = create_pool(root / "authority", "authority");
    invite_pool(authority, root / "first.invite", "first"); auto first = join_pool(root / "first.invite", root / "first");
    invite_pool(authority, root / "second.invite", "second"); auto second = join_pool(root / "second.invite", root / "second");
    auto defaults = first.defaults; defaults.cpus = 2; defaults.broker = "127.0.0.1:12345"; configure_pool(first, defaults);
    auto before = certificate_identity(first.tls.cert_file);
    save_worker_intent(first, WorkerIntent::Paused);
    request_pool_renewal(first, root / "first.request");
    request_pool_renewal(first, root / "duplicate.request"); request_pool_renewal(first, root / "first.request");
    require(read(root / "first.request") == read(root / "duplicate.request"), "retry replaced the pending key or signed request");
    write(root / "occupied-request", "unchanged");
    rejects([&] { request_pool_renewal(second, root / "occupied-request"); });
    request_pool_renewal(second, root / "second.request");
    rejects([&] { sign_pool_renewal(second, root / "first.request", root / "unauthorized.renewal"); });
    write(root / "occupied", "unchanged"); auto old_policy = pool_policy(authority.tls.membership_directory);
    rejects([&] { sign_pool_renewal(authority, root / "first.request", root / "occupied"); });
    require(pool_policy(authority.tls.membership_directory) == old_policy && read(root / "occupied") == "unchanged", "failed renewal changed policy or destination");
    auto before_signing = read(authority.tls.cert_file);
    sign_pool_renewal(authority, root / "first.request", root / "first.renewal");
    auto signed_replacement = read(root / "first.renewal");
    auto committed_policy = pool_policy(authority.tls.membership_directory);
    sign_pool_renewal(authority, root / "first.request", root / "first.renewal");
    require(read(root / "first.renewal") == signed_replacement && pool_policy(authority.tls.membership_directory) == committed_policy,
        "completed renewal replay reissued credentials or changed signed policy");
    // Restore the pre-commit bundle beside a durable signing journal, as if
    // the authority exited after preparing the response and before commit.
    Writer journal; auto original_request = read(root / "first.request");
    journal.bytes({reinterpret_cast<uint8_t const*>(original_request.data()), original_request.size()});
    journal.bytes({reinterpret_cast<uint8_t const*>(signed_replacement.data()), signed_replacement.size()});
    write(authority.directory / "renewal.pending", {reinterpret_cast<char const*>(journal.data().data()), journal.data().size()});
    write(authority.tls.cert_file, before_signing); std::filesystem::remove(root / "first.renewal");
    rejects([&] { invite_pool(authority, root / "blocked.invite", "blocked"); });
    sign_pool_renewal(authority, root / "first.request", root / "first.renewal");
    require(read(root / "first.renewal") == signed_replacement && !std::filesystem::exists(authority.directory / "renewal.pending"),
        "interrupted renewal signing did not recover its original response");
    rejects([&] { check_pool_peer(authority.tls.membership_directory, first.node_id, before); });
    rejects([&] { import_pool_renewal(second, root / "first.renewal"); });
    invite_pool(authority, root / "unrelated.invite", "unrelated");
    committed_policy = pool_policy(authority.tls.membership_directory);
    sign_pool_renewal(authority, root / "first.request", root / "first.renewal");
    require(read(root / "first.renewal") == signed_replacement && pool_policy(authority.tls.membership_directory) == committed_policy,
        "completed renewal replay rolled back an unrelated newer policy");
    accept_pool_policy(first.tls.membership_directory, pool_policy(authority.tls.membership_directory));
    auto current_generation = load_pool(first.directory)->policy_generation;
    auto pending_before_import = read(first.directory / "renewal.key");
    import_pool_renewal(first, root / "first.renewal"); auto renewed = *load_pool(first.directory);
    require(renewed.node_id == first.node_id && renewed.defaults.cpus == defaults.cpus && renewed.defaults.broker == defaults.broker &&
        certificate_node_id(renewed.tls.cert_file) == first.node_id && certificate_identity(renewed.tls.cert_file) != before &&
        !std::filesystem::exists(first.directory / "renewal.key") && renewed.policy_generation == current_generation,
        "renewal changed node/defaults, rolled back policy, or failed to rotate credentials");
    check_pool_peer(authority.tls.membership_directory, renewed.node_id, certificate_identity(renewed.tls.cert_file));
    rejects([&] { sign_pool_renewal(authority, root / "first.request", root / "replay.renewal"); });
    write(first.directory / "renewal.key", pending_before_import);
    import_pool_renewal(first, root / "first.renewal");
    require(load_worker_intent(*load_pool(first.directory)) == WorkerIntent::Paused, "renewal changed node-owned worker intent");
    import_pool_renewal(first, root / "first.renewal");
    write(first.directory / "renewal.key", pending_before_import);
    request_pool_renewal(renewed, root / "next.request");
    require(read(first.directory / "renewal.key") != pending_before_import, "completed pending key prevented a fresh rotation after interruption");
    rejects([&] { sign_pool_renewal(authority, root / "next.request", root / "first.renewal"); });
    require(pool_policy(authority.tls.membership_directory) == committed_policy && read(root / "first.renewal") == signed_replacement,
        "mismatched renewal request changed existing credentials or policy");
    sign_pool_renewal(authority, root / "next.request", root / "next.renewal");
    rejects([&] { sign_pool_renewal(authority, root / "first.request", root / "first.renewal"); });
    import_pool_renewal(renewed, root / "next.renewal"); renewed = *load_pool(first.directory);
    export_pool_policy(second, root / "old.policy");
    revoke_pool_member(authority, renewed.node_id); export_pool_policy(authority, root / "new.policy");
    rejects([&] { sign_pool_renewal(authority, root / "next.request", root / "next.renewal"); });
    auto members = pool_members(authority);
    require(std::any_of(members.begin(), members.end(), [&](auto const& member) { return member.node_id == renewed.node_id && member.revoked; }), "revocation missing from registry");
    rejects([&] { revoke_pool_member(authority, authority.node_id); });
    rejects([&] { revoke_pool_member(second, authority.node_id); });
    import_pool_policy(second, root / "new.policy");
    rejects([&] { check_pool_peer(second.tls.membership_directory, renewed.node_id, certificate_identity(renewed.tls.cert_file)); });
    rejects([&] { import_pool_policy(second, root / "old.policy"); });
    rejects([&] { import_pool_policy(second, root / "new.policy"); });
    auto tampered = read(root / "new.policy"); tampered.back() ^= 1; write(root / "tampered.policy", tampered);
    rejects([&] { import_pool_policy(renewed, root / "tampered.policy"); });
    auto foreign = create_pool(root / "foreign", "foreign"); export_pool_policy(foreign, root / "foreign.policy");
    rejects([&] { import_pool_policy(second, root / "foreign.policy"); });
    request_pool_renewal(renewed, root / "revoked.request");
    rejects([&] { sign_pool_renewal(authority, root / "revoked.request", root / "revoked.renewal"); });
    // Original files contain no policy or registry. Recovery authenticates
    // possession of their private key even after the leaf certificate expires.
    auto old_authority = legacy(create_pool(root / "old-authority", "old authority"));
    invite_pool(old_authority, root / "legacy.invite", "legacy");
    auto old_member = join_pool(root / "legacy.invite", root / "old-member");
    old_member = legacy(old_member, true, &old_authority); old_authority = legacy(*load_pool(old_authority.directory));
    require(old_member.certificate_expired && !old_member.authority_expired && !old_member.certificate_expires.empty() &&
        !old_member.authority_expires.empty() && !old_member.policy_generation, "expired legacy enrollment could not be inspected for recovery");
    request_pool_renewal(old_member, root / "legacy.request"); sign_pool_renewal(old_authority, root / "legacy.request", root / "legacy.renewal");
    import_pool_renewal(old_member, root / "legacy.renewal"); auto recovered = *load_pool(old_member.directory);
    require(!recovered.certificate_expired && recovered.node_id == old_member.node_id && recovered.policy_generation &&
        !std::filesystem::exists(old_member.directory / "key.pem"), "legacy recovery did not migrate and preserve the node");
}
void live_revocation()
{
    pool_test::Directory directory; auto root = directory.path / std::filesystem::path(u8"network-\u03bb-\u5b58"); std::filesystem::create_directory(root);
    auto authority = create_pool(root / "authority", "authority");
    invite_pool(authority, root / "first.invite", "first"); auto first = join_pool(root / "first.invite", root / "first");
    invite_pool(authority, root / "second.invite", "second"); auto second = join_pool(root / "second.invite", root / "second");
    auto persistent = connect_members(first, second);
    require(persistent.server.peer_node_id() == second.node_id && persistent.client.peer_node_id() == first.node_id, "TLS omitted stable member identity");
    auto authority_link = connect_members(authority, first);
    auto generation = load_pool(first.directory)->policy_generation;
    require(generation == load_pool(authority.directory)->policy_generation, "handshake did not synchronize signed policy");
    revoke_pool_member(authority, second.node_id);
    Writer payload; payload.string("control"); authority_link.server.send(Message::Control, payload.data());
    auto received = authority_link.client.receive();
    require(received.type == Message::Control && load_pool(first.directory)->policy_generation == load_pool(authority.directory)->policy_generation,
        "persistent connection did not propagate changed signed policy");
    rejects([&] { persistent.server.send(Message::Reply, {}); });
    rejects([&] { persistent.client.send(Message::Control, payload.data()); (void)persistent.server.receive(); });
    rejects([&] { (void)connect_members(first, second); });
    auto old_authority = create_pool(root / "old-authority", "old authority");
    invite_pool(old_authority, root / "expired.invite", "expired");
    auto expired = legacy(join_pool(root / "expired.invite", root / "expired"), true, &old_authority);
    old_authority = legacy(old_authority);
    rejects([&] { (void)connect_members(old_authority, expired); });
    request_pool_renewal(expired, root / "expired.request");
    sign_pool_renewal(old_authority, root / "expired.request", root / "expired.renewal");
    import_pool_renewal(expired, root / "expired.renewal"); auto recovered = *load_pool(expired.directory);
    auto restored = connect_members(*load_pool(old_authority.directory), recovered);
    restored.client.send(Message::Control, payload.data());
    require(restored.server.receive().payload == payload.data(), "recovered expired enrollment could not resume authenticated traffic");
}
void mixed_tls()
{
    for (bool saved_server : {false, true})
    {
        pool_test::Directory directory; auto root = directory.path;
        auto authority = create_pool(root / "authority", "authority");
        invite_pool(authority, root / "server.invite", "server"); auto server = join_pool(root / "server.invite", root / "server");
        invite_pool(authority, root / "client.invite", "client"); auto client = join_pool(root / "client.invite", root / "client");
        auto connection = connect_members(server, client, saved_server, !saved_server);
        require(connection.server.peer_node_id() == client.node_id && connection.client.peer_node_id() == server.node_id,
            "mixed TLS mode omitted verified node identity");
        Writer payload; payload.string("mixed mode"); connection.client.send(Message::Control, payload.data());
        require(connection.server.receive().payload == payload.data(), "mixed TLS request framing changed");
        connection.server.send(Message::Reply, payload.data());
        require(connection.client.receive().payload == payload.data(), "mixed TLS reply framing changed");
        // A local policy change is enforced even when the peer deliberately
        // uses explicit TLS and therefore does not exchange saved policies.
        auto const& managed = saved_server ? server : client;
        auto const& explicit_peer = saved_server ? client : server;
        revoke_pool_member(authority, explicit_peer.node_id);
        accept_pool_policy(managed.tls.membership_directory, pool_policy(authority.tls.membership_directory));
        if (saved_server)
        {
            rejects([&] { connection.server.send(Message::Reply, {}); });
            connection.client.send(Message::Control, payload.data());
            rejects([&] { (void)connection.server.receive(); });
        }
        else
        {
            rejects([&] { connection.client.send(Message::Control, payload.data()); });
            connection.server.send(Message::Reply, payload.data());
            rejects([&] { (void)connection.client.receive(); });
        }
        rejects([&] { (void)connect_members(server, client, saved_server, !saved_server); });
    }
}
void read_only_membership()
{
    pool_test::Directory directory; auto root = directory.path;
    auto authority = create_pool(root / "authority", "authority");
    invite_pool(authority, root / "probe.invite", "probe"); auto probe = join_pool(root / "probe.invite", root / "probe");
    auto const credentials = read(probe.tls.cert_file), profile = read(probe.directory / "profile");
    auto const policy = pool_policy(probe.tls.membership_directory);
    // The server has a newer policy. Diagnostic traffic must enforce the
    // probe's saved policy without learning or saving the newer registry.
    invite_pool(authority, root / "new.invite", "new member");
    auto channel = connect_members(authority, probe, true, true, true);
    Writer payload; payload.string("read-only probe"); channel.client.send(Message::Control, payload.data());
    require(channel.server.receive().payload == payload.data(), "read-only membership broke authenticated traffic");
    channel.server.send(Message::Reply, payload.data());
    require(channel.client.receive().payload == payload.data() && read(probe.tls.cert_file) == credentials &&
        read(probe.directory / "profile") == profile && pool_policy(probe.tls.membership_directory) == policy,
        "read-only diagnostic traffic persisted a peer's newer signed policy");
    revoke_pool_member(authority, probe.node_id);
    accept_pool_policy(probe.tls.membership_directory, pool_policy(authority.tls.membership_directory));
    rejects([&] { channel.client.send(Message::Control, payload.data()); });
    rejects([&] { (void)connect_members(authority, probe, true, true, true); });
}
} // namespace
int main(int argc, char** argv)
{
    try
    {
        lifecycle();
        if (argc > 1 && std::string_view(argv[1]) == "--network") { live_revocation(); mixed_tls(); read_only_membership(); }
        std::cout << "signed pool policy, revocation, credential rotation and legacy recovery passed\n"; return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
