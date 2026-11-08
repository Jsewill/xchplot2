#include "host/ComputePool.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#endif
#include "compute_pool_fixture.hpp"
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <openssl/ssl.h>
#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace group_grow::pool;
namespace fs = std::filesystem;
namespace
{
#ifdef _WIN32
using TestSocket = SOCKET;
constexpr TestSocket invalid_test_socket = INVALID_SOCKET;
void close_test_socket(TestSocket socket) { ::closesocket(socket); }
#else
using TestSocket = int;
constexpr TestSocket invalid_test_socket = -1;
void close_test_socket(TestSocket socket) { ::close(socket); }
#endif
void require(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F&& operation)
{
    bool rejected = false;
    try { operation(); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "invalid pool operation was accepted");
}
template <typename F> void transient_rejects(F&& operation)
{
    try { operation(); } catch (TransientTransport const&) { return; }
    throw std::runtime_error("transport failure did not have a transient category");
}
template <typename F> void stale_lease_rejects(F&& operation)
{
    try { operation(); } catch (StaleLease const&) { return; }
    throw std::runtime_error("fenced compute lease did not have a stale category");
}
template <typename F> void permanent_rejects(F&& operation, char const* expected = nullptr)
{
    try { operation(); }
    catch (TransientTransport const& error) { throw std::runtime_error("permanent failure was classified as transient transport: " + std::string(error.what())); }
    catch (StaleLease const& error) { throw std::runtime_error("permanent failure was classified as a stale lease: " + std::string(error.what())); }
    catch (StaleDelivery const& error) { throw std::runtime_error("permanent failure was classified as a stale delivery: " + std::string(error.what())); }
    catch (std::runtime_error const& error)
    {
        require(!expected || error.what() == std::string(expected), "permanent failure had an unexpected cause"); return;
    }
    throw std::runtime_error("permanent failure was accepted");
}
using pool_test::Directory;
using pool_test::Identity;
using pool_test::issue;
using pool_test::tls;
struct RunningBroker
{
    BrokerServer server;
    std::atomic<bool> stop{false};
    std::thread thread;
    explicit RunningBroker(BrokerOptions options) : server(std::move(options)), thread([this] { server.serve(stop); }) {}
    ~RunningBroker() { stop.store(true); thread.join(); }
};
Task task(std::string id, uint64_t sequence = 1)
{
    Task result; result.task_id = std::move(id); result.compatibility = "test-format"; result.sequence = sequence;
    result.resources = {1, 1, 1024, 0}; return result;
}
Task pack_task(std::string id, uint64_t sequence = 1)
{
    auto result = task(std::move(id), sequence); result.kind = Kind::Pack;
    result.work_units = codec_window_header_bytes + codec_chunk_header_bytes;
    result.chunk_count = 1; result.max_output_bytes = 1024; return result;
}
WorkerOffer offer(std::string instance, uint16_t port)
{
    WorkerOffer result; result.instance = std::move(instance); result.compatibility = "test-format";
    result.codec_compatibility = "test-format";
    result.endpoint = {"127.0.0.1", port}; result.capacity = {2, 1, 4096, 0};
    result.capabilities = capability(Kind::GenerateCpu) | capability(Kind::Pack); return result;
}
GenerationProfile profile(Kind kind, uint64_t host = 1024, uint64_t gpu = 2048)
{
    GenerationProfile result; result.kind = kind; result.resources = {1, 1, host, kind == Kind::GenerateGpu ? gpu : 0};
    return result;
}
void codecs()
{
    Writer writer(128); writer.u8(0xab); writer.u16(0xcdef); writer.u32(0x12345678); writer.u64(0x0123456789abcdef);
    writer.string("hello"); Reader reader(writer.data());
    require(reader.u8() == 0xab && reader.u16() == 0xcdef && reader.u32() == 0x12345678 &&
        reader.u64() == 0x0123456789abcdef && reader.string() == "hello", "wire integer round trip changed"); reader.finish();
    rejects([&] { Reader short_reader(std::span(writer.data()).first(1)); (void)short_reader.u16(); });
    rejects([&] { Reader trailing(writer.data()); trailing.finish(); });
    rejects([&] { Writer limited(3); limited.u32(1); });
    rejects([&] { Writer blob; blob.string("hello"); Reader input(blob.data()); (void)input.string(4); });
    rejects([&] { Writer nul; nul.string(std::string("a\0b", 3)); Reader input(nul.data()); (void)input.string(); });
    require(parse_endpoint("[::1]:443") == Endpoint{"::1", 443}, "IPv6 endpoint changed");
    require(numeric_host("127.0.0.1") && numeric_host("::1") && numeric_host("fe80::1%lo") &&
        !numeric_host("localhost") && !numeric_host("::1%") && !numeric_host("127.0.0.1%lo"), "numeric diagnostic routes changed");
    rejects([] { (void)parse_endpoint("::1:443"); }); rejects([] { (void)parse_endpoint("host:65536"); });
    rejects([] { (void)parse_endpoint("host:000000000000000000000000000000000000000000000000000000000000000000000000001"); });
}
// Raw authenticated connection injects frames the public Channel forbids.
void raw_frame(Endpoint endpoint, TlsOptions const& options, std::span<uint8_t const> bytes,
               bool send = true, bool orderly = false, bool reset = false)
{
    std::unique_ptr<SSL_CTX, decltype(&SSL_CTX_free)> context(SSL_CTX_new(TLS_client_method()), SSL_CTX_free);
    require(context && SSL_CTX_load_verify_locations(context.get(), options.ca_file.c_str(), nullptr) == 1 &&
        SSL_CTX_use_certificate_file(context.get(), options.cert_file.c_str(), SSL_FILETYPE_PEM) == 1 &&
        SSL_CTX_use_PrivateKey_file(context.get(), options.key_file.c_str(), SSL_FILETYPE_PEM) == 1, "raw TLS setup failed");
    SSL_CTX_set_verify(context.get(), SSL_VERIFY_PEER, nullptr);
    TestSocket socket = ::socket(AF_INET, SOCK_STREAM, 0); require(socket != invalid_test_socket, "raw socket failed");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(endpoint.port);
    require(::inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr) == 1 &&
        ::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "raw connect failed");
    std::unique_ptr<SSL, decltype(&SSL_free)> ssl(SSL_new(context.get()), SSL_free);
    require(ssl && SSL_set_fd(ssl.get(), int(socket)) == 1 && SSL_connect(ssl.get()) == 1, "raw handshake failed");
    Writer offer(5); offer.u32(0x47504d34); offer.u8(0);
    require(SSL_write(ssl.get(), offer.data().data(), int(offer.data().size())) == int(offer.data().size()), "raw membership offer failed");
    std::array<uint8_t, 5> accepted{}; size_t position = 0;
    while (position < accepted.size())
    {
        auto count = SSL_read(ssl.get(), accepted.data() + position, int(accepted.size() - position));
        require(count > 0, "raw membership reply failed"); position += size_t(count);
    }
    Reader capability(accepted);
    require(capability.u32() == 0x47504d34 && capability.u8() <= 1, "raw membership reply changed"); capability.finish();
    if (send && !bytes.empty()) require(SSL_write(ssl.get(), bytes.data(), int(bytes.size())) == int(bytes.size()), "raw send failed");
    if (!send) std::this_thread::sleep_for(std::chrono::milliseconds(600));
    if (orderly) (void)SSL_shutdown(ssl.get());
    if (reset)
    { linger value{1, 0}; require(::setsockopt(socket, SOL_SOCKET, SO_LINGER, reinterpret_cast<char const*>(&value), sizeof(value)) == 0, "raw reset setup failed"); }
    ssl.reset(); close_test_socket(socket);
}
void transport(TlsOptions server_tls, TlsOptions const& client_tls, TlsOptions const& rogue_tls, std::string const& identity)
{
    Listener listener({"127.0.0.1", 0}, server_tls);
    auto echo = std::async(std::launch::async, [&]
    {
        auto channel = listener.accept(1000); require(bool(channel), "echo connection missing");
        auto frame = channel.receive(); channel.send(frame.type, frame.payload); return channel.peer_identity();
    });
    auto channel = Channel::connect(listener.endpoint(), client_tls, identity); Writer message; message.string("direct result");
    channel.send(Message::Data, message.data()); auto frame = channel.receive();
    require(frame.type == Message::Data && frame.payload == message.data(), "direct payload changed");
    require(echo.get() == certificate_identity(client_tls.cert_file), "direct owner identity changed"); channel.close();
    auto closed_peer = std::async(std::launch::async, [&] { auto incoming = listener.accept(1000); incoming.close(); });
    auto broken = Channel::connect(listener.endpoint(), client_tls, identity); closed_peer.get();
    transient_rejects([&]
    {
        for (unsigned attempt = 0; attempt < 64; ++attempt) broken.send(Message::Data, message.data());
        (void)broken.receive();
    });
    auto reject_peer = [&](TlsOptions const& client, std::string pin)
    {
        auto peer = std::async(std::launch::async, [&] { try { auto incoming = listener.accept(1000); if (incoming) (void)incoming.receive(); } catch (...) {} });
        permanent_rejects([&] { auto denied = Channel::connect(listener.endpoint(), client, pin); denied.send(Message::Data, {}); (void)denied.receive(); }); peer.get();
    };
    reject_peer(client_tls, std::string(64, '0')); reject_peer(rogue_tls, {});
    auto corrupt = [&](std::vector<uint8_t> data, bool send = true, bool transient = false)
    {
        auto peer = std::async(std::launch::async, [&]
        {
            auto incoming = listener.accept(1000); require(bool(incoming), "malformed connection missing");
            if (transient) transient_rejects([&] { (void)incoming.receive(); });
            else permanent_rejects([&] { (void)incoming.receive(); });
        });
        raw_frame(listener.endpoint(), client_tls, data, send); peer.get();
    };
    Writer over(12); over.u32(0x47504331); over.u16(1); over.u16(uint16_t(Message::Data)); over.u32(65537); corrupt(over.data());
    Writer unknown(12); unknown.u32(0x47504331); unknown.u16(1); unknown.u16(65535); unknown.u32(0); corrupt(unknown.data());
    Writer wrong(12); wrong.u32(0x47504331); wrong.u16(2); wrong.u16(uint16_t(Message::Data)); wrong.u32(0); corrupt(wrong.data());
    Writer short_payload(12); short_payload.u32(0x47504331); short_payload.u16(1); short_payload.u16(uint16_t(Message::Data)); short_payload.u32(1); corrupt(short_payload.data(), true, true);
    corrupt({}, false, true);
    for (bool orderly : {false, true})
    {
        auto peer = std::async(std::launch::async, [&]
        { auto incoming = listener.accept(1000); require(bool(incoming), "EOF connection missing"); transient_rejects([&] { (void)incoming.receive(); }); });
        raw_frame(listener.endpoint(), client_tls, {}, true, orderly, !orderly); peer.get();
    }
    Endpoint absent;
    { Listener reserved({"127.0.0.1", 0}, server_tls); absent = reserved.endpoint(); }
    transient_rejects([&] { (void)Channel::connect(absent, client_tls); });
    auto invalid_tls = client_tls; invalid_tls.ca_file += ".missing";
    permanent_rejects([&] { (void)Channel::connect(absent, invalid_tls); });
}
void cn_only(TlsOptions const& server_tls, TlsOptions const& client_tls)
{
    Listener listener({"127.0.0.1", 0}, server_tls);
    auto server = std::async(std::launch::async, [&] { try { auto incoming = listener.accept(1000); } catch (...) {} });
    // Verify the DNS name without depending on localhost's address-family order.
    auto endpoint = listener.endpoint(); endpoint.server_name = "localhost";
    permanent_rejects([&] { (void)Channel::connect(endpoint, client_tls); }, "network certificate verification failed"); server.get();
}
void task_deadlines(TlsOptions server_tls, TlsOptions client_tls)
{
    using Clock = std::chrono::steady_clock;
    server_tls.timeout_ms = client_tls.timeout_ms = 3000;
    server_tls.max_frame_bytes = client_tls.max_frame_bytes = 1048576;
    // A connected TCP listener that never starts TLS cannot extend a task's
    // handshake deadline to the longer socket timeout.
    {
        Listener listener({"127.0.0.1", 0}, server_tls);
        auto bounded = client_tls; bounded.deadline = Clock::now() + std::chrono::milliseconds(100);
        auto connecting = std::async(std::launch::async, [&]
        { transient_rejects([&] { (void)Channel::connect(listener.endpoint(), bounded); }); });
        require(connecting.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
                "client TLS handshake ignored the task deadline"); connecting.get();
    }
    // The accepted side applies the same bound to an incomplete handshake.
    {
        auto bounded = server_tls; bounded.deadline = Clock::now() + std::chrono::milliseconds(100);
        Listener listener({"127.0.0.1", 0}, bounded);
        TestSocket socket = ::socket(AF_INET, SOCK_STREAM, 0); require(socket != invalid_test_socket, "deadline socket failed");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(listener.endpoint().port);
        ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        require(::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "deadline connect failed");
        auto accepting = std::async(std::launch::async, [&] { transient_rejects([&] { (void)listener.accept(3000); }); });
        auto const status = accepting.wait_for(std::chrono::seconds(1)); close_test_socket(socket); accepting.get();
        require(status == std::future_status::ready, "server TLS handshake ignored the task deadline");
    }
    for (bool writing : {false, true})
    {
        Listener listener({"127.0.0.1", 0}, server_tls);
        std::promise<void> authenticated, release;
        auto ready = authenticated.get_future(), released = release.get_future();
        auto peer = std::async(std::launch::async, [&]
        {
            auto incoming = listener.accept(1000); require(bool(incoming), "deadline peer did not connect");
            authenticated.set_value(); (void)released.wait_for(std::chrono::seconds(5));
        });
        auto channel = Channel::connect(listener.endpoint(), client_tls);
        auto const auth_status = ready.wait_for(std::chrono::seconds(1));
        if (auth_status != std::future_status::ready) { release.set_value(); peer.get(); }
        require(auth_status == std::future_status::ready, "deadline peer was not authenticated");
        channel.set_deadline(Clock::now() + std::chrono::milliseconds(100));
        channel.set_deadline(Clock::time_point::max()); // A later boundary cannot extend it.
        std::vector<uint8_t> payload(client_tls.max_frame_bytes, 0x5a);
        auto transfer = std::async(std::launch::async, [&]
        {
            rejects([&]
            {
                if (!writing) (void)channel.receive();
                else for (unsigned frame = 0; frame < 64; ++frame) channel.send(Message::Data, payload);
            });
        });
        auto const status = transfer.wait_for(std::chrono::seconds(1)); release.set_value();
        transfer.get(); peer.get();
        require(status == std::future_status::ready, writing ? "stalled authenticated send ignored the task deadline" :
            "stalled authenticated receive ignored the task deadline");
    }
    {
        Listener listener({"127.0.0.1", 0}, server_tls);
        BrokerClient original(listener.endpoint(), client_tls);
        std::promise<void> received, release;
        auto ready = received.get_future(), released = release.get_future();
        auto peer = std::async(std::launch::async, [&]
        {
            auto incoming = listener.accept(1000); require(bool(incoming), "deadline control peer did not connect");
            require(incoming.receive().type == Message::Control, "deadline control request missing");
            received.set_value(); (void)released.wait_for(std::chrono::seconds(5));
        });
        auto bounded = original.with_deadline(Clock::now() + std::chrono::milliseconds(200))
            .with_deadline(Clock::now() + std::chrono::seconds(3));
        auto calling = std::async(std::launch::async, [&] { rejects([&] { (void)bounded.open_owner("bounded-owner"); }); });
        auto const auth_status = ready.wait_for(std::chrono::seconds(1));
        auto const status = calling.wait_for(std::chrono::seconds(1)); release.set_value(); calling.get(); peer.get();
        require(auth_status == std::future_status::ready, "deadline control peer was not authenticated");
        require(status == std::future_status::ready, "broker call ignored its absolute task deadline");
        // A later call on the shared base must still work after the bounded
        // copy has expired. Its task deadline was never mutated.
        auto healthy = std::async(std::launch::async, [&]
        {
            auto incoming = listener.accept(1000); require(bool(incoming), "unbounded control peer did not connect");
            require(incoming.receive().type == Message::Control, "unbounded control request missing");
            Writer response; response.string("healthy-owner"); response.string(random_token()); response.string(random_token()); response.u64(1);
            incoming.send(Message::Reply, response.data());
        });
        auto session = original.open_owner("healthy-owner"); healthy.get();
        require(session.owner_id == "healthy-owner", "bounded copy changed the shared broker client deadline");
    }
}
void broker(BrokerOptions options, TlsOptions const& a_tls, TlsOptions const& b_tls,
            TlsOptions const& wa_tls, TlsOptions const& wb_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient a(endpoint, a_tls), b(endpoint, b_tls), wa(endpoint, wa_tls), wb(endpoint, wb_tls);
    wa.register_worker(offer("worker-a", 42001)); wb.register_worker(offer("worker-b", 42002));
    auto owner_a = a.open_owner("jbod-a"), owner_b = b.open_owner("jbod-b");
    auto a1 = a.acquire(owner_a, task("a1", 1)); auto a2 = a.acquire(owner_a, task("a2", 2));
    require(a1 && a2 && a1->worker_identity != a2->worker_identity, "owner could not draw from every eligible worker");
    auto worker = [&](Lease const& lease) -> BrokerClient&
    { return lease.worker_identity == certificate_identity(wa_tls.cert_file) ? wa : wb; };
    require(worker(*a1).begin(*a1, a1->owner_identity) && worker(*a2).begin(*a2, a2->owner_identity), "valid lease did not begin");
    require(!worker(*a1).begin(*a1, a1->owner_identity), "lease began twice");
    rejects([&] { a.finish(*a1); }); rejects([&] { worker(*a1).begin(*a1, a2->worker_identity); });
    auto changed = *a1; changed.task.resources.host_bytes = 1; permanent_rejects([&] { worker(*a1).renew(changed); });
    require(!b.acquire(owner_b, task("b1")), "full workers admitted another job");
    require(!a.acquire(owner_a, task("a3", 3)), "owner bypassed full worker capacity");
    worker(*a1).finish(*a1);
    require(!a.acquire(owner_a, task("a3", 3)), "rapid owner polling bypassed waiting peer");
    auto b1 = b.acquire(owner_b, task("b1")); require(b1 && b1->worker_identity == a1->worker_identity, "waiting peer did not receive released worker");
    require(worker(*b1).begin(*b1, b1->owner_identity), "peer lease did not begin");
    worker(*a2).finish(*a2); auto a3 = a.acquire(owner_a, task("a3", 3)); require(bool(a3), "owner queue did not advance");
    a.cancel(owner_a, "a3", 1); require(!worker(*a3).begin(*a3, a3->owner_identity), "cancelled grant began");
    auto retry = task("a3", 3); retry.attempt = 2; auto retried = a.acquire(owner_a, retry); require(bool(retried), "cancelled task could not retry");
    rejects([&] { (void)a.acquire(owner_a, task("a3", 3)); });
    require(worker(*retried).begin(*retried, retried->owner_identity), "retry did not begin");
    auto new_owner = a.open_owner("jbod-a"); rejects([&] { a.renew(*retried); });
    stale_lease_rejects([&] { worker(*retried).renew(*retried); });
    rejects([&] { (void)a.acquire(owner_a, task("stale", 4)); });
    auto renewed_task = task("a1"); require(!a.acquire(new_owner, renewed_task), "new owner ignored still-running old reservation");
    worker(*retried).finish(*retried); auto replacement = a.acquire(new_owner, renewed_task); require(bool(replacement), "new session could not reuse its task namespace");
    a.cancel(new_owner, "a1", 1); worker(*b1).finish(*b1);
    auto any = task("any", 2); any.kind = Kind::GenerateAny;
    auto selected = a.acquire(new_owner, any); require(selected && selected->task.kind == Kind::GenerateCpu, "generation-any did not select an eligible CPU");
    require(worker(*selected).begin(*selected, selected->owner_identity), "generation-any lease did not begin");
    worker(*selected).renew(*selected);
    // An unauthenticated peer holding a handshake slot must not block healthy
    // registrations, owner control, or worker lease renewals in other slots.
    TestSocket idle = ::socket(AF_INET, SOCK_STREAM, 0); require(idle != invalid_test_socket, "idle socket failed");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(endpoint.port);
    ::inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr);
    require(::connect(idle, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "idle connect failed");
    worker(*selected).renew(*selected); wb.register_worker(offer("worker-b", 42002));
    auto healthy = b.acquire(owner_b, task("b2", 2)); require(bool(healthy), "idle handshake blocked healthy owner");
    close_test_socket(idle); b.cancel(owner_b, "b2", 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(options.lease_ms + 30));
    require(!worker(*selected).begin(*selected, selected->owner_identity), "expired lease restarted");
    auto expired_retry = any; expired_retry.attempt = 2;
    auto new_lease = a.acquire(new_owner, expired_retry); require(bool(new_lease), "expired lease capacity was not recovered");
    rejects([&] { (void)worker(*selected).begin(*selected, selected->owner_identity); });
    a.cancel(new_owner, "any", 2);
    auto incompatible = task("bad-format", 3); incompatible.compatibility = "wrong-format";
    require(!a.acquire(new_owner, incompatible), "incompatible worker accepted task"); a.cancel(new_owner, "bad-format", 1);
}
void selection(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& first_tls, TlsOptions const& second_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), first(endpoint, first_tls), second(endpoint, second_tls);
    first.register_worker(offer("first", 43001)); second.register_worker(offer("second", 43002));
    auto session = owner.open_owner("selection"); std::string fast, slow;
    auto worker = [&](Lease const& lease) -> BrokerClient&
    { return lease.worker_identity == certificate_identity(first_tls.cert_file) ? first : second; };
    for (uint64_t sequence = 1; sequence <= 2; ++sequence)
    {
        auto lease = owner.acquire(session, task("sample-" + std::to_string(sequence), sequence));
        require(bool(lease) && worker(*lease).begin(*lease, lease->owner_identity), "cost sample did not begin");
        worker(*lease).finish(*lease);
        if (sequence == 1) { slow = lease->worker_identity; owner.delivered(*lease, {1000000000, 32, 1024}); }
        else { fast = lease->worker_identity; owner.delivered(*lease, {1000000, 32, 1024}); }
        rejects([&] { owner.delivered(*lease, {1000000, 32, 1024}); });
        rejects([&] { worker(*lease).delivered(*lease, {1000000, 32, 1024}); });
    }
    require(fast != slow, "cold exploration skipped an eligible worker");
    auto preferred = owner.acquire(session, task("preferred", 3));
    require(preferred && preferred->worker_identity == fast, "measured completion cost did not prefer faster worker");
    require(worker(*preferred).begin(*preferred, preferred->owner_identity), "preferred task did not begin");
    auto spill = owner.acquire(session, task("spill", 4));
    require(spill && spill->worker_identity == slow, "busy fast worker prevented available slow-worker admission");
    owner.cancel(session, "spill", 1); worker(*preferred).finish(*preferred);
    for (uint64_t sequence = 5; sequence <= 64; ++sequence)
    {
        auto lease = owner.acquire(session, task("age-" + std::to_string(sequence), sequence));
        require(bool(lease) && worker(*lease).begin(*lease, lease->owner_identity), "age-fair task did not begin");
        worker(*lease).finish(*lease);
        owner.delivered(*lease, {lease->worker_identity == fast ? 1000000ull : 1000000000ull, 32, 1024});
        if (lease->worker_identity == slow) return;
    }
    throw std::runtime_error("measured worker age fairness did not revisit slow worker");
}
void concrete_resources(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto advertised = offer("concrete", 45001); advertised.capacity = {2, 2, 4096, 8192};
    advertised.capabilities |= capability(Kind::GenerateGpu); advertised.generation_profiles = {profile(Kind::GenerateGpu)};
    worker.register_worker(advertised);
    auto session = owner.open_owner("concrete");
    auto cpu = task("cpu", 1); auto first = owner.acquire(session, cpu); require(bool(first), "CPU resource lease missing");
    auto cpu_second = task("cpu-second", 2); auto second = owner.acquire(session, cpu_second);
    require(second && first->task.resources == cpu.resources && second->task.resources == cpu_second.resources,
            "CPU admission did not preserve exact concurrent grants");
    owner.cancel(session, "cpu", 1); owner.cancel(session, "cpu-second", 1);
    auto any = task("gpu", 3); any.kind = Kind::GenerateAny; any.resources.host_bytes = 65536;
    auto gpu = owner.acquire(session, any);
    require(gpu && gpu->task.kind == Kind::GenerateGpu && gpu->task.resources == advertised.generation_profiles.front().resources,
        "heterogeneous GPU lease did not negotiate the selected profile");
    auto repeated = owner.acquire(session, any); require(repeated && repeated->token == gpu->token, "concrete GPU lease polling changed request identity");
    auto pack = pack_task("pack", 4);
    auto concurrent = owner.acquire(session, pack);
    require(concurrent && concurrent->task.resources == pack.resources, "GPU profile monopolized unrelated host/CPU capacity");
    auto blocked = task("blocked", 5);
    require(!owner.acquire(session, blocked), "profile grants overcommitted worker slots");
    owner.cancel(session, "blocked", 1); owner.cancel(session, "gpu", 1);
    owner.cancel(session, "pack", 1);
    auto missing = task("unspecified", 6); missing.resources.host_bytes = 0;
    rejects([&] { (void)owner.acquire(session, missing); });
}
void codec_window_limits(BrokerOptions options, TlsOptions const& owner_tls,
                         TlsOptions const& small_tls, TlsOptions const& large_tls)
{
    for (unsigned limit = 0; limit < 3; ++limit)
    {
        RunningBroker running(options); auto endpoint = running.server.endpoint();
        BrokerClient owner(endpoint, owner_tls), small_worker(endpoint, small_tls), large(endpoint, large_tls);
        auto restricted = offer("codec-small", 45002); restricted.codec_limits = {64, 4096};
        auto invalid = restricted; invalid.codec_limits.max_chunks = 0;
        rejects([&] { small_worker.register_worker(invalid); });
        invalid = restricted; invalid.codec_limits.max_task_bytes = 0;
        rejects([&] { small_worker.register_worker(invalid); });
        invalid = restricted; invalid.codec_limits.max_task_bytes = UINT64_MAX;
        rejects([&] { small_worker.register_worker(invalid); });
        small_worker.register_worker(restricted); auto session = owner.open_owner("codec-limits");
        auto request = pack_task("codec-window");
        if (limit == 0)
        {
            request.chunk_count = default_max_codec_chunks;
            request.work_units = codec_window_header_bytes + uint64_t(request.chunk_count) * codec_chunk_header_bytes;
        }
        else if (limit == 1) request.work_units = restricted.codec_limits.max_task_bytes + 1;
        else request.max_output_bytes = restricted.codec_limits.max_task_bytes + 1;
        auto malformed = request; malformed.chunk_count = 0;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = request; malformed.max_output_bytes = 0;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = request; malformed.work_units = codec_window_header_bytes;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = request; malformed.work_units = UINT64_MAX;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = request; malformed.max_output_bytes = UINT64_MAX;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = task("generation-with-codec-bounds"); malformed.chunk_count = 1;
        rejects([&] { (void)owner.acquire(session, malformed); });
        malformed = task("generation-with-codec-bounds"); malformed.max_output_bytes = 1;
        rejects([&] { (void)owner.acquire(session, malformed); });
        require(!owner.acquire(session, request), "codec task ignored a worker's count/input/output bound");
        auto fitting = offer("codec-large", 45003); large.register_worker(fitting);
        auto lease = owner.acquire(session, request);
        require(lease && lease->worker_identity == certificate_identity(large_tls.cert_file) && lease->task == request &&
            large.begin(*lease, lease->owner_identity), "codec task did not select a worker supporting its exact window");
        auto changed = *lease; ++changed.task.chunk_count;
        rejects([&] { large.renew(changed); });
        changed = *lease; ++changed.task.max_output_bytes;
        rejects([&] { large.renew(changed); });
        auto changed_offer = fitting; ++changed_offer.codec_limits.max_chunks; ++changed_offer.capacity_revision;
        rejects([&] { large.register_worker(changed_offer); });
        large.finish(*lease);
        auto changed_retry = request; ++changed_retry.attempt; ++changed_retry.max_output_bytes;
        rejects([&] { (void)owner.acquire(session, changed_retry); });
    }
}
void retry_eligibility(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& first_tls, TlsOptions const& second_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), first(endpoint, first_tls), second(endpoint, second_tls);
    auto gpu_offer = offer("gpu-rejection", 46001); gpu_offer.capacity.gpu_bytes = 8192;
    gpu_offer.capabilities |= capability(Kind::GenerateGpu); gpu_offer.generation_profiles = {profile(Kind::GenerateGpu)};
    first.register_worker(gpu_offer);
    auto session = owner.open_owner("retry"); auto any = task("member", 1); any.kind = Kind::GenerateAny;
    auto gpu = owner.acquire(session, any); require(gpu && gpu->task.kind == Kind::GenerateGpu, "initial GPU probe missing");
    require(first.begin(*gpu, gpu->owner_identity), "GPU probe did not begin");
    rejects([&] { owner.reject(*gpu, FailureCode::UnsupportedResources); });
    first.reject(*gpu, FailureCode::UnsupportedResources);
    any.attempt = 2; auto cpu = owner.acquire(session, any);
    require(cpu && cpu->task.kind == Kind::GenerateCpu && cpu->worker_identity == gpu->worker_identity,
            "unsupported GPU did not fall back to same worker CPU");
    require(first.begin(*cpu, cpu->owner_identity), "CPU fallback did not begin"); first.finish(*cpu);
    second.register_worker(offer("network-retry", 46002));
    auto request = task("network", 2); auto disconnected = owner.acquire(session, request); require(bool(disconnected), "network retry lease missing");
    auto worker = [&](Lease const& lease) -> BrokerClient&
    { return lease.worker_identity == certificate_identity(first_tls.cert_file) ? first : second; };
    require(worker(*disconnected).begin(*disconnected, disconnected->owner_identity), "network retry lease did not begin");
    owner.abandon(*disconnected); request.attempt = 2;
    require(!owner.acquire(session, request), "retry released an abandoned running reservation early");
    rejects([&] { worker(*disconnected).renew(*disconnected); });
    worker(*disconnected).finish(*disconnected);
    auto replacement = owner.acquire(session, request);
    require(replacement && replacement->worker_identity != disconnected->worker_identity, "network retry did not try an alternative worker");
    owner.cancel(session, "network", 2);
}
void queued_eligibility(BrokerOptions options, TlsOptions const& owner_tls,
                        TlsOptions const& first_tls, TlsOptions const& second_tls)
{
    for (bool incompatible : {false, true})
    {
        RunningBroker running(options); auto endpoint = running.server.endpoint();
        BrokerClient owner(endpoint, owner_tls), first(endpoint, first_tls), second(endpoint, second_tls);
        first.register_worker(offer("small-worker", 48001)); auto session = owner.open_owner("queued-eligibility");
        auto heavy = task("heavy-first", 1); heavy.resources.host_bytes = 8192;
        if (incompatible) heavy.compatibility = "other-format";
        require(!owner.acquire(session, heavy), "blocked generation unexpectedly fit the small worker");
        auto cpu = task("fitting-cpu", 2); auto cpu_lease = owner.acquire(session, cpu);
        require(cpu_lease && first.begin(*cpu_lease, cpu_lease->owner_identity), "blocked generation prevented a fitting CPU task");
        auto pack = pack_task("fitting-pack", 3);
        require(!owner.acquire(session, pack), "fitting pack task ignored the running CPU reservation");
        require(!owner.acquire(session, heavy), "blocked generation left its queue prematurely");
        first.finish(*cpu_lease); auto pack_lease = owner.acquire(session, pack);
        require(pack_lease && first.begin(*pack_lease, pack_lease->owner_identity), "blocked generation prevented a fitting pack task");
        auto later_heavy = heavy; later_heavy.task_id = "heavy-later"; later_heavy.sequence = 4;
        require(!owner.acquire(session, later_heavy), "later heavy task unexpectedly fit an existing worker");
        auto large = offer("large-worker", 48002); large.capacity.host_bytes = 16384; large.compatibility = heavy.compatibility;
        second.register_worker(large);
        auto resumed = owner.acquire(session, heavy);
        require(resumed && resumed->worker_identity == certificate_identity(second_tls.cert_file),
                "oldest blocked generation did not resume when a fitting worker registered");
        require(!owner.acquire(session, later_heavy), "later generation bypassed the older eligible task");
        require(second.begin(*resumed, resumed->owner_identity), "resumed generation did not begin"); second.finish(*resumed);
        auto next = owner.acquire(session, later_heavy);
        require(next && second.begin(*next, next->owner_identity), "eligible generation FIFO did not advance");
        second.finish(*next); first.finish(*pack_lease);
    }
}
void automatic_generation_cpus(BrokerOptions options, TlsOptions const& owner_tls,
                               TlsOptions const& small_tls, TlsOptions const& large_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), small_worker(endpoint, small_tls), large(endpoint, large_tls);
    auto small_offer = offer("small-cpu-policy", 49001); small_offer.capacity = {2, 2, 4096, 0}; small_offer.generation_threads = 2;
    auto large_offer = offer("large-cpu-policy", 49002); large_offer.capacity = {8, 1, 4096, 0}; large_offer.generation_threads = 6;
    auto invalid_offer = small_offer; invalid_offer.generation_threads = 0;
    rejects([&] { small_worker.register_worker(invalid_offer); }); invalid_offer.generation_threads = 3;
    rejects([&] { small_worker.register_worker(invalid_offer); });
    small_worker.register_worker(small_offer); large.register_worker(large_offer); auto session = owner.open_owner("automatic-cpus");
    auto first_request = task("auto-first", 1); first_request.resources.cpu_threads = 0;
    auto second_request = task("auto-second", 2); second_request.resources.cpu_threads = 0;
    auto first = owner.acquire(session, first_request), second = owner.acquire(session, second_request);
    require(first && second && first->worker_identity != second->worker_identity, "automatic CPU tasks did not use both worker sizes");
    auto const small_identity = certificate_identity(small_tls.cert_file);
    auto small_lease = first->worker_identity == small_identity ? *first : *second;
    auto large_lease = first->worker_identity == small_identity ? *second : *first;
    require(small_lease.task.resources.cpu_threads == small_offer.generation_threads &&
        large_lease.task.resources.cpu_threads == large_offer.generation_threads,
        "automatic CPU grants did not honor each worker generation policy");
    require(small_worker.begin(small_lease, small_lease.owner_identity) && large.begin(large_lease, large_lease.owner_identity),
            "automatic CPU leases did not begin");
    small_worker.finish(small_lease); owner.delivered(small_lease, {1000000000, 32, 1024});
    auto explicit_request = task("explicit-small", 3); auto explicit_lease = owner.acquire(session, explicit_request);
    require(explicit_lease && explicit_lease->worker_identity == small_identity && explicit_lease->task.resources.cpu_threads == 1 &&
        small_worker.begin(*explicit_lease, explicit_lease->owner_identity), "explicit smaller CPU request was not preserved");
    auto waiting_request = task("auto-waiting", 4); waiting_request.resources.cpu_threads = 0;
    require(!owner.acquire(session, waiting_request), "automatic CPU policy was silently reduced to available threads");
    auto pack_request = pack_task("single-codec", 5);
    auto pack = owner.acquire(session, pack_request);
    require(pack && pack->task.resources.cpu_threads == 1, "codec did not retain its single-thread grant");
    owner.cancel(session, "single-codec", 1);
    require(!owner.acquire(session, waiting_request), "automatic CPU job overlapped an incomplete policy grant");
    small_worker.finish(*explicit_lease); auto full_policy = owner.acquire(session, waiting_request);
    require(full_policy && full_policy->task.resources.cpu_threads == small_offer.generation_threads,
            "automatic CPU job did not resume with its full worker policy");
    auto repeated = owner.acquire(session, waiting_request);
    require(repeated && repeated->token == full_policy->token, "automatic CPU request polling changed its canonical lease");
    owner.cancel(session, "auto-waiting", 1);
    large.finish(large_lease); owner.delivered(large_lease, {1000000, 32, 1024});
    auto explicit_large = task("explicit-large", 6); explicit_large.resources.cpu_threads = 3;
    auto bounded = owner.acquire(session, explicit_large);
    require(bounded && bounded->worker_identity == large_lease.worker_identity && bounded->task.resources.cpu_threads == 3 &&
        large.begin(*bounded, bounded->owner_identity), "explicit CPU request ignored the selected worker policy bound");
    large.finish(*bounded);
    auto invalid_pack = pack_task("invalid-codec", 7); invalid_pack.resources.cpu_threads = 0;
    rejects([&] { (void)owner.acquire(session, invalid_pack); });
    auto excessive = task("excessive-cpus", 7); excessive.resources.cpu_threads = 7;
    require(!owner.acquire(session, excessive), "explicit CPU request exceeded all worker generation policies"); owner.cancel(session, "excessive-cpus", 1);
    auto preferred = task("measured-auto", 8); preferred.resources.cpu_threads = 0;
    auto measured = owner.acquire(session, preferred);
    require(measured && measured->worker_identity == large_lease.worker_identity && measured->task.resources.cpu_threads == large_offer.generation_threads,
            "automatic CPU scheduling did not reuse observations for the concrete grant");
    owner.cancel(session, "measured-auto", 1);
}
void capacity_refresh(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto original = offer("capacity", 47001); original.capacity.slots = 2; worker.register_worker(original);
    auto session = owner.open_owner("capacity");
    auto first = owner.acquire(session, task("retained", 1)); require(bool(first), "capacity lease missing");
    require(worker.begin(*first, first->owner_identity), "capacity lease did not begin");
    auto smaller = original.capacity; smaller.host_bytes = 2048;
    worker.finish(*first, CapacityUpdate{smaller, 2});
    // An older heartbeat arriving after completion cannot restore the old cap.
    worker.register_worker(original);
    auto large = task("large", 2); large.resources.host_bytes = 4096;
    require(!owner.acquire(session, large), "stale heartbeat inflated retained-memory capacity");
    // A repeated finish also cannot apply a newer-looking stale capacity.
    worker.finish(*first, CapacityUpdate{original.capacity, 3});
    require(!owner.acquire(session, large), "duplicate finish changed current worker capacity");
    original.capacity_revision = 3; worker.register_worker(original);
    auto admitted = owner.acquire(session, large); require(bool(admitted), "new capacity revision did not admit waiting work");
    owner.cancel(session, "large", 1);
    auto probe = owner.acquire(session, task("reject", 3)); require(bool(probe), "capacity rejection lease missing");
    require(worker.begin(*probe, probe->owner_identity), "capacity rejection lease did not begin");
    worker.reject(*probe, FailureCode::UnsupportedResources, CapacityUpdate{smaller, 4});
    original.capacity_revision = 3; worker.register_worker(original);
    auto last = task("still-large", 4); last.resources.host_bytes = 4096;
    require(!owner.acquire(session, last), "admission rejection did not atomically refresh capacity");
    owner.cancel(session, "still-large", 1);
}
void worker_route_change(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto original = offer("route-change", 47002); original.capacity.slots = 2;
    worker.register_worker(original); auto session = owner.open_owner("route-change");
    auto first = owner.acquire(session, task("old-route", 1));
    require(first && worker.begin(*first, first->owner_identity), "old-route lease did not begin");
    auto updated = original; updated.endpoint.port = 47003;
    permanent_rejects([&] { worker.register_worker(updated); });
    ++updated.capacity_revision; worker.register_worker(updated);
    worker.renew(*first);
    auto second = owner.acquire(session, task("new-route", 2));
    require(second && second->endpoint == updated.endpoint, "new lease retained an obsolete worker route");
    auto changed = *first; changed.endpoint = updated.endpoint;
    permanent_rejects([&] { worker.renew(changed); });
    worker.finish(*first); owner.cancel(session, "new-route", 1);
    worker.register_worker(original);
    auto last = owner.acquire(session, task("current-route", 3));
    require(last && last->endpoint == updated.endpoint, "late registration restored the old worker route");
}
void listen_interfaces(BrokerOptions options, TlsOptions const& owner_tls)
{
#ifdef _WIN32
    DWORD loopback = 0; require(::GetBestInterface(::inet_addr("127.0.0.1"), &loopback) == NO_ERROR, "cannot find loopback interface");
    options.tls.listen_interfaces = {std::to_string(loopback)};
#else
    options.tls.listen_interfaces = {"lo"};
#endif
    options.tls.listen_lan = true;
    {
        RunningBroker running(options); BrokerClient owner(running.server.endpoint(), owner_tls);
        owner.open_owner("allowed-loopback");
    }
    options.tls.listen_interfaces = {"nonexistent-test-interface"}; options.tls.listen_lan = false;
    RunningBroker running(options); BrokerClient owner(running.server.endpoint(), owner_tls);
    rejects([&] { owner.open_owner("blocked-loopback"); });
}
void mixed_backend_selection(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto mixed = offer("mixed-backends", 49501); mixed.capacity.gpu_bytes = 8192;
    mixed.capabilities |= capability(Kind::GenerateGpu); mixed.generation_profiles = {profile(Kind::GenerateGpu)};
    worker.register_worker(mixed); auto session = owner.open_owner("mixed-backends");
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        auto request = task("mixed-" + std::to_string(sequence), sequence); request.kind = Kind::GenerateAny;
        auto grant = owner.acquire(session, request);
        require(grant && worker.begin(*grant, grant->owner_identity), "mixed backend sample did not begin");
        require(grant->task.kind == (sequence == 1 ? Kind::GenerateGpu : Kind::GenerateCpu),
            "auto did not compare CPU and GPU candidates on the same offer");
        worker.finish(*grant);
        owner.delivered(*grant, {grant->task.kind == Kind::GenerateGpu ? 1000000000ull : 1000000ull, 32, 1024});
    }
}
void batched_profiles(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto batch = offer("batch-profile", 49502); batch.capacity = {2, 2, 4096, 0};
    auto model = profile(Kind::GenerateCpu); model.member_host_bytes = 512; model.max_batch_members = 4;
    batch.generation_profiles = {model};
    auto invalid = batch; invalid.generation_profiles.push_back(model);
    rejects([&] { worker.register_worker(invalid); });
    invalid = batch; invalid.generation_profiles.front().max_batch_members = max_generation_batch_members + 1;
    rejects([&] { worker.register_worker(invalid); });
    worker.register_worker(batch); auto session = owner.open_owner("batch-profile");
    auto request = task("batch", 1); request.work_units = 4;
    require(owner.generation_hint(session, request) == 4, "batch hint ignored the advertised maximum");
    auto grant = owner.acquire(session, request);
    require(grant && grant->task.work_units == request.work_units && grant->task.resources.host_bytes == 2560 &&
        worker.begin(*grant, grant->owner_identity), "batch profile did not reserve the exact immutable count");
    require(owner.generation_hint(session, request) == 4, "busy workers hid their supported batch shape");
    auto full = task("second-batch", 2); full.work_units = 4;
    require(!owner.acquire(session, full), "batches overcommitted host memory");
    auto changed = request; changed.work_units = 2; changed.attempt = 2;
    rejects([&] { (void)owner.acquire(session, changed); });
    auto updated = batch.generation_profiles; updated.front().resources.host_bytes = 512;
    auto capacity = batch.capacity; capacity.host_bytes = 3584;
    worker.finish(*grant, CapacityUpdate{capacity, 2, updated});
    worker.register_worker(batch);
    auto next = owner.acquire(session, full);
    require(next && next->task.work_units == 4 && next->task.resources.host_bytes == 2048,
        "retained capacity/profile update was not applied atomically");
    owner.cancel(session, full.task_id, full.attempt);
    auto unavailable = task("too-many", 3); unavailable.work_units = 5;
    require(!owner.acquire(session, unavailable), "batch exceeded the worker profile maximum");
    owner.cancel(session, unavailable.task_id, unavailable.attempt);
}
void weighted_shares(BrokerOptions options, TlsOptions const& first_tls, TlsOptions const& second_tls,
                     TlsOptions const& worker_tls, uint64_t heavy_members = 1)
{
    // Keep queued requests live through the full scheduling-order check.
    options.lease_ms = 10000; options.max_tasks = 64; options.owner_shares = {{"heavy", 3}, {"light", 1}};
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient first(endpoint, first_tls), second(endpoint, second_tls), worker(endpoint, worker_tls);
    auto advertised = offer("weighted", 49503); auto model = profile(Kind::GenerateCpu); model.max_batch_members = 3;
    advertised.generation_profiles = {model}; worker.register_worker(advertised);
    auto heavy = first.open_owner("heavy"), light = second.open_owner("light");
    auto heavy_task = [&](std::string id, uint64_t sequence)
    { auto result = task(std::move(id), sequence); result.work_units = heavy_members; return result; };
    auto initial = first.acquire(heavy, heavy_task("initial", 1));
    require(initial && worker.begin(*initial, initial->owner_identity), "weighted initial task missing");
    for (uint64_t sequence = 2; sequence <= 17; ++sequence)
    {
        require(!first.acquire(heavy, heavy_task("heavy-" + std::to_string(sequence), sequence)), "busy weighted worker granted heavy task");
        require(!second.acquire(light, task("light-" + std::to_string(sequence), sequence)), "busy weighted worker granted light task");
    }
    worker.finish(*initial);
    uint64_t heavy_sequence = 2, light_sequence = 2;
    unsigned heavy_count = 0, light_count = 0;
    for (unsigned sample = 0; sample < 16; ++sample)
    {
        auto a = first.acquire(heavy, heavy_task("heavy-" + std::to_string(heavy_sequence), heavy_sequence));
        auto b = second.acquire(light, task("light-" + std::to_string(light_sequence), light_sequence));
        require(bool(a) != bool(b), "weighted scheduler failed to reserve exactly one slot");
        auto const& grant = a ? *a : *b;
        require(worker.begin(grant, grant.owner_identity), "weighted task did not begin");
        if (a) { ++heavy_count; ++heavy_sequence; } else { ++light_count; ++light_sequence; }
        worker.finish(grant);
    }
    require(heavy_count == (heavy_members == 1 ? 12 : 8) && light_count == (heavy_members == 1 ? 4 : 8),
        "configured owner shares did not provide weighted member progress across batch lengths");
    for (; heavy_sequence <= 17; ++heavy_sequence) first.cancel(heavy, "heavy-" + std::to_string(heavy_sequence), 1);
    for (; light_sequence <= 17; ++light_sequence) second.cancel(light, "light-" + std::to_string(light_sequence), 1);
    auto lent = second.acquire(light, task("idle-share-lent", 18));
    require(lent && worker.begin(*lent, lent->owner_identity), "idle owner share was not lent to waiting work"); worker.finish(*lent);
}
void idle_lending_fairness(BrokerOptions options, TlsOptions const& first_tls, TlsOptions const& second_tls,
                          TlsOptions const& worker_tls)
{
    options.lease_ms = 10000;
    for (bool blocked : {false, true})
    {
        options.max_tasks = 64;
        RunningBroker running(options); auto endpoint = running.server.endpoint();
        BrokerClient first(endpoint, first_tls), second(endpoint, second_tls), worker(endpoint, worker_tls);
        worker.register_worker(offer("idle-lending", 49507)); auto lender = first.open_owner("lender");
        OwnerSession peer; uint64_t peer_sequence = 1;
        if (blocked)
        {
            peer = second.open_owner("peer"); auto incompatible = task("incompatible", peer_sequence++);
            incompatible.compatibility = "other-format";
            require(!second.acquire(peer, incompatible), "incompatible peer unexpectedly received idle capacity");
        }
        uint64_t next = 1;
        for (; next <= 12; ++next)
        {
            auto request = task("loan-" + std::to_string(next), next); auto grant = first.acquire(lender, request);
            require(grant && worker.begin(*grant, grant->owner_identity), "owner could not use lent idle capacity");
            worker.finish(*grant);
        }
        if (blocked) second.cancel(peer, "incompatible", 1);
        else peer = second.open_owner("peer");
        auto active_request = task("peer-active", peer_sequence++);
        auto queued_request = task("peer-queued", peer_sequence++);
        auto later_request = task("peer-later", peer_sequence++);
        auto active = second.acquire(peer, active_request);
        require(active && worker.begin(*active, active->owner_identity), "peer did not receive the idle worker");
        require(!second.acquire(peer, queued_request) && !second.acquire(peer, later_request), "peer overcommitted the worker");
        auto returning = task("returning-lender", next);
        require(!first.acquire(lender, returning), "returning lender bypassed running peer work");
        worker.finish(*active);
        auto resumed = first.acquire(lender, returning);
        require(resumed && worker.begin(*resumed, resumed->owner_identity),
            "idle-capacity lending left historical service debt against a later owner");
        worker.finish(*resumed);
        second.cancel(peer, queued_request.task_id, queued_request.attempt);
        second.cancel(peer, later_request.task_id, later_request.attempt);
    }
}
void brief_idle_preserves_debt(BrokerOptions options, TlsOptions const& first_tls, TlsOptions const& second_tls,
                               TlsOptions const& worker_tls)
{
    options.lease_ms = 10000; options.max_tasks = 64;
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient batch(endpoint, first_tls), steady(endpoint, second_tls), worker(endpoint, worker_tls);
    auto advertised = offer("brief-idle", 49508); auto model = profile(Kind::GenerateCpu);
    model.max_batch_members = max_generation_batch_members; advertised.generation_profiles = {model}; worker.register_worker(advertised);
    auto batch_owner = batch.open_owner("batch"), steady_owner = steady.open_owner("steady");
    auto initial = steady.acquire(steady_owner, task("steady-1"));
    require(initial && worker.begin(*initial, initial->owner_identity), "steady initial task missing");
    auto large = task("batch-first"); large.work_units = max_generation_batch_members;
    require(!batch.acquire(batch_owner, large), "batch bypassed running steady work");
    for (uint64_t sequence = 2; sequence <= 9; ++sequence)
        require(!steady.acquire(steady_owner, task("steady-" + std::to_string(sequence), sequence)), "steady work overcommitted the worker");
    worker.finish(*initial); auto granted = batch.acquire(batch_owner, large);
    require(granted && worker.begin(*granted, granted->owner_identity), "waiting batch did not begin"); worker.finish(*granted);
    // The batch owner is now briefly idle, while the peer still has work.
    auto returning = large; returning.task_id = "batch-returning"; ++returning.sequence;
    require(!batch.acquire(batch_owner, returning), "briefly idle batch bypassed the steady reservation");
    for (uint64_t sequence = 2; sequence <= 5; ++sequence)
    {
        auto request = task("steady-" + std::to_string(sequence), sequence); auto grant = steady.acquire(steady_owner, request);
        require(grant && worker.begin(*grant, grant->owner_identity), "brief idle erased batch service debt");
        require(!batch.acquire(batch_owner, returning), "batch owner evaded its member charge by becoming idle"); worker.finish(*grant);
    }
    for (uint64_t sequence = 6; sequence <= 9; ++sequence) steady.cancel(steady_owner, "steady-" + std::to_string(sequence), 1);
    batch.cancel(batch_owner, returning.task_id, returning.attempt);
}
void persistent_control(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    options.control_threads = 2; options.max_connections = 16;
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    std::vector<BrokerClient> idle;
    for (unsigned i = 0; i < 8; ++i)
    { idle.emplace_back(endpoint, owner_tls); (void)idle.back().open_owner("idle-" + std::to_string(i)); }
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    worker.register_worker(offer("persistent", 49504)); auto session = owner.open_owner("persistent");
    auto request = task("probe");
    for (unsigned i = 0; i < 32; ++i) require(owner.generation_hint(session, request) == 1, "idle channels occupied control handlers");
    auto short_task = owner.with_deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
    require(short_task.generation_hint(session, request) == 1, "bounded persistent request failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    require(owner.generation_hint(session, request) == 1, "earlier task deadline poisoned reused control channel");
    transient_rejects([&] { (void)short_task.generation_hint(session, request); });
    auto metrics = owner.metrics();
    require(metrics.connections == 1 && metrics.reused + 1 == metrics.requests && metrics.request_bytes && metrics.response_bytes,
        "sequential control requests did not reuse one authenticated channel");
    std::cout << "persistent control requests=" << metrics.requests << " connections=" << metrics.connections <<
        " request_bytes=" << metrics.request_bytes << " response_bytes=" << metrics.response_bytes << '\n';
    auto small_frames = worker_tls; small_frames.max_frame_bytes = 4096;
    BrokerClient small_worker(endpoint, small_frames);
    auto profiled = offer("persistent", 49504);
    for (uint8_t k = 18; k <= 28; k += 2)
        for (uint8_t strength = 0; strength <= 32; ++strength)
        { auto value = profile(Kind::GenerateCpu); value.k = k; value.strength = strength; profiled.generation_profiles.push_back(value); }
    profiled.capacity_revision = 2;
    small_worker.register_worker(profiled);
    require(small_worker.metrics().request_bytes > small_frames.max_frame_bytes,
        "small data frames did not exercise independent control profile bounds");
}
void control_faults(BrokerOptions options, TlsOptions const& client_tls)
{
    using Clock = std::chrono::steady_clock;
    {
        options.idle_ms = 20;
        RunningBroker running(options); BrokerClient client(running.server.endpoint(), client_tls);
        auto session = client.open_owner("idle-reconnect");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        require(client.generation_hint(session, task("probe")) == 0 && client.metrics().connections == 2,
            "closed idle channel did not reconnect before sending a new request");
    }
    {
        Listener listener({"127.0.0.1", 0}, options.tls); BrokerClient client(listener.endpoint(), client_tls);
        std::promise<void> received, release; auto ready = received.get_future(), released = release.get_future();
        auto peer = std::async(std::launch::async, [&]
        {
            auto stalled = listener.accept(1000); require(stalled && stalled.receive().type == Message::Control, "stalled control request missing");
            received.set_value();
            auto healthy = listener.accept(1000); require(healthy && healthy.receive().type == Message::Control, "independent control request missing");
            Writer response; response.string("independent"); response.string(random_token()); response.string(random_token()); response.u64(1);
            healthy.send(Message::Reply, response.data()); released.wait();
        });
        auto stalled = std::async(std::launch::async, [&]
        { transient_rejects([&] { client.with_deadline(Clock::now() + std::chrono::milliseconds(200)).open_owner("stalled"); }); });
        ready.wait();
        auto independent = client.open_owner("independent");
        require(independent.owner_id == "independent", "stalled request blocked another control channel");
        stalled.get(); release.set_value(); peer.get();
        require(client.metrics().connections == 2 && client.metrics().requests == 2,
            "failed non-idempotent control request was replayed");
    }
    {
        Listener listener({"127.0.0.1", 0}, options.tls); BrokerClient client(listener.endpoint(), client_tls);
        std::promise<void> full, release; auto ready = full.get_future(), released = release.get_future();
        auto peer = std::async(std::launch::async, [&]
        {
            std::vector<Channel> channels;
            for (unsigned i = 0; i < control_channel_limit; ++i)
            {
                channels.push_back(listener.accept(1000));
                require(channels.back() && channels.back().receive().type == Message::Control, "bounded control request missing");
            }
            full.set_value(); released.wait();
        });
        std::vector<std::future<void>> calls;
        for (unsigned i = 0; i < control_channel_limit; ++i) calls.push_back(std::async(std::launch::async, [&]
        { rejects([&] { client.open_owner("bounded"); }); }));
        ready.wait();
        transient_rejects([&] { client.with_deadline(Clock::now() + std::chrono::milliseconds(30)).open_owner("excess"); });
        require(client.metrics().connections == control_channel_limit, "control pool exceeded its connection budget");
        release.set_value(); peer.get(); for (auto& call : calls) call.get();
    }
}
void stale_delivery(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    worker.register_worker(offer("stale-delivery", 49505)); auto session = owner.open_owner("stale-delivery");
    auto grant = owner.acquire(session, task("stale"));
    require(grant && worker.begin(*grant, grant->owner_identity), "stale delivery task did not begin");
    owner.open_owner("stale-delivery");
    bool stale = false;
    try { owner.delivered(*grant, {1000000, 32, 1024}); } catch (StaleDelivery const&) { stale = true; }
    require(stale, "fenced delivery was indistinguishable from a transient broker outage");
    worker.finish(*grant);
}
void compact_heartbeat(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    options.worker_ms = 160;
    RunningBroker running(options); auto endpoint = running.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto advertised = offer("heartbeat-original", 49506);
    for (uint8_t k = 18; k <= 28; k += 2)
        for (uint8_t strength = 0; strength <= 32; ++strength)
        { auto value = profile(Kind::GenerateCpu); value.k = k; value.strength = strength; advertised.generation_profiles.push_back(value); }
    require(!worker.heartbeat(advertised.instance, 1), "unknown heartbeat registered a worker implicitly");
    worker.register_worker(advertised); auto before = worker.metrics();
    for (unsigned i = 0; i < 6; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        require(worker.heartbeat(advertised.instance, advertised.capacity_revision), "current heartbeat did not preserve worker liveness");
    }
    auto after = worker.metrics();
    require(after.request_bytes - before.request_bytes < before.request_bytes,
        "compact heartbeats resent the resource profile table");
    require(!worker.heartbeat(advertised.instance, 2), "heartbeat revision mismatch was accepted");
    auto session = owner.open_owner("heartbeat");
    auto lease = owner.acquire(session, task("live")); require(bool(lease), "live compact-heartbeat worker was unavailable");
    owner.cancel(session, lease->task.task_id, lease->task.attempt);
    // A different process using the certificate cannot keep this instance alive.
    for (unsigned i = 0; i < 6; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        require(!worker.heartbeat("heartbeat-replacement", 1), "another instance refreshed the worker registration");
    }
    require(!worker.heartbeat(advertised.instance, 1), "expired worker was revived by a heartbeat");
    advertised.instance = "heartbeat-replacement"; worker.register_worker(advertised);
    for (unsigned i = 0; i < 6; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        require(!worker.heartbeat("heartbeat-original", 1), "old process refreshed its replacement instance");
    }
    require(!worker.heartbeat(advertised.instance, 1), "old process kept its replacement alive");
    worker.register_worker(advertised);
    require(worker.heartbeat(advertised.instance, 1), "full registration did not restore expired worker liveness");
}
void queued_expiry(BrokerOptions options, TlsOptions const& owner_tls,
                   TlsOptions const& other_tls, TlsOptions const& worker_tls)
{
    {
        RunningBroker running(options); auto endpoint = running.server.endpoint();
        BrokerClient abandoned(endpoint, owner_tls), live(endpoint, other_tls), worker(endpoint, worker_tls);
        auto lost = abandoned.open_owner("abandoned-queue"), active = live.open_owner("active-queue");
        auto missing = task("expired-request"), waiting = task("polled-request");
        require(!abandoned.acquire(lost, missing), "unregistered worker received a task");
        rejects([&] { abandoned.with_deadline(std::chrono::steady_clock::now()).cancel(lost, missing.task_id, 1); });
        require(!live.acquire(active, waiting), "unregistered worker received a live task");
        auto same_owner = task("same-owner-expired", 2);
        require(!live.acquire(active, same_owner), "unregistered worker received a second task");
        auto const until = std::chrono::steady_clock::now() + std::chrono::milliseconds(options.lease_ms * 2);
        do
        {
            require(!live.acquire(active, waiting), "unregistered worker received a polled task");
            std::this_thread::sleep_for(std::chrono::milliseconds(options.lease_ms / 4));
        } while (std::chrono::steady_clock::now() < until);
        rejects([&] { (void)abandoned.acquire(lost, missing); });
        rejects([&] { (void)live.acquire(active, same_owner); });
        worker.register_worker(offer("returned-worker", 49003));
        auto granted = live.acquire(active, waiting);
        require(granted && worker.begin(*granted, granted->owner_identity),
            "abandoned queued work consumed the returning worker's capacity");
        missing.attempt = 2;
        require(!abandoned.acquire(lost, missing), "queued retry bypassed the running owner's reservation");
        worker.finish(*granted);
        auto retry = abandoned.acquire(lost, missing);
        require(retry && worker.begin(*retry, retry->owner_identity), "expired queued task could not retry with a new attempt");
        rejects([&] { abandoned.cancel(lost, missing.task_id, 1); });
        worker.renew(*retry);
        worker.finish(*retry);
    }
    {
        options.max_tasks = 1;
        RunningBroker running(options);
        BrokerClient owner(running.server.endpoint(), owner_tls);
        auto session = owner.open_owner("expired-queue-capacity");
        require(!owner.acquire(session, task("abandoned")), "workerless task was granted");
        std::this_thread::sleep_for(std::chrono::milliseconds(options.lease_ms + 100));
        require(!owner.acquire(session, task("fresh", 2)), "workerless replacement task was granted");
        rejects([&] { (void)owner.acquire(session, task("abandoned")); });
    }
}
void automatic_lifecycle(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    options.max_owners = 1; options.lease_ms = 500;
    RunningBroker running(options);
    BrokerClient owner(running.server.endpoint(), owner_tls), worker(running.server.endpoint(), worker_tls);
    auto first = owner.open_owner("automatic-first", true);
    owner.close_owner(first);
    auto abandoned = owner.open_owner("automatic-abandoned", true);
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    auto current = owner.open_owner("automatic-current", true);
    rejects([&] { owner.close_owner(abandoned); });
    worker.register_worker(offer("departing", 44001));
    auto granted = owner.acquire(current, task("departing-task"));
    require(granted && worker.begin(*granted, granted->owner_identity), "departing worker task was not running");
    worker.unregister_worker("unrelated-instance");
    worker.renew(*granted); // A stale shutdown must not revoke a newer process.
    worker.unregister_worker("departing");
    require(!worker.heartbeat("departing", 1), "graceful worker departure retained registration");
    worker.register_worker(offer("replacement", 44002));
    auto next = owner.acquire(current, task("replacement-task", 2));
    require(next && next->worker_instance == "replacement", "departing worker retained capacity after shutdown");
    stale_lease_rejects([&] { worker.renew(*granted); });
    owner.close_owner(current);
    auto persistent = owner.open_owner("persistent");
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    (void)owner.generation_hint(persistent, task("hint"));
}
void restart(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    OwnerSession session; Lease lease;
    {
        RunningBroker running(options); options.listen = running.server.endpoint();
        BrokerClient owner(options.listen, owner_tls), worker(options.listen, worker_tls);
        worker.register_worker(offer("before-restart", 44001)); session = owner.open_owner("restart");
        auto granted = owner.acquire(session, task("member")); require(bool(granted), "pre-restart lease missing"); lease = *granted;
        require(worker.begin(lease, lease.owner_identity), "pre-restart lease did not begin");
    }
    RunningBroker running(options);
    BrokerClient owner(options.listen, owner_tls), worker(options.listen, worker_tls);
    require(!worker.heartbeat("before-restart", 1), "restarted broker retained an old worker registry");
    worker.register_worker(offer("after-restart", 44001));
    bool stale_session = false;
    try { (void)owner.acquire(session, task("member")); } catch (StaleOwnerSession const&) { stale_session = true; }
    require(stale_session, "broker restart did not produce a typed stale owner session");
    permanent_rejects([&] { (void)owner.begin(lease, lease.owner_identity); });
    stale_lease_rejects([&] { (void)worker.begin(lease, lease.owner_identity); });
    auto next = owner.open_owner("restart"); auto granted = owner.acquire(next, task("member"));
    require(granted && granted->epoch != lease.epoch && worker.begin(*granted, granted->owner_identity), "restart did not fence and reconcile work");
    worker.finish(*granted);
}
void live_status(BrokerOptions options, TlsOptions const& owner_tls, TlsOptions const& worker_tls)
{
    options.max_tasks = 128; options.max_owners = 128; options.lease_ms = 10000;
    RunningBroker live(options); auto endpoint = live.server.endpoint();
    BrokerClient owner(endpoint, owner_tls), worker(endpoint, worker_tls);
    auto session = owner.open_owner("status-owner");
    require(!owner.acquire(session, task("private-task-input")), "workerless status task unexpectedly ran");
    auto page = owner.status({StatusSection::Jobs});
    require(page.queued == 1 && page.job_records.size() == 1 && page.job_records.front().waiting == WaitReason::NoWorkers,
        "workerless queue status omitted its reason");
    auto offered = offer("status-worker", 9999); auto const first_epoch = worker.register_worker(offered);
    auto alternate = endpoint; alternate.server_name = "localhost";
    BrokerClient routed(alternate, worker_tls);
    require(routed.register_worker(offered) == first_epoch, "alternate route changed authenticated broker epoch");
    offered.producer_build = "source-build-a"; worker.register_worker(offered);
    offered.producer_build = "source-build-b"; worker.register_worker(offered);
    require(owner.status().worker_records.front().producer_build == offered.producer_build,
        "source provenance became a registration gate or was not retained for diagnostics");
    auto task_control = worker.with_deadline(std::chrono::steady_clock::now() + std::chrono::seconds(5));
    auto const before_reroute = worker.metrics().connections; worker.reroute(alternate);
    require(task_control.heartbeat(offered.instance, offered.capacity_revision) && worker.metrics().connections == before_reroute + 1,
        "task control copy did not follow authenticated broker route replacement");
    auto lease = owner.acquire(session, task("private-task-input")); require(bool(lease), "status fixture was not granted");
    require(!worker.retire_worker(offered.instance), "idle rotation revoked an unused granted lease");
    require(lease->broker_identity == certificate_identity(options.tls.cert_file), "lease omitted its authenticated broker authority");
    auto altered_authority = *lease; altered_authority.broker_identity = certificate_identity(owner_tls.cert_file);
    permanent_rejects([&] { worker.begin(altered_authority, altered_authority.owner_identity); });
    altered_authority = *lease; altered_authority.broker_node_id = "altered-node";
    permanent_rejects([&] { worker.begin(altered_authority, altered_authority.owner_identity); });
    auto second = task("second-status-task", 2); require(!owner.acquire(session, second), "status fixture exceeded one slot");
    page = owner.status({StatusSection::Jobs}); bool found_reserved = false;
    for (auto const& job : page.job_records) found_reserved |= job.state == JobState::Queued && job.waiting == WaitReason::SlotReserved;
    require(found_reserved, "status did not identify the reserved slot");
    auto workers = owner.status(); require(workers.worker_records.size() == 1 && workers.worker_records[0].reserved == lease->task.resources &&
        workers.worker_records[0].capacity == offered.capacity && workers.worker_records[0].endpoint == offered.endpoint,
        "status confused broker reservations with registered worker capacity");
    worker.worker_admission(offered.instance, WorkerAdmission::Paused);
    require(!worker.begin(*lease, lease->owner_identity), "paused broker admitted an unused grant");
    page = owner.status({StatusSection::Jobs}); bool found_paused = false;
    for (auto const& job : page.job_records) found_paused |= job.state == JobState::Queued && job.waiting == WaitReason::WorkerPaused;
    require(found_paused, "paused queue status omitted its reason");
    rejects([&] { owner.worker_admission(offered.instance, WorkerAdmission::Accepting); });
    worker.worker_admission(offered.instance, WorkerAdmission::Accepting);
    require(worker.begin(*lease, lease->owner_identity), "resumed broker did not admit existing grant");
    require(!worker.retire_worker(offered.instance), "idle rotation revoked running work");
    page = owner.status({StatusSection::Jobs}); require(page.running == 1 && page.queued == 1, "live status lost running work");
    // The diagnostic wire must never expose the private lease/session keys or
    // task identifiers, even though this authenticated peer may read status.
    auto channel = Channel::connect(endpoint, owner_tls); Writer query(65536);
    query.u32(protocol_version); query.u8(15); query.u8(uint8_t(StatusSection::Jobs)); query.u64(0); query.u32(max_status_page_size);
    channel.send(Message::Control, query.data()); auto raw = channel.receive(); require(raw.type == Message::Reply, "status wire request failed");
    std::string encoded(raw.payload.begin(), raw.payload.end());
    require(encoded.find(session.token) == std::string::npos && encoded.find(lease->token) == std::string::npos &&
        encoded.find(lease->epoch) == std::string::npos && encoded.find("private-task-input") == std::string::npos,
        "status leaked private task or authorization material");
    worker.finish(*lease);
    owner.cancel(session, second.task_id, second.attempt);
    require(worker.retire_worker(offered.instance) && owner.status().worker_records.empty(), "idle authority did not retire after reservations finished");
    for (unsigned index = 0; index < 40; ++index) owner.open_owner("paged-owner-" + std::to_string(index));
    uint64_t cursor = 0, seen = 0;
    do { auto owners = owner.status({StatusSection::Owners, cursor, 7}); require(owners.owner_records.size() <= 7, "unbounded owner status page");
        seen += owners.owner_records.size(); cursor = owners.next_cursor; } while (cursor);
    require(seen == 41, "status pagination omitted or repeated owners");
    rejects([&] { owner.status({StatusSection::Jobs, 0, max_status_page_size + 1}); });
    rejects([&] { owner.status({StatusSection::Jobs, 0, 0}); });
    require(owner.status({StatusSection::Jobs, UINT64_MAX}).job_records.empty(), "status cursor overflow was not bounded");
}
void queued_capacity_status(BrokerOptions options, TlsOptions const& first_tls, TlsOptions const& second_tls,
                            TlsOptions const& worker_tls)
{
    options.lease_ms = 10000; options.worker_ms = 1000; options.max_tasks = 64;
    RunningBroker live(options); auto endpoint = live.server.endpoint();
    BrokerClient first(endpoint, first_tls), second(endpoint, second_tls), worker(endpoint, worker_tls);
    auto a = first.open_owner("queued-capacity-a"), b = second.open_owner("queued-capacity-b");
    struct Request { BrokerClient* client; OwnerSession owner; Task task; };
    std::vector<Request> requests;
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        requests.push_back({&first, a, task("a-" + std::to_string(sequence), sequence)});
        requests.push_back({&second, b, task("b-" + std::to_string(sequence), sequence)});
    }
    for (auto const& request : requests)
        require(!request.client->acquire(request.owner, request.task), "workerless queue unexpectedly received a grant");
    auto advertised = offer("queued-capacity-worker", 49509); advertised.capacity = {3, 3, 3072, 0};
    worker.register_worker(advertised);
    auto check_status = [&]
    {
        auto page = first.status();
        require(page.worker_records.size() == 1 && page.worker_records.front().reserved == advertised.capacity,
            "scheduling a queued wave lost or exceeded the worker's reservations");
        auto owners = first.status({StatusSection::Owners});
        uint64_t queued = 0, granted = 0, running = 0, cancelled = 0;
        for (auto const& owner : owners.owner_records)
        { queued += owner.queued; granted += owner.granted; running += owner.running; cancelled += owner.cancelled; }
        require(owners.owner_records.size() == 2 && queued == page.queued && granted == page.granted &&
            running == page.running && cancelled == page.cancelled,
            "owner status counters disagree with live reservations");
        return page;
    };
    auto page = check_status();
    require(page.granted == 3 && page.queued == 3, "one scheduling pass overcommitted a queued wave");
    std::vector<std::pair<Request const*, Lease>> running;
    for (auto const& request : requests)
        if (auto grant = request.client->acquire(request.owner, request.task))
        {
            require(worker.begin(*grant, grant->owner_identity), "queued wave grant did not begin");
            running.emplace_back(&request, *grant);
        }
    require(running.size() == 3, "queued wave did not fill the advertised capacity");
    auto const& [request, cancelled] = running.front();
    request->client->cancel(request->owner, request->task.task_id, request->task.attempt);
    page = check_status();
    require(page.cancelled == 1 && page.running == 2 && page.queued == 3,
        "cancelled running work released capacity before worker acknowledgement");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    advertised.instance = "queued-capacity-replacement"; worker.register_worker(advertised);
    page = check_status();
    require(page.cancelled == 1 && page.running == 2 && page.queued == 3,
        "a replacement worker escaped the old incarnation's live reservations");
    worker.finish(cancelled); page = check_status();
    require(!page.cancelled && page.running == 2 && page.granted == 1 && page.queued == 2,
        "worker acknowledgement did not replenish exactly the released reservation");
    auto jobs = first.status({StatusSection::Jobs});
    for (auto const& job : jobs.job_records)
        if (job.state == JobState::Granted)
            require(job.worker_instance == advertised.instance, "new grant reused a departed worker incarnation");
    auto const closed = running.back().first;
    closed->client->close_owner(closed->owner);
    auto owners = first.status({StatusSection::Owners}); jobs = first.status({StatusSection::Jobs});
    auto const survivor = closed->owner.owner_id == a.owner_id ? b.owner_id : a.owner_id;
    require(owners.owner_records.size() == 1 && owners.owner_records.front().owner_id == survivor,
        "closed owner retained live diagnostic counters");
    std::array<uint64_t, 4> counts{}; bool fenced = false;
    for (auto const& job : jobs.job_records)
    {
        if (job.owner_id == survivor)
        { counts[0] += job.state == JobState::Queued; counts[1] += job.state == JobState::Granted;
          counts[2] += job.state == JobState::Running; counts[3] += job.state == JobState::Cancelled; }
        if (job.owner_id == closed->owner.owner_id && job.state == JobState::Cancelled) fenced = true;
    }
    auto const& owner = owners.owner_records.front();
    require(fenced && owner.queued == counts[0] && owner.granted == counts[1] && owner.running == counts[2] && owner.cancelled == counts[3],
        "closed-owner reservations leaked into the surviving owner's status counters");
}
} // namespace
int main() try
{
    codecs(); Directory directory;
    auto ca = issue(directory.path, "ca"); auto broker_id = issue(directory.path, "broker", &ca);
    auto a = issue(directory.path, "owner-a", &ca), b = issue(directory.path, "owner-b", &ca);
    auto wa = issue(directory.path, "worker-a", &ca), wb = issue(directory.path, "worker-b", &ca);
    auto rogue_ca = issue(directory.path, "rogue-ca"), rogue = issue(directory.path, "rogue", &rogue_ca);
    auto cn = issue(directory.path, "localhost", &ca, false);
    auto broker_tls = tls(ca, broker_id), a_tls = tls(ca, a), b_tls = tls(ca, b);
    transport(broker_tls, a_tls, tls(ca, rogue), certificate_identity(broker_tls.cert_file));
    cn_only(tls(ca, cn), a_tls);
    task_deadlines(broker_tls, a_tls);
    BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = broker_tls;
    options.lease_ms = 800; options.worker_ms = 3000; options.max_tasks = 4;
    broker(options, a_tls, b_tls, tls(ca, wa), tls(ca, wb));
    selection(options, a_tls, tls(ca, wa), tls(ca, wb));
    concrete_resources(options, a_tls, tls(ca, wa));
    codec_window_limits(options, a_tls, tls(ca, wa), tls(ca, wb));
    retry_eligibility(options, a_tls, tls(ca, wa), tls(ca, wb));
    queued_eligibility(options, a_tls, tls(ca, wa), tls(ca, wb));
    automatic_generation_cpus(options, a_tls, tls(ca, wa), tls(ca, wb));
    capacity_refresh(options, a_tls, tls(ca, wa));
    worker_route_change(options, a_tls, tls(ca, wa));
    listen_interfaces(options, a_tls);
    mixed_backend_selection(options, a_tls, tls(ca, wa));
    batched_profiles(options, a_tls, tls(ca, wa));
    weighted_shares(options, a_tls, b_tls, tls(ca, wa));
    weighted_shares(options, a_tls, b_tls, tls(ca, wa), 3);
    idle_lending_fairness(options, a_tls, b_tls, tls(ca, wa));
    brief_idle_preserves_debt(options, a_tls, b_tls, tls(ca, wa));
    persistent_control(options, a_tls, tls(ca, wa));
    control_faults(options, a_tls);
    stale_delivery(options, a_tls, tls(ca, wa));
    compact_heartbeat(options, a_tls, tls(ca, wa));
    queued_expiry(options, a_tls, b_tls, tls(ca, wa));
    automatic_lifecycle(options, a_tls, tls(ca, wa));
    restart(options, a_tls, tls(ca, wa));
    live_status(options, a_tls, tls(ca, wa));
    queued_capacity_status(options, a_tls, b_tls, tls(ca, wa));
    std::cout << "network pool transport, authentication, fairness, fencing, and recovery tests passed\n"; return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
