// Bounded authenticated transport for shared compute.
#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace group_grow::pool
{
// A connection failure or progress timeout may be retried by the owner under
// its original task deadline. Authentication, TLS alerts, framing and content
// errors use permanent exceptions. No transport operation replays a request.
class TransientTransport : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct Endpoint
{
    std::string host;
    uint16_t port = 0;
    // Discovery can supply a numeric route separately from the certificate's
    // stable DNS identity. This name is still verified against the pool CA.
    std::string server_name;
    bool operator==(Endpoint const&) const = default;
};
struct TlsOptions
{
    std::string ca_file, cert_file, key_file;
    // Per-operation socket/TLS/frame timeout. OS hostname resolution remains
    // synchronous; the task deadline cannot interrupt it.
    unsigned timeout_ms = 30000;
    uint32_t max_frame_bytes = 1048576;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    // When set, accepted connections must arrive on one of these interfaces.
    // Client connections are unaffected. Loopback can be included explicitly.
    std::vector<std::string> listen_interfaces;
    // Evaluate active LAN interfaces on each accept so automatic sharing
    // follows interface changes. Explicit names above are also allowed.
    bool listen_lan = false;
    // Saved enrollment enforces local revocation; policies are exchanged when
    // both authenticated peers advertise saved enrollment.
    std::string membership_directory;
    // Diagnostic probes enforce saved membership without learning/persisting
    // peer policies. No policy exchange is negotiated on these channels.
    bool read_only_membership = false;
};
// A numeric route avoids the synchronous hostname resolver. Scoped IPv6
// interface suffixes are local routing information and are accepted here.
bool numeric_host(std::string const&);
// Network integers are big-endian. Strings and blobs carry bounded lengths;
// decoders must call finish() to reject unrecognized trailing fields.
class Writer
{
public:
    explicit Writer(uint32_t limit = 1048576) : limit_(limit) {}
    void u8(uint8_t value);
    void u16(uint16_t value);
    void u32(uint32_t value);
    void u64(uint64_t value);
    void bytes(std::span<uint8_t const> value);
    void string(std::string const& value);
    std::vector<uint8_t> const& data() const { return data_; }
private:
    void reserve(size_t bytes);
    uint32_t limit_;
    std::vector<uint8_t> data_;
};
class Reader
{
public:
    explicit Reader(std::span<uint8_t const> data) : data_(data) {}
    uint8_t u8();
    uint16_t u16();
    uint32_t u32();
    uint64_t u64();
    std::span<uint8_t const> bytes(uint32_t limit);
    std::string string(uint32_t limit = 4096);
    void finish() const;
    size_t remaining() const { return data_.size() - position_; }
private:
    std::span<uint8_t const> take(size_t bytes);
    std::span<uint8_t const> data_;
    size_t position_ = 0;
};
enum class Message : uint16_t
{
    Control = 1, Reply = 2, Error = 3, Policy = 4,
    Work = 16, Result = 17, Data = 18, End = 19, Cancel = 20
};
struct Frame { Message type; std::vector<uint8_t> payload; };
// One owner thread per channel; move it to transfer ownership.
class Channel
{
public:
    Channel();
    ~Channel();
    Channel(Channel&&) noexcept;
    Channel& operator=(Channel&&) noexcept;
    Channel(Channel const&) = delete;
    Channel& operator=(Channel const&) = delete;
    // Hostname/IP SAN verification is mandatory. A granted worker connection
    // also pins the fingerprint from its authenticated broker registration.
    static Channel connect(Endpoint const&, TlsOptions const&,
                           std::string const& expected_peer = {},
                           std::chrono::steady_clock::time_point connect_deadline = std::chrono::steady_clock::time_point::max());
    void send(Message, std::span<uint8_t const> payload);
    Frame receive();
    // Shorten future operations on an exclusively owned channel; never extend
    // a deadline already imposed by its creator or an earlier task boundary.
    void set_deadline(std::chrono::steady_clock::time_point);
    // Reusable control channels get a fresh request deadline. Their immutable
    // creator deadline remains a ceiling; direct task channels use set_deadline.
    void reset_deadline(std::chrono::steady_clock::time_point);
    // A nonblocking readiness probe; receiving an incomplete frame still uses
    // the ordinary bounded frame deadline.
    bool readable() const;
    // Returns channels.size() if none is ready; one poll covers the registry.
    static size_t first_readable(std::span<Channel const* const> channels);
    // Detect a peer EOF already available on the nonblocking socket. This does
    // not consume application bytes or wait for a request/response.
    bool peer_closed();
    std::string const& peer_identity() const;
    std::string const& peer_node_id() const;
    explicit operator bool() const noexcept { return bool(impl_); }
    uint32_t max_frame_bytes() const;
    static uint64_t memory_bytes(uint32_t max_frame_bytes);
    void close() noexcept;
private:
    struct Impl;
    explicit Channel(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class Listener;
};
class Listener
{
public:
    Listener(Endpoint const&, TlsOptions const&, unsigned backlog = 32);
    ~Listener();
    Listener(Listener const&) = delete;
    Listener& operator=(Listener const&) = delete;
    // Polling accept returns an empty channel on timeout, allowing shutdown.
    // Each accepted handshake and each frame has one absolute deadline.
    Channel accept(unsigned poll_ms = 250);
    Endpoint endpoint() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::string random_token();
std::string certificate_identity(std::string const& cert_file);
// Load the ordinary TLS context and matching credentials without opening a
// listener or connection, for deterministic embedding preflight.
void validate_tls_credentials(TlsOptions const&);
std::string certificate_node_id(std::string const& cert_file);
Endpoint parse_endpoint(std::string const&);
// Address of the local interface used to reach this peer; sends no data.
std::string local_address(Endpoint const&);
} // namespace group_grow::pool
