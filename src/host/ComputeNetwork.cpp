#include "host/ComputeNetwork.hpp"
#include "host/ComputeIdentity.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#ifdef interface
#undef interface
#endif
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace group_grow::pool
{
namespace
{
void require(bool condition, char const* message)
{ if (!condition) throw std::runtime_error(message); }
void transport_require(bool condition, char const* message)
{ if (!condition) throw TransientTransport(message); }
#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle invalid_socket = INVALID_SOCKET;
using PollFd = WSAPOLLFD;
using AddressSize = int;
int socket_error() { return ::WSAGetLastError(); }
bool interrupted(int error) { return error == WSAEINTR; }
bool retry_socket(int error) { return error == WSAEWOULDBLOCK || error == WSAEINTR || error == WSAEINPROGRESS; }
void clear_socket_error() { ::WSASetLastError(0); errno = 0; }
void close_socket(SocketHandle socket) { ::closesocket(socket); }
int poll_sockets(PollFd* items, size_t count, int timeout)
{ if (!count) return 0; return ::WSAPoll(items, ULONG(count), timeout); }
void initialize_sockets()
{
    struct Winsock
    {
        Winsock() { WSADATA data{}; require(::WSAStartup(MAKEWORD(2, 2), &data) == 0, "cannot initialize Windows network sockets"); }
        ~Winsock() { ::WSACleanup(); }
    };
    static Winsock state;
}
#else
using SocketHandle = int;
constexpr SocketHandle invalid_socket = -1;
using PollFd = pollfd;
using AddressSize = socklen_t;
int socket_error() { return errno; }
bool interrupted(int error) { return error == EINTR; }
bool retry_socket(int error) { return error == EAGAIN || error == EWOULDBLOCK || error == EINTR || error == EINPROGRESS; }
void clear_socket_error() { errno = 0; }
void close_socket(SocketHandle socket) { ::close(socket); }
int poll_sockets(PollFd* items, size_t count, int timeout) { return ::poll(items, count, timeout); }
void initialize_sockets() {}
#endif
bool transient_errno(int error)
{
    switch (error)
    {
#ifdef _WIN32
    case WSAEWOULDBLOCK: case WSAECONNABORTED: case WSAECONNREFUSED: case WSAECONNRESET:
    case WSAEHOSTDOWN: case WSAEHOSTUNREACH: case WSAENETDOWN: case WSAENETRESET:
    case WSAENETUNREACH: case WSAENOBUFS: case WSAETIMEDOUT: case WSAEMFILE:
    case WSAENOTCONN: case WSAESHUTDOWN: return true;
#else
    case EAGAIN: case ECONNABORTED: case ECONNREFUSED: case ECONNRESET:
    case EHOSTDOWN: case EHOSTUNREACH: case ENETDOWN: case ENETRESET:
    case ENETUNREACH: case ENOBUFS: case EPIPE: case ETIMEDOUT:
    case EMFILE: case ENFILE: case ENOMEM: return true;
#endif
    default: return false;
    }
}
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
Deadline deadline(unsigned milliseconds)
{ return Clock::now() + std::chrono::milliseconds(milliseconds); }
Deadline operation_deadline(TlsOptions const& options)
{ return std::min(deadline(options.timeout_ms), options.deadline); }
void check_deadline(Deadline until)
{ transport_require(Clock::now() < until, "network operation timed out"); }
bool wait_fd(SocketHandle fd, short events, Deadline until, bool optional = false)
{
    for (;;)
    {
        auto const left = std::chrono::duration_cast<std::chrono::milliseconds>(until - Clock::now()).count();
        if (left <= 0)
        {
            if (optional) return false;
            throw TransientTransport("network operation timed out");
        }
        PollFd item{fd, events, 0};
        int const result = poll_sockets(&item, 1, int(std::min<int64_t>(left + 1, INT_MAX)));
        if (result < 0 && interrupted(socket_error())) continue;
        transport_require(result >= 0, "network poll failed");
        if (!result) continue;
        transport_require(!(item.revents & POLLNVAL), "network socket is closed");
        if (item.revents & (events | POLLERR | POLLHUP)) return true;
    }
}
void nonblocking(SocketHandle fd)
{
#ifdef _WIN32
    u_long enabled = 1; int const no_delay = 1;
    require(::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char const*>(&no_delay), sizeof(no_delay)) == 0 &&
        ::ioctlsocket(fd, FIONBIO, &enabled) == 0 && ::SetHandleInformation(reinterpret_cast<HANDLE>(fd), HANDLE_FLAG_INHERIT, 0),
        "cannot configure network socket");
#else
    int const flags = ::fcntl(fd, F_GETFL);
    int const no_delay = 1;
    require(::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay)) == 0,
            "cannot configure network TCP latency");
    require(flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 &&
        ::fcntl(fd, F_SETFD, FD_CLOEXEC) == 0, "cannot configure network socket");
#endif
}
struct Socket
{
    SocketHandle value = invalid_socket;
    ~Socket() { if (value != invalid_socket) close_socket(value); }
    SocketHandle release() { auto const result = value; value = invalid_socket; return result; }
};
bool allowed_interface(SocketHandle fd, TlsOptions const& options)
{
    auto const& names = options.listen_interfaces;
    if (names.empty() && !options.listen_lan) return true;
    sockaddr_storage local{}; AddressSize size = sizeof(local);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &size)) return false;
    if (local.ss_family == AF_INET6 && IN6_IS_ADDR_V4MAPPED(&reinterpret_cast<sockaddr_in6&>(local).sin6_addr))
    {
        in_addr ipv4{}; std::memcpy(&ipv4, reinterpret_cast<sockaddr_in6&>(local).sin6_addr.s6_addr + 12, sizeof(ipv4));
        local = {}; auto& value = reinterpret_cast<sockaddr_in&>(local); value.sin_family = AF_INET; value.sin_addr = ipv4;
    }
#ifdef _WIN32
    ULONG bytes = 16384; std::vector<uint8_t> storage(bytes); ULONG result = ERROR_BUFFER_OVERFLOW;
    for (unsigned attempt = 0; attempt < 3 && result == ERROR_BUFFER_OVERFLOW; ++attempt)
    {
        if (bytes > 1048576) return false;
        storage.resize(bytes);
        result = ::GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
            nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data()), &bytes);
    }
    if (result != NO_ERROR) return false;
    for (auto item = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data()); item; item = item->Next)
    {
        std::string friendly;
        if (item->FriendlyName)
        {
            auto size = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, item->FriendlyName, -1, nullptr, 0, nullptr, nullptr);
            if (size > 0 && size <= 1024)
            { friendly.resize(size_t(size)); ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, item->FriendlyName, -1, friendly.data(), size, nullptr, nullptr); friendly.pop_back(); }
        }
        bool const lan = options.listen_lan && item->OperStatus == IfOperStatusUp && !(item->Flags & IP_ADAPTER_NO_MULTICAST) &&
            item->IfType != IF_TYPE_SOFTWARE_LOOPBACK && item->IfType != IF_TYPE_PPP && item->IfType != IF_TYPE_TUNNEL;
        auto index = std::to_string(local.ss_family == AF_INET6 ? item->Ipv6IfIndex : item->IfIndex);
        bool const named = std::any_of(names.begin(), names.end(), [&](auto const& name)
        { return name == item->AdapterName || name == friendly || name == index || (name == "lo" && item->IfType == IF_TYPE_SOFTWARE_LOOPBACK); });
        if (!lan && !named) continue;
        for (auto address = item->FirstUnicastAddress; address; address = address->Next)
        {
            auto value = address->Address.lpSockaddr;
            if (!value || value->sa_family != local.ss_family) continue;
            if (local.ss_family == AF_INET && reinterpret_cast<sockaddr_in&>(local).sin_addr.s_addr == reinterpret_cast<sockaddr_in*>(value)->sin_addr.s_addr) return true;
            if (local.ss_family == AF_INET6 && !std::memcmp(&reinterpret_cast<sockaddr_in6&>(local).sin6_addr, &reinterpret_cast<sockaddr_in6*>(value)->sin6_addr, sizeof(in6_addr))) return true;
        }
    }
#else
    ifaddrs* interfaces = nullptr;
    if (::getifaddrs(&interfaces)) return false;
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> cleanup(interfaces, freeifaddrs);
    for (auto item = interfaces; item; item = item->ifa_next)
    {
        bool const lan = options.listen_lan && (item->ifa_flags & IFF_UP) &&
            (item->ifa_flags & IFF_MULTICAST) && !(item->ifa_flags & (IFF_LOOPBACK | IFF_POINTOPOINT));
        if (!item->ifa_addr || item->ifa_addr->sa_family != local.ss_family ||
            (!lan && std::find(names.begin(), names.end(), item->ifa_name) == names.end())) continue;
        if (local.ss_family == AF_INET && reinterpret_cast<sockaddr_in&>(local).sin_addr.s_addr ==
            reinterpret_cast<sockaddr_in*>(item->ifa_addr)->sin_addr.s_addr) return true;
        if (local.ss_family == AF_INET6 && !std::memcmp(&reinterpret_cast<sockaddr_in6&>(local).sin6_addr,
            &reinterpret_cast<sockaddr_in6*>(item->ifa_addr)->sin6_addr, sizeof(in6_addr))) return true;
    }
#endif
    return false;
}
// OpenSSL's default socket BIO writes can raise SIGPIPE on Linux. Use the
// socket flag here so the library does not change the process signal policy.
int socket_bio_write(BIO* bio, char const* bytes, int count)
{
    BIO_clear_retry_flags(bio);
    auto const fd = SocketHandle(reinterpret_cast<uintptr_t>(BIO_get_data(bio)));
#ifdef _WIN32
    int const result = ::send(fd, bytes, count, 0);
#else
    int const result = int(::send(fd, bytes, size_t(count), MSG_NOSIGNAL));
#endif
    if (result < 0 && retry_socket(socket_error())) BIO_set_retry_write(bio);
    return result;
}
int socket_bio_read(BIO* bio, char* bytes, int count)
{
    BIO_clear_retry_flags(bio);
    auto const fd = SocketHandle(reinterpret_cast<uintptr_t>(BIO_get_data(bio)));
    int const result = int(::recv(fd, bytes, count, 0));
    if (result < 0 && retry_socket(socket_error())) BIO_set_retry_read(bio);
    return result;
}
long socket_bio_control(BIO* bio, int command, long, void* pointer)
{
    if (command == BIO_C_GET_FD)
    {
        int const fd = int(reinterpret_cast<uintptr_t>(BIO_get_data(bio)));
        if (pointer) *static_cast<int*>(pointer) = fd;
        return fd;
    }
    if (command == BIO_CTRL_FLUSH) return 1;
    return 0;
}
BIO_METHOD const* socket_bio_method()
{
    static std::unique_ptr<BIO_METHOD, decltype(&BIO_meth_free)> method([]
    {
        auto result = BIO_meth_new(BIO_TYPE_SOURCE_SINK | BIO_TYPE_DESCRIPTOR | BIO_get_new_index(), "pool socket");
        if (!result) return static_cast<BIO_METHOD*>(nullptr);
        if (BIO_meth_set_write(result, socket_bio_write) != 1 || BIO_meth_set_read(result, socket_bio_read) != 1 ||
            BIO_meth_set_ctrl(result, socket_bio_control) != 1)
        { BIO_meth_free(result); return static_cast<BIO_METHOD*>(nullptr); }
        return result;
    }(), BIO_meth_free);
    require(bool(method), "cannot create TLS socket transport"); return method.get();
}
void set_socket(SSL* ssl, SocketHandle fd)
{
    BIO* bio = BIO_new(socket_bio_method()); require(bio, "cannot create TLS socket transport");
    BIO_set_data(bio, reinterpret_cast<void*>(uintptr_t(fd))); BIO_set_init(bio, 1);
    SSL_set_bio(ssl, bio, bio);
}
using Context = std::shared_ptr<SSL_CTX>;
Context context(TlsOptions const& options, bool server)
{
    require(!options.ca_file.empty() && !options.cert_file.empty() && !options.key_file.empty(),
            "network TLS requires CA, certificate, and private-key files");
    require(options.timeout_ms && options.timeout_ms <= 3600000 && options.max_frame_bytes &&
        options.max_frame_bytes <= 67108864, "invalid network timeout or frame limit");
    Context result(SSL_CTX_new(TLS_method()), SSL_CTX_free);
    require(bool(result), "cannot create TLS context");
    require(SSL_CTX_set_min_proto_version(result.get(), TLS1_3_VERSION) == 1,
            "cannot require TLS 1.3");
    SSL_CTX_set_verify(result.get(), SSL_VERIFY_PEER |
        (server ? SSL_VERIFY_FAIL_IF_NO_PEER_CERT : 0), nullptr);
    SSL_CTX_set_verify_depth(result.get(), 4);
    SSL_CTX_set_max_cert_list(result.get(), 65536);
    SSL_CTX_set_options(result.get(), SSL_OP_NO_COMPRESSION | SSL_OP_NO_RENEGOTIATION);
    SSL_CTX_set_session_cache_mode(result.get(), SSL_SESS_CACHE_OFF);
    require(SSL_CTX_load_verify_locations(result.get(), options.ca_file.c_str(), nullptr) == 1,
            "cannot load network TLS CA");
    require(SSL_CTX_use_certificate_chain_file(result.get(), options.cert_file.c_str()) == 1 &&
        SSL_CTX_use_PrivateKey_file(result.get(), options.key_file.c_str(), SSL_FILETYPE_PEM) == 1 &&
        SSL_CTX_check_private_key(result.get()) == 1, "cannot load matching network TLS certificate and key");
    return result;
}
std::string hex(std::span<unsigned char const> bytes)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (auto value : bytes) { result.push_back(digits[value >> 4]); result.push_back(digits[value & 15]); }
    return result;
}
std::string fingerprint(X509* certificate)
{
    require(certificate, "network peer has no certificate");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned bytes = 0;
    require(X509_digest(certificate, EVP_sha256(), digest.data(), &bytes) == 1,
            "cannot identify network certificate");
    return hex(std::span(digest).first(bytes));
}
std::string node_identity(X509* certificate)
{
    require(certificate, "network peer has no certificate");
    auto name = X509_get_subject_name(certificate); auto index = X509_NAME_get_index_by_NID(name, NID_commonName, -1);
    if (index < 0 || X509_NAME_get_index_by_NID(name, NID_commonName, index) >= 0) return {};
    auto value = X509_NAME_ENTRY_get_data(X509_NAME_get_entry(name, index));
    auto data = ASN1_STRING_get0_data(value); auto size = ASN1_STRING_length(value);
    if (size != 32) return {};
    std::string node(reinterpret_cast<char const*>(data), size_t(size));
    if (node.find_first_not_of("0123456789abcdef") != node.npos) return {};
    auto host = "xchplot2-" + node + ".local";
    if (X509_check_host(certificate, host.c_str(), host.size(), X509_CHECK_FLAG_NEVER_CHECK_SUBJECT, nullptr) != 1) return {};
    return node;
}
[[noreturn]] void ssl_failure(SSL* ssl, int result, int error, int socket_error)
{
    require(SSL_get_verify_result(ssl) == X509_V_OK, "network certificate verification failed");
    if (error == SSL_ERROR_ZERO_RETURN)
        throw TransientTransport("authenticated network connection failed or closed");
    if (error == SSL_ERROR_SSL || error == SSL_ERROR_SYSCALL)
    {
        // OpenSSL 3 reports an unauthenticated TCP EOF through its SSL error
        // queue. Explicit TLS alerts and other protocol errors stay permanent.
        bool eof = false, system = false, other = false;
        while (auto const code = ERR_get_error())
        {
            if (ERR_GET_LIB(code) == ERR_LIB_SSL && ERR_GET_REASON(code) == SSL_R_UNEXPECTED_EOF_WHILE_READING) eof = true;
            else if (ERR_GET_LIB(code) == ERR_LIB_SYS && transient_errno(ERR_GET_REASON(code))) system = true;
            else other = true;
        }
        if (eof && !other) throw TransientTransport("authenticated network connection closed unexpectedly");
        if (error == SSL_ERROR_SYSCALL && !other && (result == 0 || system || transient_errno(socket_error)))
            throw TransientTransport("authenticated network connection failed or closed");
    }
    throw std::runtime_error("authenticated network TLS connection failed");
}
void ssl_wait(SSL* ssl, int result, Deadline until)
{
    int const error_number = socket_error();
    int const error = SSL_get_error(ssl, result);
    auto socket = SocketHandle(reinterpret_cast<uintptr_t>(BIO_get_data(SSL_get_rbio(ssl))));
    if (error == SSL_ERROR_WANT_READ) { wait_fd(socket, POLLIN, until); return; }
    if (error == SSL_ERROR_WANT_WRITE) { wait_fd(socket, POLLOUT, until); return; }
    ssl_failure(ssl, result, error, error_number);
}
void handshake(SSL* ssl, bool server, Deadline until)
{
    for (;;)
    {
        check_deadline(until); ERR_clear_error(); clear_socket_error();
        int const result = server ? SSL_accept(ssl) : SSL_connect(ssl);
        if (result == 1) { check_deadline(until); break; }
        ssl_wait(ssl, result, until);
    }
    require(SSL_get_verify_result(ssl) == X509_V_OK, "network certificate verification failed");
}
bool valid_message(uint16_t value)
{
    switch (Message(value))
    {
    case Message::Control: case Message::Reply: case Message::Error: case Message::Policy:
    case Message::Work: case Message::Result: case Message::Data: case Message::End: case Message::Cancel:
        return true;
    }
    return false;
}
} // namespace

void Writer::reserve(size_t bytes)
{
    require(bytes <= limit_ && data_.size() <= limit_ - bytes, "network message exceeds its byte limit");
    auto const required = data_.size() + bytes;
    if (required > data_.capacity())
    {
        auto const grown = std::min<size_t>(limit_, std::max<size_t>(64, data_.capacity() * 2));
        data_.reserve(std::max(required, grown));
    }
}
void Writer::u8(uint8_t value) { reserve(1); data_.push_back(value); }
void Writer::u16(uint16_t value) { reserve(2); u8(uint8_t(value >> 8)); u8(uint8_t(value)); }
void Writer::u32(uint32_t value) { reserve(4); u16(uint16_t(value >> 16)); u16(uint16_t(value)); }
void Writer::u64(uint64_t value) { reserve(8); u32(uint32_t(value >> 32)); u32(uint32_t(value)); }
void Writer::bytes(std::span<uint8_t const> value)
{
    require(value.size() <= UINT32_MAX, "network blob is too large");
    reserve(value.size() + 4); u32(uint32_t(value.size()));
    data_.insert(data_.end(), value.begin(), value.end());
}
void Writer::string(std::string const& value)
{ bytes({reinterpret_cast<uint8_t const*>(value.data()), value.size()}); }
std::span<uint8_t const> Reader::take(size_t bytes)
{
    require(bytes <= remaining(), "truncated network message");
    auto result = data_.subspan(position_, bytes); position_ += bytes; return result;
}
uint8_t Reader::u8() { return take(1)[0]; }
uint16_t Reader::u16() { uint16_t const high = u8(); return uint16_t((high << 8) | u8()); }
uint32_t Reader::u32() { uint32_t const high = u16(); return (high << 16) | u16(); }
uint64_t Reader::u64() { uint64_t const high = u32(); return (high << 32) | u32(); }
std::span<uint8_t const> Reader::bytes(uint32_t limit)
{ uint32_t const size = u32(); require(size <= limit, "network blob exceeds its byte limit"); return take(size); }
std::string Reader::string(uint32_t limit)
{
    auto const value = bytes(limit);
    require(std::find(value.begin(), value.end(), uint8_t(0)) == value.end(), "network string contains NUL");
    if (value.empty()) return {};
    return {reinterpret_cast<char const*>(value.data()), value.size()};
}
void Reader::finish() const { require(!remaining(), "network message has trailing bytes"); }

struct Channel::Impl
{
    Context context;
    SSL* ssl = nullptr;
    SocketHandle fd = invalid_socket;
    TlsOptions options;
    Deadline creator_deadline = Deadline::max();
    std::string peer;
    std::string peer_node, local_node, local_certificate;
    std::vector<uint8_t> sent_policy;
    bool policy_exchange = false;
    ~Impl() { if (ssl) SSL_free(ssl); if (fd != invalid_socket) close_socket(fd); }
    void transfer(std::span<uint8_t> bytes, bool write, Deadline until)
    {
        size_t position = 0;
        while (position < bytes.size())
        {
            if (!options.membership_directory.empty() && !membership_exchange) check_membership();
            check_deadline(until); size_t moved = 0;
            ERR_clear_error(); clear_socket_error();
            int const result = write ? SSL_write_ex(ssl, bytes.data() + position, bytes.size() - position, &moved) :
                SSL_read_ex(ssl, bytes.data() + position, bytes.size() - position, &moved);
            if (result == 1) { transport_require(moved, "network transfer made no progress"); position += moved; }
            else ssl_wait(ssl, result, until);
            transport_require(Clock::now() <= until, "network operation timed out");
        }
    }
    void check_membership() const
    {
        check_pool_peer(options.membership_directory, peer_node, peer);
        check_pool_peer(options.membership_directory, local_node, local_certificate);
    }
    void policy_blob(bool write, Deadline until)
    {
        constexpr uint32_t limit = 262144;
        std::array<uint8_t, 4> header{};
        if (write)
        {
            auto policy = pool_policy(options.membership_directory); Writer size(4); size.u32(uint32_t(policy.size()));
            transfer({const_cast<uint8_t*>(size.data().data()), size.data().size()}, true, until);
            transfer(policy, true, until); sent_policy = std::move(policy);
        }
        else
        {
            transfer(header, false, until); Reader input(header); auto size = input.u32();
            require(size <= limit, "network pool policy exceeds its byte limit");
            std::vector<uint8_t> policy(size); transfer(policy, false, until); accept_pool_policy(options.membership_directory, policy);
        }
    }
    void enroll(bool server, Deadline until)
    {
        std::unique_ptr<X509, decltype(&X509_free)> certificate(SSL_get1_peer_certificate(ssl), X509_free);
        peer_node = node_identity(certificate.get());
        local_node = node_identity(SSL_get_certificate(ssl)); local_certificate = fingerprint(SSL_get_certificate(ssl));
        // Learn a newer signed policy before applying its revocations. A peer
        // cannot relax the pinned authority's policy, even with expired data.
        // An old local policy may reject a peer that has just rotated. Defer
        // membership checks only during this bounded signed-policy exchange.
        membership_exchange = true;
        try
        {
            Writer offer(5); offer.u32(0x47504d34); offer.u8(!options.membership_directory.empty());
            std::array<uint8_t, 5> remote{};
            auto send_offer = [&] { transfer({const_cast<uint8_t*>(offer.data().data()), offer.data().size()}, true, until); };
            if (server) { transfer(remote, false, until); send_offer(); }
            else { send_offer(); transfer(remote, false, until); }
            Reader input(remote); require(input.u32() == 0x47504d34, "incompatible network membership protocol");
            auto saved = input.u8(); require(saved <= 1, "invalid network membership capability"); input.finish();
            policy_exchange = saved && !options.membership_directory.empty();
            if (policy_exchange)
            {
                auto policy = pool_policy(options.membership_directory);
                std::array<uint8_t, 32> local_digest{}, remote_digest{}; unsigned size = 0;
                require(EVP_Digest(policy.data(), policy.size(), local_digest.data(), &size, EVP_sha256(), nullptr) == 1 && size == local_digest.size(),
                    "cannot identify signed pool policy");
                if (server) { transfer(remote_digest, false, until); transfer(local_digest, true, until); }
                else { transfer(local_digest, true, until); transfer(remote_digest, false, until); }
                if (local_digest != remote_digest) { policy_blob(!server, until); policy_blob(server, until); }
                else sent_policy = std::move(policy);
            }
        }
        catch (...) { membership_exchange = false; throw; }
        membership_exchange = false;
        if (!options.membership_directory.empty()) check_membership();
    }
    bool membership_exchange = false;
};
Channel::Channel() = default;
Channel::Channel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Channel::~Channel() = default;
Channel::Channel(Channel&&) noexcept = default;
Channel& Channel::operator=(Channel&&) noexcept = default;
Channel Channel::connect(Endpoint const& endpoint, TlsOptions const& options, std::string const& expected_peer, Deadline connect_deadline)
{
    require(!endpoint.host.empty() && endpoint.port, "network endpoint requires host and port");
    initialize_sockets();
    auto const ceiling = std::min(options.deadline, connect_deadline);
    check_deadline(ceiling);
    auto impl = std::make_unique<Impl>(); impl->options = options; impl->creator_deadline = options.deadline;
    impl->context = context(options, false);
    check_deadline(ceiling);
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    int const resolved = ::getaddrinfo(endpoint.host.c_str(), std::to_string(endpoint.port).c_str(), &hints, &addresses);
    if (resolved == EAI_AGAIN
#ifndef _WIN32
        || (resolved == EAI_SYSTEM && transient_errno(errno))
#endif
       )
        throw TransientTransport("cannot resolve network endpoint temporarily");
    require(resolved == 0, "cannot resolve network endpoint");
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> cleanup(addresses, freeaddrinfo);
    auto const until = std::min(operation_deadline(options), ceiling);
    check_deadline(until);
    bool temporary = false;
    for (auto item = addresses; item; item = item->ai_next)
    {
        check_deadline(until);
        Socket socket{::socket(item->ai_family, item->ai_socktype, item->ai_protocol)};
        if (socket.value == invalid_socket) { temporary = temporary || transient_errno(socket_error()); continue; }
        nonblocking(socket.value);
        if (::connect(socket.value, item->ai_addr, item->ai_addrlen) != 0)
        {
            if (!retry_socket(socket_error())) { temporary = temporary || transient_errno(socket_error()); continue; }
            wait_fd(socket.value, POLLOUT, until);
            int error = 0; AddressSize size = sizeof(error);
            if (::getsockopt(socket.value, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &size) != 0)
            { temporary = temporary || transient_errno(socket_error()); continue; }
            if (error) { temporary = temporary || transient_errno(error); continue; }
        }
        impl->fd = socket.release(); break;
    }
    if (impl->fd == invalid_socket && temporary) throw TransientTransport("cannot connect to network endpoint");
    require(impl->fd != invalid_socket, "cannot connect to network endpoint");
    impl->ssl = SSL_new(impl->context.get());
    require(impl->ssl, "cannot initialize TLS connection"); set_socket(impl->ssl, impl->fd);
    auto const& server_name = endpoint.server_name.empty() ? endpoint.host : endpoint.server_name;
    std::array<uint8_t, 16> binary{};
    bool const numeric = ::inet_pton(AF_INET, server_name.c_str(), binary.data()) == 1 ||
        ::inet_pton(AF_INET6, server_name.c_str(), binary.data()) == 1;
    if (numeric)
        require(X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(impl->ssl), server_name.c_str()) == 1,
                "cannot set TLS IP verification");
    else
    {
        X509_VERIFY_PARAM_set_hostflags(SSL_get0_param(impl->ssl), X509_CHECK_FLAG_NEVER_CHECK_SUBJECT);
        require(SSL_set_tlsext_host_name(impl->ssl, server_name.c_str()) == 1 &&
            SSL_set1_host(impl->ssl, server_name.c_str()) == 1, "cannot set TLS hostname verification");
    }
    handshake(impl->ssl, false, until);
    std::unique_ptr<X509, decltype(&X509_free)> peer(SSL_get1_peer_certificate(impl->ssl), X509_free);
    impl->peer = fingerprint(peer.get());
    require(expected_peer.empty() || expected_peer == impl->peer, "network worker certificate differs from its lease");
    impl->enroll(false, until);
    check_deadline(until); return Channel(std::move(impl));
}
void Channel::send(Message type, std::span<uint8_t const> payload)
{
    require(bool(impl_), "network channel is closed");
    auto const until = operation_deadline(impl_->options);
    require(payload.size() <= impl_->options.max_frame_bytes && valid_message(uint16_t(type)) && type != Message::Policy, "invalid network frame");
    if (!impl_->options.membership_directory.empty()) impl_->check_membership();
    if (impl_->policy_exchange)
    {
        auto policy = pool_policy(impl_->options.membership_directory);
        if (policy != impl_->sent_policy)
        {
            Writer update(12); update.u32(0x47504331); update.u16(1); update.u16(uint16_t(Message::Policy)); update.u32(uint32_t(policy.size()));
            impl_->transfer({const_cast<uint8_t*>(update.data().data()), update.data().size()}, true, until);
            impl_->transfer(policy, true, until); impl_->sent_policy = std::move(policy);
        }
    }
    Writer header(12); header.u32(0x47504331); header.u16(1); header.u16(uint16_t(type)); header.u32(uint32_t(payload.size()));
    // Control frames fit one TLS write without a heap allocation. Bulk frames
    // keep their existing no-copy payload path.
    if (payload.size() <= 4096)
    {
        std::array<uint8_t, 4108> bytes;
        std::copy(header.data().begin(), header.data().end(), bytes.begin());
        std::copy(payload.begin(), payload.end(), bytes.begin() + header.data().size());
        impl_->transfer(std::span(bytes).first(header.data().size() + payload.size()), true, until); return;
    }
    impl_->transfer({const_cast<uint8_t*>(header.data().data()), header.data().size()}, true, until);
    impl_->transfer({const_cast<uint8_t*>(payload.data()), payload.size()}, true, until);
}
Frame Channel::receive()
{
    require(bool(impl_), "network channel is closed");
    auto const until = operation_deadline(impl_->options);
    unsigned updates = 0;
    for (;;)
    {
        if (!impl_->options.membership_directory.empty()) impl_->check_membership();
        std::array<uint8_t, 12> header{};
        impl_->transfer(header, false, until);
        Reader reader(header);
        require(reader.u32() == 0x47504331 && reader.u16() == 1, "incompatible network frame protocol");
        auto const type = reader.u16(); auto const bytes = reader.u32(); reader.finish();
        bool const policy = Message(type) == Message::Policy;
        require(valid_message(type) && bytes <= (policy ? 262144 : impl_->options.max_frame_bytes) &&
            (!policy || (impl_->policy_exchange && ++updates <= 4)), "invalid network frame type or size");
        Frame result{Message(type), std::vector<uint8_t>(bytes)};
        impl_->transfer(result.payload, false, until); check_deadline(until);
        if (policy) { accept_pool_policy(impl_->options.membership_directory, result.payload); impl_->check_membership(); continue; }
        if (!impl_->options.membership_directory.empty()) impl_->check_membership();
        return result;
    }
}
void Channel::set_deadline(std::chrono::steady_clock::time_point until)
{
    require(bool(impl_), "network channel is closed");
    impl_->options.deadline = std::min(impl_->options.deadline, until);
}
void Channel::reset_deadline(Deadline until)
{
    require(bool(impl_), "network channel is closed");
    impl_->options.deadline = std::min(impl_->creator_deadline, until);
}
bool Channel::readable() const
{
    Channel const* channel = this; return first_readable({&channel, 1}) == 0;
}
size_t Channel::first_readable(std::span<Channel const* const> channels)
{
    require(channels.size() <= 4096, "too many channels in readiness poll");
    std::array<PollFd, 4096> items;
    for (size_t i = 0; i < channels.size(); ++i)
    {
        require(channels[i] && channels[i]->impl_, "network channel is closed");
        if (SSL_pending(channels[i]->impl_->ssl)) return i;
        items[i] = {channels[i]->impl_->fd, POLLIN, 0};
    }
    int result;
    do { result = poll_sockets(items.data(), channels.size(), 0); } while (result < 0 && interrupted(socket_error()));
    transport_require(result >= 0, "network readiness poll failed");
    if (result) for (size_t i = 0; i < channels.size(); ++i) if (items[i].revents) return i;
    return channels.size();
}
bool Channel::peer_closed()
{
    if (!readable()) return false;
    uint8_t byte; size_t count = 0; ERR_clear_error(); clear_socket_error();
    int const result = SSL_peek_ex(impl_->ssl, &byte, 1, &count);
    if (result == 1) return false;
    int const error_number = socket_error();
    int const error = SSL_get_error(impl_->ssl, result);
    if (error == SSL_ERROR_WANT_READ || error == SSL_ERROR_WANT_WRITE) return false;
    try { ssl_failure(impl_->ssl, result, error, error_number); }
    catch (TransientTransport const&) { return true; }
}
std::string const& Channel::peer_identity() const
{ require(bool(impl_), "network channel is closed"); return impl_->peer; }
std::string const& Channel::peer_node_id() const
{ require(bool(impl_), "network channel is closed"); return impl_->peer_node; }
uint32_t Channel::max_frame_bytes() const
{ require(bool(impl_), "network channel is closed"); return impl_->options.max_frame_bytes; }
uint64_t Channel::memory_bytes(uint32_t bytes)
{
    // Budget one full receive frame and TLS/socket/certificate scratch. Chain
    // size and depth are bounded in context(); reserve additional scratch for
    // the saved credential bundle, signed policy and member registry. This is
    // admission allowance, rather than a measurement of OpenSSL resident pages.
    return uint64_t(bytes) + 4194304;
}
void Channel::close() noexcept { impl_.reset(); }
struct Listener::Impl
{
    Context context;
    Socket socket;
    Endpoint endpoint;
    TlsOptions options;
};
Listener::Listener(Endpoint const& endpoint, TlsOptions const& options, unsigned backlog) : impl_(std::make_unique<Impl>())
{
    require(backlog && backlog <= 4096, "invalid network listener backlog");
    initialize_sockets();
    impl_->options = options; impl_->context = context(options, true); impl_->endpoint = endpoint;
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    require(::getaddrinfo(endpoint.host.empty() ? nullptr : endpoint.host.c_str(), std::to_string(endpoint.port).c_str(),
        &hints, &addresses) == 0, "cannot resolve listener address");
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> cleanup(addresses, freeaddrinfo);
    for (auto item = addresses; item; item = item->ai_next)
    {
        Socket socket{::socket(item->ai_family, item->ai_socktype, item->ai_protocol)};
        if (socket.value == invalid_socket) continue;
        int reuse = 1;
#ifdef _WIN32
        // Windows SO_REUSEADDR permits another process to hijack a listening
        // address. Exclusive binding preserves the POSIX listener ownership.
        require(::setsockopt(socket.value, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<char const*>(&reuse), sizeof(reuse)) == 0,
            "cannot require exclusive network listener ownership");
#else
        ::setsockopt(socket.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
        if (item->ai_family == AF_INET6 && endpoint.host == "::")
        { int no = 0; ::setsockopt(socket.value, IPPROTO_IPV6, IPV6_V6ONLY, reinterpret_cast<char const*>(&no), sizeof(no)); }
        nonblocking(socket.value);
        if (::bind(socket.value, item->ai_addr, item->ai_addrlen) != 0 || ::listen(socket.value, int(backlog)) != 0) continue;
        impl_->socket.value = socket.release(); break;
    }
    if (impl_->socket.value == invalid_socket && endpoint.host == "::")
    {
        // Automatic listeners still work on hosts with IPv6 disabled.
        Socket socket{::socket(AF_INET, SOCK_STREAM, 0)};
        if (socket.value != invalid_socket)
        {
            int reuse = 1;
#ifdef _WIN32
            require(::setsockopt(socket.value, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<char const*>(&reuse), sizeof(reuse)) == 0,
                "cannot require exclusive network listener ownership");
#else
            ::setsockopt(socket.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
            nonblocking(socket.value);
            sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(endpoint.port);
            if (!::bind(socket.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) && !::listen(socket.value, int(backlog)))
            { impl_->socket.value = socket.release(); impl_->endpoint.host = "0.0.0.0"; }
        }
    }
    require(impl_->socket.value != invalid_socket, "cannot bind network listener");
    sockaddr_storage address{}; AddressSize size = sizeof(address);
    require(::getsockname(impl_->socket.value, reinterpret_cast<sockaddr*>(&address), &size) == 0,
            "cannot inspect network listener");
    if (address.ss_family == AF_INET) impl_->endpoint.port = ntohs(reinterpret_cast<sockaddr_in*>(&address)->sin_port);
    else impl_->endpoint.port = ntohs(reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
}
Listener::~Listener() = default;
Channel Listener::accept(unsigned poll_ms)
{
    require(poll_ms, "network accept poll must be positive");
    check_deadline(impl_->options.deadline);
    if (!wait_fd(impl_->socket.value, POLLIN, std::min(deadline(poll_ms), impl_->options.deadline), true))
    { check_deadline(impl_->options.deadline); return {}; }
    Socket socket{::accept(impl_->socket.value, nullptr, nullptr)};
    if (socket.value == invalid_socket && retry_socket(socket_error())) return {};
    require(socket.value != invalid_socket, "cannot accept network connection"); nonblocking(socket.value);
    if (!allowed_interface(socket.value, impl_->options)) return {};
    auto channel = std::make_unique<Channel::Impl>();
    channel->context = impl_->options.membership_directory.empty() ? impl_->context : context(impl_->options, true);
    channel->fd = socket.release(); channel->options = impl_->options; channel->creator_deadline = impl_->options.deadline;
    channel->ssl = SSL_new(channel->context.get());
    require(channel->ssl, "cannot initialize accepted TLS connection"); set_socket(channel->ssl, channel->fd);
    auto const until = operation_deadline(impl_->options); handshake(channel->ssl, true, until);
    std::unique_ptr<X509, decltype(&X509_free)> peer(SSL_get1_peer_certificate(channel->ssl), X509_free);
    channel->peer = fingerprint(peer.get()); channel->enroll(true, until); check_deadline(until); return Channel(std::move(channel));
}
Endpoint Listener::endpoint() const { return impl_->endpoint; }
std::string random_token()
{
    std::array<unsigned char, 32> value{};
    require(RAND_bytes(value.data(), int(value.size())) == 1, "cannot create network lease identity");
    return hex(value);
}
std::string certificate_identity(std::string const& path)
{
    std::unique_ptr<BIO, decltype(&BIO_free)> file(BIO_new_file(path.c_str(), "r"), BIO_free);
    require(bool(file), "cannot read network certificate");
    std::unique_ptr<X509, decltype(&X509_free)> certificate(PEM_read_bio_X509(file.get(), nullptr, nullptr, nullptr), X509_free);
    return fingerprint(certificate.get());
}
void validate_tls_credentials(TlsOptions const& options)
{ (void)context(options, false); }
std::string certificate_node_id(std::string const& path)
{
    std::unique_ptr<BIO, decltype(&BIO_free)> file(BIO_new_file(path.c_str(), "r"), BIO_free);
    require(bool(file), "cannot read network certificate");
    std::unique_ptr<X509, decltype(&X509_free)> certificate(PEM_read_bio_X509(file.get(), nullptr, nullptr, nullptr), X509_free);
    return node_identity(certificate.get());
}
Endpoint parse_endpoint(std::string const& value)
{
    size_t delimiter = value.rfind(':');
    require(delimiter != std::string::npos && delimiter && delimiter + 1 < value.size() && value.size() - delimiter - 1 <= 5, "endpoint must be HOST:PORT");
    std::string host = value.substr(0, delimiter);
    if (host.front() == '[')
    { require(host.back() == ']', "invalid bracketed endpoint"); host = host.substr(1, host.size() - 2); }
    else require(host.find(':') == std::string::npos, "IPv6 endpoint must use brackets");
    uint32_t port = 0;
    for (size_t i = delimiter + 1; i < value.size(); ++i)
    {
        require(value[i] >= '0' && value[i] <= '9', "invalid endpoint port");
        port = port * 10 + uint32_t(value[i] - '0'); require(port <= UINT16_MAX, "endpoint port out of range");
    }
    require(!host.empty() && port, "endpoint requires host and nonzero port");
    return {std::move(host), uint16_t(port)};
}
std::string local_address(Endpoint const& endpoint)
{
    initialize_sockets();
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_DGRAM;
    addrinfo* addresses = nullptr;
    require(::getaddrinfo(endpoint.host.c_str(), std::to_string(endpoint.port).c_str(), &hints, &addresses) == 0,
            "cannot resolve broker route");
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> cleanup(addresses, freeaddrinfo);
    for (auto item = addresses; item; item = item->ai_next)
    {
#ifdef _WIN32
        Socket socket{::socket(item->ai_family, SOCK_DGRAM, 0)};
        if (socket.value != invalid_socket && !::SetHandleInformation(reinterpret_cast<HANDLE>(socket.value), HANDLE_FLAG_INHERIT, 0)) continue;
#else
        Socket socket{::socket(item->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, 0)};
#endif
        if (socket.value == invalid_socket || ::connect(socket.value, item->ai_addr, item->ai_addrlen)) continue;
        sockaddr_storage address{}; AddressSize length = sizeof(address);
        std::array<char, NI_MAXHOST> host{};
        if (!::getsockname(socket.value, reinterpret_cast<sockaddr*>(&address), &length) &&
            !::getnameinfo(reinterpret_cast<sockaddr*>(&address), length, host.data(), host.size(), nullptr, 0, NI_NUMERICHOST))
            return host.data();
    }
    throw TransientTransport("cannot determine local route to broker");
}
} // namespace group_grow::pool
