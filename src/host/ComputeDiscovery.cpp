#include "host/ComputeDiscovery.hpp"
#include "host/ComputePool.hpp"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
#include <mswsock.h>
using socklen_t = int;
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <mdns.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#ifdef _WIN32
// COM headers included by IP Helper define this ordinary C++ identifier.
#undef interface
#endif

namespace group_grow::pool
{
namespace
{
using Clock = std::chrono::steady_clock;
#ifdef _WIN32
using Descriptor = SOCKET;
using Poll = WSAPOLLFD;
constexpr Descriptor invalid_socket = INVALID_SOCKET;
struct SocketRuntime
{
    SocketRuntime()
    { WSADATA data{}; if (::WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("cannot initialize compute discovery sockets"); }
    ~SocketRuntime() { ::WSACleanup(); }
};
void close_socket(Descriptor fd) { ::closesocket(fd); }
int poll_sockets(Poll* sockets, size_t count, int timeout)
{
    if (!count) { std::this_thread::sleep_for(std::chrono::milliseconds(timeout)); return 0; }
    return ::WSAPoll(sockets, ULONG(count), timeout);
}
#else
using Descriptor = int;
using Poll = pollfd;
constexpr Descriptor invalid_socket = -1;
struct SocketRuntime {};
void close_socket(Descriptor fd) { ::close(fd); }
int poll_sockets(Poll* sockets, size_t count, int timeout) { return ::poll(sockets, count, timeout); }
#endif
constexpr std::string_view broker_service = "_xchplot2-broker._tcp.local.";
constexpr std::string_view worker_service = "_xchplot2-worker._tcp.local.";
mdns_string_t view(std::string_view value) { return {value.data(), value.size()}; }
std::string text(mdns_string_t value) { return value.str ? std::string(value.str, value.length) : std::string{}; }
void require(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
struct Interface
{
    std::string name;
    unsigned index = 0;
    sockaddr_storage address{};
};
std::string address(sockaddr_storage const&, unsigned);
std::vector<Interface> interfaces(std::vector<std::string> const& selected, Endpoint const& endpoint)
{
#ifdef _WIN32
    ULONG size = 16384;
    std::vector<uint8_t> storage(size);
    ULONG result = ERROR_BUFFER_OVERFLOW;
    for (unsigned attempt = 0; attempt < 3 && result == ERROR_BUFFER_OVERFLOW; ++attempt)
    {
        storage.resize(size);
        result = ::GetAdaptersAddresses(AF_UNSPEC,
            GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
            nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data()), &size);
    }
    if (result == ERROR_NO_DATA) return {};
    require(result == NO_ERROR, "cannot enumerate compute network interfaces");
    std::vector<Interface> output; std::set<std::pair<unsigned, int>> seen;
    for (auto adapter = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data()); adapter; adapter = adapter->Next)
    {
        if (adapter->OperStatus != IfOperStatusUp) continue;
        std::string friendly;
        if (adapter->FriendlyName)
        {
            int length = ::WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, nullptr, 0, nullptr, nullptr);
            if (length > 0)
            {
                friendly.resize(size_t(length));
                ::WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, friendly.data(), length, nullptr, nullptr);
                friendly.pop_back();
            }
        }
        for (auto item = adapter->FirstUnicastAddress; item; item = item->Next)
        {
            if (!item->Address.lpSockaddr) continue;
            auto const family = item->Address.lpSockaddr->sa_family;
            if (family != AF_INET && family != AF_INET6) continue;
            unsigned const index = family == AF_INET ? adapter->IfIndex : adapter->Ipv6IfIndex;
            bool const explicit_name = std::any_of(selected.begin(), selected.end(), [&](auto const& name)
            { return name == adapter->AdapterName || name == friendly || name == std::to_string(index) ||
                (name == "lo" && adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK); });
            if ((!selected.empty() && !explicit_name) || !index) continue;
            if (!explicit_name && ((adapter->Flags & IP_ADAPTER_NO_MULTICAST) ||
                adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK || adapter->IfType == IF_TYPE_PPP || adapter->IfType == IF_TYPE_TUNNEL)) continue;
            Interface value; value.name = adapter->AdapterName; value.index = index;
            std::memcpy(&value.address, item->Address.lpSockaddr, family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6));
            if (!endpoint.host.empty() && endpoint.host != "::" && endpoint.host != "0.0.0.0" &&
                address(value.address, index) != endpoint.host) continue;
            if (endpoint.host == "0.0.0.0" && family != AF_INET) continue;
            if (!seen.emplace(index, family).second) continue;
            output.push_back(std::move(value));
            if (output.size() == 64) return output;
        }
    }
    return output;
#else
    ifaddrs* addresses = nullptr;
    require(!::getifaddrs(&addresses), "cannot enumerate compute network interfaces");
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> cleanup(addresses, freeifaddrs);
    std::vector<Interface> result; std::set<std::pair<unsigned, int>> seen;
    for (auto item = addresses; item; item = item->ifa_next)
    {
        if (!item->ifa_addr || !(item->ifa_flags & IFF_UP)) continue;
        bool const explicit_name = std::find(selected.begin(), selected.end(), item->ifa_name) != selected.end();
        if (!selected.empty() && !explicit_name) continue;
        if (!explicit_name && (!(item->ifa_flags & IFF_MULTICAST) ||
            (item->ifa_flags & (IFF_LOOPBACK | IFF_POINTOPOINT)))) continue;
        auto const family = item->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) continue;
        auto const index = ::if_nametoindex(item->ifa_name);
        if (!index) continue;
        Interface value; value.name = item->ifa_name; value.index = index;
        std::memcpy(&value.address, item->ifa_addr, family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6));
        // Match the bound address before choosing one address per family. An
        // explicitly bound secondary address must not be hidden by the first
        // address returned for that interface.
        if (!endpoint.host.empty() && endpoint.host != "::" && endpoint.host != "0.0.0.0" &&
            address(value.address, index) != endpoint.host) continue;
        if (endpoint.host == "0.0.0.0" && family != AF_INET) continue;
        if (!seen.emplace(index, family).second) continue;
        result.push_back(std::move(value));
        if (result.size() == 64) break;
    }
    return result;
#endif
}
struct Socket
{
    Descriptor fd = invalid_socket;
#ifdef _WIN32
    LPFN_WSARECVMSG receive_message = nullptr;
#endif
    Interface interface;
    mutable Clock::time_point next_response{};
    Socket(Interface value) : interface(std::move(value))
    {
        auto const family = interface.address.ss_family;
#ifdef _WIN32
        fd = ::WSASocketW(family, SOCK_DGRAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
        if (fd == invalid_socket) return;
        auto option = [&](int level, int key, auto const& value)
        { return ::setsockopt(fd, level, key, reinterpret_cast<char const*>(&value), int(sizeof(value))) == 0; };
        u_long nonblocking = 1; int yes = 1, hops = 255;
        GUID identifier = WSAID_WSARECVMSG; DWORD returned = 0;
        bool okay = !::ioctlsocket(fd, FIONBIO, &nonblocking) &&
            !::WSAIoctl(fd, SIO_GET_EXTENSION_FUNCTION_POINTER, &identifier, sizeof(identifier),
                &receive_message, sizeof(receive_message), &returned, nullptr, nullptr) &&
            option(SOL_SOCKET, SO_REUSEADDR, yes);
        if (family == AF_INET)
        {
            auto const& address = reinterpret_cast<sockaddr_in const&>(interface.address);
            ip_mreq membership{}; ::inet_pton(AF_INET, "224.0.0.251", &membership.imr_multiaddr);
            membership.imr_interface = address.sin_addr;
            sockaddr_in local{}; local.sin_family = AF_INET; local.sin_port = htons(MDNS_PORT);
            okay = okay && option(IPPROTO_IP, IP_ADD_MEMBERSHIP, membership) &&
                option(IPPROTO_IP, IP_MULTICAST_IF, address.sin_addr) && option(IPPROTO_IP, IP_PKTINFO, yes) &&
                option(IPPROTO_IP, IP_RECVTTL, yes) && option(IPPROTO_IP, IP_MULTICAST_TTL, hops) &&
                option(IPPROTO_IP, IP_TTL, hops) && !::bind(fd, reinterpret_cast<sockaddr*>(&local), sizeof(local));
        }
        else
        {
            ipv6_mreq membership{}; ::inet_pton(AF_INET6, "ff02::fb", &membership.ipv6mr_multiaddr);
            membership.ipv6mr_interface = interface.index;
            sockaddr_in6 local{}; local.sin6_family = AF_INET6; local.sin6_port = htons(MDNS_PORT);
            okay = okay && option(IPPROTO_IPV6, IPV6_V6ONLY, yes) && option(IPPROTO_IPV6, IPV6_JOIN_GROUP, membership) &&
                option(IPPROTO_IPV6, IPV6_MULTICAST_IF, interface.index) && option(IPPROTO_IPV6, IPV6_PKTINFO, yes) &&
                option(IPPROTO_IPV6, IPV6_HOPLIMIT, yes) && option(IPPROTO_IPV6, IPV6_MULTICAST_HOPS, hops) &&
                option(IPPROTO_IPV6, IPV6_UNICAST_HOPS, hops) && !::bind(fd, reinterpret_cast<sockaddr*>(&local), sizeof(local));
        }
#else
        fd = ::socket(family, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) return;
        int yes = 1, no = 0, hops = 255;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
        bool okay = false;
        if (family == AF_INET)
        {
            auto address = *reinterpret_cast<sockaddr_in*>(&interface.address);
            ip_mreqn membership{}; membership.imr_multiaddr.s_addr = ::inet_addr("224.0.0.251");
            membership.imr_address = address.sin_addr; membership.imr_ifindex = int(interface.index);
            sockaddr_in bind_address{}; bind_address.sin_family = AF_INET; bind_address.sin_port = htons(MDNS_PORT);
            okay = !::setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership, sizeof(membership)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_IF, &membership, sizeof(membership)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_ALL, &no, sizeof(no)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_PKTINFO, &yes, sizeof(yes)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_RECVTTL, &yes, sizeof(yes)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &hops, sizeof(hops)) &&
                !::setsockopt(fd, IPPROTO_IP, IP_TTL, &hops, sizeof(hops)) &&
                !::bind(fd, reinterpret_cast<sockaddr*>(&bind_address), sizeof(bind_address));
        }
        else
        {
            ipv6_mreq membership{}; ::inet_pton(AF_INET6, "ff02::fb", &membership.ipv6mr_multiaddr);
            membership.ipv6mr_interface = interface.index;
            sockaddr_in6 bind_address{}; bind_address.sin6_family = AF_INET6; bind_address.sin6_port = htons(MDNS_PORT);
            okay = !::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &yes, sizeof(yes)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_JOIN_GROUP, &membership, sizeof(membership)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_MULTICAST_IF, &interface.index, sizeof(interface.index)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_RECVPKTINFO, &yes, sizeof(yes)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_RECVHOPLIMIT, &yes, sizeof(yes)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &hops, sizeof(hops)) &&
                !::setsockopt(fd, IPPROTO_IPV6, IPV6_UNICAST_HOPS, &hops, sizeof(hops)) &&
                !::bind(fd, reinterpret_cast<sockaddr*>(&bind_address), sizeof(bind_address));
        }
#endif
        if (!okay) { close_socket(fd); fd = invalid_socket; }
    }
    ~Socket() { if (fd != invalid_socket) close_socket(fd); }
    void send(std::span<uint8_t const> bytes, sockaddr_storage const* peer = nullptr, socklen_t peer_size = 0) const
    {
        sockaddr_storage destination{};
        if (!peer)
        {
            destination.ss_family = interface.address.ss_family;
            if (destination.ss_family == AF_INET)
            {
                auto& address = reinterpret_cast<sockaddr_in&>(destination);
                address.sin_port = htons(MDNS_PORT); ::inet_pton(AF_INET, "224.0.0.251", &address.sin_addr);
                peer_size = sizeof(address);
            }
            else
            {
                auto& address = reinterpret_cast<sockaddr_in6&>(destination);
                address.sin6_port = htons(MDNS_PORT); address.sin6_scope_id = interface.index;
                ::inet_pton(AF_INET6, "ff02::fb", &address.sin6_addr); peer_size = sizeof(address);
            }
            peer = &destination;
        }
#ifdef _WIN32
        ::sendto(fd, reinterpret_cast<char const*>(bytes.data()), int(bytes.size()), 0,
            reinterpret_cast<sockaddr const*>(peer), peer_size);
#else
        ::sendto(fd, bytes.data(), bytes.size(), 0, reinterpret_cast<sockaddr const*>(peer), peer_size);
#endif
    }
    void query(std::string_view service) const
    {
        alignas(4) std::array<uint8_t, 4096> buffer{};
        auto header = reinterpret_cast<mdns_header_t*>(buffer.data()); header->questions = htons(1);
        void* data = mdns_string_make(buffer.data(), buffer.size(), buffer.data() + sizeof(*header),
            service.data(), service.size(), nullptr);
        if (!data || MDNS_POINTER_DIFF(data, buffer.data()) > buffer.size() - 4) return;
        data = mdns_htons(data, MDNS_RECORDTYPE_PTR); data = mdns_htons(data, MDNS_CLASS_IN);
        send({buffer.data(), MDNS_POINTER_DIFF(data, buffer.data())});
    }
    void publish(mdns_record_t answer, std::span<mdns_record_t const> extra, bool goodbye,
        sockaddr_storage const* peer, socklen_t peer_size, uint16_t query_id,
        uint16_t query_type, std::string const& query_name) const
    {
        // The pinned library's send helpers take int sockets. Keep full-width
        // Winsock handles here and reuse its record builders and compression.
        alignas(4) std::array<uint8_t, 4096> buffer{};
        auto header = reinterpret_cast<mdns_header_t*>(buffer.data());
        header->query_id = htons(peer ? query_id : 0); header->flags = htons(0x8400);
        header->questions = htons(peer ? 1 : 0); header->answer_rrs = htons(1);
        header->additional_rrs = htons(mdns_answer_get_record_count(extra.data(), extra.size()));
        mdns_string_table_t strings{{0}, 0, 0}; void* data = buffer.data() + sizeof(*header);
        if (peer) data = mdns_answer_add_question_unicast(buffer.data(), buffer.size(), data,
            mdns_record_type_t(query_type), query_name.data(), query_name.size(), &strings);
        uint16_t const klass = MDNS_CLASS_IN | (peer || goodbye ? 0 : MDNS_CACHE_FLUSH);
        uint32_t const ttl = goodbye ? 0 : peer ? 10 : 60;
        if (peer) answer.ttl = ttl;
        mdns_record_update_rclass_ttl(&answer, klass, ttl);
        if (data) data = mdns_answer_add_record(buffer.data(), buffer.size(), data, answer, &strings);
        for (auto record : extra)
        {
            mdns_record_update_rclass_ttl(&record, klass, ttl);
            if (data) data = mdns_answer_add_record(buffer.data(), buffer.size(), data, record, &strings);
        }
        if (data) data = mdns_answer_add_txt_record(buffer.data(), buffer.size(), data,
            extra.data(), extra.size(), klass, ttl, &strings);
        if (data) send({buffer.data(), MDNS_POINTER_DIFF(data, buffer.data())}, peer, peer_size);
    }
};
uint16_t word(std::span<uint8_t const> bytes, size_t offset)
{ return uint16_t((uint16_t(bytes[offset]) << 8) | bytes[offset + 1]); }
uint32_t integer(std::span<uint8_t const> bytes, size_t offset)
{ return (uint32_t(word(bytes, offset)) << 16) | word(bytes, offset + 2); }
std::string name(std::span<uint8_t const> bytes, size_t& offset)
{
    std::array<char, 256> buffer{};
    auto const result = mdns_string_extract(bytes.data(), bytes.size(), &offset, buffer.data(), buffer.size());
    require(result.length && result.length < buffer.size(), "invalid discovery DNS name");
    auto value = text(result);
    require(value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == value.npos,
        "invalid discovery name characters");
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return value;
}
std::string address(sockaddr_storage const& value, unsigned interface)
{
    std::array<char, INET6_ADDRSTRLEN> result{};
    if (value.ss_family == AF_INET)
    {
        auto const& ip = reinterpret_cast<sockaddr_in const&>(value);
        if (!::inet_ntop(AF_INET, &ip.sin_addr, result.data(), result.size())) return {};
        return result.data();
    }
    auto const& ip = reinterpret_cast<sockaddr_in6 const&>(value);
    if (!::inet_ntop(AF_INET6, &ip.sin6_addr, result.data(), result.size())) return {};
    std::string output = result.data();
    if (IN6_IS_ADDR_LINKLOCAL(&ip.sin6_addr)) output += '%' + std::to_string(interface);
    return output;
}
} // namespace

struct ComputeDiscovery::Impl
{
    SocketRuntime runtime;
    DiscoveryOptions options;
    std::string instance, hostname, service, version = std::to_string(protocol_version);
    std::vector<std::unique_ptr<Socket>> sockets;
    std::vector<std::string> topology;
    std::atomic<bool> stopping{false};
    std::thread thread;
    mutable std::mutex mutex;
    struct Candidate
    {
        DiscoveredCompute service;
        bool matching = false;
        Clock::time_point srv_expires{}, txt_expires{};
    };
    std::map<std::string, Candidate> candidates;
    std::string diagnostic;
    explicit Impl(DiscoveryOptions value) : options(std::move(value))
    {
        require(options.role.empty() || options.role == "broker" || options.role == "worker", "invalid discovery role");
        require(options.pool_id.size() == 64 && options.node_id.size() == 32 && !options.server_name.empty(),
            "LAN discovery requires saved pool membership");
        require(options.role.empty() || options.endpoint.port, "advertised compute service needs a listener");
        // A random process suffix lets concurrent plotters on one machine
        // publish independent brokers without fighting for a DNS-SD name.
        service = options.role == "worker" ? worker_service : broker_service;
        hostname = "xchplot2-" + options.node_id + ".local.";
        instance = options.node_id + '-' + random_token().substr(0, 16) + '.' + service;
        refresh_interfaces();
        thread = std::thread([this] { loop(); });
    }
    ~Impl()
    { stopping.store(true); if (thread.joinable()) thread.join(); }
    bool refresh_interfaces()
    {
        auto available = interfaces(options.interfaces, options.endpoint);
        std::vector<std::string> current;
        for (auto const& interface : available)
            current.push_back(interface.name + ':' + std::to_string(interface.index) + ':' + address(interface.address, interface.index));
        if (current == topology && !sockets.empty()) return false;
        for (auto const& socket : sockets) publish(*socket, true);
        sockets.clear(); topology = std::move(current);
        for (auto const& interface : available)
        {
            auto socket = std::make_unique<Socket>(interface);
            if (socket->fd != invalid_socket) sockets.push_back(std::move(socket));
        }
        std::lock_guard lock(mutex); candidates.clear();
        if (sockets.empty()) diagnostic = "LAN discovery unavailable on the selected interfaces; direct broker addresses still work";
        else diagnostic = "LAN discovery active";
        return true;
    }
    void publish(Socket const& socket, bool goodbye = false, sockaddr_storage const* peer = nullptr,
        socklen_t peer_size = 0, uint16_t query_id = 0, uint16_t query_type = MDNS_RECORDTYPE_PTR,
        std::string const& query_name = {})
    {
        if (options.role.empty()) return;
        mdns_record_t ptr{}; ptr.name = view(service); ptr.type = MDNS_RECORDTYPE_PTR; ptr.data.ptr.name = view(instance); ptr.ttl = 30;
        std::array<mdns_record_t, 5> extra{};
        auto& srv = extra[0]; srv.name = view(instance); srv.type = MDNS_RECORDTYPE_SRV;
        srv.data.srv.name = view(hostname); srv.data.srv.port = options.endpoint.port; srv.ttl = 30;
        auto& ip = extra[1]; ip.name = view(hostname); ip.ttl = 30;
        if (socket.interface.address.ss_family == AF_INET)
        { ip.type = MDNS_RECORDTYPE_A; ip.data.a.addr = reinterpret_cast<sockaddr_in const&>(socket.interface.address); }
        else
        { ip.type = MDNS_RECORDTYPE_AAAA; ip.data.aaaa.addr = reinterpret_cast<sockaddr_in6 const&>(socket.interface.address); }
        for (size_t i = 2; i < extra.size(); ++i)
        { extra[i].name = view(instance); extra[i].type = MDNS_RECORDTYPE_TXT; extra[i].ttl = 30; }
        extra[2].data.txt = {view("pool"), view(options.pool_id)};
        extra[3].data.txt = {view("name"), view(options.name)};
        extra[4].data.txt = {view("version"), view(version)};
        socket.publish(ptr, extra, goodbye, peer, peer_size, query_id, query_type, query_name);
    }
    void receive(Socket const& socket)
    {
        alignas(4) std::array<uint8_t, 16384> buffer{};
#ifdef _WIN32
        alignas(WSACMSGHDR) std::array<uint8_t, 256> control{};
        sockaddr_storage peer{}; WSABUF io{ULONG(buffer.size()), reinterpret_cast<char*>(buffer.data())};
        WSAMSG message{}; message.name = reinterpret_cast<sockaddr*>(&peer); message.namelen = sizeof(peer);
        message.lpBuffers = &io; message.dwBufferCount = 1;
        message.Control = {ULONG(control.size()), reinterpret_cast<char*>(control.data())};
        DWORD count = 0;
        if (socket.receive_message(socket.fd, &message, &count, nullptr, nullptr) ||
            count < 12 || (message.dwFlags & (MSG_TRUNC | MSG_CTRUNC))) return;
        unsigned incoming = 0; int hops = 0;
        for (auto item = WSA_CMSG_FIRSTHDR(&message); item; item = WSA_CMSG_NXTHDR(&message, item))
        {
            if (item->cmsg_level == IPPROTO_IP && item->cmsg_type == IP_PKTINFO && item->cmsg_len >= WSA_CMSG_LEN(sizeof(IN_PKTINFO)))
                incoming = reinterpret_cast<IN_PKTINFO*>(WSA_CMSG_DATA(item))->ipi_ifindex;
            else if (item->cmsg_level == IPPROTO_IPV6 && item->cmsg_type == IPV6_PKTINFO && item->cmsg_len >= WSA_CMSG_LEN(sizeof(IN6_PKTINFO)))
                incoming = reinterpret_cast<IN6_PKTINFO*>(WSA_CMSG_DATA(item))->ipi6_ifindex;
            else if (((item->cmsg_level == IPPROTO_IP && item->cmsg_type == IP_TTL) ||
                      (item->cmsg_level == IPPROTO_IPV6 && item->cmsg_type == IPV6_HOPLIMIT)) && item->cmsg_len >= WSA_CMSG_LEN(sizeof(int)))
                std::memcpy(&hops, WSA_CMSG_DATA(item), sizeof(hops));
        }
        if (incoming != socket.interface.index || hops != 255) return;
        parse(socket, {buffer.data(), size_t(count)}, peer, message.namelen);
#else
        alignas(cmsghdr) std::array<uint8_t, 256> control{};
        sockaddr_storage peer{}; iovec io{buffer.data(), buffer.size()};
        msghdr message{}; message.msg_name = &peer; message.msg_namelen = sizeof(peer);
        message.msg_iov = &io; message.msg_iovlen = 1; message.msg_control = control.data(); message.msg_controllen = control.size();
        auto const count = ::recvmsg(socket.fd, &message, MSG_DONTWAIT);
        if (count < 12 || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) return;
        unsigned incoming = 0; int hops = 0;
        for (auto item = CMSG_FIRSTHDR(&message); item; item = CMSG_NXTHDR(&message, item))
        {
            if (item->cmsg_level == IPPROTO_IP && item->cmsg_type == IP_PKTINFO && item->cmsg_len >= CMSG_LEN(sizeof(in_pktinfo)))
                incoming = unsigned(reinterpret_cast<in_pktinfo*>(CMSG_DATA(item))->ipi_ifindex);
            else if (item->cmsg_level == IPPROTO_IPV6 && item->cmsg_type == IPV6_PKTINFO && item->cmsg_len >= CMSG_LEN(sizeof(in6_pktinfo)))
                incoming = reinterpret_cast<in6_pktinfo*>(CMSG_DATA(item))->ipi6_ifindex;
            else if (((item->cmsg_level == IPPROTO_IP && item->cmsg_type == IP_TTL) ||
                      (item->cmsg_level == IPPROTO_IPV6 && item->cmsg_type == IPV6_HOPLIMIT)) && item->cmsg_len >= CMSG_LEN(sizeof(int)))
                std::memcpy(&hops, CMSG_DATA(item), sizeof(hops));
        }
        if (incoming != socket.interface.index || hops != 255) return;
        parse(socket, {buffer.data(), size_t(count)}, peer, message.msg_namelen);
#endif
    }
    void parse(Socket const& socket, std::span<uint8_t const> bytes, sockaddr_storage const& peer, socklen_t peer_size)
    {
        auto const flags = word(bytes, 2);
        if ((flags & 0x780f) || (flags & 0x0200)) return;
        bool const response = flags & 0x8000;
        size_t offset = 12; unsigned records = 0;
        for (unsigned i = 0; i < word(bytes, 4); ++i)
        {
            if (++records > 128) return;
            auto question = name(bytes, offset);
            if (offset + 4 > bytes.size()) return;
            auto const type = word(bytes, offset), klass = word(bytes, offset + 2); offset += 4;
            if (!response && (klass & 0x7fff) == MDNS_CLASS_IN &&
                (question == service || question == instance || question == hostname))
            {
                if (Clock::now() < socket.next_response) continue;
                socket.next_response = Clock::now() + std::chrono::milliseconds(100);
                publish(socket, false, klass & MDNS_UNICAST_RESPONSE ? &peer : nullptr,
                    peer_size, word(bytes, 0), type, question);
            }
        }
        auto const source = address(peer, socket.interface.index);
        auto const source_port = peer.ss_family == AF_INET ? reinterpret_cast<sockaddr_in const&>(peer).sin_port :
            reinterpret_cast<sockaddr_in6 const&>(peer).sin6_port;
        if (!response || source.empty() || ntohs(source_port) != MDNS_PORT) return;
        unsigned const total = unsigned(word(bytes, 6)) + word(bytes, 8) + word(bytes, 10);
        for (unsigned i = 0; i < total; ++i)
        {
            if (++records > 128) return;
            auto record_name = name(bytes, offset);
            if (offset + 10 > bytes.size()) return;
            auto const type = word(bytes, offset), klass = word(bytes, offset + 2), length = word(bytes, offset + 8);
            auto const ttl = integer(bytes, offset + 4); offset += 10;
            if (length > bytes.size() - offset) return;
            auto const data_offset = offset; offset += length;
            std::string role;
            if (record_name.ends_with('.' + std::string(broker_service))) role = "broker";
            else if (record_name.ends_with('.' + std::string(worker_service))) role = "worker";
            if (role.empty() || (klass & 0x7fff) != MDNS_CLASS_IN ||
                (type != MDNS_RECORDTYPE_SRV && type != MDNS_RECORDTYPE_TXT)) continue;
            auto key = record_name + '|' + source;
            std::lock_guard lock(mutex);
            std::erase_if(candidates, [](auto const& item)
            { return item.second.srv_expires < Clock::now() && item.second.txt_expires < Clock::now(); });
            if (!ttl) { candidates.erase(key); continue; }
            if (!candidates.contains(key) && candidates.size() == 128) continue;
            auto& candidate = candidates[key]; candidate.service.instance = record_name; candidate.service.role = role;
            auto expires = Clock::now() + std::chrono::seconds(std::min<uint32_t>(ttl, 120));
            if (type == MDNS_RECORDTYPE_SRV)
            {
                if (length < 8) continue;
                size_t target_offset = data_offset + 6; auto target = name(bytes, target_offset);
                if (target_offset > offset || !target.starts_with("xchplot2-") || !target.ends_with(".local.")) continue;
                candidate.service.endpoint = {source, word(bytes, data_offset + 4), options.server_name};
                candidate.srv_expires = expires;
            }
            else
            {
                std::array<mdns_record_txt_t, 8> fields{};
                auto count = mdns_record_parse_txt(bytes.data(), bytes.size(), data_offset, length, fields.data(), fields.size());
                std::string pool, protocol, label;
                for (size_t j = 0; j < count; ++j)
                {
                    auto field = text(fields[j].key), value = text(fields[j].value);
                    if (field == "pool") pool = value;
                    else if (field == "version") protocol = value;
                    else if (field == "name") label = value;
                }
                candidate.matching = pool == options.pool_id && protocol == version;
                if (label.size() > 64 || std::any_of(label.begin(), label.end(), [](unsigned char c) { return c < 32 || c == 127; }))
                    label = "unnamed machine";
                candidate.service.name = std::move(label); candidate.txt_expires = expires;
            }
        }
    }
    void loop()
    {
        std::vector<Poll> polls;
        for (auto const& socket : sockets) polls.push_back({socket->fd, POLLIN, 0});
        auto next_query = Clock::now(), next_announce = Clock::now(), next_interfaces = Clock::now() + std::chrono::seconds(5);
        while (!stopping.load())
        {
            if (Clock::now() >= next_interfaces)
            {
                try
                {
                    if (refresh_interfaces())
                    {
                        polls.clear();
                        for (auto const& socket : sockets) polls.push_back({socket->fd, POLLIN, 0});
                        next_query = next_announce = Clock::now();
                    }
                }
                catch (std::exception const&) { /* Retry when the network inventory settles. */ }
                next_interfaces = Clock::now() + std::chrono::seconds(5);
            }
            if (Clock::now() >= next_query)
            {
                for (auto const& socket : sockets)
                    for (auto service : {broker_service, worker_service})
                        socket->query(service);
                next_query = Clock::now() + std::chrono::seconds(5);
            }
            if (Clock::now() >= next_announce)
            {
                for (auto const& socket : sockets) publish(*socket);
                next_announce = Clock::now() + std::chrono::seconds(10);
            }
            if (poll_sockets(polls.data(), polls.size(), 100) > 0)
                for (size_t i = 0; i < polls.size(); ++i)
                    if (polls[i].revents & POLLIN) try { receive(*sockets[i]); } catch (std::exception const&) {}
        }
        for (auto const& socket : sockets) publish(*socket, true);
    }
    std::vector<DiscoveredCompute> services() const
    {
        std::lock_guard lock(mutex); std::vector<DiscoveredCompute> result;
        for (auto const& [key, candidate] : candidates)
            if (candidate.matching && candidate.service.endpoint.port &&
                candidate.srv_expires > Clock::now() && candidate.txt_expires > Clock::now())
                result.push_back(candidate.service);
        return result;
    }
};
ComputeDiscovery::ComputeDiscovery(DiscoveryOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
ComputeDiscovery::~ComputeDiscovery() = default;
std::vector<DiscoveredCompute> ComputeDiscovery::services() const { return impl_->services(); }
std::vector<Endpoint> ComputeDiscovery::brokers() const
{
    std::vector<Endpoint> result;
    for (auto const& service : services())
        if (service.role == "broker" &&
            std::find(result.begin(), result.end(), service.endpoint) == result.end())
            result.push_back(service.endpoint);
    return result;
}
std::string ComputeDiscovery::status() const { std::lock_guard lock(impl_->mutex); return impl_->diagnostic; }
} // namespace group_grow::pool
