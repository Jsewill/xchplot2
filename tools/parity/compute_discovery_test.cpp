#include "host/ComputeDiscovery.hpp"
#include "host/ComputePool.hpp"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <chrono>
#include <iostream>
#include <thread>

using namespace group_grow::pool;
namespace
{
// Send an independent DNS-SD response, without an announcer refreshing its
// lease. Loopback's subnet supplies separate source routes for one instance.
void response(std::string const& source, std::string const& pool, std::string const& instance,
    uint32_t ttl = 1, int hops = 255, uint16_t source_port = 5353)
{
    std::vector<uint8_t> bytes{0, 0, 0x84, 0, 0, 0, 0, 2, 0, 0, 0, 0};
    auto word = [&](uint16_t value) { bytes.push_back(uint8_t(value >> 8)); bytes.push_back(uint8_t(value)); };
    auto name = [&](std::string const& value)
    {
        size_t begin = 0;
        while (begin < value.size())
        {
            auto end = value.find('.', begin); if (end == value.npos) end = value.size();
            bytes.push_back(uint8_t(end - begin));
            bytes.insert(bytes.end(), value.begin() + begin, value.begin() + end); begin = end + 1;
        }
        bytes.push_back(0);
    };
    auto record = [&](uint16_t type, uint16_t size)
    { name(instance); word(type); word(1); word(uint16_t(ttl >> 16)); word(uint16_t(ttl)); word(size); };
    std::string const target = "xchplot2-test.local.";
    record(33, uint16_t(6 + target.size() + 1)); word(0); word(0); word(18447); name(target);
    std::string const pool_field = "pool=" + pool, version_field = "version=" + std::to_string(protocol_version),
        name_field = "name=route test";
    record(16, uint16_t(pool_field.size() + version_field.size() + name_field.size() + 3));
    for (auto const& field : {pool_field, version_field, name_field})
    { bytes.push_back(uint8_t(field.size())); bytes.insert(bytes.end(), field.begin(), field.end()); }
    auto fd = ::socket(AF_INET, SOCK_DGRAM, 0);
#ifdef _WIN32
    if (fd == INVALID_SOCKET) throw std::runtime_error("cannot create discovery fixture socket");
    struct Close { SOCKET fd; ~Close() { ::closesocket(fd); } } close{fd};
    auto option = [&](int key, auto const& value)
    { return ::setsockopt(fd, IPPROTO_IP, key, reinterpret_cast<char const*>(&value), int(sizeof(value))); };
    int yes = 1; ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char const*>(&yes), sizeof(yes));
#else
    if (fd < 0) throw std::runtime_error("cannot create discovery fixture socket");
    struct Close { int fd; ~Close() { ::close(fd); } } close{fd};
    int yes = 1; ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    auto option = [&](int key, auto const& value)
    { return ::setsockopt(fd, IPPROTO_IP, key, &value, sizeof(value)); };
#endif
    sockaddr_in local{}; local.sin_family = AF_INET; local.sin_port = htons(source_port);
    ::inet_pton(AF_INET, source.c_str(), &local.sin_addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&local), sizeof(local)) ||
        option(IP_MULTICAST_IF, local.sin_addr) || option(IP_MULTICAST_TTL, hops))
        throw std::runtime_error("cannot bind discovery fixture route");
    sockaddr_in group{}; group.sin_family = AF_INET; group.sin_port = htons(5353);
    ::inet_pton(AF_INET, "224.0.0.251", &group.sin_addr);
#ifdef _WIN32
    if (::sendto(fd, reinterpret_cast<char const*>(bytes.data()), int(bytes.size()), 0,
        reinterpret_cast<sockaddr*>(&group), sizeof(group)) != int(bytes.size()))
#else
    if (::sendto(fd, bytes.data(), bytes.size(), 0, reinterpret_cast<sockaddr*>(&group), sizeof(group)) != ssize_t(bytes.size()))
#endif
        throw std::runtime_error("cannot send discovery fixture response");
}
}
template <typename F> void eventually(F&& condition, char const* message)
{
    auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!condition())
    {
        if (std::chrono::steady_clock::now() >= until) throw std::runtime_error(message);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
int main()
{
    try
    {
        DiscoveryOptions options; options.pool_id = random_token(); options.node_id = random_token().substr(0, 32);
        auto const enrolled_pool = options.pool_id;
        options.name = "test storage"; options.server_name = "pool.local"; options.interfaces = {"lo"};
        ComputeDiscovery browser(options);
        if (browser.status() != "LAN discovery active") throw std::runtime_error(browser.status());
        options.role = "broker"; options.endpoint = {"127.0.0.1", 18444};
        auto first = std::make_unique<ComputeDiscovery>(options);
        eventually([&] { return browser.brokers().size() == 1; }, "broker was not discovered over loopback multicast");
        auto endpoint = browser.brokers().front();
        if (endpoint.host != "127.0.0.1" || endpoint.port != 18444 || endpoint.server_name != "pool.local")
            throw std::runtime_error("discovery changed the route or trusted TLS name");
        options.endpoint.port = 18445;
        auto second = std::make_unique<ComputeDiscovery>(options);
        eventually([&] { return browser.brokers().size() == 2; }, "concurrent brokers on one machine collided");
        options.pool_id = random_token(); options.endpoint.port = 18446;
        ComputeDiscovery other_pool(options);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (browser.brokers().size() != 2) throw std::runtime_error("discovery included a foreign pool");
        first.reset();
        eventually([&] { return browser.brokers().size() == 1 && browser.brokers().front().port == 18445; },
            "departing broker was not removed");
        second.reset(); eventually([&] { return browser.brokers().empty(); }, "last broker remained after goodbye");
        auto const route_instance = "route-test._xchplot2-broker._tcp.local.";
        response("127.0.0.2", options.pool_id, route_instance, 30);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (!browser.brokers().empty()) throw std::runtime_error("discovery included a foreign packet pool");
        response("127.0.0.4", enrolled_pool, route_instance, 30, 254);
        response("127.0.0.5", enrolled_pool, route_instance, 30, 255, 15353);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (!browser.brokers().empty()) throw std::runtime_error("discovery accepted a routed or wrong-port response");
        response("127.0.0.2", enrolled_pool, route_instance);
        response("127.0.0.3", enrolled_pool, route_instance);
        eventually([&] { return browser.brokers().size() == 2; }, "alternate routes for one broker were discarded");
        for (auto const& route : browser.brokers())
            if ((route.host != "127.0.0.2" && route.host != "127.0.0.3") || route.server_name != "pool.local")
                throw std::runtime_error("alternate route changed the trusted TLS name");
        eventually([&] { return browser.brokers().empty(); }, "silent broker advertisements did not expire");
        std::cout << "LAN discovery, concurrent brokers, pool filtering, alternate routes and expiry passed\n";
        return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
