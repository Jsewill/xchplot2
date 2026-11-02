#pragma once
#include "host/ComputeNetwork.hpp"
#include <memory>

namespace group_grow::pool
{
struct DiscoveredCompute
{
    std::string name, instance, role;
    Endpoint endpoint;
};
struct DiscoveryOptions
{
    std::string pool_id, node_id, name, server_name;
    std::vector<std::string> interfaces;
    // Empty role only browses. Other values are "broker" and "worker".
    std::string role;
    Endpoint endpoint;
};
// Default discovery is confined to active multicast LAN interfaces. Explicit
// names also permit a VPN or loopback interface. Advertisements are untrusted;
// every selected peer must authenticate to the enrolled pool over TLS.
class ComputeDiscovery
{
public:
    explicit ComputeDiscovery(DiscoveryOptions);
    ~ComputeDiscovery();
    ComputeDiscovery(ComputeDiscovery const&) = delete;
    ComputeDiscovery& operator=(ComputeDiscovery const&) = delete;
    std::vector<DiscoveredCompute> services() const;
    // Preserve alternate address families and interfaces for route failover.
    std::vector<Endpoint> brokers() const;
    std::string status() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace group_grow::pool
