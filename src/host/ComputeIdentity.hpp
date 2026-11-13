#pragma once
#include "host/ComputeNetwork.hpp"
#include <filesystem>
#include <optional>

namespace group_grow::pool
{
struct ComputeDefaults
{
    std::string broker;
    std::vector<std::string> interfaces;
    uint32_t cpus = 0;
    uint32_t max_brokers = 128;
    uint64_t host_bytes = 0, gpu_bytes = 0;
    std::string backend = "auto";
    bool discovery = true;
};
struct PoolIdentity
{
    std::filesystem::path directory;
    std::string name, node_id, pool_id, server_name;
    TlsOptions tls;
    ComputeDefaults defaults;
    bool can_invite = false;
    bool certificate_expired = false;
    bool authority_expired = false;
    std::string certificate_expires, authority_expires;
    uint64_t policy_generation = 0;
};
struct PoolMember
{
    std::string node_id, name, certificate;
    bool revoked = false;
};
enum class WorkerIntent : uint8_t { Active = 1, Paused, Draining };
// Local operator intent is bound to the enrolled pool/node, not a renewable
// certificate or transient worker instance. Only the owning worker applies it.
WorkerIntent load_worker_intent(PoolIdentity const&);
void save_worker_intent(PoolIdentity const&, WorkerIntent);
std::filesystem::path pool_directory();
std::optional<PoolIdentity> load_pool(std::filesystem::path const&);
PoolIdentity create_pool(std::filesystem::path const&, std::string const& name);
void invite_pool(PoolIdentity const&, std::filesystem::path const& invitation, std::string const& name);
PoolIdentity join_pool(std::filesystem::path const& invitation, std::filesystem::path const& directory);
void configure_pool(PoolIdentity const&, ComputeDefaults const&);
void validate_compute_defaults(ComputeDefaults const&);
std::vector<PoolMember> pool_members(PoolIdentity const&);
void revoke_pool_member(PoolIdentity const&, std::string const& node);
void export_pool_policy(PoolIdentity const&, std::filesystem::path const&);
void import_pool_policy(PoolIdentity const&, std::filesystem::path const&);
void request_pool_renewal(PoolIdentity const&, std::filesystem::path const&);
void sign_pool_renewal(PoolIdentity const&, std::filesystem::path const& request,
                       std::filesystem::path const& replacement);
void import_pool_renewal(PoolIdentity const&, std::filesystem::path const& replacement);
// Transport integration: signed policy bytes only; never exports a private key.
std::vector<uint8_t> pool_policy(std::string const& directory);
void accept_pool_policy(std::string const& directory, std::span<uint8_t const>);
void check_pool_peer(std::string const& directory, std::string const& node, std::string const& certificate);
} // namespace group_grow::pool
