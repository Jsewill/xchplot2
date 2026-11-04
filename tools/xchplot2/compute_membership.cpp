#include "group_commands.hpp"
#include "compute_output.hpp"
#include "host/ComputeIdentity.hpp"
#include <iostream>

namespace
{
void require(bool condition, char const* message)
{ if (!condition) throw std::invalid_argument(message); }
std::filesystem::path path(std::string const& value)
{ return std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str())); }
void help(std::string const& command)
{
    std::cout << "Usage: xchplot2 compute " << command;
    if (command == "revoke") std::cout << " NODE";
    else if (command == "renew-sign") std::cout << " REQUEST REPLACEMENT";
    else if (command != "members") std::cout << " FILE";
    std::cout << " [--directory DIR]\n";
    std::cout << pos2gpu::compute_output_help;
    if (command == "members") std::cout << "Show the signed member registry. Older enrollments appear after renewal.\n";
    else if (command == "revoke") std::cout << "Revoke a node on the authority machine. Connected members exchange the signed policy automatically.\n"
        "Use policy-export and policy-import for offline members.\n";
    else if (command == "renew-request") std::cout << "Generate a replacement key and a signed renewal request, preserving this machine's node identity.\n"
        "Transfer the request to the authority machine; expired credentials can request recovery.\n";
    else if (command == "renew-sign") std::cout << "Authorize a renewal request on the authority machine. Transfer REPLACEMENT back to the requesting machine.\n";
    else if (command == "renew-import") std::cout << "Install replacement credentials and signed policy together. Restart active compute services afterward.\n";
    else if (command == "policy-export") std::cout << "Export the current signed policy for transfer to disconnected members.\n";
    else std::cout << "Install a newer signed policy. Stale or foreign policies are rejected.\n";
}
} // namespace

int pos2gpu::run_compute_membership(int argc, char** argv)
{
    namespace pool = group_grow::pool;
    try
    {
        require(argc > 0, "missing compute membership command"); std::string command = argv[0];
        require(command == "members" || command == "revoke" || command == "renew-request" || command == "renew-sign" ||
            command == "renew-import" || command == "policy-export" || command == "policy-import", "unknown compute membership command");
        auto directory = pool::pool_directory(); std::vector<std::string> arguments;
        ComputeOutputOptions output;
        for (int i = 1; i < argc; ++i)
        {
            std::string value = argv[i];
            if (value == "--help" || value == "-h" || value == "--help-all") { help(command); return 0; }
            if (output.parse(value)) continue;
            if (value == "--directory") { require(i + 1 < argc, "missing --directory value"); directory = path(argv[++i]); }
            else { require(!value.starts_with('-'), "unknown compute membership option"); arguments.push_back(std::move(value)); }
        }
        output.validate();
        auto expected = command == "members" ? 0 : command == "renew-sign" ? 2 : 1;
        require(arguments.size() == size_t(expected), "incorrect compute membership arguments; use --help");
        auto identity = pool::load_pool(directory); require(bool(identity), "shared compute is not configured; use compute init or compute join");
        if (command == "members")
        {
            if (output.details()) std::cout << "Policy generation: " << identity->policy_generation << '\n';
            if (!identity->policy_generation) std::cout << "Legacy enrollment: export policy on the authority machine to migrate.\n";
            for (auto const& member : pool::pool_members(*identity))
                std::cout << member.node_id << ' ' << (member.revoked ? "revoked" : "active") << ' ' << member.name << '\n';
            if (output.details())
                std::cout << "Certificate expires: " << identity->certificate_expires << "\nAuthority expires: " << identity->authority_expires << '\n';
            if (identity->authority_expired) std::cout << "The pool authority has expired; create a new pool and re-enroll its members.\n";
            else if (identity->certificate_expired) std::cout << "This machine's certificate needs renewal before it can connect.\n";
        }
        else if (command == "revoke")
        { pool::revoke_pool_member(*identity, arguments[0]); std::cout << "Revoked member " << arguments[0] << ". Export policy for disconnected members.\n"; }
        else if (command == "renew-request")
        { pool::request_pool_renewal(*identity, path(arguments[0])); std::cout << "Created renewal request. Run compute renew-sign REQUEST REPLACEMENT on the authority machine.\n"; }
        else if (command == "renew-sign")
        { pool::sign_pool_renewal(*identity, path(arguments[0]), path(arguments[1])); std::cout << "Signed replacement credentials. Transfer the replacement and run compute renew-import FILE.\n"; }
        else if (command == "renew-import")
        { pool::import_pool_renewal(*identity, path(arguments[0])); std::cout << "Installed replacement credentials. Restart active compute services.\n"; }
        else if (command == "policy-export")
        { pool::export_pool_policy(*identity, path(arguments[0])); std::cout << "Exported signed pool policy.\n"; }
        else { pool::import_pool_policy(*identity, path(arguments[0])); std::cout << "Installed newer signed pool policy.\n"; }
        return 0;
    }
    catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
}
