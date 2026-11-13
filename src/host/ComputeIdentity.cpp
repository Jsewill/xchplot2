#include "host/ComputeIdentity.hpp"
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/x509v3.h>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/file.h>
#include <unistd.h>
#endif
#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>

namespace group_grow::pool
{
namespace
{
namespace fs = std::filesystem;
std::string path_text(fs::path const& path)
{
    auto value = path.u8string(); return {reinterpret_cast<char const*>(value.data()), value.size()};
}
fs::path path_from_text(std::string const& value)
{ return fs::path(reinterpret_cast<char8_t const*>(value.c_str())); }
void require(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using Certificate = std::unique_ptr<X509, decltype(&X509_free)>;
using Bio = std::unique_ptr<BIO, decltype(&BIO_free)>;
struct Identity
{
    Key key{nullptr, EVP_PKEY_free};
    Certificate cert{nullptr, X509_free};
};
struct File
{
#ifdef _WIN32
    HANDLE fd = INVALID_HANDLE_VALUE;
    ~File() { if (fd != INVALID_HANDLE_VALUE) ::CloseHandle(fd); }
#else
    int fd;
    ~File() { if (fd >= 0) ::close(fd); }
#endif
};
#ifndef _WIN32
void sync_directory(fs::path const& path)
{
    File directory{::open(path.c_str(), O_DIRECTORY | O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
    require(directory.fd >= 0 && !::fsync(directory.fd), "cannot sync pool configuration directory");
}
#endif
void remove_file(fs::path const& path, bool optional = true)
{
#ifdef _WIN32
    require(::DeleteFileW(path.c_str()) || (optional && ::GetLastError() == ERROR_FILE_NOT_FOUND), "cannot remove pool file");
#else
    require(!::unlink(path.c_str()) || (optional && errno == ENOENT), "cannot remove pool file");
#endif
}
void replace_file(fs::path const& temporary, fs::path const& destination)
{
#ifdef _WIN32
    pos2gpu::move_private_path(temporary, destination, true);
#else
    require(!::rename(temporary.c_str(), destination.c_str()), "cannot replace pool file");
    sync_directory(destination.parent_path());
#endif
}
void write_private(fs::path const& path, std::span<uint8_t const> bytes)
{
#ifdef _WIN32
    File file{pos2gpu::create_private_file_handle(path)};
    while (!bytes.empty())
    {
        DWORD count = 0;
        require(::WriteFile(file.fd, bytes.data(), DWORD(std::min<size_t>(bytes.size(), MAXDWORD)), &count, nullptr) && count,
            "cannot write private pool file"); bytes = bytes.subspan(count);
    }
    require(::FlushFileBuffers(file.fd), "cannot sync private pool file");
#else
    File file{::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600)};
    require(file.fd >= 0, "cannot create private pool file (destination must not exist)");
    while (!bytes.empty())
    {
        auto count = ::write(file.fd, bytes.data(), bytes.size());
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "cannot write private pool file"); bytes = bytes.subspan(size_t(count));
    }
    require(!::fsync(file.fd), "cannot sync private pool file");
#endif
}
void write_private(fs::path const& path, std::string const& value)
{ write_private(path, {reinterpret_cast<uint8_t const*>(value.data()), value.size()}); }
std::string read_private(fs::path const& path)
{
#ifdef _WIN32
    File file{pos2gpu::create_private_file_handle(path, GENERIC_READ, OPEN_EXISTING)};
    LARGE_INTEGER size{};
    require(::GetFileSizeEx(file.fd, &size) && size.QuadPart > 0 && size.QuadPart <= 1048576,
        "private pool files must contain at most 1 MiB");
    std::string result(size_t(size.QuadPart), '\0'); size_t position = 0;
    while (position < result.size())
    {
        DWORD count = 0;
        require(::ReadFile(file.fd, result.data() + position, DWORD(result.size() - position), &count, nullptr) && count,
            "cannot read private pool file"); position += count;
    }
    return result;
#else
    File file{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
    struct stat status{};
    require(file.fd >= 0 && !::fstat(file.fd, &status) && S_ISREG(status.st_mode) &&
        status.st_uid == ::geteuid() && !(status.st_mode & 0077) && status.st_size > 0 && status.st_size <= 1048576,
        "pool files must be owned by this user, private (mode 600), and at most 1 MiB");
    std::string result(size_t(status.st_size), '\0'); size_t position = 0;
    while (position < result.size())
    {
        auto count = ::read(file.fd, result.data() + position, result.size() - position);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "cannot read private pool file"); position += size_t(count);
    }
    return result;
#endif
}
std::string pem(X509* value)
{
    Bio output(BIO_new(BIO_s_mem()), BIO_free);
    require(output && PEM_write_bio_X509(output.get(), value) == 1, "cannot encode pool certificate");
    char* bytes = nullptr; auto size = BIO_get_mem_data(output.get(), &bytes); return {bytes, size_t(size)};
}
std::string pem(EVP_PKEY* value)
{
    Bio output(BIO_new(BIO_s_mem()), BIO_free);
    require(output && PEM_write_bio_PrivateKey(output.get(), value, nullptr, nullptr, 0, nullptr, nullptr) == 1,
        "cannot encode pool key");
    char* bytes = nullptr; auto size = BIO_get_mem_data(output.get(), &bytes); return {bytes, size_t(size)};
}
std::string expires(X509* value)
{
    Bio output(BIO_new(BIO_s_mem()), BIO_free);
    require(output && ASN1_TIME_print(output.get(), X509_get0_notAfter(value)) == 1, "cannot read pool certificate expiry");
    char* data = nullptr; auto size = BIO_get_mem_data(output.get(), &data);
    require(size > 0 && size <= 128, "invalid pool certificate expiry"); return {data, size_t(size)};
}
Certificate read_certificate(std::string const& value)
{
    Bio input(BIO_new_mem_buf(value.data(), int(value.size())), BIO_free);
    Certificate result(input ? PEM_read_bio_X509(input.get(), nullptr, nullptr, nullptr) : nullptr, X509_free);
    require(bool(result), "invalid pool certificate"); return result;
}
Key read_key(std::string const& value)
{
    Bio input(BIO_new_mem_buf(value.data(), int(value.size())), BIO_free);
    Key result(input ? PEM_read_bio_PrivateKey(input.get(), nullptr,
        [](char*, int, int, void*) { return 0; }, nullptr) : nullptr, EVP_PKEY_free);
    require(bool(result), "invalid pool key"); return result;
}
std::string fingerprint(X509* certificate)
{
    std::array<uint8_t, 32> digest{}; unsigned size = 0;
    require(X509_digest(certificate, EVP_sha256(), digest.data(), &size) == 1 && size == digest.size(),
        "cannot identify pool authority");
    std::string result;
    for (auto byte : digest) { result += "0123456789abcdef"[byte >> 4]; result += "0123456789abcdef"[byte & 15]; }
    return result;
}
std::string server_name(std::string const& pool_id)
{ return "xchplot2-pool-" + pool_id.substr(0, 32) + ".local"; }
void extension(X509* certificate, X509* issuer, int id, std::string const& value)
{
    X509V3_CTX context{}; X509V3_set_ctx(&context, issuer, certificate, nullptr, nullptr, 0);
    std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> item(
        X509V3_EXT_conf_nid(nullptr, &context, id, const_cast<char*>(value.c_str())), X509_EXTENSION_free);
    require(item && X509_add_ext(certificate, item.get(), -1) == 1, "cannot create pool certificate extension");
}
Identity issue(std::string const& node, Identity const* authority = nullptr, EVP_PKEY* public_key = nullptr)
{
    Identity result; result.key.reset(EVP_EC_gen("prime256v1")); result.cert.reset(X509_new());
    require(result.key && result.cert, "cannot create pool identity");
    std::array<unsigned char, 20> serial{};
    require(RAND_bytes(serial.data(), serial.size()) == 1, "cannot create pool certificate serial");
    serial[0] &= 0x7f;
    std::unique_ptr<BIGNUM, decltype(&BN_free)> number(BN_bin2bn(serial.data(), serial.size(), nullptr), BN_free);
    require(number && BN_to_ASN1_INTEGER(number.get(), X509_get_serialNumber(result.cert.get())) &&
        X509_set_version(result.cert.get(), 2) == 1 &&
        X509_gmtime_adj(X509_getm_notBefore(result.cert.get()), -300) &&
        X509_gmtime_adj(X509_getm_notAfter(result.cert.get()), (authority ? 1L : 10L) * 365 * 24 * 3600) &&
        X509_set_pubkey(result.cert.get(), public_key ? public_key : result.key.get()) == 1, "cannot prepare pool identity");
    if (authority && ASN1_TIME_compare(X509_get0_notAfter(result.cert.get()), X509_get0_notAfter(authority->cert.get())) > 0)
        require(X509_set1_notAfter(result.cert.get(), X509_get0_notAfter(authority->cert.get())) == 1, "cannot bound node certificate by pool authority validity");
    auto subject = X509_get_subject_name(result.cert.get());
    require(X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
        reinterpret_cast<unsigned char const*>(node.c_str()), -1, -1, 0) == 1, "cannot name pool certificate");
    auto issuer = authority ? authority->cert.get() : result.cert.get();
    auto signing_key = authority ? authority->key.get() : result.key.get();
    require(X509_set_issuer_name(result.cert.get(), X509_get_subject_name(issuer)) == 1, "cannot set pool issuer");
    extension(result.cert.get(), issuer, NID_basic_constraints, authority ? "critical,CA:FALSE" : "critical,CA:TRUE,pathlen:0");
    extension(result.cert.get(), issuer, NID_key_usage, authority ? "critical,digitalSignature" : "critical,keyCertSign,cRLSign");
    extension(result.cert.get(), issuer, NID_subject_key_identifier, "hash");
    if (authority)
    {
        extension(result.cert.get(), issuer, NID_ext_key_usage, "serverAuth,clientAuth");
        extension(result.cert.get(), issuer, NID_authority_key_identifier, "keyid:always");
        extension(result.cert.get(), issuer, NID_subject_alt_name,
            "DNS:xchplot2-" + node + ".local,DNS:" + server_name(fingerprint(issuer)));
    }
    require(X509_sign(result.cert.get(), signing_key, EVP_sha256()) > 0, "cannot sign pool certificate");
    return result;
}
void validate_name(std::string const& name)
{
    require(!name.empty() && name.size() <= 64 &&
        std::all_of(name.begin(), name.end(), [](unsigned char c) { return c >= 32 && c != 127; }),
        "pool machine name must contain 1..64 printable bytes");
}
void validate_node(std::string const& node)
{ require(node.size() == 32 && node.find_first_not_of("0123456789abcdef") == node.npos, "invalid pool node identity"); }
void validate_defaults(ComputeDefaults const& value)
{
    if (!value.broker.empty()) (void)parse_endpoint(value.broker);
    require(value.cpus <= 1024 && value.max_brokers && value.max_brokers <= 4096 &&
        (value.backend == "auto" || value.backend == "cpu" || value.backend == "gpu"),
        "invalid saved compute resources");
    require(value.interfaces.size() <= 32, "too many compute network interfaces");
    for (auto const& name : value.interfaces)
#ifdef _WIN32
        require(!name.empty() && name.size() <= 256 && std::all_of(name.begin(), name.end(), [](unsigned char c) { return c >= 32 && c != 127; }),
            "invalid compute network interface name");
#else
        require(!name.empty() && name.size() < 16 && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-") == name.npos,
            "invalid compute network interface name");
#endif
}
std::string profile(std::string const& name, std::string const& node, ComputeDefaults const& defaults)
{
    validate_name(name); validate_node(node); validate_defaults(defaults);
    std::ostringstream output;
    output << "xchplot2-compute-2\n" << std::quoted(name) << ' ' << node << '\n'
        << std::quoted(defaults.broker) << ' ' << defaults.cpus << ' ' << defaults.host_bytes << ' '
        << defaults.gpu_bytes << ' ' << defaults.backend << ' ' << defaults.discovery << ' ' << defaults.max_brokers << '\n';
    for (auto const& item : defaults.interfaces) output << std::quoted(item) << '\n';
    return output.str();
}
void validate_identity(X509* ca, Identity const& identity, std::string const& node, bool allow_expired = false)
{
    require(X509_check_ca(ca) > 0 && X509_check_private_key(identity.cert.get(), identity.key.get()) == 1,
        "pool certificate and key do not match");
    std::unique_ptr<X509_STORE, decltype(&X509_STORE_free)> store(X509_STORE_new(), X509_STORE_free);
    std::unique_ptr<X509_STORE_CTX, decltype(&X509_STORE_CTX_free)> context(X509_STORE_CTX_new(), X509_STORE_CTX_free);
    require(store && context && X509_STORE_add_cert(store.get(), ca) == 1 &&
        X509_STORE_CTX_init(context.get(), store.get(), identity.cert.get(), nullptr) == 1,
        "cannot verify pool invitation");
    X509_STORE_CTX_set_purpose(context.get(), X509_PURPOSE_SSL_CLIENT);
    if (allow_expired) X509_VERIFY_PARAM_set_flags(X509_STORE_CTX_get0_param(context.get()), X509_V_FLAG_NO_CHECK_TIME);
    auto const host = server_name(fingerprint(ca)), node_host = "xchplot2-" + node + ".local";
    require(X509_verify_cert(context.get()) == 1 &&
        X509_check_host(identity.cert.get(), host.c_str(), host.size(), X509_CHECK_FLAG_NEVER_CHECK_SUBJECT, nullptr) == 1 &&
        X509_check_host(identity.cert.get(), node_host.c_str(), node_host.size(), X509_CHECK_FLAG_NEVER_CHECK_SUBJECT, nullptr) == 1,
        "pool certificate is expired, untrusted, or belongs to another node");
}
constexpr uint32_t policy_limit = 262144;
constexpr uint32_t member_limit = 1024;
constexpr char policy_marker[] = "\n# xchplot2-compute-policy-1 ";
constexpr char request_marker[] = "\n# xchplot2-compute-renewal-request-1 ";
struct Policy
{
    uint64_t generation = 0;
    std::vector<PoolMember> members;
    std::vector<uint8_t> encoded;
};
std::span<uint8_t const> bytes(std::string const& value)
{ return {reinterpret_cast<uint8_t const*>(value.data()), value.size()}; }
std::vector<uint8_t> signature(EVP_PKEY* key, std::span<uint8_t const> data)
{
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    size_t size = 0;
    require(context && EVP_DigestSignInit(context.get(), nullptr, EVP_sha256(), nullptr, key) == 1 &&
        EVP_DigestSign(context.get(), nullptr, &size, data.data(), data.size()) == 1 && size <= 1024,
        "cannot sign pool membership data");
    std::vector<uint8_t> result(size);
    require(EVP_DigestSign(context.get(), result.data(), &size, data.data(), data.size()) == 1,
        "cannot sign pool membership data");
    result.resize(size); return result;
}
void verify_signature(EVP_PKEY* key, std::span<uint8_t const> data, std::span<uint8_t const> signed_data)
{
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(context && EVP_DigestVerifyInit(context.get(), nullptr, EVP_sha256(), nullptr, key) == 1 &&
        EVP_DigestVerify(context.get(), signed_data.data(), signed_data.size(), data.data(), data.size()) == 1,
        "invalid pool membership signature");
}
std::vector<uint8_t> signed_data(std::span<uint8_t const> payload, EVP_PKEY* key)
{
    Writer output(policy_limit); output.bytes(payload); output.bytes(signature(key, payload)); return output.data();
}
std::string hex(std::span<uint8_t const> data)
{
    std::string result; result.reserve(data.size() * 2);
    for (auto value : data) { result += "0123456789abcdef"[value >> 4]; result += "0123456789abcdef"[value & 15]; }
    return result;
}
std::string digest(std::span<uint8_t const> data)
{
    std::array<uint8_t, 32> value{}; unsigned size = 0;
    require(EVP_Digest(data.data(), data.size(), value.data(), &size, EVP_sha256(), nullptr) == 1 && size == value.size(),
        "cannot identify pool renewal request"); return hex(value);
}
std::vector<uint8_t> unhex(std::string_view value)
{
    require(value.size() <= policy_limit * 2 && !(value.size() & 1), "invalid saved pool policy");
    std::vector<uint8_t> result; result.reserve(value.size() / 2);
    auto digit = [](char c) -> uint8_t
    {
        if (c >= '0' && c <= '9') return uint8_t(c - '0');
        require(c >= 'a' && c <= 'f', "invalid saved pool policy"); return uint8_t(c - 'a' + 10);
    };
    for (size_t i = 0; i < value.size(); i += 2) result.push_back(uint8_t((digit(value[i]) << 4) | digit(value[i + 1])));
    return result;
}
Policy decode_policy(std::span<uint8_t const> encoded, X509* ca)
{
    Policy result;
    if (encoded.empty()) return result;
    require(encoded.size() <= policy_limit, "pool policy exceeds its byte limit");
    Reader envelope(encoded); auto payload = envelope.bytes(policy_limit); auto signed_bytes = envelope.bytes(1024); envelope.finish();
    Key key(X509_get_pubkey(ca), EVP_PKEY_free); verify_signature(key.get(), payload, signed_bytes);
    Reader input(payload);
    require(input.string(64) == "xchplot2-compute-policy-1" && input.string(64) == fingerprint(ca), "pool policy belongs to another authority");
    result.generation = input.u64(); auto count = input.u32();
    require(result.generation && count && count <= member_limit, "invalid pool policy generation or member count");
    for (uint32_t i = 0; i < count; ++i)
    {
        PoolMember member{input.string(32), input.string(64), input.string(64), false}; auto revoked = input.u8();
        validate_node(member.node_id); validate_name(member.name);
        require(member.certificate.size() == 64 && member.certificate.find_first_not_of("0123456789abcdef") == member.certificate.npos && revoked <= 1 &&
            (result.members.empty() || result.members.back().node_id < member.node_id), "invalid or unordered pool member");
        member.revoked = revoked; result.members.push_back(std::move(member));
    }
    input.finish(); result.encoded.assign(encoded.begin(), encoded.end()); return result;
}
void sign_policy(Policy& policy, Identity const& authority)
{
    require(policy.generation < std::numeric_limits<uint64_t>::max() && policy.members.size() <= member_limit, "pool policy capacity exhausted");
    std::sort(policy.members.begin(), policy.members.end(), [](auto const& a, auto const& b) { return a.node_id < b.node_id; });
    Writer payload(policy_limit); payload.string("xchplot2-compute-policy-1"); payload.string(fingerprint(authority.cert.get()));
    payload.u64(++policy.generation); payload.u32(uint32_t(policy.members.size()));
    for (auto const& member : policy.members)
    { payload.string(member.node_id); payload.string(member.name); payload.string(member.certificate); payload.u8(member.revoked); }
    policy.encoded = signed_data(payload.data(), authority.key.get());
}
std::string credential_text(fs::path const& directory)
{
    if (fs::exists(directory / "credentials.pem")) return read_private(directory / "credentials.pem");
    return read_private(directory / "cert.pem") + read_private(directory / "key.pem");
}
Policy saved_policy(fs::path const& directory, X509* ca)
{
    auto text = credential_text(directory); auto position = text.find(policy_marker);
    if (position == text.npos) return {};
    auto encoded = std::string_view(text).substr(position + sizeof(policy_marker) - 1);
    require(!encoded.empty() && encoded.back() == '\n', "invalid saved pool policy"); encoded.remove_suffix(1);
    return decode_policy(unhex(encoded), ca);
}
Identity saved_identity(fs::path const& directory)
{
    auto text = credential_text(directory); Identity identity;
    identity.cert = read_certificate(text); identity.key = read_key(text); return identity;
}
void save_credentials(fs::path const& directory, Identity const& identity, Policy const& policy)
{
    auto temporary = directory / ("credentials.partial." + random_token());
    try
    {
        auto encoded = pem(identity.cert.get()) + pem(identity.key.get());
        if (!policy.encoded.empty()) encoded += policy_marker + hex(policy.encoded) + '\n';
        write_private(temporary, encoded);
        replace_file(temporary, directory / "credentials.pem");
    }
    catch (...) { std::error_code ignored; fs::remove(temporary, ignored); throw; }
}
struct PoolLock
{
    File file;
#ifdef _WIN32
    explicit PoolLock(fs::path const& directory) : file{pos2gpu::create_private_file_handle(directory / "membership.lock", GENERIC_READ | GENERIC_WRITE, OPEN_ALWAYS)}
    {
        OVERLAPPED position{};
        require(::LockFileEx(file.fd, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &position), "cannot lock private pool membership");
    }
#else
    explicit PoolLock(fs::path const& directory) : file{::open((directory / "membership.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600)}
    {
        struct stat status{};
        require(file.fd >= 0 && !::fstat(file.fd, &status) && S_ISREG(status.st_mode) && status.st_uid == ::geteuid() && !(status.st_mode & 0077) &&
            !::flock(file.fd, LOCK_EX), "cannot lock private pool membership");
    }
#endif
};
Identity authority_identity(PoolIdentity const& pool)
{
    require(pool.can_invite, "run this command on the machine that initialized the pool");
    Identity result; result.cert = read_certificate(read_private(pool.directory / "ca.pem")); result.key = read_key(read_private(pool.directory / "authority.key"));
    require(X509_check_private_key(result.cert.get(), result.key.get()) == 1, "pool authority key does not match");
    require(X509_cmp_current_time(X509_get0_notAfter(result.cert.get())) > 0 && X509_cmp_current_time(X509_get0_notBefore(result.cert.get())) <= 0,
        "pool authority certificate is expired or not yet valid; create a new pool and re-enroll its members"); return result;
}
void register_member(Policy& policy, std::string const& node, std::string const& name, X509* certificate)
{
    auto found = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& member) { return member.node_id == node; });
    if (found == policy.members.end())
    { require(policy.members.size() < member_limit, "pool membership registry is full"); policy.members.push_back({node, name, fingerprint(certificate), false}); }
    else { require(!found->revoked, "pool member has been revoked"); found->certificate = fingerprint(certificate); }
}
Policy authority_policy(PoolIdentity const& pool, Identity const& authority)
{
    require(!fs::exists(pool.directory / "renewal.pending"), "finish the pending compute renew-sign export before changing pool membership");
    auto result = saved_policy(pool.directory, authority.cert.get());
    if (!result.generation)
    {
        auto identity = saved_identity(pool.directory); register_member(result, pool.node_id, pool.name, identity.cert.get());
    }
    return result;
}
bool active_member(Policy const& policy, std::string const& node, std::string const& certificate)
{
    auto found = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& member) { return member.node_id == node; });
    return found != policy.members.end() && !found->revoked && found->certificate == certificate;
}
struct Renewal
{
    Identity identity;
    Policy policy;
    std::string request;
};
Renewal decode_renewal(std::span<uint8_t const> encoded, X509* ca, std::string const& pool, std::string const& node)
{
    Reader envelope(encoded); auto payload = envelope.bytes(policy_limit); auto signed_bytes = envelope.bytes(1024); envelope.finish();
    Key authority(X509_get_pubkey(ca), EVP_PKEY_free); verify_signature(authority.get(), payload, signed_bytes);
    Reader input(payload); Renewal result;
    require(input.string(64) == "xchplot2-compute-renewal-2" && input.string(64) == pool && input.string(32) == node,
        "replacement credentials belong to another pool or node");
    result.request = input.string(64);
    require(result.request.size() == 64 && result.request.find_first_not_of("0123456789abcdef") == result.request.npos,
        "invalid replacement renewal request identity");
    result.identity.cert = read_certificate(input.string(65536)); result.identity.key.reset(X509_get_pubkey(result.identity.cert.get()));
    result.policy = decode_policy(input.bytes(policy_limit), ca); input.finish(); validate_identity(ca, result.identity, node);
    require(active_member(result.policy, node, fingerprint(result.identity.cert.get())), "replacement certificate is not active in its signed policy");
    return result;
}
void export_private(fs::path const& destination, std::span<uint8_t const> value)
{
    auto temporary = destination; temporary += ".partial." + random_token();
    try
    {
        write_private(temporary, value);
#ifdef _WIN32
        pos2gpu::move_private_path(temporary, destination, false);
#else
        require(!::link(temporary.c_str(), destination.c_str()), "pool export destination already exists or cannot be published");
        remove_file(temporary, false);
        sync_directory(destination.has_parent_path() ? destination.parent_path() : fs::path("."));
#endif
    }
    catch (...) { std::error_code ignored; fs::remove(temporary, ignored); throw; }
}
PoolIdentity publish(fs::path const& directory, std::string const& name, std::string const& node,
    std::string const& ca, Identity const& identity, std::string const& authority = {}, Policy const& policy = {})
{
    require(!directory.empty() && !directory.filename().empty() && directory.filename() != "." && directory.filename() != "..",
        "pool directory must have a name");
    auto const parent = directory.has_parent_path() ? directory.parent_path() : fs::path(".");
    fs::create_directories(parent);
#ifdef _WIN32
    auto temporary = directory; temporary += ".partial." + random_token();
    pos2gpu::create_private_directory(temporary);
#else
    std::string temporary = directory.string() + ".partial.XXXXXX";
    require(::mkdtemp(temporary.data()), "cannot create private pool configuration directory");
#endif
    try
    {
        write_private(fs::path(temporary) / "ca.pem", ca);
        auto encoded = pem(identity.cert.get()) + pem(identity.key.get());
        if (!policy.encoded.empty()) encoded += policy_marker + hex(policy.encoded) + '\n';
        write_private(fs::path(temporary) / "credentials.pem", encoded);
        if (!authority.empty()) write_private(fs::path(temporary) / "authority.key", authority);
        write_private(fs::path(temporary) / "profile", profile(name, node, {}));
#ifdef _WIN32
        pos2gpu::move_private_path(temporary, directory, false);
#else
        sync_directory(temporary);
        require(!::syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(), AT_FDCWD, directory.c_str(), RENAME_NOREPLACE),
            "pool configuration already exists or could not be published");
        sync_directory(parent);
#endif
    }
    catch (...) { std::error_code ignored; fs::remove_all(temporary, ignored); throw; }
    return *load_pool(directory);
}
} // namespace

std::filesystem::path pool_directory()
{
#ifdef _WIN32
    auto value = ::_wgetenv(L"XDG_CONFIG_HOME");
    if (value && *value) return fs::path(value) / "xchplot2/compute";
    value = ::_wgetenv(L"LOCALAPPDATA");
    require(value && *value, "LOCALAPPDATA or XDG_CONFIG_HOME is required for saved compute setup");
    return fs::path(value) / "xchplot2/compute";
#else
    auto value = std::getenv("XDG_CONFIG_HOME");
    if (value && *value) return fs::path(value) / "xchplot2/compute";
    value = std::getenv("HOME"); require(value && *value, "HOME or XDG_CONFIG_HOME is required for saved compute setup");
    return fs::path(value) / ".config/xchplot2/compute";
#endif
}
std::optional<PoolIdentity> load_pool(fs::path const& directory)
{
#ifdef _WIN32
    auto attributes = ::GetFileAttributesW(directory.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
    {
        auto error = ::GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return {};
        throw std::runtime_error("cannot inspect pool configuration directory");
    }
    File saved{::CreateFileW(directory.c_str(), READ_CONTROL | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
    pos2gpu::validate_private_file_handle(saved.fd, true);
#else
    struct stat status{};
    if (::lstat(directory.c_str(), &status))
    { if (errno == ENOENT) return {}; throw std::runtime_error("cannot inspect pool configuration directory"); }
    require(S_ISDIR(status.st_mode) && status.st_uid == ::geteuid() && !(status.st_mode & 0077),
        "pool configuration directory must be owned by this user and private (mode 700)");
#endif
    PoolIdentity result; result.directory = directory;
    std::istringstream input(read_private(directory / "profile")); std::string version;
    auto& defaults = result.defaults;
    require(bool(input >> version >> std::quoted(result.name) >> result.node_id >> std::quoted(defaults.broker)
        >> defaults.cpus >> defaults.host_bytes >> defaults.gpu_bytes >> defaults.backend >> defaults.discovery) &&
        (version == "xchplot2-compute-1" || version == "xchplot2-compute-2"), "invalid saved pool profile");
    if (version == "xchplot2-compute-2") require(bool(input >> defaults.max_brokers), "invalid saved compute broker limit");
    input >> std::ws;
    while (input.peek() != std::char_traits<char>::eof())
    {
        std::string name; require(bool(input >> std::quoted(name)), "invalid saved pool interface");
        defaults.interfaces.push_back(std::move(name)); input >> std::ws;
    }
    validate_name(result.name); validate_node(result.node_id); validate_defaults(defaults);
    auto ca = read_certificate(read_private(directory / "ca.pem"));
    auto identity = saved_identity(directory); validate_identity(ca.get(), identity, result.node_id, true);
    result.certificate_expired = X509_cmp_current_time(X509_get0_notAfter(identity.cert.get())) <= 0 ||
        X509_cmp_current_time(X509_get0_notBefore(identity.cert.get())) > 0;
    result.authority_expired = X509_cmp_current_time(X509_get0_notAfter(ca.get())) <= 0 || X509_cmp_current_time(X509_get0_notBefore(ca.get())) > 0;
    result.certificate_expires = expires(identity.cert.get()); result.authority_expires = expires(ca.get());
    result.pool_id = fingerprint(ca.get()); result.server_name = server_name(result.pool_id);
    auto bundled = fs::exists(directory / "credentials.pem");
    result.tls = {path_text(directory / "ca.pem"), path_text(directory / (bundled ? "credentials.pem" : "cert.pem")),
        path_text(directory / (bundled ? "credentials.pem" : "key.pem"))};
    result.tls.membership_directory = path_text(directory); result.policy_generation = saved_policy(directory, ca.get()).generation;
    result.can_invite = fs::exists(directory / "authority.key"); return result;
}
PoolIdentity create_pool(fs::path const& directory, std::string const& name)
{
    validate_name(name); auto ca = issue("xchplot2 compute pool");
    auto node = random_token().substr(0, 32); auto identity = issue(node, &ca);
    Policy policy; register_member(policy, node, name, identity.cert.get()); sign_policy(policy, ca);
    return publish(directory, name, node, pem(ca.cert.get()), identity, pem(ca.key.get()), policy);
}
void invite_pool(PoolIdentity const& pool, fs::path const& invitation, std::string const& name)
{
    validate_name(name); PoolLock lock(pool.directory); auto ca = authority_identity(pool);
    require(!fs::exists(invitation), "invitation destination already exists");
    auto node = random_token().substr(0, 32); auto identity = issue(node, &ca);
    auto policy = authority_policy(pool, ca); register_member(policy, node, name, identity.cert.get()); sign_policy(policy, ca);
    Writer output; output.string("xchplot2-compute-invitation-2"); output.string(name); output.string(node);
    output.string(pem(ca.cert.get())); output.string(pem(identity.cert.get())); output.string(pem(identity.key.get()));
    output.bytes(policy.encoded); auto local = saved_identity(pool.directory); save_credentials(pool.directory, local, policy);
    export_private(invitation, output.data());
}
PoolIdentity join_pool(fs::path const& invitation, fs::path const& directory)
{
    auto encoded = read_private(invitation);
    Reader input({reinterpret_cast<uint8_t const*>(encoded.data()), encoded.size()});
    auto version = input.string(64);
    require(version == "xchplot2-compute-invitation-1" || version == "xchplot2-compute-invitation-2", "unsupported pool invitation");
    auto name = input.string(64), node = input.string(32); validate_name(name); validate_node(node);
    auto ca_text = input.string(65536); auto ca = read_certificate(ca_text);
    Identity identity; identity.cert = read_certificate(input.string(65536)); identity.key = read_key(input.string(65536));
    Policy policy; if (version == "xchplot2-compute-invitation-2") policy = decode_policy(input.bytes(policy_limit), ca.get());
    input.finish(); validate_identity(ca.get(), identity, node);
    auto member = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& value) { return value.node_id == node; });
    require(!policy.generation || (member != policy.members.end() && !member->revoked && member->certificate == fingerprint(identity.cert.get())),
        "invitation is not an active member of its signed policy");
    return publish(directory, name, node, ca_text, identity, {}, policy);
}
void validate_compute_defaults(ComputeDefaults const& defaults) { validate_defaults(defaults); }
void configure_pool(PoolIdentity const& pool, ComputeDefaults const& defaults)
{
    auto encoded = profile(pool.name, pool.node_id, defaults);
    auto temporary = pool.directory / ("profile.partial." + random_token());
    try
    {
        write_private(temporary, encoded);
        replace_file(temporary, pool.directory / "profile");
    }
    catch (...) { std::error_code ignored; fs::remove(temporary, ignored); throw; }
}
WorkerIntent load_worker_intent(PoolIdentity const& pool)
{
    auto const path = pool.directory / "worker-control";
    if (!fs::exists(fs::symlink_status(path))) return WorkerIntent::Active;
    std::istringstream input(read_private(path));
    std::string version, pool_id, node, intent;
    require(bool(input >> version >> pool_id >> node >> intent) && version == "xchplot2-compute-control-1" &&
        pool_id == pool.pool_id && node == pool.node_id, "worker control belongs to another pool or node");
    input >> std::ws; require(input.eof(), "invalid saved worker control");
    if (intent == "active") return WorkerIntent::Active;
    if (intent == "paused") return WorkerIntent::Paused;
    require(intent == "draining", "invalid saved worker intent"); return WorkerIntent::Draining;
}
void save_worker_intent(PoolIdentity const& pool, WorkerIntent intent)
{
    require(intent >= WorkerIntent::Active && intent <= WorkerIntent::Draining, "invalid worker intent");
    PoolLock lock(pool.directory);
    auto const current = load_pool(pool.directory);
    require(current && current->pool_id == pool.pool_id && current->node_id == pool.node_id,
        "worker control enrollment changed");
    auto const value = intent == WorkerIntent::Active ? "active" : intent == WorkerIntent::Paused ? "paused" : "draining";
    auto const encoded = "xchplot2-compute-control-1\n" + pool.pool_id + '\n' + pool.node_id + '\n' + value + '\n';
    auto const temporary = pool.directory / ("worker-control.partial." + random_token());
    try
    {
        write_private(temporary, encoded);
        replace_file(temporary, pool.directory / "worker-control");
    }
    catch (...) { std::error_code ignored; fs::remove(temporary, ignored); throw; }
}
std::vector<PoolMember> pool_members(PoolIdentity const& pool)
{
    auto ca = read_certificate(read_private(pool.directory / "ca.pem")); return saved_policy(pool.directory, ca.get()).members;
}
void revoke_pool_member(PoolIdentity const& pool, std::string const& node)
{
    validate_node(node); PoolLock lock(pool.directory); auto authority = authority_identity(pool);
    auto policy = authority_policy(pool, authority);
    require(node != pool.node_id, "the pool authority cannot revoke its own node");
    auto found = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& value) { return value.node_id == node; });
    // Legacy members were never recorded by the original invitation format.
    if (found == policy.members.end())
    {
        require(policy.members.size() < member_limit, "pool membership registry is full");
        policy.members.push_back({node, "legacy member", std::string(64, '0'), true});
    }
    else { require(!found->revoked, "pool member is already revoked"); found->revoked = true; }
    sign_policy(policy, authority); auto identity = saved_identity(pool.directory); save_credentials(pool.directory, identity, policy);
}
std::vector<uint8_t> pool_policy(std::string const& directory)
{
    auto saved = path_from_text(directory);
    auto ca = read_certificate(read_private(saved / "ca.pem")); return saved_policy(saved, ca.get()).encoded;
}
void accept_pool_policy(std::string const& directory, std::span<uint8_t const> encoded)
{
    if (encoded.empty()) return;
    auto saved = path_from_text(directory); PoolLock lock(saved); auto ca = read_certificate(read_private(saved / "ca.pem"));
    auto incoming = decode_policy(encoded, ca.get()); auto current = saved_policy(saved, ca.get());
    if (incoming.generation < current.generation) return; // Peers may legitimately lag; never roll back.
    require(incoming.generation != current.generation || incoming.encoded == current.encoded, "conflicting pool policies have the same generation");
    if (incoming.generation == current.generation) return;
    auto identity = saved_identity(saved); save_credentials(saved, identity, incoming);
}
void check_pool_peer(std::string const& directory, std::string const& node, std::string const& certificate)
{
    auto saved = path_from_text(directory);
    validate_node(node); auto ca = read_certificate(read_private(saved / "ca.pem")); auto policy = saved_policy(saved, ca.get());
    auto found = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& value) { return value.node_id == node; });
    require(found == policy.members.end() || (!found->revoked && found->certificate == certificate), "pool member or certificate has been revoked");
}
void export_pool_policy(PoolIdentity const& pool, fs::path const& destination)
{
    PoolLock lock(pool.directory); auto ca = read_certificate(read_private(pool.directory / "ca.pem")); auto policy = saved_policy(pool.directory, ca.get());
    if (!policy.generation)
    {
        auto authority = authority_identity(pool); policy = authority_policy(pool, authority); sign_policy(policy, authority);
        auto identity = saved_identity(pool.directory); save_credentials(pool.directory, identity, policy);
    }
    export_private(destination, policy.encoded);
}
void import_pool_policy(PoolIdentity const& pool, fs::path const& source)
{
    auto encoded = read_private(source); PoolLock lock(pool.directory);
    auto ca = read_certificate(read_private(pool.directory / "ca.pem")); auto incoming = decode_policy(bytes(encoded), ca.get());
    auto current = saved_policy(pool.directory, ca.get()); require(incoming.generation > current.generation, "pool policy is stale or already installed");
    auto identity = saved_identity(pool.directory); save_credentials(pool.directory, identity, incoming);
}
void request_pool_renewal(PoolIdentity const& pool, fs::path const& destination)
{
    PoolLock lock(pool.directory); auto pending = pool.directory / "renewal.key";
    std::vector<uint8_t> request;
    if (fs::exists(pending))
    {
        auto candidate = read_key(read_private(pending)); auto current = saved_identity(pool.directory);
        // A completed import may have exited before removing its duplicate
        // pending key. The atomic credential bundle proves it is installed.
        if (EVP_PKEY_eq(candidate.get(), current.key.get()) == 1) remove_file(pending, false);
    }
    if (fs::exists(pending))
    {
        auto saved = read_private(pending); auto position = saved.find(request_marker);
        require(position != saved.npos && saved.back() == '\n', "pending renewal key has no recoverable request");
        auto encoded = std::string_view(saved).substr(position + sizeof(request_marker) - 1); encoded.remove_suffix(1);
        request = unhex(encoded);
    }
    else
    {
        auto identity = saved_identity(pool.directory);
        Key replacement(EVP_EC_gen("prime256v1"), EVP_PKEY_free); require(bool(replacement), "cannot generate replacement pool key");
        Bio public_key(BIO_new(BIO_s_mem()), BIO_free);
        require(public_key && PEM_write_bio_PUBKEY(public_key.get(), replacement.get()) == 1, "cannot encode replacement public key");
        char* data = nullptr; auto size = BIO_get_mem_data(public_key.get(), &data);
        Writer payload(policy_limit); payload.string("xchplot2-compute-renewal-request-1"); payload.string(pool.pool_id);
        payload.string(pool.node_id); payload.string(pool.name); payload.string(pem(identity.cert.get())); payload.string({data, size_t(size)});
        request = signed_data(payload.data(), identity.key.get());
        auto encoded = pem(replacement.get()) + request_marker + hex(request) + '\n'; export_private(pending, bytes(encoded));
    }
    if (fs::exists(destination)) require(std::ranges::equal(bytes(read_private(destination)), request), "renewal request destination contains different data");
    else export_private(destination, request);
}
void sign_pool_renewal(PoolIdentity const& pool, fs::path const& request, fs::path const& replacement)
{
    auto encoded = read_private(request); Reader envelope(bytes(encoded)); auto payload = envelope.bytes(policy_limit); auto signed_bytes = envelope.bytes(1024); envelope.finish();
    Reader input(payload);
    require(input.string(64) == "xchplot2-compute-renewal-request-1" && input.string(64) == pool.pool_id, "renewal request belongs to another pool");
    auto node = input.string(32), name = input.string(64); validate_node(node); validate_name(name);
    Identity old; old.cert = read_certificate(input.string(65536)); old.key.reset(X509_get_pubkey(old.cert.get()));
    auto public_text = input.string(65536); input.finish(); verify_signature(old.key.get(), payload, signed_bytes);
    Bio public_input(BIO_new_mem_buf(public_text.data(), int(public_text.size())), BIO_free);
    Key public_key(public_input ? PEM_read_bio_PUBKEY(public_input.get(), nullptr, nullptr, nullptr) : nullptr, EVP_PKEY_free);
    require(public_key && EVP_PKEY_is_a(public_key.get(), "EC") && EVP_PKEY_get_bits(public_key.get()) == 256, "invalid replacement pool public key");
    PoolLock lock(pool.directory); auto authority = authority_identity(pool); validate_identity(authority.cert.get(), old, node, true);
    auto journal = pool.directory / "renewal.pending";
    if (fs::exists(journal))
    {
        auto pending = read_private(journal); Reader record(bytes(pending));
        auto original = record.bytes(policy_limit), response = record.bytes(policy_limit); record.finish();
        require(std::ranges::equal(original, bytes(encoded)), "finish the pending renewal request before signing another");
        auto recovery = decode_renewal(response, authority.cert.get(), pool.pool_id, node);
        require(recovery.request == digest(bytes(encoded)) && EVP_PKEY_eq(recovery.identity.key.get(), public_key.get()) == 1,
            "pending replacement does not match its renewal request");
        auto const& next_policy = recovery.policy;
        auto current = saved_policy(pool.directory, authority.cert.get());
        require(next_policy.generation >= current.generation && (next_policy.generation != current.generation || next_policy.encoded == current.encoded),
            "pending renewal policy conflicts with current policy");
        auto local = saved_identity(pool.directory); save_credentials(pool.directory, local, next_policy);
        if (fs::exists(replacement)) require(std::ranges::equal(bytes(read_private(replacement)), response), "replacement destination contains different data");
        else export_private(replacement, response);
        remove_file(journal, false);
#ifndef _WIN32
        sync_directory(pool.directory);
#endif
        return;
    }
    auto policy = authority_policy(pool, authority);
    if (fs::exists(replacement))
    {
        auto completed = read_private(replacement); auto response = decode_renewal(bytes(completed), authority.cert.get(), pool.pool_id, node);
        require(response.request == digest(bytes(encoded)) && EVP_PKEY_eq(response.identity.key.get(), public_key.get()) == 1,
            "replacement destination belongs to another renewal request");
        require(response.policy.generation <= policy.generation &&
            (response.policy.generation != policy.generation || response.policy.encoded == policy.encoded) &&
            active_member(policy, node, fingerprint(response.identity.cert.get())), "replacement credentials have been revoked or superseded");
        return;
    }
    auto found = std::find_if(policy.members.begin(), policy.members.end(), [&](auto const& value) { return value.node_id == node; });
    require(found == policy.members.end() || (!found->revoked && found->certificate == fingerprint(old.cert.get())), "renewal request uses revoked or superseded credentials");
    auto next = issue(node, &authority, public_key.get()); register_member(policy, node, name, next.cert.get()); sign_policy(policy, authority);
    Writer output(policy_limit); output.string("xchplot2-compute-renewal-2"); output.string(pool.pool_id); output.string(node); output.string(digest(bytes(encoded)));
    output.string(pem(next.cert.get())); output.bytes(policy.encoded);
    auto response = signed_data(output.data(), authority.key.get()); Writer record;
    record.bytes(bytes(encoded)); record.bytes(response); export_private(journal, record.data());
    auto local = saved_identity(pool.directory); save_credentials(pool.directory, local, policy);
    export_private(replacement, response);
    remove_file(journal, false);
#ifndef _WIN32
    sync_directory(pool.directory);
#endif
}
void import_pool_renewal(PoolIdentity const& pool, fs::path const& replacement)
{
    auto encoded = read_private(replacement);
    PoolLock lock(pool.directory); auto ca = read_certificate(read_private(pool.directory / "ca.pem"));
    auto response = decode_renewal(bytes(encoded), ca.get(), pool.pool_id, pool.node_id);
    auto& identity = response.identity; auto& policy = response.policy;
    auto pending = pool.directory / "renewal.key";
    if (fs::exists(pending)) identity.key = read_key(read_private(pending));
    else identity.key = std::move(saved_identity(pool.directory).key);
    validate_identity(ca.get(), identity, pool.node_id);
    auto current = saved_policy(pool.directory, ca.get());
    require(policy.generation != current.generation || policy.encoded == current.encoded, "replacement pool policy is conflicting");
    // A later unrelated enrollment or revocation must not strand an already
    // signed replacement. Keep the newer policy and verify it still grants
    // this exact replacement certificate; never install the older policy.
    if (policy.generation < current.generation) policy = std::move(current);
    require(active_member(policy, pool.node_id, fingerprint(identity.cert.get())), "replacement certificate is not active in its policy");
    save_credentials(pool.directory, identity, policy);
    remove_file(pending);
    for (auto name : {"cert.pem", "key.pem"})
        remove_file(pool.directory / name);
#ifndef _WIN32
    sync_directory(pool.directory);
#endif
}
} // namespace group_grow::pool
