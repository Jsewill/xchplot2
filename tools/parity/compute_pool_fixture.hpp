// Local ephemeral CA/certificates for standalone network-compute tests.
#pragma once
#include "host/ComputeNetwork.hpp"
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <unistd.h>
#endif

namespace pool_test
{
using group_grow::pool::TlsOptions;
namespace fs = std::filesystem;
inline std::string utf8_path(fs::path const& path)
{ auto const value = path.u8string(); return {reinterpret_cast<char const*>(value.data()), value.size()}; }
inline void require(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
struct Directory
{
    fs::path path;
    Directory()
    {
#ifdef _WIN32
        for (unsigned attempt = 0; attempt < 32; ++attempt)
        {
            path = fs::temp_directory_path() / ("grow-pool-test-" + group_grow::pool::random_token());
            try { pos2gpu::create_private_directory(path); return; }
            catch (std::system_error const& error)
            { if (error.code().value() != ERROR_ALREADY_EXISTS) throw; }
        }
        throw std::runtime_error("cannot create unique pool test directory");
#else
        std::string value = (fs::temp_directory_path() / "grow-pool-test-XXXXXX").string();
        require(::mkdtemp(value.data()), "cannot create pool test directory"); path = value;
#endif
    }
    ~Directory() { std::error_code error; fs::remove_all(path, error); }
};
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using Certificate = std::unique_ptr<X509, decltype(&X509_free)>;
struct Identity
{
    Key key{nullptr, EVP_PKEY_free};
    Certificate cert{nullptr, X509_free};
    fs::path key_file, cert_file;
};
inline void extension(X509* certificate, X509* issuer, int id, char const* value)
{
    X509V3_CTX context{}; X509V3_set_ctx(&context, issuer, certificate, nullptr, nullptr, 0);
    std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> item(
        X509V3_EXT_conf_nid(nullptr, &context, id, const_cast<char*>(value)), X509_EXTENSION_free);
    require(item && X509_add_ext(certificate, item.get(), -1) == 1, "cannot create test certificate extension");
}
inline Identity issue(fs::path const& path, std::string const& name, Identity const* issuer = nullptr, bool san = true)
{
    Identity result; result.key.reset(EVP_EC_gen("prime256v1")); result.cert.reset(X509_new());
    require(result.key && result.cert, "cannot create test TLS identity");
    require(X509_set_version(result.cert.get(), 2) == 1 && ASN1_INTEGER_set(X509_get_serialNumber(result.cert.get()), 1) == 1 &&
        X509_gmtime_adj(X509_getm_notBefore(result.cert.get()), -60) &&
        X509_gmtime_adj(X509_getm_notAfter(result.cert.get()), 3600) &&
        X509_set_pubkey(result.cert.get(), result.key.get()) == 1, "cannot prepare test TLS identity");
    auto subject = X509_get_subject_name(result.cert.get());
    require(X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
        reinterpret_cast<unsigned char const*>(name.c_str()), -1, -1, 0) == 1, "cannot name test certificate");
    auto signing_cert = issuer ? issuer->cert.get() : result.cert.get();
    auto signing_key = issuer ? issuer->key.get() : result.key.get();
    require(X509_set_issuer_name(result.cert.get(), X509_get_subject_name(signing_cert)) == 1, "cannot set test issuer");
    extension(result.cert.get(), signing_cert, NID_basic_constraints, issuer ? "critical,CA:FALSE" : "critical,CA:TRUE");
    extension(result.cert.get(), signing_cert, NID_key_usage, issuer ? "critical,digitalSignature" : "critical,keyCertSign,cRLSign");
    if (issuer)
    {
        extension(result.cert.get(), signing_cert, NID_ext_key_usage, "serverAuth,clientAuth");
        if (san) extension(result.cert.get(), signing_cert, NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1");
    }
    require(X509_sign(result.cert.get(), signing_key, EVP_sha256()) > 0, "cannot sign test certificate");
    result.key_file = path / (name + ".key"); result.cert_file = path / (name + ".crt");
    std::unique_ptr<BIO, decltype(&BIO_free)> key_file(BIO_new_file(utf8_path(result.key_file).c_str(), "w"), BIO_free);
    std::unique_ptr<BIO, decltype(&BIO_free)> cert_file(BIO_new_file(utf8_path(result.cert_file).c_str(), "w"), BIO_free);
    require(key_file && cert_file && PEM_write_bio_PrivateKey(key_file.get(), result.key.get(), nullptr, nullptr, 0, nullptr, nullptr) == 1 &&
        PEM_write_bio_X509(cert_file.get(), result.cert.get()) == 1, "cannot save test TLS identity");
    return result;
}
inline TlsOptions tls(Identity const& ca, Identity const& identity)
{ return {utf8_path(ca.cert_file), utf8_path(identity.cert_file), utf8_path(identity.key_file), 500, 65536}; }
} // namespace pool_test
