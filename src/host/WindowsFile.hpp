#pragma once

// Windows counterpart of mkstemp's exclusive creation and owner-only access.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#include <aclapi.h>
#include <fcntl.h>
#include <io.h>
#ifdef interface
#undef interface
#endif

#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace pos2gpu {

struct PrivateFileSecurity
{
    HANDLE token = nullptr;
    LPWSTR sid_text = nullptr;
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    std::vector<unsigned char> token_user;
    SECURITY_ATTRIBUTES attributes{};
    PrivateFileSecurity()
    {
        try
        {
            auto fail = [] { throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "private file security"); };
            if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &token)) fail();
            DWORD size = 0; ::GetTokenInformation(token, TokenUser, nullptr, 0, &size);
            if (!size || size > 65536) fail();
            token_user.resize(size);
            if (!::GetTokenInformation(token, TokenUser, token_user.data(), size, &size)) fail();
            if (!::ConvertSidToStringSidW(sid(), &sid_text)) fail();
            // Explicit ownership and a protected DACL also apply when elevated.
            std::wstring const sddl = L"O:" + std::wstring(sid_text) + L"D:P(A;;FA;;;" + sid_text + L")";
            if (!::ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) fail();
            attributes = {sizeof(attributes), descriptor, FALSE};
        }
        catch (...) { close(); throw; }
    }
    ~PrivateFileSecurity() { close(); }
    PrivateFileSecurity(PrivateFileSecurity const&) = delete;
    PrivateFileSecurity& operator=(PrivateFileSecurity const&) = delete;
    PSID sid() const { return reinterpret_cast<TOKEN_USER const*>(token_user.data())->User.Sid; }
private:
    void close()
    {
        if (descriptor) ::LocalFree(descriptor);
        if (sid_text) ::LocalFree(sid_text);
        if (token) ::CloseHandle(token);
    }
};

inline void validate_private_file_handle(HANDLE file, bool directory)
{
    BY_HANDLE_FILE_INFORMATION information{};
    if (file == INVALID_HANDLE_VALUE || ::GetFileType(file) != FILE_TYPE_DISK || !::GetFileInformationByHandle(file, &information) ||
        bool(information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != directory ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
        throw std::runtime_error("private storage must be a regular file or directory, without reparse points");
    DWORD volume_flags = 0;
    if (!::GetVolumeInformationByHandleW(file, nullptr, 0, nullptr, nullptr, &volume_flags, nullptr, 0) || !(volume_flags & FILE_PERSISTENT_ACLS))
        throw std::runtime_error("private files require an ACL-capable filesystem (NTFS/ReFS)");
    PrivateFileSecurity security; PSID owner = nullptr; PACL acl = nullptr; PSECURITY_DESCRIPTOR descriptor = nullptr;
    auto error = ::GetSecurityInfo(file, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner, nullptr, &acl, nullptr, &descriptor);
    struct Cleanup { PSECURITY_DESCRIPTOR value; ~Cleanup() { if (value) ::LocalFree(value); } } cleanup{descriptor};
    SECURITY_DESCRIPTOR_CONTROL control = 0; DWORD revision = 0;
    if (error != ERROR_SUCCESS || !owner || !::EqualSid(owner, security.sid()) || !acl ||
        !::GetSecurityDescriptorControl(descriptor, &control, &revision) || !(control & SE_DACL_PROTECTED))
        throw std::runtime_error("private storage must be owned by this user with a protected owner-only DACL");
    for (DWORD index = 0; index < acl->AceCount; ++index)
    {
        void* entry = nullptr;
        if (!::GetAce(acl, index, &entry)) throw std::runtime_error("cannot inspect private storage DACL");
        auto header = static_cast<ACE_HEADER const*>(entry);
        if (header->AceType == ACCESS_DENIED_ACE_TYPE) continue;
        if (header->AceType != ACCESS_ALLOWED_ACE_TYPE || !::EqualSid(const_cast<DWORD*>(&static_cast<ACCESS_ALLOWED_ACE const*>(entry)->SidStart), security.sid()))
            throw std::runtime_error("private storage DACL grants access to another user");
    }
}

inline HANDLE create_private_file_handle(std::filesystem::path const& path, DWORD access = GENERIC_READ | GENERIC_WRITE,
                                        DWORD creation = CREATE_NEW, DWORD flags = 0,
                                        DWORD sharing = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
{
    if (path.filename().native().find(L':') != std::wstring::npos)
        throw std::invalid_argument("alternate data streams are not supported: " + path.string());
    PrivateFileSecurity security;
    HANDLE file = ::CreateFileW(path.c_str(), access | READ_CONTROL, sharing, &security.attributes, creation,
        flags | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "open private file: " + path.string());
    try { validate_private_file_handle(file, false); }
    catch (...) { ::CloseHandle(file); if (creation == CREATE_NEW) ::DeleteFileW(path.c_str()); throw; }
    return file;
}

inline void create_private_directory(std::filesystem::path const& path)
{
    PrivateFileSecurity security;
    if (!::CreateDirectoryW(path.c_str(), &security.attributes))
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "create private directory: " + path.string());
    HANDLE directory = ::CreateFileW(path.c_str(), READ_CONTROL | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    try { validate_private_file_handle(directory, true); }
    catch (...) { if (directory != INVALID_HANDLE_VALUE) ::CloseHandle(directory); ::RemoveDirectoryW(path.c_str()); throw; }
    ::CloseHandle(directory);
}

inline void move_private_path(std::filesystem::path const& source, std::filesystem::path const& destination, bool replace)
{
    if (!::MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0)))
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "publish private storage: " + destination.string());
}

inline int create_private_temp(std::string& path, DWORD flags = 0)
{
    std::string const prefix = path.substr(0, path.size() - 6); // caller's XXXXXX template
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        path = prefix + std::to_string((std::uint64_t(random()) << 32) | random());
        auto const native = std::filesystem::path(reinterpret_cast<char8_t const*>(path.c_str()));
        HANDLE file;
        try { file = create_private_file_handle(native, GENERIC_READ | GENERIC_WRITE, CREATE_NEW, flags, FILE_SHARE_DELETE); }
        catch (std::system_error const& error) { if (error.code().value() == ERROR_FILE_EXISTS || error.code().value() == ERROR_ALREADY_EXISTS) continue; throw; }
        struct Cleanup {
            HANDLE file;
            std::filesystem::path const& path;
            ~Cleanup() {
                if (file != INVALID_HANDLE_VALUE) {
                    ::CloseHandle(file);
                    ::DeleteFileW(path.c_str());
                }
            }
        } cleanup{file, native};
        int const fd = ::_open_osfhandle(reinterpret_cast<std::intptr_t>(file), _O_RDWR | _O_BINARY);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open file descriptor: " + path);
        cleanup.file = INVALID_HANDLE_VALUE; // descriptor now owns the handle
        return fd;
    }
    throw std::runtime_error("cannot create a unique temporary file: " + path);
}

} // namespace pos2gpu
#endif
