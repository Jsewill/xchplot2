#pragma once

// Windows counterpart of mkstemp's exclusive creation and owner-only access.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#include <fcntl.h>
#include <io.h>

#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace pos2gpu {

inline int create_private_temp(std::string& path, DWORD flags = 0)
{
    // Alternate data streams inherit the existing file's security descriptor.
    // They cannot provide the private, newly created file promised here.
    if (std::filesystem::path(path).filename().native().find(L':') != std::wstring::npos)
        throw std::invalid_argument("alternate data streams are not supported: " + path);
    struct Security {
        HANDLE token = nullptr;
        LPWSTR sid = nullptr;
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        ~Security() {
            if (descriptor) ::LocalFree(descriptor);
            if (sid) ::LocalFree(sid);
            if (token) ::CloseHandle(token);
        }
    } security;
    auto fail = [&] {
        throw std::system_error(static_cast<int>(::GetLastError()),
                                std::system_category(), "create private file: " + path);
    };
    if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &security.token)) fail();
    DWORD size = 0;
    ::GetTokenInformation(security.token, TokenUser, nullptr, 0, &size);
    if (!size) fail();
    std::vector<unsigned char> token(size);
    if (!::GetTokenInformation(security.token, TokenUser, token.data(), size, &size)) fail();
    auto const* user = reinterpret_cast<TOKEN_USER const*>(token.data());
    if (!::ConvertSidToStringSidW(user->User.Sid, &security.sid)) fail();
    // Set the owner explicitly, including for an elevated process, and block
    // inherited ACEs. CRT _S_IREAD/_S_IWRITE alone do not restrict other users.
    std::wstring const sddl = L"O:" + std::wstring(security.sid)
        + L"D:P(A;;FA;;;" + security.sid + L")";
    if (!::ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, &security.descriptor, nullptr)) fail();
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), security.descriptor, FALSE};
    std::string const prefix = path.substr(0, path.size() - 6); // caller's XXXXXX template
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        path = prefix + std::to_string((std::uint64_t(random()) << 32) | random());
        auto const native = std::filesystem::path(path);
        HANDLE file = ::CreateFileW(native.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_DELETE, &attributes, CREATE_NEW, flags, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            if (::GetLastError() == ERROR_FILE_EXISTS) continue;
            fail();
        }
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
        DWORD volume_flags = 0;
        if (!::GetVolumeInformationByHandleW(file, nullptr, 0, nullptr, nullptr,
                                             &volume_flags, nullptr, 0)) fail();
        if (!(volume_flags & FILE_PERSISTENT_ACLS))
            throw std::runtime_error("private files require an ACL-capable filesystem (NTFS/ReFS): " + path);
        int const fd = ::_open_osfhandle(reinterpret_cast<std::intptr_t>(file), _O_RDWR | _O_BINARY);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open file descriptor: " + path);
        cleanup.file = INVALID_HANDLE_VALUE; // descriptor now owns the handle
        return fd;
    }
    throw std::runtime_error("cannot create a unique temporary file: " + path);
}

} // namespace pos2gpu
#endif
