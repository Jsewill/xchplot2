#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

namespace pos2gpu {
// Read exactly n bytes from the OS cryptographic RNG, before saving a new job.
inline void read_random_bytes(uint8_t* out, size_t n)
{
#ifdef _WIN32
    if (n > std::numeric_limits<ULONG>::max())
        throw std::invalid_argument("entropy request exceeds the Windows buffer limit");
    auto const status = ::BCryptGenRandom(nullptr, out, static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status != 0)
        throw std::runtime_error("BCryptGenRandom failed: " + std::to_string(status));
#else
    std::ifstream f("/dev/urandom", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open /dev/urandom");
    f.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n));
    if (f.gcount() != static_cast<std::streamsize>(n))
        throw std::runtime_error("short read from /dev/urandom");
#endif
}
} // namespace pos2gpu
