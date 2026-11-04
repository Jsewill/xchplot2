#pragma once

#include <cstddef>
#include <string_view>

namespace pos2gpu
{
inline std::size_t utf8_sequence_size(std::string_view value, std::size_t at)
{
    if (at >= value.size()) return 0;
    unsigned char const c = value[at];
    if (c < 0x80) return 1;
    std::size_t const count = c >= 0xc2 && c <= 0xdf ? 2 :
        c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!count || count > value.size() - at) return 0;
    for (std::size_t j = 1; j < count; ++j)
    {
        unsigned char const next = value[at + j];
        if (next < 0x80 || next > 0xbf || (j == 1 &&
            ((c == 0xe0 && next < 0xa0) || (c == 0xed && next >= 0xa0) ||
             (c == 0xf0 && next < 0x90) || (c == 0xf4 && next >= 0x90)))) return 0;
    }
    return count;
}
inline bool valid_utf8(std::string_view value)
{
    for (std::size_t at = 0; at < value.size();)
    { auto const count = utf8_sequence_size(value, at); if (!count) return false; at += count; }
    return true;
}
}
