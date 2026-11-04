#pragma once

#include "host/Utf8.hpp"
#include <cstddef>
#include <ostream>
#include <string_view>

namespace pos2gpu
{
inline void write_json_string(std::ostream& out, std::string_view value)
{
    static constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (std::size_t i = 0; i < value.size(); ++i) {
        unsigned char const c = value[i];
        if (c >= 0x80) {
            // Preserve valid UTF-8; escape invalid filesystem bytes individually.
            auto const n = utf8_sequence_size(value, i);
            if (n) { out.write(value.data() + i, n); i += n - 1; }
            else out << "\\u00" << hex[c >> 4] << hex[c & 15];
            continue;
        }
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << c;
    }
    out << '"';
}
}
