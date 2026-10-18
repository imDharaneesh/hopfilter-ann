#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace hopfilter {

/// Incremental CRC-32 (IEEE 802.3 polynomial, as used by zip and PNG).
class Crc32 {
public:
    void update(std::span<const std::byte> bytes) noexcept;

    [[nodiscard]] std::uint32_t value() const noexcept { return ~state_; }

private:
    std::uint32_t state_ = 0xFFFFFFFFU;
};

}  // namespace hopfilter
