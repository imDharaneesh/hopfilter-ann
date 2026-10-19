#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace hopfilter {

class Crc32 {
public:
    void update(std::span<const std::byte> bytes) noexcept;

    [[nodiscard]] std::uint32_t value() const noexcept { return ~state_; }

private:
    std::uint32_t state_ = 0xFFFFFFFFU;
};

}
