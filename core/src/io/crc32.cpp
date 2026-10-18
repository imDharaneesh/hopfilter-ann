#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <hopfilter/io/crc32.hpp>

namespace hopfilter {
namespace {

constexpr std::uint32_t kReflectedPolynomial = 0xEDB88320U;
constexpr std::size_t kTableSize = 256;
constexpr int kBitsPerByte = 8;
constexpr std::uint32_t kLowByteMask = 0xFFU;

constexpr std::array<std::uint32_t, kTableSize> makeTable() {
    std::array<std::uint32_t, kTableSize> table{};
    for (std::uint32_t entry = 0; entry < kTableSize; ++entry) {
        std::uint32_t value = entry;
        for (int bit = 0; bit < kBitsPerByte; ++bit) {
            value = (value & 1U) != 0 ? (value >> 1U) ^ kReflectedPolynomial : value >> 1U;
        }
        table[entry] = value;
    }
    return table;
}

constexpr auto kTable = makeTable();

}  // namespace

void Crc32::update(std::span<const std::byte> bytes) noexcept {
    for (const std::byte byte : bytes) {
        const auto index = (state_ ^ std::to_integer<std::uint32_t>(byte)) & kLowByteMask;
        state_ = kTable[index] ^ (state_ >> static_cast<std::uint32_t>(kBitsPerByte));
    }
}

}  // namespace hopfilter
