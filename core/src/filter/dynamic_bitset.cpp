#include <bit>
#include <cstddef>
#include <cstdint>

#include <hopfilter/filter/dynamic_bitset.hpp>

namespace hopfilter {

DynamicBitset::DynamicBitset(std::size_t size)
    : words_((size + kWordBits - 1) / kWordBits, 0), size_(size) {}

void DynamicBitset::set(std::size_t position) noexcept {
    if (position < size_) {
        words_[position / kWordBits] |= std::uint64_t{1} << (position % kWordBits);
    }
}

std::size_t DynamicBitset::popcount() const noexcept {
    std::size_t count = 0;
    for (const std::uint64_t word : words_) {
        count += static_cast<std::size_t>(std::popcount(word));
    }
    return count;
}

}
