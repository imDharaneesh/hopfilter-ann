#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hopfilter {

class DynamicBitset {
public:
    DynamicBitset() = default;

    explicit DynamicBitset(std::size_t size);

    void set(std::size_t position) noexcept;

    [[nodiscard]] bool test(std::size_t position) const noexcept {
        return position < size_ &&
               ((words_[position / kWordBits] >> (position % kWordBits)) & 1U) != 0;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] std::size_t popcount() const noexcept;

    template <typename Visit>
    void forEachSet(const Visit& visit) const {
        for (std::size_t word = 0; word < words_.size(); ++word) {
            std::uint64_t bits = words_[word];
            while (bits != 0) {
                visit((word * kWordBits) + static_cast<std::size_t>(std::countr_zero(bits)));
                bits &= bits - 1;
            }
        }
    }

    friend bool operator==(const DynamicBitset&, const DynamicBitset&) = default;

private:
    static constexpr std::size_t kWordBits = 64;

    std::vector<std::uint64_t> words_;
    std::size_t size_ = 0;
};

}
