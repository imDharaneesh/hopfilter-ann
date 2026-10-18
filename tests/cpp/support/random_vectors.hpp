#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace hopfilter::testing {

/// Uniform floats in [-1, 1) from raw generator bits, identical on every standard library.
class RandomVectors {
public:
    explicit RandomVectors(std::uint64_t seed) : generator_(seed) {}

    [[nodiscard]] float nextUniform() {
        constexpr int kMantissaBits = 24;
        constexpr int kDiscardedBits = 64 - kMantissaBits;
        constexpr float kScale = 1.0F / static_cast<float>(1U << kMantissaBits);
        const auto bits = static_cast<std::uint32_t>(generator_() >> kDiscardedBits);
        return (static_cast<float>(bits) * kScale * 2.0F) - 1.0F;
    }

    [[nodiscard]] std::vector<float> vector(std::size_t dimension) {
        std::vector<float> values(dimension);
        for (float& value : values) {
            value = nextUniform();
        }
        return values;
    }

    [[nodiscard]] std::vector<std::vector<float>> vectors(std::size_t count,
                                                          std::size_t dimension) {
        std::vector<std::vector<float>> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back(vector(dimension));
        }
        return result;
    }

private:
    std::mt19937_64 generator_;
};

}  // namespace hopfilter::testing
