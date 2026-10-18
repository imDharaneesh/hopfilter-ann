#include <array>
#include <cstddef>
#include <numeric>
#include <span>

#include <hopfilter/distance/kernels.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kAccumulators = 4;

template <typename PairTerm>
float blockedSum(std::span<const float> lhs, std::span<const float> rhs, PairTerm term) noexcept {
    std::array<float, kAccumulators> partial{};
    const std::size_t blockedEnd = lhs.size() - (lhs.size() % kAccumulators);
    for (std::size_t index = 0; index < blockedEnd; index += kAccumulators) {
        for (std::size_t lane = 0; lane < kAccumulators; ++lane) {
            partial[lane] += term(lhs[index + lane], rhs[index + lane]);
        }
    }
    float total = std::accumulate(partial.begin(), partial.end(), 0.0F);
    for (std::size_t index = blockedEnd; index < lhs.size(); ++index) {
        total += term(lhs[index], rhs[index]);
    }
    return total;
}

}

float squaredL2(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return blockedSum(lhs, rhs, [](float left, float right) {
        const float difference = left - right;
        return difference * difference;
    });
}

float dotProduct(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return blockedSum(lhs, rhs, [](float left, float right) { return left * right; });
}

}
