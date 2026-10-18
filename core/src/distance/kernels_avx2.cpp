#include "kernels_avx2.hpp"

#include <cstddef>
#include <immintrin.h>

namespace hopfilter {
namespace {

constexpr std::size_t kLanes = 8;
constexpr std::size_t kUnrolledLanes = 2 * kLanes;

float horizontalSum(__m256 sum) noexcept {
    __m128 pair = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    pair = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
    pair = _mm_add_ss(pair, _mm_movehdup_ps(pair));
    return _mm_cvtss_f32(pair);
}

__m256 squaredDifference(const float* lhs, const float* rhs, __m256 accumulator) noexcept {
    const __m256 difference = _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
    return _mm256_fmadd_ps(difference, difference, accumulator);
}

__m256 product(const float* lhs, const float* rhs, __m256 accumulator) noexcept {
    return _mm256_fmadd_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs), accumulator);
}

template <typename LaneTerm, typename ScalarTerm>
float vectorSum(const float* lhs,
                const float* rhs,
                std::size_t size,
                LaneTerm laneTerm,
                ScalarTerm scalarTerm) noexcept {
    __m256 first = _mm256_setzero_ps();
    __m256 second = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + kUnrolledLanes <= size; index += kUnrolledLanes) {
        first = laneTerm(lhs + index, rhs + index, first);
        second = laneTerm(lhs + index + kLanes, rhs + index + kLanes, second);
    }
    for (; index + kLanes <= size; index += kLanes) {
        first = laneTerm(lhs + index, rhs + index, first);
    }
    float total = horizontalSum(_mm256_add_ps(first, second));
    for (; index < size; ++index) {
        total += scalarTerm(lhs[index], rhs[index]);
    }
    return total;
}

}

float avx2SquaredL2(const float* lhs, const float* rhs, std::size_t size) noexcept {
    return vectorSum(lhs, rhs, size, squaredDifference, [](float left, float right) {
        const float difference = left - right;
        return difference * difference;
    });
}

float avx2DotProduct(const float* lhs, const float* rhs, std::size_t size) noexcept {
    return vectorSum(lhs, rhs, size, product, [](float left, float right) { return left * right; });
}

}
