#pragma once

#include <cstddef>

namespace hopfilter {

float avx2SquaredL2(const float* lhs, const float* rhs, std::size_t size) noexcept;
float avx2DotProduct(const float* lhs, const float* rhs, std::size_t size) noexcept;

}
