#pragma once

#include <span>

namespace hopfilter {

/// Sum of squared differences. Both spans must have the same size.
[[nodiscard]] float squaredL2(std::span<const float> lhs, std::span<const float> rhs) noexcept;

/// Dot product. Both spans must have the same size.
[[nodiscard]] float dotProduct(std::span<const float> lhs, std::span<const float> rhs) noexcept;

}  // namespace hopfilter
