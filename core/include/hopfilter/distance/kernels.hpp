#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace hopfilter {

[[nodiscard]] float squaredL2(std::span<const float> lhs, std::span<const float> rhs) noexcept;

[[nodiscard]] float dotProduct(std::span<const float> lhs, std::span<const float> rhs) noexcept;

[[nodiscard]] float scalarSquaredL2(std::span<const float> lhs,
                                    std::span<const float> rhs) noexcept;
[[nodiscard]] float scalarDotProduct(std::span<const float> lhs,
                                     std::span<const float> rhs) noexcept;

enum class KernelSet : std::uint8_t { Scalar, Avx2 };

struct DistanceKernels {
    KernelSet set;
    float (*squaredL2)(std::span<const float>, std::span<const float>) noexcept;
    float (*dotProduct)(std::span<const float>, std::span<const float>) noexcept;
};

[[nodiscard]] bool isSupported(KernelSet set) noexcept;

[[nodiscard]] KernelSet bestSupportedKernelSet() noexcept;

[[nodiscard]] const DistanceKernels& activeKernels() noexcept;

void useKernelSet(KernelSet set);

[[nodiscard]] std::string_view kernelSetName(KernelSet set) noexcept;

}
