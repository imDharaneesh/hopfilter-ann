#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace hopfilter {

/// Sum of squared differences using the active kernel set. Both spans must have the same size.
[[nodiscard]] float squaredL2(std::span<const float> lhs, std::span<const float> rhs) noexcept;

/// Dot product using the active kernel set. Both spans must have the same size.
[[nodiscard]] float dotProduct(std::span<const float> lhs, std::span<const float> rhs) noexcept;

/// Portable reference implementations, always available.
[[nodiscard]] float scalarSquaredL2(std::span<const float> lhs,
                                    std::span<const float> rhs) noexcept;
[[nodiscard]] float scalarDotProduct(std::span<const float> lhs,
                                     std::span<const float> rhs) noexcept;

/// Instruction sets the distance kernels can be compiled for.
enum class KernelSet : std::uint8_t { Scalar, Avx2 };

/// One implementation of every distance kernel.
struct DistanceKernels {
    KernelSet set;
    float (*squaredL2)(std::span<const float>, std::span<const float>) noexcept;
    float (*dotProduct)(std::span<const float>, std::span<const float>) noexcept;
};

/// Whether ``set`` was compiled into this binary and the CPU and OS can run it.
[[nodiscard]] bool isSupported(KernelSet set) noexcept;

/// The fastest supported set; the default active set.
[[nodiscard]] KernelSet bestSupportedKernelSet() noexcept;

/// The kernels currently used by ``squaredL2`` and ``dotProduct``.
[[nodiscard]] const DistanceKernels& activeKernels() noexcept;

/// Switches every distance computation to ``set``; safe while other threads search.
/// Throws InvalidArgumentError if ``set`` is not supported.
void useKernelSet(KernelSet set);

/// Lower-case name, e.g. "avx2".
[[nodiscard]] std::string_view kernelSetName(KernelSet set) noexcept;

}
