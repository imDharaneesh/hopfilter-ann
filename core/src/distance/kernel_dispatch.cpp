#include <atomic>
#include <span>
#include <string>
#include <string_view>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/kernels.hpp>

#include "cpu_features.hpp"

#ifdef HOPFILTER_HAVE_AVX2_KERNELS
#include "kernels_avx2.hpp"
#endif

namespace hopfilter {
namespace {

constexpr DistanceKernels kScalarKernels{KernelSet::Scalar, &scalarSquaredL2, &scalarDotProduct};

#ifdef HOPFILTER_HAVE_AVX2_KERNELS
float avx2SquaredL2Span(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return avx2SquaredL2(lhs.data(), rhs.data(), lhs.size());
}

float avx2DotProductSpan(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return avx2DotProduct(lhs.data(), rhs.data(), lhs.size());
}

constexpr DistanceKernels kAvx2Kernels{KernelSet::Avx2, &avx2SquaredL2Span, &avx2DotProductSpan};
#endif

const DistanceKernels* kernelsFor(KernelSet set) noexcept {
#ifdef HOPFILTER_HAVE_AVX2_KERNELS
    if (set == KernelSet::Avx2) {
        return cpuSupportsAvx2Fma() ? &kAvx2Kernels : nullptr;
    }
#endif
    return set == KernelSet::Scalar ? &kScalarKernels : nullptr;
}

std::atomic<const DistanceKernels*>& activePointer() noexcept {
    static std::atomic<const DistanceKernels*> active{kernelsFor(bestSupportedKernelSet())};
    return active;
}

}

float squaredL2(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return activePointer().load(std::memory_order_relaxed)->squaredL2(lhs, rhs);
}

float dotProduct(std::span<const float> lhs, std::span<const float> rhs) noexcept {
    return activePointer().load(std::memory_order_relaxed)->dotProduct(lhs, rhs);
}

bool isSupported(KernelSet set) noexcept {
    return kernelsFor(set) != nullptr;
}

KernelSet bestSupportedKernelSet() noexcept {
    return isSupported(KernelSet::Avx2) ? KernelSet::Avx2 : KernelSet::Scalar;
}

const DistanceKernels& activeKernels() noexcept {
    return *activePointer().load(std::memory_order_relaxed);
}

void useKernelSet(KernelSet set) {
    const DistanceKernels* kernels = kernelsFor(set);
    if (kernels == nullptr) {
        throw InvalidArgumentError("kernel set '" + std::string{kernelSetName(set)} +
                                   "' is not supported on this machine");
    }
    activePointer().store(kernels, std::memory_order_relaxed);
}

std::string_view kernelSetName(KernelSet set) noexcept {
    return set == KernelSet::Avx2 ? "avx2" : "scalar";
}

}
