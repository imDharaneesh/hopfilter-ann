#include "cpu_features.hpp"

#if defined(HOPFILTER_HAVE_AVX2_KERNELS) && defined(_MSC_VER)

#include <array>
#include <cstdint>
#include <intrin.h>

namespace hopfilter {
namespace {

constexpr int kFeatureLeaf = 1;
constexpr int kExtendedFeatureLeaf = 7;
constexpr std::uint32_t kFmaBit = 1U << 12U;
constexpr std::uint32_t kOsXsaveBit = 1U << 27U;
constexpr std::uint32_t kAvxBit = 1U << 28U;
constexpr std::uint32_t kAvx2Bit = 1U << 5U;
constexpr std::uint64_t kYmmStateMask = 0x6U;

std::array<std::uint32_t, 4> cpuid(int leaf) noexcept {
    std::array<int, 4> registers{};
    __cpuidex(registers.data(), leaf, 0);
    return {static_cast<std::uint32_t>(registers[0]),
            static_cast<std::uint32_t>(registers[1]),
            static_cast<std::uint32_t>(registers[2]),
            static_cast<std::uint32_t>(registers[3])};
}

bool hasAll(std::uint32_t value, std::uint32_t bits) noexcept {
    return (value & bits) == bits;
}

}

bool cpuSupportsAvx2Fma() noexcept {
    if (cpuid(0)[0] < static_cast<std::uint32_t>(kExtendedFeatureLeaf)) {
        return false;
    }
    if (!hasAll(cpuid(kFeatureLeaf)[2], kFmaBit | kOsXsaveBit | kAvxBit)) {
        return false;
    }
    if ((_xgetbv(0) & kYmmStateMask) != kYmmStateMask) {
        return false;
    }
    return hasAll(cpuid(kExtendedFeatureLeaf)[1], kAvx2Bit);
}

}

#elif defined(HOPFILTER_HAVE_AVX2_KERNELS)

namespace hopfilter {

bool cpuSupportsAvx2Fma() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

}

#else

namespace hopfilter {

bool cpuSupportsAvx2Fma() noexcept {
    return false;
}

}

#endif
