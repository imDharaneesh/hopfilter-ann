#include <cstddef>
#include <cstdint>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/filter/selectivity.hpp>

namespace hopfilter {
namespace {

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

    [[nodiscard]] std::uint64_t next() noexcept {
        constexpr std::uint64_t kIncrement = 0x9E3779B97F4A7C15ULL;
        constexpr std::uint64_t kFirstMultiplier = 0xBF58476D1CE4E5B9ULL;
        constexpr std::uint64_t kSecondMultiplier = 0x94D049BB133111EBULL;
        state_ += kIncrement;
        std::uint64_t mixed = state_;
        mixed = (mixed ^ (mixed >> 30U)) * kFirstMultiplier;
        mixed = (mixed ^ (mixed >> 27U)) * kSecondMultiplier;
        return mixed ^ (mixed >> 31U);
    }

private:
    std::uint64_t state_;
};

double fractionMatching(const FilterPredicate& predicate, std::size_t universeSize) {
    std::size_t matching = 0;
    for (std::size_t id = 0; id < universeSize; ++id) {
        matching += predicate.matches(static_cast<NodeId>(id)) ? std::size_t{1} : std::size_t{0};
    }
    return static_cast<double>(matching) / static_cast<double>(universeSize);
}

}

double ExactSelectivityEstimator::estimate(const FilterPredicate& predicate,
                                           std::size_t universeSize) const {
    return universeSize == 0 ? 0.0 : fractionMatching(predicate, universeSize);
}

SamplingSelectivityEstimator::SamplingSelectivityEstimator(std::size_t sampleSize,
                                                           std::uint64_t seed)
    : sampleSize_(sampleSize), seed_(seed) {
    if (sampleSize == 0) {
        throw InvalidArgumentError("selectivity sample size must be positive");
    }
}

double SamplingSelectivityEstimator::estimate(const FilterPredicate& predicate,
                                              std::size_t universeSize) const {
    if (universeSize <= sampleSize_) {
        return universeSize == 0 ? 0.0 : fractionMatching(predicate, universeSize);
    }
    SplitMix64 random{seed_};
    std::size_t matching = 0;
    for (std::size_t draw = 0; draw < sampleSize_; ++draw) {
        const auto id = static_cast<NodeId>(random.next() % universeSize);
        matching += predicate.matches(id) ? std::size_t{1} : std::size_t{0};
    }
    return static_cast<double>(matching) / static_cast<double>(sampleSize_);
}

}
