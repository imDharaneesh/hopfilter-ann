#include <string>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/search/filtered_search_config.hpp>

namespace hopfilter {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw InvalidArgumentError("filtered search config: " + message);
    }
}

}

void validate(const FilteredSearchConfig& config) {
    require(config.overFetchFactor >= 1.0, "overFetchFactor must be at least 1");
    require(config.rejectionWindow > 0, "rejectionWindow must be positive");
    require(config.rejectionThreshold >= 0.0 && config.rejectionThreshold <= 1.0,
            "rejectionThreshold must be in [0, 1]");
    require(config.efGrowthFactor >= 1.0, "efGrowthFactor must be at least 1");
    require(config.maxEf > 0, "maxEf must be positive");
    require(config.thresholds.exactBelow >= 0.0 &&
                config.thresholds.exactBelow <= config.thresholds.postFilterAbove,
            "thresholds need 0 <= exactBelow <= postFilterAbove");
    require(config.selectivitySample > 0, "selectivitySample must be positive");
    require(config.thresholds.distantShare >= 0.0, "distantShare must not be negative");
    require(
        config.thresholds.distantExactBelow >= 0.0 && config.thresholds.distantExactBelow <= 1.0,
        "distantExactBelow must be in [0, 1]");
    require(config.localProbe > 0, "localProbe must be positive");
}

}
