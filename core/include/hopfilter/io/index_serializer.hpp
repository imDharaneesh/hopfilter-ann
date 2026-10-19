#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>

#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/index/hnsw_index.hpp>

namespace hopfilter {

class IndexSerializer {
public:
    static constexpr std::uint32_t kFormatVersion = 2;
    static constexpr std::uint32_t kOldestReadableVersion = 1;

    static void save(const HnswIndex& index, std::ostream& sink);

    [[nodiscard]] static std::unique_ptr<HnswIndex> load(
        std::istream& source, const MetricRegistry& registry = MetricRegistry::withBuiltins());

    static void saveToFile(const HnswIndex& index, const std::filesystem::path& path);

    [[nodiscard]] static std::unique_ptr<HnswIndex> loadFromFile(
        const std::filesystem::path& path,
        const MetricRegistry& registry = MetricRegistry::withBuiltins());
};

}
