#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>

#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/index/hnsw_index.hpp>

namespace hopfilter {

/// Versioned binary save and load of an HnswIndex.
///
/// Layout (little-endian): magic "HOPF", format version, the IndexConfig, every stored
/// vector, the graph (entry point, then each node's level and per-level neighbor ids),
/// and a CRC-32 of everything before it. Loading validates the header before
/// allocating storage, bounds-checks every count and id, and verifies the checksum
/// before returning an index; any problem raises SerializationError.
///
/// Compatibility rule: version N reads files written by versions N and N-1.
class IndexSerializer {
public:
    static constexpr std::uint32_t kFormatVersion = 1;

    static void save(const HnswIndex& index, std::ostream& sink);

    [[nodiscard]] static std::unique_ptr<HnswIndex> load(
        std::istream& source, const MetricRegistry& registry = MetricRegistry::withBuiltins());

    static void saveToFile(const HnswIndex& index, const std::filesystem::path& path);

    [[nodiscard]] static std::unique_ptr<HnswIndex> loadFromFile(
        const std::filesystem::path& path,
        const MetricRegistry& registry = MetricRegistry::withBuiltins());
};

}
