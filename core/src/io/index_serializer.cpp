#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <new>
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/io/binary_stream.hpp>
#include <hopfilter/io/index_serializer.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

constexpr std::array<char, 4> kMagic{'H', 'O', 'P', 'F'};
constexpr std::size_t kMaxMetricNameLength = 64;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw SerializationError("corrupt index: " + message);
    }
}

void writeConfig(BinaryWriter& writer, const IndexConfig& config) {
    writer.write(static_cast<std::uint64_t>(config.dimension));
    writer.write(static_cast<std::uint64_t>(config.capacity));
    writer.write(static_cast<std::uint64_t>(config.maxNeighbors));
    writer.write(static_cast<std::uint64_t>(config.efConstruction));
    writer.write(config.seed);
    writer.writeString(config.metricName);
}

void writeGraph(BinaryWriter& writer, const HnswGraph& graph) {
    writer.write(graph.entryPoint().value_or(kInvalidNodeId));
    for (NodeId id = 0; id < graph.size(); ++id) {
        writer.write(graph.levelOf(id));
        for (LevelIndex level = 0; level <= graph.levelOf(id); ++level) {
            const auto neighbors = graph.neighbors(id, level);
            writer.write(static_cast<std::uint32_t>(neighbors.size()));
            writer.writeAll(neighbors);
        }
    }
}

void readHeader(BinaryReader& reader) {
    std::array<char, kMagic.size()> magic{};
    reader.readAll(std::span<char>{magic});
    if (magic != kMagic) {
        throw SerializationError("not a hopfilter index (bad magic bytes)");
    }
    const auto version = reader.read<std::uint32_t>();
    if (version != IndexSerializer::kFormatVersion) {
        throw SerializationError("unsupported index format version " + std::to_string(version));
    }
}

IndexConfig readConfig(BinaryReader& reader) {
    IndexConfig config;
    config.dimension = reader.read<std::uint64_t>();
    config.capacity = reader.read<std::uint64_t>();
    config.maxNeighbors = reader.read<std::uint64_t>();
    config.efConstruction = reader.read<std::uint64_t>();
    config.seed = reader.read<std::uint64_t>();
    config.metricName = reader.readString(kMaxMetricNameLength);
    require(config.dimension > 0 && config.capacity > 0, "empty shape");
    require(config.capacity < kInvalidNodeId, "capacity out of range");
    require(config.maxNeighbors >= 2 && config.efConstruction >= config.maxNeighbors,
            "invalid degree parameters");
    return config;
}

std::unique_ptr<VectorStore> readVectors(BinaryReader& reader, const IndexConfig& config) {
    const auto size = reader.read<std::uint64_t>();
    require(size <= config.capacity, "more vectors than capacity");
    auto vectors = std::make_unique<VectorStore>(config.dimension, config.capacity);
    std::vector<float> row(config.dimension);
    for (std::uint64_t index = 0; index < size; ++index) {
        reader.readAll(std::span<float>{row});
        static_cast<void>(vectors->append(row));
    }
    return vectors;
}

void readNodeLinks(BinaryReader& reader, HnswGraph& graph, NodeId id, std::size_t nodeCount) {
    std::vector<NodeId> neighbors;
    for (LevelIndex level = 0; level <= graph.levelOf(id); ++level) {
        const auto count = reader.read<std::uint32_t>();
        require(count <= graph.maxNeighbors(level), "neighbor list exceeds degree bound");
        neighbors.resize(count);
        reader.readAll(std::span<NodeId>{neighbors});
        for (const NodeId neighbor : neighbors) {
            require(neighbor < nodeCount && neighbor != id, "neighbor id out of range");
        }
        graph.setNeighbors(id, level, neighbors);
    }
}

std::unique_ptr<HnswGraph> readGraph(BinaryReader& reader,
                                     const IndexConfig& config,
                                     std::size_t nodeCount) {
    auto graph = std::make_unique<HnswGraph>(
        config.capacity, config.maxNeighborsLevelZero(), config.maxNeighbors);
    const auto entryPoint = reader.read<NodeId>();
    require(nodeCount == 0 ? entryPoint == kInvalidNodeId : entryPoint < nodeCount,
            "entry point out of range");
    for (NodeId id = 0; id < nodeCount; ++id) {
        const auto level = reader.read<LevelIndex>();
        require(level <= ExponentialLayerAssigner::kMaxLevel, "node level out of range");
        static_cast<void>(graph->addNode(level));
        readNodeLinks(reader, *graph, id, nodeCount);
    }
    if (nodeCount > 0) {
        graph->setEntryPoint(entryPoint);
    }
    return graph;
}

std::unique_ptr<HnswIndex> parse(std::istream& source, const MetricRegistry& registry) {
    BinaryReader reader{source};
    readHeader(reader);
    IndexConfig config = readConfig(reader);
    auto vectors = readVectors(reader, config);
    auto graph = readGraph(reader, config, vectors->size());
    reader.verifyChecksum();
    return HnswIndexBuilder{}
        .withConfig(std::move(config))
        .withMetricRegistry(registry)
        .withStorage(std::move(vectors), std::move(graph))
        .build();
}

}

void IndexSerializer::save(const HnswIndex& index, std::ostream& sink) {
    BinaryWriter writer{sink};
    writer.writeAll(std::span<const char>{kMagic});
    writer.write(kFormatVersion);
    writeConfig(writer, index.config());
    writer.write(static_cast<std::uint64_t>(index.size()));
    for (NodeId id = 0; id < index.size(); ++id) {
        writer.writeAll(index.vectors().at(id));
    }
    writeGraph(writer, index.graph());
    const std::uint32_t checksum = writer.checksum();
    writer.write(checksum);
}

std::unique_ptr<HnswIndex> IndexSerializer::load(std::istream& source,
                                                 const MetricRegistry& registry) {
    try {
        return parse(source, registry);
    } catch (const InvalidArgumentError& error) {
        throw SerializationError(std::string{"corrupt index: "} + error.what());
    } catch (const IndexStateError& error) {
        throw SerializationError(std::string{"corrupt index: "} + error.what());
    } catch (const std::bad_alloc&) {
        throw SerializationError("corrupt index: declared shape is too large to allocate");
    }
}

void IndexSerializer::saveToFile(const HnswIndex& index, const std::filesystem::path& path) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        throw SerializationError("cannot open " + path.string() + " for writing");
    }
    save(index, file);
}

std::unique_ptr<HnswIndex> IndexSerializer::loadFromFile(const std::filesystem::path& path,
                                                         const MetricRegistry& registry) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        throw SerializationError("cannot open " + path.string() + " for reading");
    }
    return load(file, registry);
}

}
