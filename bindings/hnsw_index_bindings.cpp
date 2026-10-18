#include "hnsw_index_bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/io/index_serializer.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter::bindings {
namespace {

namespace py = pybind11;

using FloatMatrix = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IdMatrix = py::array_t<std::int64_t>;
using DistanceMatrix = py::array_t<float>;

constexpr std::int64_t kMissingId = -1;
constexpr std::size_t kMatrixRank = 2;

std::unique_ptr<HnswIndex> createIndex(std::size_t dimension,
                                       std::size_t capacity,
                                       std::string metric,
                                       std::size_t maxNeighbors,
                                       std::size_t efConstruction,
                                       std::uint64_t seed) {
    IndexConfig config;
    config.dimension = dimension;
    config.capacity = capacity;
    config.metricName = std::move(metric);
    config.maxNeighbors = maxNeighbors;
    config.efConstruction = efConstruction;
    config.seed = seed;
    return HnswIndexBuilder{}.withConfig(std::move(config)).build();
}

void requireMatrixOf(const FloatMatrix& matrix, std::size_t dimension, const char* name) {
    if (static_cast<std::size_t>(matrix.ndim()) != kMatrixRank ||
        static_cast<std::size_t>(matrix.shape(1)) != dimension) {
        throw InvalidArgumentError(std::string{name} + " must have shape (n, " +
                                   std::to_string(dimension) + ")");
    }
}

std::span<const float> row(const FloatMatrix& matrix, py::ssize_t index, std::size_t dimension) {
    return {matrix.data(index, 0), dimension};
}

IdMatrix addVectors(HnswIndex& index, const FloatMatrix& vectors) {
    requireMatrixOf(vectors, index.dimension(), "vectors");
    const py::ssize_t count = vectors.shape(0);
    IdMatrix ids(count);
    auto output = ids.mutable_unchecked<1>();
    {
        const py::gil_scoped_release release;
        for (py::ssize_t position = 0; position < count; ++position) {
            output(position) = index.insert(row(vectors, position, index.dimension()));
        }
    }
    return ids;
}

py::tuple searchBatch(const HnswIndex& index,
                      const FloatMatrix& queries,
                      std::size_t k,
                      std::size_t ef) {
    requireMatrixOf(queries, index.dimension(), "queries");
    if (k == 0) {
        throw InvalidArgumentError("k must be positive");
    }
    const py::ssize_t count = queries.shape(0);
    const auto width = static_cast<py::ssize_t>(k);
    IdMatrix ids({count, width});
    DistanceMatrix distances({count, width});
    auto idView = ids.mutable_unchecked<2>();
    auto distanceView = distances.mutable_unchecked<2>();
    {
        const py::gil_scoped_release release;
        for (py::ssize_t query = 0; query < count; ++query) {
            const auto results =
                index.search(row(queries, query, index.dimension()), k, SearchRequest{ef});
            for (py::ssize_t slot = 0; slot < width; ++slot) {
                const bool found = static_cast<std::size_t>(slot) < results.size();
                idView(query, slot) =
                    found ? results[static_cast<std::size_t>(slot)].id : kMissingId;
                distanceView(query, slot) = found ? results[static_cast<std::size_t>(slot)].distance
                                                  : std::numeric_limits<float>::infinity();
            }
        }
    }
    return py::make_tuple(std::move(ids), std::move(distances));
}

}

void bindHnswIndex(py::module_& module) {
    py::class_<HnswIndex>(module, "HnswIndex")
        .def(py::init(&createIndex),
             py::arg("dimension"),
             py::arg("capacity"),
             py::arg("metric") = std::string{kL2MetricName},
             py::arg("max_neighbors") = kDefaultMaxNeighbors,
             py::arg("ef_construction") = kDefaultEfConstruction,
             py::arg("seed") = kDefaultSeed)
        .def("add", &addVectors, py::arg("vectors"))
        .def("search",
             &searchBatch,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef") = kDefaultEfSearch)
        .def(
            "save",
            [](const HnswIndex& index, const std::filesystem::path& path) {
                IndexSerializer::saveToFile(index, path);
            },
            py::arg("path"))
        .def_static(
            "load",
            [](const std::filesystem::path& path) { return IndexSerializer::loadFromFile(path); },
            py::arg("path"))
        .def("__len__", &HnswIndex::size)
        .def_property_readonly("dimension", &HnswIndex::dimension)
        .def_property_readonly("capacity", &HnswIndex::capacity)
        .def_property_readonly(
            "metric", [](const HnswIndex& index) { return std::string{index.metric().name()}; })
        .def_property_readonly("max_neighbors",
                               [](const HnswIndex& index) { return index.config().maxNeighbors; })
        .def_property_readonly("ef_construction", [](const HnswIndex& index) {
            return index.config().efConstruction;
        });
}

}
