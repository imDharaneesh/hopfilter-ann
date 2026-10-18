#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <hopfilter/distance/kernels.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/io/index_serializer.hpp>
#include <hopfilter/search/search_context.hpp>

namespace {

template <typename Value>
struct Matrix {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<Value> values;

    [[nodiscard]] std::span<const Value> row(std::size_t index) const {
        return std::span<const Value>{values}.subspan(index * columns, columns);
    }
};

template <typename Value>
Matrix<Value> readMatrix(const std::filesystem::path& path) {
    std::ifstream source{path, std::ios::binary};
    if (!source) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::array<char, 2 * sizeof(std::uint32_t)> header{};
    source.read(header.data(), static_cast<std::streamsize>(header.size()));
    std::array<std::uint32_t, 2> shape{};
    std::memcpy(shape.data(), header.data(), header.size());
    Matrix<Value> matrix{shape[0], shape[1], std::vector<Value>(std::size_t{shape[0]} * shape[1])};
    std::vector<char> payload(matrix.values.size() * sizeof(Value));
    source.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!source) {
        throw std::runtime_error("truncated matrix in " + path.string());
    }
    std::memcpy(matrix.values.data(), payload.data(), payload.size());
    return matrix;
}

struct Options {
    std::filesystem::path index;
    std::filesystem::path queries;
    std::filesystem::path truth;
    std::size_t k = 10;
    std::size_t ef = 64;
    std::size_t repeats = 5;
    std::string kernels = "default";
};

Options parseOptions(int argc, char** argv) {
    const std::vector<std::string> arguments(argv + 1, argv + argc);
    if (arguments.size() < 3) {
        throw std::invalid_argument(
            "usage: hopfilter_search_bench INDEX QUERIES TRUTH [k] [ef] [repeats] [kernels]");
    }
    Options options{arguments[0], arguments[1], arguments[2]};
    if (arguments.size() > 3) {
        options.k = std::stoul(arguments[3]);
    }
    if (arguments.size() > 4) {
        options.ef = std::stoul(arguments[4]);
    }
    if (arguments.size() > 5) {
        options.repeats = std::stoul(arguments[5]);
    }
    if (arguments.size() > 6) {
        options.kernels = arguments[6];
    }
    return options;
}

double recallOf(const std::vector<hopfilter::SearchResults>& found,
                const Matrix<std::int32_t>& truth,
                std::size_t k) {
    std::size_t hits = 0;
    for (std::size_t query = 0; query < found.size(); ++query) {
        const auto expected = truth.row(query).first(k);
        const std::unordered_set<std::int32_t> wanted(expected.begin(), expected.end());
        for (const auto& neighbor : found[query]) {
            hits += wanted.contains(static_cast<std::int32_t>(neighbor.id)) ? 1 : 0;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(found.size() * k);
}

double timedPass(const hopfilter::HnswIndex& index,
                 const Matrix<float>& queries,
                 const Options& options,
                 std::vector<hopfilter::SearchResults>& found) {
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t query = 0; query < queries.rows; ++query) {
        found[query] =
            index.search(queries.row(query), options.k, hopfilter::SearchRequest{options.ef});
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
    return static_cast<double>(queries.rows) / elapsed.count();
}

}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.kernels == "scalar") {
            hopfilter::useKernelSet(hopfilter::KernelSet::Scalar);
        }
        const auto index = hopfilter::IndexSerializer::loadFromFile(options.index);
        const auto queries = readMatrix<float>(options.queries);
        const auto truth = readMatrix<std::int32_t>(options.truth);
        std::vector<hopfilter::SearchResults> found(queries.rows);
        std::vector<double> rates;
        timedPass(*index, queries, options, found);
        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
            rates.push_back(timedPass(*index, queries, options, found));
        }
        std::ranges::sort(rates);
        std::cout << "kernels=" << hopfilter::kernelSetName(hopfilter::activeKernels().set)
                  << " ef=" << options.ef << " recall=" << recallOf(found, truth, options.k)
                  << " qps_median=" << rates[rates.size() / 2] << " qps_min=" << rates.front()
                  << " qps_max=" << rates.back() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
