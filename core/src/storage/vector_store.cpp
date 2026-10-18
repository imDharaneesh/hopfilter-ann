#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <string>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

std::size_t paddedStride(Dimension dimension) {
    const std::size_t lines =
        (dimension + VectorStore::kFloatsPerAlignment - 1) / VectorStore::kFloatsPerAlignment;
    return lines * VectorStore::kFloatsPerAlignment;
}

void validateShape(Dimension dimension, std::size_t capacity) {
    if (dimension == 0 || capacity == 0) {
        throw InvalidArgumentError("dimension and capacity must be positive");
    }
    if (capacity >= kInvalidNodeId) {
        throw InvalidArgumentError("capacity must be below " + std::to_string(kInvalidNodeId));
    }
    if (paddedStride(dimension) > std::numeric_limits<std::size_t>::max() / capacity) {
        throw InvalidArgumentError("dimension * capacity overflows");
    }
}

float* allocateZeroed(std::size_t floats) {
    auto* data = static_cast<float*>(
        ::operator new[](floats * sizeof(float), std::align_val_t{VectorStore::kAlignmentBytes}));
    std::fill_n(data, floats, 0.0F);
    return data;
}

}  // namespace

VectorStore::VectorStore(Dimension dimension, std::size_t capacity)
    : dimension_(dimension),
      stride_((validateShape(dimension, capacity), paddedStride(dimension))),
      capacity_(capacity),
      data_(allocateZeroed(stride_ * capacity)) {}

NodeId VectorStore::append(std::span<const float> vector) {
    if (vector.size() != dimension_) {
        throw InvalidArgumentError("expected a vector of dimension " + std::to_string(dimension_) +
                                   ", got " + std::to_string(vector.size()));
    }
    if (size_ == capacity_) {
        throw IndexStateError("vector store is full (capacity " + std::to_string(capacity_) + ")");
    }
    std::ranges::copy(vector, data_.get() + (size_ * stride_));
    return static_cast<NodeId>(size_++);
}

std::span<const float> VectorStore::at(NodeId id) const noexcept {
    return {data_.get() + (static_cast<std::size_t>(id) * stride_), dimension_};
}

}  // namespace hopfilter
