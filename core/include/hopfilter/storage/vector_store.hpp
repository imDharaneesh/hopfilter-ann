#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <span>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

/// Fixed-capacity, contiguous storage for equally sized float vectors.
///
/// One allocation aligned to a cache line holds every row; each row is padded with
/// zeros to a whole number of cache lines so SIMD kernels can use aligned loads.
/// Capacity never grows, so spans returned by ``at`` stay valid for the store's
/// lifetime. Reads are thread-safe; ``append`` is not and must be serialized.
class VectorStore {
public:
    static constexpr std::size_t kAlignmentBytes = 64;
    static constexpr std::size_t kFloatsPerAlignment = kAlignmentBytes / sizeof(float);

    /// Throws InvalidArgumentError if ``dimension`` or ``capacity`` is zero, or if
    /// ``capacity`` does not fit in a NodeId.
    VectorStore(Dimension dimension, std::size_t capacity);

    /// Copies ``vector`` into the next free row and returns its id.
    /// Throws InvalidArgumentError on a size mismatch and IndexStateError when full.
    NodeId append(std::span<const float> vector);

    /// The stored vector ``id`` (without padding). ``id`` must be below ``size()``.
    [[nodiscard]] std::span<const float> at(NodeId id) const noexcept;

    [[nodiscard]] Dimension dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    struct AlignedDeleter {
        void operator()(float* data) const noexcept {
            ::operator delete[](data, std::align_val_t{kAlignmentBytes});
        }
    };

    Dimension dimension_;
    std::size_t stride_;
    std::size_t capacity_;
    std::size_t size_ = 0;
    std::unique_ptr<float[], AlignedDeleter> data_;
};

}
