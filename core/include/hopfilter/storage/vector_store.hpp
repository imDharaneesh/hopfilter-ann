#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <span>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

class VectorStore {
public:
    static constexpr std::size_t kAlignmentBytes = 64;
    static constexpr std::size_t kFloatsPerAlignment = kAlignmentBytes / sizeof(float);

    VectorStore(Dimension dimension, std::size_t capacity);

    NodeId append(std::span<const float> vector);

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
    std::unique_ptr<float, AlignedDeleter> data_;
};

}
