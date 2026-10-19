#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>

namespace hopfilter {

class NodeLocks {
public:
    static constexpr std::size_t kDefaultStripes = std::size_t{1} << 16U;

    explicit NodeLocks(std::size_t stripes = kDefaultStripes)
        : mutexes_(stripes > 0 ? stripes : throw InvalidArgumentError("stripes must be positive")) {
    }

    [[nodiscard]] std::unique_lock<std::mutex> lock(NodeId id) const {
        return std::unique_lock{mutexes_[stripeOf(id)]};
    }

    [[nodiscard]] std::size_t stripeOf(NodeId id) const noexcept { return id % mutexes_.size(); }

    [[nodiscard]] std::size_t stripes() const noexcept { return mutexes_.size(); }

private:
    mutable std::vector<std::mutex> mutexes_;
};

}
