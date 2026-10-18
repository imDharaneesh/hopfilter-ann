#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include <hopfilter/common/types.hpp>
#include <hopfilter/common/visited_set.hpp>

namespace hopfilter {
namespace {

constexpr std::uint32_t kNeverVisited = 0;
constexpr std::uint32_t kFirstEpoch = 1;

}  // namespace

VisitedSet::VisitedSet(std::size_t capacity) : VisitedSet(capacity, kFirstEpoch) {}

VisitedSet::VisitedSet(std::size_t capacity, std::uint32_t initialEpoch)
    : stamps_(capacity, kNeverVisited),
      epoch_(initialEpoch == kNeverVisited ? kFirstEpoch : initialEpoch) {}

void VisitedSet::clear() noexcept {
    ++epoch_;
    if (epoch_ == kNeverVisited) {
        std::ranges::fill(stamps_, kNeverVisited);
        epoch_ = kFirstEpoch;
    }
}

bool VisitedSet::markVisited(NodeId id) noexcept {
    if (stamps_[id] == epoch_) {
        return false;
    }
    stamps_[id] = epoch_;
    return true;
}

bool VisitedSet::isVisited(NodeId id) const noexcept {
    return stamps_[id] == epoch_;
}

VisitedSetPool::Lease::Lease(VisitedSetPool& pool, std::unique_ptr<VisitedSet> set) noexcept
    : pool_(&pool), set_(std::move(set)) {}

VisitedSetPool::Lease::~Lease() {
    if (!set_) {
        return;
    }
    try {
        pool_->release(std::move(set_));
    } catch (const std::exception&) {
        // Losing a pooled set only costs one allocation on a later acquire.
        set_ = nullptr;
    }
}

VisitedSetPool::VisitedSetPool(std::size_t capacity) : capacity_(capacity) {}

VisitedSetPool::Lease VisitedSetPool::acquire() {
    std::unique_ptr<VisitedSet> set;
    {
        const std::scoped_lock lock{mutex_};
        if (!idle_.empty()) {
            set = std::move(idle_.back());
            idle_.pop_back();
        }
    }
    if (set) {
        set->clear();
    } else {
        set = std::make_unique<VisitedSet>(capacity_);
    }
    return Lease{*this, std::move(set)};
}

std::size_t VisitedSetPool::idleCount() const {
    const std::scoped_lock lock{mutex_};
    return idle_.size();
}

void VisitedSetPool::release(std::unique_ptr<VisitedSet> set) {
    const std::scoped_lock lock{mutex_};
    idle_.push_back(std::move(set));
}

}  // namespace hopfilter
