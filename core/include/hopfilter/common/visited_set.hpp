#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

/// Tracks which nodes a traversal has visited, cleared in O(1) between traversals.
///
/// Each node stores the epoch it was last visited in; ``reset`` starts a new epoch.
/// Only when the epoch counter wraps around are the stamps actually cleared.
/// Not thread-safe: each concurrent traversal needs its own instance.
class VisitedSet {
public:
    explicit VisitedSet(std::size_t capacity);

    /// Starts with ``initialEpoch``; lets tests reach the wrap-around quickly.
    VisitedSet(std::size_t capacity, std::uint32_t initialEpoch);

    void clear() noexcept;

    /// Marks ``id`` and returns true if it was not yet visited since the last clear.
    [[nodiscard]] bool markVisited(NodeId id) noexcept;

    [[nodiscard]] bool isVisited(NodeId id) const noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept { return stamps_.size(); }

private:
    std::vector<std::uint32_t> stamps_;
    std::uint32_t epoch_;
};

/// Hands out reset VisitedSets to concurrent traversals and takes them back.
///
/// ``acquire`` and lease destruction take a short lock; the traversal itself runs
/// without one. The pool must outlive every lease it hands out.
class VisitedSetPool {
public:
    /// Exclusive use of one VisitedSet; returns it to the pool when destroyed.
    class Lease {
    public:
        Lease(VisitedSetPool& pool, std::unique_ptr<VisitedSet> set) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept = default;
        Lease& operator=(Lease&& other) noexcept = delete;
        ~Lease();

        [[nodiscard]] VisitedSet& get() const noexcept { return *set_; }

    private:
        VisitedSetPool* pool_;
        std::unique_ptr<VisitedSet> set_;
    };

    explicit VisitedSetPool(std::size_t capacity);

    [[nodiscard]] Lease acquire();

    /// Number of sets currently waiting in the pool.
    [[nodiscard]] std::size_t idleCount() const;

private:
    void release(std::unique_ptr<VisitedSet> set);

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedSet>> idle_;
};

}
