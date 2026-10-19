#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <hopfilter/common/prefetch.hpp>
#include <hopfilter/common/types.hpp>

namespace hopfilter {

class VisitedSet {
public:
    explicit VisitedSet(std::size_t capacity);

    VisitedSet(std::size_t capacity, std::uint32_t initialEpoch);

    void clear() noexcept;

    [[nodiscard]] bool markVisited(NodeId id) noexcept;

    [[nodiscard]] bool isVisited(NodeId id) const noexcept;

    void prefetch(NodeId id) const noexcept { prefetchForRead(&stamps_[id]); }

    [[nodiscard]] std::size_t capacity() const noexcept { return stamps_.size(); }

private:
    std::vector<std::uint32_t> stamps_;
    std::uint32_t epoch_;
};

class VisitedSetPool {
public:
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

    [[nodiscard]] std::size_t idleCount() const;

private:
    void release(std::unique_ptr<VisitedSet> set);

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedSet>> idle_;
};

}
