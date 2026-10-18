#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/types.hpp>
#include <hopfilter/distance/metrics.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

// Base at the origin; A and B point the same way (B just behind A), C points elsewhere.
constexpr NodeId kBase = 0;
constexpr NodeId kA = 1;
constexpr NodeId kB = 2;
constexpr NodeId kC = 3;

class NeighborSelectorTest : public ::testing::Test {
protected:
    NeighborSelectorTest() : store_(2, 4) {
        for (const auto& point : std::vector<std::vector<float>>{
                 {0.0F, 0.0F}, {1.0F, 0.0F}, {1.2F, 0.1F}, {0.0F, 1.3F}}) {
            static_cast<void>(store_.append(point));
        }
    }

    [[nodiscard]] std::vector<Neighbor> candidatesOfBase() const {
        const NodeDistances distances = this->distances();
        std::vector<Neighbor> candidates;
        for (const NodeId id : {kA, kB, kC}) {
            candidates.push_back({id, distances.between(kBase, id)});
        }
        return candidates;
    }

    [[nodiscard]] NodeDistances distances() const { return NodeDistances{metric_, store_}; }

private:
    L2Metric metric_;
    VectorStore store_;
};

TEST_F(NeighborSelectorTest, CandidatesArriveClosestFirst) {
    const auto candidates = candidatesOfBase();

    EXPECT_LT(candidates[0].distance, candidates[1].distance);
    EXPECT_LT(candidates[1].distance, candidates[2].distance);
}

TEST_F(NeighborSelectorTest, SimpleSelectorKeepsTheClosest) {
    const auto selected = SimpleNeighborSelector{}.select(candidatesOfBase(), 2, distances());

    EXPECT_EQ(selected, (std::vector<NodeId>{kA, kB}));
}

TEST_F(NeighborSelectorTest, HeuristicSkipsCandidateShadowedBySelectedNeighbor) {
    const auto selected = HeuristicNeighborSelector{}.select(candidatesOfBase(), 2, distances());

    EXPECT_EQ(selected, (std::vector<NodeId>{kA, kC}));
}

TEST_F(NeighborSelectorTest, HeuristicCanReturnFewerThanMax) {
    const auto selected = HeuristicNeighborSelector{}.select(candidatesOfBase(), 3, distances());

    EXPECT_EQ(selected, (std::vector<NodeId>{kA, kC}));
}

TEST_F(NeighborSelectorTest, KeepPrunedConnectionsRefillsFreeSlots) {
    const HeuristicNeighborSelector selector{true};

    const auto selected = selector.select(candidatesOfBase(), 3, distances());

    EXPECT_EQ(selected, (std::vector<NodeId>{kA, kC, kB}));
}

TEST_F(NeighborSelectorTest, SelectorsHandleEmptyCandidates) {
    const std::vector<Neighbor> none;

    EXPECT_TRUE(SimpleNeighborSelector{}.select(none, 4, distances()).empty());
    EXPECT_TRUE(HeuristicNeighborSelector{true}.select(none, 4, distances()).empty());
}

}  // namespace
}  // namespace hopfilter
