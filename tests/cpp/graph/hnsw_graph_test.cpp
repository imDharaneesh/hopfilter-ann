#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>

namespace hopfilter {
namespace {

std::vector<NodeId> asVector(std::span<const NodeId> ids) {
    return {ids.begin(), ids.end()};
}

TEST(HnswGraphTest, NewNodesHaveDenseIdsAndNoNeighbors) {
    HnswGraph graph{4, 4, 2};

    EXPECT_EQ(graph.addNode(0), 0U);
    EXPECT_EQ(graph.addNode(2), 1U);

    EXPECT_EQ(graph.size(), 2U);
    EXPECT_EQ(graph.levelOf(1), 2);
    EXPECT_TRUE(graph.neighbors(0, 0).empty());
    EXPECT_TRUE(graph.neighbors(1, 2).empty());
}

TEST(HnswGraphTest, NeighborsAreStoredPerLevel) {
    HnswGraph graph{4, 4, 2};
    for (int node = 0; node < 3; ++node) {
        static_cast<void>(graph.addNode(1));
    }
    const std::vector<NodeId> levelZero{1, 2};
    const std::vector<NodeId> levelOne{2};

    graph.setNeighbors(0, 0, levelZero);
    graph.setNeighbors(0, 1, levelOne);

    EXPECT_EQ(asVector(graph.neighbors(0, 0)), levelZero);
    EXPECT_EQ(asVector(graph.neighbors(0, 1)), levelOne);
    EXPECT_TRUE(graph.neighbors(1, 0).empty());
}

TEST(HnswGraphTest, SetNeighborsReplacesPreviousList) {
    HnswGraph graph{3, 4, 2};
    for (int node = 0; node < 3; ++node) {
        static_cast<void>(graph.addNode(0));
    }
    graph.setNeighbors(0, 0, std::vector<NodeId>{1, 2});

    graph.setNeighbors(0, 0, std::vector<NodeId>{2});

    EXPECT_EQ(asVector(graph.neighbors(0, 0)), std::vector<NodeId>{2});
}

TEST(HnswGraphTest, DegreeBoundsDifferBetweenLevelZeroAndUpperLevels) {
    HnswGraph graph{5, 4, 2};
    for (int node = 0; node < 5; ++node) {
        static_cast<void>(graph.addNode(1));
    }

    EXPECT_EQ(graph.maxNeighbors(0), 4U);
    EXPECT_EQ(graph.maxNeighbors(1), 2U);
    EXPECT_NO_THROW(graph.setNeighbors(0, 0, std::vector<NodeId>{1, 2, 3, 4}));
    EXPECT_THROW(graph.setNeighbors(0, 1, std::vector<NodeId>{1, 2, 3}), InvalidArgumentError);
}

TEST(HnswGraphTest, RejectsNeighborsAboveNodeLevel) {
    HnswGraph graph{2, 4, 2};
    static_cast<void>(graph.addNode(0));
    static_cast<void>(graph.addNode(0));

    EXPECT_THROW(graph.setNeighbors(0, 1, std::vector<NodeId>{1}), InvalidArgumentError);
}

TEST(HnswGraphTest, EntryPointCarriesItsLevel) {
    HnswGraph graph{3, 4, 2};
    static_cast<void>(graph.addNode(0));
    static_cast<void>(graph.addNode(3));

    EXPECT_FALSE(graph.entryPoint().has_value());
    graph.setEntryPoint(1);

    EXPECT_EQ(graph.entryPoint(), NodeId{1});
    EXPECT_EQ(graph.maxLevel(), 3);
}

TEST(HnswGraphTest, RejectsAddWhenFull) {
    HnswGraph graph{1, 4, 2};
    static_cast<void>(graph.addNode(0));

    EXPECT_THROW(static_cast<void>(graph.addNode(0)), IndexStateError);
}

TEST(HnswGraphTest, RejectsInvalidShape) {
    EXPECT_THROW(HnswGraph(0, 4, 2), InvalidArgumentError);
    EXPECT_THROW(HnswGraph(4, 0, 2), InvalidArgumentError);
    EXPECT_THROW(HnswGraph(4, 4, 0), InvalidArgumentError);
    EXPECT_THROW(HnswGraph(kInvalidNodeId, 4, 2), InvalidArgumentError);
}

}  // namespace
}  // namespace hopfilter
