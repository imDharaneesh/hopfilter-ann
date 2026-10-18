#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

TEST(VectorStoreTest, AppendReturnsSequentialIdsAndCopiesData) {
    VectorStore store{3, 4};
    std::vector<float> first{1.0F, 2.0F, 3.0F};
    const std::vector<float> second{4.0F, 5.0F, 6.0F};

    EXPECT_EQ(store.append(first), 0U);
    EXPECT_EQ(store.append(second), 1U);
    first[0] = 99.0F;

    EXPECT_EQ(store.size(), 2U);
    const auto stored = store.at(0);
    EXPECT_EQ(std::vector<float>(stored.begin(), stored.end()), (std::vector<float>{1, 2, 3}));
    EXPECT_EQ(store.at(1)[2], 6.0F);
}

TEST(VectorStoreTest, RowsAreCacheLineAlignedAndPaddedWithZeros) {
    VectorStore store{20, 3};
    const std::vector<float> ones(20, 1.0F);
    for (int row = 0; row < 3; ++row) {
        static_cast<void>(store.append(ones));
    }

    EXPECT_EQ(store.stride(), 32U);
    for (NodeId id = 0; id < 3; ++id) {
        const auto address = std::bit_cast<std::uintptr_t>(store.at(id).data());
        EXPECT_EQ(address % VectorStore::kAlignmentBytes, 0U);
        const float* padding = store.at(id).data() + store.dimension();
        for (std::size_t offset = 0; offset < store.stride() - store.dimension(); ++offset) {
            EXPECT_EQ(padding[offset], 0.0F);
        }
    }
}

TEST(VectorStoreTest, StrideEqualsDimensionWhenAlreadyAligned) {
    const VectorStore store{128, 1};

    EXPECT_EQ(store.stride(), 128U);
}

TEST(VectorStoreTest, RejectsWrongDimension) {
    VectorStore store{3, 2};
    const std::vector<float> tooShort{1.0F, 2.0F};

    EXPECT_THROW(static_cast<void>(store.append(tooShort)), InvalidArgumentError);
    EXPECT_EQ(store.size(), 0U);
}

TEST(VectorStoreTest, RejectsAppendWhenFull) {
    VectorStore store{2, 1};
    const std::vector<float> vector{1.0F, 2.0F};
    static_cast<void>(store.append(vector));

    EXPECT_THROW(static_cast<void>(store.append(vector)), IndexStateError);
}

TEST(VectorStoreTest, RejectsInvalidShape) {
    EXPECT_THROW(VectorStore(0, 10), InvalidArgumentError);
    EXPECT_THROW(VectorStore(10, 0), InvalidArgumentError);
    EXPECT_THROW(VectorStore(10, kInvalidNodeId), InvalidArgumentError);
}

}  // namespace
}  // namespace hopfilter
