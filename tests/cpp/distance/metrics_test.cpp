#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/kernels.hpp>
#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/distance/metrics.hpp>

namespace hopfilter {
namespace {

TEST(L2MetricTest, ReturnsSquaredDistance) {
    const std::vector<float> origin{0.0F, 0.0F};
    const std::vector<float> point{3.0F, 4.0F};

    EXPECT_FLOAT_EQ(L2Metric{}.distance(origin, point), 25.0F);
    EXPECT_EQ(L2Metric{}.name(), "l2");
}

TEST(L2MetricTest, PrepareLeavesVectorUnchanged) {
    std::vector<float> vector{3.0F, 4.0F};

    L2Metric{}.prepare(vector);

    EXPECT_EQ(vector, (std::vector<float>{3.0F, 4.0F}));
}

TEST(InnerProductMetricTest, LargerDotProductIsCloser) {
    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> aligned{2.0F, 0.0F};
    const std::vector<float> orthogonal{0.0F, 5.0F};
    const InnerProductMetric metric;

    EXPECT_FLOAT_EQ(metric.distance(query, aligned), -1.0F);
    EXPECT_FLOAT_EQ(metric.distance(query, orthogonal), 1.0F);
    EXPECT_LT(metric.distance(query, aligned), metric.distance(query, orthogonal));
}

TEST(CosineMetricTest, PrepareNormalizesToUnitLength) {
    std::vector<float> vector{3.0F, 4.0F};

    CosineMetric{}.prepare(vector);

    EXPECT_FLOAT_EQ(vector[0], 0.6F);
    EXPECT_FLOAT_EQ(vector[1], 0.8F);
}

TEST(CosineMetricTest, IgnoresMagnitudeAfterPrepare) {
    const CosineMetric metric;
    std::vector<float> lhs{1.0F, 1.0F};
    std::vector<float> rhs{10.0F, 10.0F};
    std::vector<float> opposite{-2.0F, -2.0F};
    metric.prepare(lhs);
    metric.prepare(rhs);
    metric.prepare(opposite);

    EXPECT_NEAR(metric.distance(lhs, rhs), 0.0F, 1e-6F);
    EXPECT_NEAR(metric.distance(lhs, opposite), 2.0F, 1e-6F);
}

TEST(CosineMetricTest, ZeroVectorStaysZero) {
    std::vector<float> zero{0.0F, 0.0F, 0.0F};

    CosineMetric{}.prepare(zero);

    EXPECT_EQ(zero, (std::vector<float>{0.0F, 0.0F, 0.0F}));
}

TEST(MetricRegistryTest, BuiltinsCreateMetricsByName) {
    const auto registry = MetricRegistry::withBuiltins();

    EXPECT_EQ(registry.names(), (std::vector<std::string>{"cosine", "ip", "l2"}));
    for (const std::string& name : registry.names()) {
        EXPECT_EQ(registry.create(name)->name(), name);
    }
}

template <typename Action>
std::string invalidArgumentMessage(Action action) {
    try {
        action();
    } catch (const InvalidArgumentError& error) {
        return error.what();
    }
    return "<no InvalidArgumentError thrown>";
}

TEST(MetricRegistryTest, UnknownNameListsKnownMetrics) {
    const auto registry = MetricRegistry::withBuiltins();

    EXPECT_EQ(invalidArgumentMessage([&] { static_cast<void>(registry.create("manhattan")); }),
              "unknown metric 'manhattan'; known metrics: cosine, ip, l2");
}

TEST(MetricRegistryTest, CustomMetricCanBeRegistered) {
    auto registry = MetricRegistry::withBuiltins();

    registry.registerMetric("squared-l2-alias", [] { return std::make_unique<L2Metric>(); });

    EXPECT_EQ(registry.create("squared-l2-alias")->name(), "l2");
}

TEST(MetricRegistryTest, RejectsDuplicateAndEmptyRegistrations) {
    auto registry = MetricRegistry::withBuiltins();
    const MetricRegistry::Factory makeL2 = [] { return std::make_unique<L2Metric>(); };

    EXPECT_THROW(registry.registerMetric("l2", makeL2), InvalidArgumentError);
    EXPECT_THROW(registry.registerMetric("", makeL2), InvalidArgumentError);
    EXPECT_THROW(registry.registerMetric("empty", nullptr), InvalidArgumentError);
}

}
}
