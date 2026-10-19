#pragma once

#include <span>
#include <string_view>

namespace hopfilter {

class DistanceMetric {
public:
    virtual ~DistanceMetric() = default;

    [[nodiscard]] virtual float distance(std::span<const float> lhs,
                                         std::span<const float> rhs) const noexcept = 0;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    virtual void prepare(std::span<float> vector) const noexcept;
};

}
