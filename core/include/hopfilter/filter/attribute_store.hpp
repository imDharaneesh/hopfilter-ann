#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace hopfilter {

using CategoryCode = std::uint32_t;
using CategoricalColumn = std::vector<CategoryCode>;
using NumericColumn = std::vector<double>;

class AttributeStore {
public:
    template <typename Column>
    using Columns = std::map<std::string, Column, std::less<>>;

    void setCategorical(std::string name, CategoricalColumn codes);
    void setNumeric(std::string name, NumericColumn values);

    [[nodiscard]] const CategoricalColumn& categorical(std::string_view name) const;
    [[nodiscard]] const NumericColumn& numeric(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] bool empty() const noexcept { return categorical_.empty() && numeric_.empty(); }

    [[nodiscard]] const Columns<CategoricalColumn>& categoricalColumns() const noexcept {
        return categorical_;
    }
    [[nodiscard]] const Columns<NumericColumn>& numericColumns() const noexcept { return numeric_; }

private:
    Columns<CategoricalColumn> categorical_;
    Columns<NumericColumn> numeric_;
};

}
