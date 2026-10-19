#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/filter/attribute_store.hpp>

namespace hopfilter {
namespace {

void requireName(const std::string& name) {
    if (name.empty()) {
        throw InvalidArgumentError("attribute name must be non-empty");
    }
}

[[noreturn]] void throwMissing(std::string_view name, std::string_view kind, bool otherKindExists) {
    const std::string reason = otherKindExists ? "' is not " : "' does not exist as ";
    throw InvalidArgumentError("attribute '" + std::string{name} + reason + std::string{kind});
}

}

void AttributeStore::setCategorical(std::string name, CategoricalColumn codes) {
    requireName(name);
    numeric_.erase(name);
    categorical_.insert_or_assign(std::move(name), std::move(codes));
}

void AttributeStore::setNumeric(std::string name, NumericColumn values) {
    requireName(name);
    categorical_.erase(name);
    numeric_.insert_or_assign(std::move(name), std::move(values));
}

const CategoricalColumn& AttributeStore::categorical(std::string_view name) const {
    const auto found = categorical_.find(name);
    if (found == categorical_.end()) {
        throwMissing(name, "categorical", numeric_.contains(name));
    }
    return found->second;
}

const NumericColumn& AttributeStore::numeric(std::string_view name) const {
    const auto found = numeric_.find(name);
    if (found == numeric_.end()) {
        throwMissing(name, "numeric", categorical_.contains(name));
    }
    return found->second;
}

std::vector<std::string> AttributeStore::names() const {
    std::vector<std::string> result;
    result.reserve(categorical_.size() + numeric_.size());
    for (const auto& [name, column] : categorical_) {
        result.push_back(name);
    }
    for (const auto& [name, column] : numeric_) {
        result.push_back(name);
    }
    std::ranges::sort(result);
    return result;
}

}
