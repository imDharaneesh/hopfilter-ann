#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>

namespace hopfilter {

template <typename Product>
class NamedFactories {
public:
    using Factory = std::function<std::unique_ptr<Product>()>;

    explicit NamedFactories(std::string kind) : kind_(std::move(kind)) {}

    void add(std::string name, Factory factory) {
        if (name.empty() || !factory) {
            throw InvalidArgumentError(kind_ + " name and factory must be non-empty");
        }
        const auto position = std::ranges::lower_bound(factories_, name, {}, &Entry::first);
        if (position != factories_.end() && position->first == name) {
            throw InvalidArgumentError(kind_ + " '" + name + "' is already registered");
        }
        factories_.emplace(position, std::move(name), std::move(factory));
    }

    [[nodiscard]] std::unique_ptr<Product> create(std::string_view name) const {
        const auto found = std::ranges::find(factories_, name, &Entry::first);
        if (found == factories_.end()) {
            throw InvalidArgumentError("unknown " + kind_ + " '" + std::string{name} +
                                       "'; known: " + joinedNames());
        }
        return found->second();
    }

    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(factories_.size());
        for (const auto& [name, factory] : factories_) {
            result.push_back(name);
        }
        return result;
    }

private:
    using Entry = std::pair<std::string, Factory>;

    [[nodiscard]] std::string joinedNames() const {
        std::string result;
        for (const auto& [name, factory] : factories_) {
            result += result.empty() ? name : ", " + name;
        }
        return result;
    }

    std::string kind_;
    std::vector<Entry> factories_;
};

}
