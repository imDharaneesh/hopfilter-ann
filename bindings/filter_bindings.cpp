#include "filter_bindings.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_expression.hpp>

namespace hopfilter::bindings {
namespace {

namespace py = pybind11;

using BoolVector = py::array_t<bool, py::array::c_style | py::array::forcecast>;

FilterExpression maskOf(const BoolVector& mask) {
    if (mask.ndim() != 1) {
        throw InvalidArgumentError("mask must be one-dimensional");
    }
    const auto values = mask.unchecked<1>();
    DynamicBitset bits{static_cast<std::size_t>(values.shape(0))};
    for (py::ssize_t position = 0; position < values.shape(0); ++position) {
        if (values(position)) {
            bits.set(static_cast<std::size_t>(position));
        }
    }
    return FilterExpression::mask(std::move(bits));
}

std::string reprOf(const FilterExpression& expression) {
    const std::string text = expression.describe();
    const bool wrapped = text.size() >= 2 && text.front() == '(' && text.back() == ')';
    return "Filter(" + (wrapped ? text.substr(1, text.size() - 2) : text) + ")";
}

[[noreturn]] void refuseTruthValue([[maybe_unused]] const FilterExpression& expression) {
    throw py::type_error(
        "a Filter has no truth value; combine filters with &, | and ~ instead of and, or, not");
}

}

void bindFilter(py::module_& module) {
    py::class_<FilterExpression>(module, "Filter")
        .def_static("equals", &FilterExpression::equals, py::arg("column"), py::arg("value"))
        .def_static("between",
                    &FilterExpression::between,
                    py::arg("column"),
                    py::arg("low"),
                    py::arg("high"))
        .def_static("mask", &maskOf, py::arg("mask"))
        .def_static("all_of", &FilterExpression::allOf, py::arg("filters"))
        .def_static("any_of", &FilterExpression::anyOf, py::arg("filters"))
        .def("__and__",
             [](const FilterExpression& lhs, const FilterExpression& rhs) { return lhs & rhs; })
        .def("__or__",
             [](const FilterExpression& lhs, const FilterExpression& rhs) { return lhs | rhs; })
        .def("__invert__", [](const FilterExpression& expression) { return !expression; })
        .def("__bool__", &refuseTruthValue)
        .def("__repr__", &reprOf);
}

}
