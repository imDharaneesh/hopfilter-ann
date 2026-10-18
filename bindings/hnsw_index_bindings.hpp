#pragma once

#include <pybind11/pybind11.h>

namespace hopfilter::bindings {

void bindHnswIndex(pybind11::module_& module);
void bindKernelSelection(pybind11::module_& module);

}
