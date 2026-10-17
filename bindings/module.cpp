#include <string>

#include <pybind11/pybind11.h>

#include <hopfilter/common/version.hpp>

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native core of hopfilter: HNSW index and filtered search.";
    module.def("version", [] { return std::string{hopfilter::version()}; });
    module.attr("__version__") = std::string{hopfilter::version()};
}
