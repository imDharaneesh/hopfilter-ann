find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE
    module.cpp
    error_translation.cpp
    hnsw_index_bindings.cpp
)
target_link_libraries(_core PRIVATE hopfilter::core)

hopfilter_set_warnings(_core)

install(TARGETS _core LIBRARY DESTINATION hopfilter)
