cmake_minimum_required(VERSION 3.24)

project(hopfilter VERSION 0.1.0 LANGUAGES CXX)

option(HOPFILTER_BUILD_TESTS "Build the C++ unit tests" OFF)
option(HOPFILTER_BUILD_BENCHMARKS "Build the C++ benchmark harnesses" OFF)
option(HOPFILTER_BUILD_PYTHON "Build the Python extension module" ${SKBUILD})
option(HOPFILTER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(HOPFILTER_SANITIZER "" CACHE STRING "Sanitizer to enable: address, undefined or thread")
set_property(CACHE HOPFILTER_SANITIZER PROPERTY STRINGS "" address undefined thread)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(Modules)

add_subdirectory(core)

if(HOPFILTER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

if(HOPFILTER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/cpp)
endif()

if(HOPFILTER_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
