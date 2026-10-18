include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(hopfilter_core_tests
    common/types_test.cpp
    common/version_test.cpp
    distance/kernels_test.cpp
    distance/metrics_test.cpp
)
target_include_directories(hopfilter_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hopfilter_core_tests PRIVATE hopfilter::core GTest::gtest_main)
target_compile_definitions(hopfilter_core_tests
    PRIVATE HOPFILTER_EXPECTED_VERSION="${PROJECT_VERSION}"
)

hopfilter_set_warnings(hopfilter_core_tests)
hopfilter_enable_sanitizer(hopfilter_core_tests)

gtest_discover_tests(hopfilter_core_tests DISCOVERY_MODE PRE_TEST)
