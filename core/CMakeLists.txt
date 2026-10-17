add_library(hopfilter_core STATIC
    src/common/version.cpp
)
add_library(hopfilter::core ALIAS hopfilter_core)

target_include_directories(hopfilter_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(hopfilter_core PUBLIC cxx_std_20)

set_source_files_properties(src/common/version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HOPFILTER_VERSION=\"${PROJECT_VERSION}\""
)

hopfilter_set_warnings(hopfilter_core)
hopfilter_enable_sanitizer(hopfilter_core)
