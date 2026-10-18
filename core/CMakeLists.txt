find_package(Threads REQUIRED)

hopfilter_add_module(common
    SOURCES
        src/common/version.cpp
        src/common/visited_set.cpp
    DEPENDS
        Threads::Threads
)
set_source_files_properties(src/common/version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HOPFILTER_VERSION=\"${PROJECT_VERSION}\""
)

hopfilter_add_module(distance
    SOURCES
        src/distance/kernels.cpp
        src/distance/metrics.cpp
        src/distance/metric_registry.cpp
    DEPENDS
        hopfilter::common
)

hopfilter_add_module(storage
    SOURCES
        src/storage/vector_store.cpp
    DEPENDS
        hopfilter::common
)

add_library(hopfilter_core INTERFACE)
add_library(hopfilter::core ALIAS hopfilter_core)
get_property(hopfilterModules GLOBAL PROPERTY HOPFILTER_MODULES)
target_link_libraries(hopfilter_core INTERFACE ${hopfilterModules})
