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
        src/distance/cpu_features.cpp
        src/distance/kernel_dispatch.cpp
        src/distance/kernels_scalar.cpp
        src/distance/metrics.cpp
        src/distance/metric_registry.cpp
    DEPENDS
        hopfilter::common
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|x64)$")
    target_sources(hopfilter_distance PRIVATE src/distance/kernels_avx2.cpp)
    target_compile_definitions(hopfilter_distance PRIVATE HOPFILTER_HAVE_AVX2_KERNELS=1)
    if(MSVC)
        set(hopfilterAvx2Flags /arch:AVX2)
    else()
        set(hopfilterAvx2Flags -mavx2 -mfma)
    endif()
    set_source_files_properties(src/distance/kernels_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "${hopfilterAvx2Flags}"
    )
endif()

hopfilter_add_module(storage
    SOURCES
        src/storage/vector_store.cpp
    DEPENDS
        hopfilter::common
)

hopfilter_add_module(graph
    SOURCES
        src/graph/hnsw_graph.cpp
        src/graph/layer_assigner.cpp
        src/graph/neighbor_selector.cpp
    DEPENDS
        hopfilter::common
        hopfilter::distance
        hopfilter::storage
)

hopfilter_add_module(search
    SOURCES
        src/search/search_layer.cpp
        src/search/search_strategy.cpp
    DEPENDS
        hopfilter::common
        hopfilter::graph
)

hopfilter_add_module(index
    SOURCES
        src/index/hnsw_index.cpp
        src/index/hnsw_index_builder.cpp
    DEPENDS
        hopfilter::common
        hopfilter::distance
        hopfilter::storage
        hopfilter::graph
        hopfilter::search
)

hopfilter_add_module(io
    SOURCES
        src/io/binary_stream.cpp
        src/io/crc32.cpp
        src/io/index_serializer.cpp
    DEPENDS
        hopfilter::index
)

add_library(hopfilter_core INTERFACE)
add_library(hopfilter::core ALIAS hopfilter_core)
get_property(hopfilterModules GLOBAL PROPERTY HOPFILTER_MODULES)
target_link_libraries(hopfilter_core INTERFACE ${hopfilterModules})
