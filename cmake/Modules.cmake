function(hopfilter_add_module name)
    cmake_parse_arguments(MODULE "" "" "SOURCES;DEPENDS" ${ARGN})
    set(target hopfilter_${name})
    add_library(${target} STATIC ${MODULE_SOURCES})
    add_library(hopfilter::${name} ALIAS ${target})
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    )
    target_link_libraries(${target} PUBLIC ${MODULE_DEPENDS})
    target_compile_features(${target} PUBLIC cxx_std_20)
    hopfilter_set_warnings(${target})
    hopfilter_enable_sanitizer(${target})
    set_property(GLOBAL APPEND PROPERTY HOPFILTER_MODULES ${target})
endfunction()
