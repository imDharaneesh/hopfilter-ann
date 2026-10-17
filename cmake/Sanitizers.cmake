set(HOPFILTER_SUPPORTED_SANITIZERS address undefined thread)

function(hopfilter_enable_sanitizer target)
    if(NOT HOPFILTER_SANITIZER)
        return()
    endif()

    if(NOT HOPFILTER_SANITIZER IN_LIST HOPFILTER_SUPPORTED_SANITIZERS)
        message(FATAL_ERROR "Unknown HOPFILTER_SANITIZER '${HOPFILTER_SANITIZER}'")
    endif()

    if(MSVC)
        if(NOT HOPFILTER_SANITIZER STREQUAL "address")
            message(FATAL_ERROR "MSVC only supports HOPFILTER_SANITIZER=address")
        endif()
        target_compile_options(${target} PRIVATE /fsanitize=address)
        return()
    endif()

    set(sanitizerFlags -fsanitize=${HOPFILTER_SANITIZER} -fno-omit-frame-pointer)
    if(HOPFILTER_SANITIZER STREQUAL "undefined")
        list(APPEND sanitizerFlags -fno-sanitize-recover=all)
    endif()

    target_compile_options(${target} PRIVATE ${sanitizerFlags})
    target_link_options(${target} PUBLIC ${sanitizerFlags})
endfunction()
