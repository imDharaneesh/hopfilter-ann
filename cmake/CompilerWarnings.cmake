function(hopfilter_set_warnings target)
    set(msvcWarnings /W4 /permissive- /utf-8 /w14242 /w14254 /w14263 /w14265 /w14287 /w14296)
    set(gccClangWarnings
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wnull-dereference
    )

    if(HOPFILTER_WARNINGS_AS_ERRORS)
        list(APPEND msvcWarnings /WX)
        list(APPEND gccClangWarnings -Werror)
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE ${msvcWarnings})
    else()
        target_compile_options(${target} PRIVATE ${gccClangWarnings})
    endif()
endfunction()
