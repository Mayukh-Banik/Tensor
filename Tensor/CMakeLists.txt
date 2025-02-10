# find_package(CUDA REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
    "*.h"
    "*.hpp"
    "*.cuh"
)

add_library(TENSOR INTERFACE)

set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

target_sources(TENSOR INTERFACE ${SOURCE_FILES})

target_include_directories(TENSOR INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(TENSOR
    INTERFACE
)
