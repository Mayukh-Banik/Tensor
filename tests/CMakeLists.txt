# set(TEST_SOURCES main_test.cpp)
cmake_minimum_required(VERSION 3.28.3)

file(GLOB_RECURSE TEST_SOURCES
    "*.cpp"
    "*.cu"
)

set(CMAKE_CXX_STANDARD 17)

# Create the test executable
add_executable(tests ${TEST_SOURCES})

# Include directories
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link the test executable against Google Test and the main project
target_link_libraries(tests PRIVATE
    gtest
    gtest_main
    TENSOR
)

set_target_properties(TENSOR PROPERTIES LINKER_LANGUAGE CUDA)

# Optionally, set CUDA separable compilation if needed
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add a CMake target for running the tests
add_test(NAME tests COMMAND tests)