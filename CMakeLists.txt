cmake_minimum_required(VERSION 3.28.3)

project(temp LANGUAGES CXX CUDA)

add_subdirectory(Tensor)
add_subdirectory(googletest)
add_subdirectory(tests)

