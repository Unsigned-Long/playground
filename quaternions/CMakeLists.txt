cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Release")

project(quaternion VERSION 1.0)

include_directories(./include ./thirdparty/include)

aux_source_directory(./src SRC_DIR)

add_library(Quaternion SHARED ${SRC_DIR})

add_executable(quaternion ./main.cpp)

target_link_libraries(quaternion Quaternion)