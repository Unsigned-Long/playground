cmake_minimum_required(VERSION 3.10)

project(gaussian VERSION 1.0)

include_directories(./include)

add_executable(gaussian ./main.cpp)


target_link_libraries(gaussian mgl)