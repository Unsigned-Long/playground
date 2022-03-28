cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")

project(mono-camera VERSION 1.0 LANGUAGES CXX)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)