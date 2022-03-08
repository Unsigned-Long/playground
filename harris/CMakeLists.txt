cmake_minimum_required(VERSION 3.10)

project(harris VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(./include ${OPENCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})

aux_source_directory(./src SRC_DIR)

add_library(Harris SHARED  ${SRC_DIR})

add_executable(harris ./main.cpp ${SRC_DIR})

target_link_libraries(harris ${OpenCV_LIBS})

target_link_libraries(Harris ${OpenCV_LIBS})