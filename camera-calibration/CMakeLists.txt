cmake_minimum_required(VERSION 3.10)

project(cameraCal VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

find_package(Ceres)

find_package(Eigen3)

include_directories(./include ${CERES_INCLUDE_DIRS} ./thirdparty/include ${EIGEN_INCLUDE_DIRS})

aux_source_directory(./src SRC_DIR)

add_executable(cameraCal ./main.cpp ${SRC_DIR})

target_link_libraries(cameraCal ${CERES_LIBRARIES})
