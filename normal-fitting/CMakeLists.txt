cmake_minimum_required(VERSION 3.10)

project(normalFitting VERSION 1.0)

find_package(Ceres)

include_directories(./include ./thirdparty/include ${CERES_INCLUDE_DIRS})

add_executable(normalFitting ./main.cpp)

target_link_libraries(normalFitting ${CERES_LIBRARIES})

