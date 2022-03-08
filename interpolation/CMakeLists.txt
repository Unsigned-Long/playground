cmake_minimum_required(VERSION 3.10)

project(interploation VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL 1.3 REQUIRED common kdtree)

include_directories(./include ./thirdparty/include ${PCL_INCLUDE_DIRS})

aux_source_directory(./src SRC_DIR)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(interploation ./main.cpp ${SRC_DIR})

target_link_libraries(interploation ${PCL_LIBRARIES})
