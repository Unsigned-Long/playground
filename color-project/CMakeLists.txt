cmake_minimum_required(VERSION 3.10)

project(colorProject VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV )

include_directories(./include ${OPENCV_INCLUDE_DIRS})

aux_source_directory(./src SRC_DIR)

add_executable(colorProject ./main.cpp ${SRC_DIR})

add_library(colorPrj SHARED ${SRC_DIR})

target_link_libraries(colorProject ${OpenCV_LIBS})

target_link_libraries(colorPrj ${OpenCV_LIBS})