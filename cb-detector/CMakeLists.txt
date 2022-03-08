cmake_minimum_required(VERSION 3.10)

project(cbdetector VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)

find_package(Ceres)

find_package(PCL REQUIRED)

include_directories(./include ${OPENCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

aux_source_directory(./src SRC_DIR)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(cbdetector ./main.cpp  ${SRC_DIR})

target_link_libraries(cbdetector ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})