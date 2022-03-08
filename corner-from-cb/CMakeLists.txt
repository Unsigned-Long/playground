cmake_minimum_required(VERSION 3.10)

project(cornerSelector VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(Ceres)

find_package(PCL REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(./include ${OPENCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ./thirdparty/include)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

link_directories(./thirdparty/lib)

aux_source_directory(./src SRC_DIR)

add_executable(cornerSelector ./main.cpp ${SRC_DIR})

target_link_libraries(cornerSelector ${OpenCV_LIBS} ${PCL_LIBRARIES} Harris ${CERES_LIBRARIES})
