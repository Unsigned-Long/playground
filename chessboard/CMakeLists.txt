cmake_minimum_required(VERSION 3.10)

project(chessboard VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chessboard ./main.cpp)

target_link_libraries(chessboard ${OpenCV_LIBRARIES})

