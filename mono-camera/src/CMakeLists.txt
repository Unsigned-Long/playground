set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(Eigen3)

find_package(OpenCV)

include_directories(./include ${EIGEN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} main.cpp src/camera.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
