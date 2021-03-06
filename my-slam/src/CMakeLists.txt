set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(Eigen3)

find_package(OpenCV)

find_package(Sophus)

find_package(Ceres)

include_directories(./include 
                    ${EIGEN_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS} 
                    ${SOPHUS_INCLUDE_DIRS}
                    ${CERES_INCLUDE_DIRS})

aux_source_directory(./src SRC_FILES)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SRC_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} 
                      PRIVATE
                      ${OpenCV_LIBS} 
                      Sophus::Sophus
                      ${CERES_LIBRARIES})
