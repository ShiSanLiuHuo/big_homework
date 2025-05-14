cmake_minimum_required(VERSION 3.22.1)
project(big_homework)

add_compile_options(-g)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(big_homework big_homework.cpp)

target_include_directories(big_homework PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(big_homework PRIVATE ${OpenCV_LIBRARIES})