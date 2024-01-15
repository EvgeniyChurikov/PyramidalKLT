cmake_minimum_required(VERSION 3.27)
project(PyramidalKLT)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(PyramidalKLT
        main.cpp
        Tracker.h
        Tracker.cpp)
target_link_libraries (PyramidalKLT Eigen3::Eigen)
target_link_libraries(PyramidalKLT ${OpenCV_LIBS})
