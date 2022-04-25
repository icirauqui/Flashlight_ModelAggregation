cmake_minimum_required(VERSION 3.10)

project(mdlAgg) 

#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(Gloo_DIR /opt/gloo/share/cmake)
set(DNNL_DIR /opt/onednn/lib/cmake/dnnl)

add_executable(mdlAgg ../src/main.cpp)

find_package(flashlight CONFIG REQUIRED)
target_link_libraries(mdlAgg PRIVATE flashlight::flashlight)