cmake_minimum_required(VERSION 3.18)

project(Open3DReconstructionCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Open3D REQUIRED)

add_executable(poisson_reconstruction_cpp poisson_reconstruction.cpp)
target_link_libraries(poisson_reconstruction_cpp PRIVATE Open3D::Open3D)
