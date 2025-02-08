cmake_minimum_required(VERSION 3.10)
project(Simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(FetchContent)

# 拉取并设置 pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.0
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# 生成 Python 模块
pybind11_add_module(simulator src/simulator/simulator.cpp)
install(TARGETS simulator LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/Lib/site-packages)