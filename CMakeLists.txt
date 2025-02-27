cmake_minimum_required(VERSION 3.10)
project(Simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(PYBIND11_DIR include/pybind11)

add_subdirectory(${PYBIND11_DIR} EXCLUDE_FROM_ALL)

include_directories("${CMAKE_SOURCE_DIR}/src/simulator")
include_directories("${CMAKE_SOURCE_DIR}/src/simulator/envs")
include_directories("${CMAKE_SOURCE_DIR}/src/simulator/objects")

set(OUTPUT_BASE_DIR "${CMAKE_SOURCE_DIR}/python/aerodrome/simulator")

# 1. 收集需要处理的源文件
file(GLOB ENV_FILES "src/simulator/envs/*.cpp")
file(GLOB OBJECT_FILES "src/simulator/objects/*.cpp")
file(GLOB BASE_ENV_FILES "src/simulator/*.cpp")

# 2. 合并所有文件
list(APPEND ALL_FILES ${ENV_FILES} ${OBJECT_FILES} ${BASE_ENV_FILES})

# 3. 为每个 .h 文件生成对应的 .pyd 文件
foreach(FILE ${ALL_FILES})
    # 获取文件的基本名称，不带路径
    get_filename_component(FILE_NAME ${FILE} NAME_WE)

    # 提取文件的相对路径
    get_filename_component(FILE_DIR ${FILE} DIRECTORY)

    # 创建输出目录
    string(REPLACE "src/simulator" "python/aerodrome/simulator" FILE_OUTPUT_DIR ${FILE_DIR})
    set(OUTPUT_DIR "${FILE_OUTPUT_DIR}")
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # 使用 pybind11_add_module 来生成模块
    pybind11_add_module(${FILE_NAME} ${FILE})

    # 设置生成的 .pyd 文件的输出路径
    set_target_properties(${FILE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )

    # 处理调试/发布版的输出路径
    set_target_properties(${FILE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    )

    # add_custom_command(TARGET ${FILE_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${FILE_NAME}.lib
    #     COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${FILE_NAME}.exp
    #     COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${FILE_NAME}.pdb
    # )
endforeach()