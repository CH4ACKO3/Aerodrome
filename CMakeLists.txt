cmake_minimum_required(VERSION 3.10)
project(Simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

include_directories(src/simulator)

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/aerodrome/envs/c_envs")

file(GLOB ENV_FILES "src/simulator/envs/*.cpp")

# 2. 为每个 .cpp 文件生成单独的 .pyd 文件
foreach(ENV_FILE ${ENV_FILES})
    # 获取文件名（去除路径）
    get_filename_component(ENV_NAME ${ENV_FILE} NAME_WE)

    # 使用 pybind11_add_module 生成模块
    pybind11_add_module(${ENV_NAME} ${ENV_FILE})

    # 设置输出目录（可选，指定最终 .pyd 文件的输出位置）
    set_target_properties(${ENV_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )

    set_target_properties(${ENV_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    )

    add_custom_command(TARGET ${ENV_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${ENV_NAME}.lib
        COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${ENV_NAME}.exp
        COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_DIR}/${ENV_NAME}.pdb
    )


    # 安装模块到 Python 的 site-packages 目录
    # install(TARGETS ${ENV_NAME}
    #         LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/simulator)
endforeach()