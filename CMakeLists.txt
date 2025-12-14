cmake_minimum_required(VERSION 3.18)

project(cudaMixtureProject LANGUAGES CXX CUDA)

# ===== 基本设置 =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 编译设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成位置无关代码（.so 必须）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== 构建共享库 =====
add_library(cudaMixture SHARED
    cudaMixture.cu
)

# 头文件可见性
target_include_directories(cudaMixture
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include> # 安装的头文件搜索路径
)

# CUDA 编译属性
set_target_properties(cudaMixture PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 如果你用到了 device code 的链接（__device__ 全局变量等）
# set_target_properties(cudaMixture PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ===== 可执行文件（可选，用来测试）=====
add_executable(main
    main.cpp
)

target_link_libraries(main
    PRIVATE cudaMixture
)



# ================= 安装规则 =================

include(GNUInstallDirs)

# 安装共享库 libcudaMixture.so
install(
    TARGETS cudaMixture
    EXPORT cudaMixtureTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   # lib/
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# 安装头文件
install(
    FILES cudaMixture.H
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# （可选）导出 CMake target，方便 find_package
install(
    EXPORT cudaMixtureTargets
    FILE cudaMixtureTargets.cmake
    NAMESPACE cudaMixture::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaMixture
)


# cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cudaMixture