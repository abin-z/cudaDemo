# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.20)

# 如果没有指定构建类型，默认设置为 Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置 C++ 标准为 C++17，并强制要求支持 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求支持指定的 C++ 标准
set(CMAKE_CXX_EXTENSIONS OFF)       # 禁用编译器特定扩展，确保代码更具可移植性

# 定义项目名称及支持的语言（注意这里添加了 CUDA）
# 这样 CMake 会自动启用对 .cu 文件的识别和 nvcc 的调用，
# 不需要手动指定编译器或修改任何规则
project(cudaDemo LANGUAGES C CXX CUDA)

# 如果 .cu 文件存在，CMake 会根据扩展名自动使用 nvcc 编译器，
# 并且链接阶段也会正确处理 CUDA 运行时库。
# 只要文件扩展名是 .cu，并在 project() 启用了 CUDA，
# 就不需要做任何额外设置，CMake 会自动配置 nvcc。

# 设置输出目录
# 静态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build_output/lib)
# 动态库输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build_output/bin)
# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build_output/bin)

# # 仅在 Linux/macOS 上启用 pthread
# if (UNIX)
#     find_package(Threads REQUIRED)
#     set(THREADS_PREFER_PTHREAD_FLAG ON)
#     link_libraries(Threads::Threads)
# endif()

# 添加子目录
# 每个子目录对应一个模块
# add_subdirectory(external/fmt)  # 添加 fmt 库
add_subdirectory(1_hello)       # 示例 1：Hello World 模块
add_subdirectory(1_hello_cuda)  # 示例 2：Hello CUDA 模块（包含 .cu 文件，自动由 nvcc 编译）
