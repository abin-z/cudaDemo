# cudaDemo

**该仓库主要记录CUDA学习相关内容**

## CUDA简介:

CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的并行计算平台和编程模型，允许你使用 C、C++ 或 Fortran 这样的语言，在 GPU 上进行通用计算（GPGPU）。其核心思想是利用 GPU 的强大并行处理能力来加速计算密集型任务，比如图像处理、深度学习、科学计算等。

CUDA 主要由以下几个部分组成：

1. **CUDA C/C++ 语言扩展**：在普通 C/C++ 上增加了一些关键词（如 `__global__`、`__device__`、`__shared__`）来定义 GPU 上的函数与变量。
2. **内存管理**：提供了设备内存（global、shared、constant、local 等）和主机内存之间的数据管理 API。
3. **执行模型**：通过 **线程块（block）** 和 **网格（grid）** 的组织方式，管理大量线程的并发执行。
4. **库支持**：如 cuBLAS（线性代数）、cuFFT（傅里叶变换）、Thrust（STL 风格）、cuDNN（深度学习）等。



开发环境工具链（WSL2 + CUDA）说明:  

| 项目                | 配置                                                   |
| ------------------- | ---------------------------------------------------------- |
| **显卡**            | NVIDIA GeForce RTX5070                                     |
| **WSL 版本**        | **WSL 2**（必需，WSL 1 不支持 GPU 加速）                   |
| **Linux 发行版**    | Ubuntu 24.04 或 Ubuntu 22.04                               |
| **NVIDIA GPU 驱动** | 安装 [WSL 专用驱动](https://developer.nvidia.com/cuda/wsl) |
| **CUDA Toolkit**    | CUDA Toolkit 12.9（适配驱动）                              |

[Win11安装WSL2 部署Ubuntu 迁移至非系统盘D盘](https://www.bilibili.com/video/BV1Yk7JzTEjH/?spm_id_from=333.337.search-card.all.click&vd_source=b406ed5db011e57f04f8df4e7af4a1f3)

WSL2 常用指令清单, 更多指令请前往微软[wsl官网](https://learn.microsoft.com/zh-cn/windows/wsl/)

```bash
# 查看 WSL 版本（需新版 Windows 才支持）
wsl --version

# 查看所有已安装的 Linux 子系统及其运行状态
wsl -l -v           # 或 wsl --list --verbose

# 查看所有可用的在线 Linux 发行版（可用于安装）
wsl --list --online

# 查看默认子系统和已安装的子系统列表
wsl --list          # 或 wsl -l

# 停掉所有正在运行的子系统
wsl --shutdown

# 导出某个子系统为 .tar 文件（备份）
wsl --export Ubuntu-24.04 D:\WSL\ubuntu-2404.tar

# 从 .tar 备份导入一个新子系统到指定目录
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu24 D:\WSL\ubuntu-2404.tar

# 注销并删除某个子系统（不可恢复，慎用）
wsl --unregister Ubuntu-24.04

# 设置默认子系统
wsl --set-default Ubuntu-24.04

# 设置某个子系统使用 WSL2（默认新系统应使用 WSL2）
wsl --set-version Ubuntu-24.04 2

# 设置以后安装的子系统默认使用 WSL2
wsl --set-default-version 2

# 启动默认子系统
wsl

# 启动指定子系统
wsl -d Ubuntu-24.04

# 启动指定子系统并执行命令
wsl -d Ubuntu-24.04 -- lsb_release -a
```

### 核函数(Kernel function)

在 CUDA 中，**核函数（Kernel Function）** 是一种特殊的函数，它在 **GPU 上运行**，可以由 CPU 发起大量并发线程来执行

1. 核函数在GPU上进行并行执行

2. 限定词为 `__global__` 修饰

3. 返回值必须是`void`

4. 形式:

   ```c++
   // 使用 __global__ 关键字声明
   // CUDA 核函数, 运行在 GPU 上, 可以从 CPU 调用
   // 需要使用 <<<...>>> 语法来指定执行配置
   __global__ void kernel_function(argument arg)
   {
     printf("Hello from GPU!\n");  // 不能使用 std::cout, 因为它在 GPU 上运行
   }
   
   void __global__ kernel_function(argument arg)
   {
     printf("Hello from GPU!\n");  // 不能使用 std::cout, 因为它在 GPU 上运行
   }
   ```

核函数注意事项: 

- 核函数只能访问GPU内存
- 核函数不能使用可变长参数
- 核函数不能使用静态变量
- 核函数不能使用函数指针
- 核函数不能使用iostream的内容
- 核函数具有异步性, CPU代码需要调用同步函数





###  CUDA 线程模型

多个 Grid（通常只有一个）        👇
Grid（网格）中有多个 Block       👇
Block（线程块）中有多个 Thread   👇
每个 Thread 执行核函数





### 线程 & 块 相关内建变量

| 变量名      | 类型   | 含义                               | 举例          |
| ----------- | ------ | ---------------------------------- | ------------- |
| `threadIdx` | `dim3` | 当前线程在 **所在 block 中的索引** | `threadIdx.x` |
| `blockIdx`  | `dim3` | 当前 block 在 **grid 中的索引**    | `blockIdx.x`  |
| `blockDim`  | `dim3` | 每个 block 中线程的数量            | `blockDim.x`  |
| `gridDim`   | `dim3` | grid 中 block 的数量               | `gridDim.x`   |

#### 如何计算全局线程索引？

```cpp
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

适用于 1D 情况。对于 2D/3D 的 block，可以这样：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 内存地址空间修饰符（用于函数或变量声明）

| 关键词            | 含义                                   |
| ----------------- | -------------------------------------- |
| `__global__`      | 核函数：主机调用，设备执行             |
| `__device__`      | 设备函数/变量：只能被 GPU 调用/访问    |
| `__host__`        | 主机函数：只能在 CPU 上调用（默认）    |
| `__shared__`      | 声明共享内存，线程块内共享             |
| `__constant__`    | 声明只读的常量内存（GPU 所有线程共享） |
| `__restrict__`    | 编译优化用，声明指针无别名             |
| `__syncthreads()` | 所有 block 内线程同步函数（很常用）    |



## CUDA 常用内存管理函数分类总览

------

#### 1. **设备内存分配与释放**

| 函数原型                                            | 功能说明                          |
| --------------------------------------------------- | --------------------------------- |
| `cudaMalloc(void** devPtr, size_t size)`            | 在设备（GPU）上分配内存           |
| `cudaFree(void* devPtr)`                            | 释放设备上分配的内存              |
| `cudaMallocManaged(void** ptr, size_t size)`        | 分配“统一内存”（主机和设备共享）  |
| `cudaMemset(void* devPtr, int value, size_t count)` | 在设备内存中设置值（类似 memset） |

####  2. **主机与设备内存拷贝**

| 函数原型                                                     | 功能说明               |
| ------------------------------------------------------------ | ---------------------- |
| `cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)` | 主机 ↔ 设备 内存拷贝   |
| `cudaMemcpyAsync(...)`                                       | 异步版本，适用于流操作 |
| `cudaMemcpyKind` 的常见值：                                  |                        |
| - `cudaMemcpyHostToDevice`                                   | 主机 → 设备            |
| - `cudaMemcpyDeviceToHost`                                   | 设备 → 主机            |
| - `cudaMemcpyDeviceToDevice`                                 | 设备内存之间复制       |
| - `cudaMemcpyHostToHost`                                     | 主机内存之间复制       |

####  3. **主机端可固定内存（页锁定内存）**

| 函数原型                                                     | 功能说明                         |
| ------------------------------------------------------------ | -------------------------------- |
| `cudaHostAlloc(void** ptr, size_t size, unsigned int flags)` | 分配页锁定主机内存，加快数据传输 |
| `cudaFreeHost(void* ptr)`                                    | 释放页锁定主机内存               |

**用途：** 适用于高性能场景中主机和设备之间频繁传输数据。

#### 4. **内存信息查询函数**

| 函数                                          | 功能说明                  |
| --------------------------------------------- | ------------------------- |
| `cudaMemGetInfo(size_t* free, size_t* total)` | 获取当前 GPU 剩余和总内存 |

#### 5. **错误检查与调试**

| 函数                              | 功能说明                     |
| --------------------------------- | ---------------------------- |
| `cudaGetLastError()`              | 获取上一次 CUDA API 的错误码 |
| `cudaPeekAtLastError()`           | 获取但不清除最后的错误       |
| `cudaGetErrorString(cudaError_t)` | 将错误码转成字符串           |
