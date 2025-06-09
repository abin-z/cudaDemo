# cudaDemo

**该仓库主要记录CUDA学习相关内容**

开发环境工具链（WSL2 + CUDA）说明:  

| 项目                                 | 推荐配置                                                   |
| ------------------------------------ | ---------------------------------------------------------- |
| **显卡**                             | NVIDIA GeForce RTX5070                                     |
| **WSL 版本**                         | **WSL 2**（必需，WSL 1 不支持 GPU 加速）                   |
| **Linux 发行版**                     | Ubuntu 20.04 或 Ubuntu 22.04                               |
| **NVIDIA GPU 驱动**                  | 安装 [WSL 专用驱动](https://developer.nvidia.com/cuda/wsl) |
| **CUDA Toolkit**                     | CUDA Toolkit 12.x（适配驱动）                              |
| **NVIDIA Container Toolkit（可选）** | 用于 GPU 加速 Docker 容器（适合进阶）                      |
