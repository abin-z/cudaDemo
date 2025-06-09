# cudaDemo

**该仓库主要记录CUDA学习相关内容**

开发环境工具链（WSL2 + CUDA）说明:  

| 项目                                 | 推荐配置                                                   |
| ------------------------------------ | ---------------------------------------------------------- |
| **显卡**                             | NVIDIA GeForce RTX5070                                     |
| **WSL 版本**                         | **WSL 2**（必需，WSL 1 不支持 GPU 加速）                   |
| **Linux 发行版**                     | Ubuntu 24.04 或 Ubuntu 22.04                               |
| **NVIDIA GPU 驱动**                  | 安装 [WSL 专用驱动](https://developer.nvidia.com/cuda/wsl) |
| **CUDA Toolkit**                     | CUDA Toolkit 12.x（适配驱动）                              |
| **NVIDIA Container Toolkit（可选）** | 用于 GPU 加速 Docker 容器（适合进阶）                      |

[Win11安装WSL2 部署Ubuntu 迁移至非系统盘D盘](https://www.bilibili.com/video/BV1Yk7JzTEjH/?spm_id_from=333.337.search-card.all.click&vd_source=b406ed5db011e57f04f8df4e7af4a1f3)

WSL2 常用指令清单

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

