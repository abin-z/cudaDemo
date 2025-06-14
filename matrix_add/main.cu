#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

const int N = 1024;  // 矩阵维度 N x N

__global__ void matrix_add(const float* A, const float* B, float* C, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width)
  {
    int idx = row * width + col;
    C[idx] = A[idx] + B[idx];
  }
}

int main()
{
  const int size = N * N;
  const size_t bytes = size * sizeof(float);

  // 分配主机内存（使用 pinned 内存提高传输效率）
  float* h_A;
  float* h_B;
  float* h_C;
  cudaMallocHost(&h_A, bytes);
  cudaMallocHost(&h_B, bytes);
  cudaMallocHost(&h_C, bytes);

  // 初始化输入矩阵
  for (int i = 0; i < size; ++i)
  {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // 分配设备内存
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  // 拷贝输入数据到设备
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  // 设置 CUDA kernel 启动参数
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // ===== 性能测试开始 =====
  auto start = std::chrono::high_resolution_clock::now();

  matrix_add<<<grid, block>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();  // 确保 kernel 执行完成

  auto end = std::chrono::high_resolution_clock::now();
  // ===== 性能测试结束 =====

  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "CUDA Kernel Execution Time: " << elapsed_ms << " ms" << std::endl;

  // 拷贝结果回主机
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  // 简单验证结果
  bool success = true;
  for (int i = 0; i < 10; ++i)
  {
    if (h_C[i] != 3.0f)
    {
      success = false;
      break;
    }
  }

  std::cout << "Matrix Add Result: " << (success ? "PASS ✅" : "FAIL ❌") << std::endl;

  // 清理资源
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
