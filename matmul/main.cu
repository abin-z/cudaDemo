#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// 矩阵大小
const int N = 1024;  // 矩阵为 N x N

// CUDA 核函数：矩阵乘法 C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width) {
        float val = 0;
        for (int k = 0; k < width; ++k) {
            val += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = val;
    }
}

// CPU 矩阵乘法（朴素实现）
void matmul_cpu(const float* A, const float* B, float* C, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float val = 0;
            for (int k = 0; k < width; ++k) {
                val += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = val;
        }
    }
}

int main() {
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // 分配并初始化 host 内存
    std::vector<float> h_A(size, 1.0f);
    std::vector<float> h_B(size, 2.0f);
    std::vector<float> h_C_cpu(size, 0.0f);
    std::vector<float> h_C_gpu(size, 0.0f);

    // CPU 计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU matrix multiplication time: " << cpu_time << " ms\n";

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 拷贝输入数据到设备
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // 设置 CUDA kernel 参数
    dim3 block(16, 16);
    dim3 grid((N + block.x -1) / block.x, (N + block.y -1) / block.y);

    // CUDA 计算
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_event, stop_event);
    std::cout << "CUDA matrix multiplication time: " << gpu_time << " ms\n";

    // 拷贝结果回 host
    cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // 简单验证结果（前10个元素是否相等）
    bool match = true;
    for (int i = 0; i < 10; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            match = false;
            break;
        }
    }
    std::cout << "Result match: " << (match ? "PASS ✅" : "FAIL ❌") << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
