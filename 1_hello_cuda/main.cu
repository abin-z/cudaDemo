// main.cu
#include <iostream>

// 使用 __global__ 关键字声明
// CUDA 核函数, 运行在 GPU 上, 可以从 CPU 调用
// 需要使用 <<<...>>> 语法来指定执行配置
__global__ void hello_cuda()
{
  printf("Hello from GPU!\n");  // 不能使用 std::cout, 因为它在 GPU 上运行
}

int main(int argc, char** argv)
{
  hello_cuda<<<1, 2>>>();
  cudaDeviceSynchronize();  // 同步函数, 等待 GPU 完成
  std::cout << "Hello from CPU!" << std::endl;
  return 0;
}
