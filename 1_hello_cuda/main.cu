// main.cu
#include <iostream>

// CUDA 核函数, 运行在 GPU 上
__global__ void hello_cuda()
{
  printf("Hello from GPU!\n");
}

int main()
{
  hello_cuda<<<1, 2>>>();
  cudaDeviceSynchronize();
  std::cout << "Hello from CPU!" << std::endl;
  return 0;
}
