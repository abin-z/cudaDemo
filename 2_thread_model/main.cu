#include <stdio.h>

// CUDA 核函数，运行在 GPU 上，可以从 CPU 调用
__global__ void thread_model(int x)
{
  const int gx = gridDim.x;   // 网格大小
  const int bx = blockDim.x;  // 块大小

  const int bid = blockIdx.x;       // 块索引
  const int tid = threadIdx.x;      // 线程索引
  const int gtid = bid * bx + tid;  // 全局线程索引
  printf("Hello from GPU! Thread ID: %d, Block ID: %d, Global Thread ID: %d\n", tid, bid, gtid);
  printf("Hello from GPU! Grid size: %d, Block size: %d\n", gx, bx);
}

int main(int argc, char** argv)
{
  // 启动 CUDA 核函数，指定网格和块的大小
  thread_model<<<2, 3>>>(0);
  cudaDeviceSynchronize();  // 必须调用, 等待 GPU 执行完毕, 否则
                            // 程序可能会提前结束，导致输出不完整
  return 0;
}