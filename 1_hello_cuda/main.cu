// main.cu
#include <iostream>

__global__ void hello_cuda() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_cuda<<<2, 2>>>();
    cudaDeviceSynchronize();
    std::cout << "Hello from CPU!" << std::endl;
    return 0;
}
