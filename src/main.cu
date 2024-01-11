#include <stdio.h>

__global__ void hello()
{
    printf("(%d, %d): Hello CUDA\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
