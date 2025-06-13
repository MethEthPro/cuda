#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(){
    int N = 10 * 1024 * 1024; 
    size_t size = N * sizeof(float);

    float *h_A;
    cudaMallocHost(&h_A,size);    // Host memory
    float *d_A;

    cudaMalloc((void**)&d_A, size);           // Device memory

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);


    cudaEventElapsedTime(&time, start, stop);

    float bandwidth = (float)size / (time * 1e6);  // Convert to GB/s
    printf("Host to Device bandwidth: %.2f GB/s\n", bandwidth);

    // Cleanup
    cudaFreeHost(h_A);
    cudaFree(d_A);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
