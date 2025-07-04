#include <stdio.h>
#include <stdlib.h>
#include<float.h>
#include <cuda_runtime.h>

// kernel to find minimum using parallel reduction
__global__ void minParallelReduction(float* A, float *B,int N) {
    extern __shared__ float sh[];

    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x * 2;  // Each block processes 2*blockDim.x elements

    // Each thread loads two elements from global to shared memory
    sh[tid] = (gid < N) ? A[gid] : FLT_MAX;
    sh[tid+blockDim.x] = (gid+blockDim.x<N)? A[gid+blockDim.x] : FLT_MAX;
    
    __syncthreads();



    // Perform reduction in shared memory
    for (int s = blockDim.x ; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] = fminf(sh[tid], sh[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        B[blockIdx.x] = sh[0];
    }
}

#define NUM_THREADS 256

int main() {
    int N = 1024*1024;
    int currentSize = N;
    size_t size = sizeof(float) * N;
    size_t shmem = sizeof(float) * NUM_THREADS * 2;

    float* h_A = (float*)malloc(size);

    // Initialize with decreasing values (to test min logic)
    for (int i = 0; i < N; i++) {
        h_A[i] = float(1024*1024*2 - i); // min should be 1024*1024*2-(1024*1024-1) = 1048577
    }

    float *d_A, *d_B;
    float *d_original;

    cudaMalloc((void**)&d_A, size);
    d_original = d_A;  // Keep reference to original allocation

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    while(currentSize > 1){

        int blocks = (currentSize + NUM_THREADS * 2 - 1) / (NUM_THREADS * 2);
        
        printf("Iteration: currentSize=%d, blocks=%d\n", currentSize, blocks);

        cudaMalloc((void**)&d_B, sizeof(float) * blocks);

        minParallelReduction<<<blocks, NUM_THREADS, shmem>>>(d_A, d_B, currentSize);
        cudaDeviceSynchronize();

        // Free previous input if it's not the original input
        if (d_A != d_original) {
            cudaFree(d_A);
        }


        // Prepare for next iteration
        d_A = d_B;
        currentSize = blocks;
    }

    float result;
    cudaMemcpy(&result, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Global minimum: %f\n", result);
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_original);
    free(h_A);

    return 0;
}