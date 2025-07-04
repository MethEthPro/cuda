#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// kernel to find minimum using parallel reduction
__global__ void minParallelReduction(float* A, float *B) {
    extern __shared__ float sh[];

    int tid = threadIdx.x;
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread loads one element from global to shared memory
    sh[tid] = A[tid];
    sh[tid+blockDim.x] = A[tid+blockDim.x];
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

#define NUM_BLOCKS 1
#define NUM_THREADS 256

int main() {
    int N = 512; // 512 elements
    size_t size = sizeof(float) * N;

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(sizeof(float) * NUM_BLOCKS);

    // Initialize with decreasing values (to test min logic)
    for (int i = 0; i < N; i++) {
        h_A[i] = float(512 - i); // min should be 1
    }

    float *d_A, *d_B;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, sizeof(float) * NUM_BLOCKS);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    minParallelReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * NUM_THREADS*2>>>(d_A, d_B);

    cudaMemcpy(h_B, d_B, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BLOCKS; i++) {
        printf("Minimum from block %d: %f\n", i, h_B[i]);
    }

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
