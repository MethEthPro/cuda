#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Atomic add and sub
__global__ void addsub(int* x, int* y) {
    atomicAdd(x, threadIdx.x);
    atomicSub(y, threadIdx.x);
}

__global__ void exchange(int* x) {
    atomicExch(x, threadIdx.x);
}

__global__ void minmax(int* x, int* y) {
    atomicMin(x, threadIdx.x);
    atomicMax(y, threadIdx.x);
}

__global__ void incDec(unsigned int* x, unsigned int* y) {
    unsigned int limit = 5;

    atomicAdd(x, threadIdx.x);     // Add tid to x
    atomicSub(y, threadIdx.x);     // Subtract tid from y

    atomicInc(x, limit);           // Wraps to 0 if x >= limit
    atomicDec(y, limit);           // Wraps to limit if y == 0 or y > limit
}

__global__ void CAS(int* x) {
    int expected = 10;
    int new_val = 77;
    atomicCAS(x, expected, new_val);  // x becomes 77 only if it was 10
}

int main() {
    // Host variables
    int *h_A = (int*)malloc(sizeof(int));
    int *h_B = (int*)malloc(sizeof(int));
    int *h_C = (int*)malloc(sizeof(int));
    int *h_D = (int*)malloc(sizeof(int));
    int *h_E = (int*)malloc(sizeof(int));
    unsigned int *h_F = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int *h_G = (unsigned int*)malloc(sizeof(unsigned int));
    int *h_H = (int*)malloc(sizeof(int));

    // Device variables
    int *d_A, *d_B, *d_C, *d_D, *d_E, *d_H;
    unsigned int *d_F, *d_G;

    cudaMalloc(&d_A, sizeof(int));
    cudaMalloc(&d_B, sizeof(int));
    cudaMalloc(&d_C, sizeof(int));
    cudaMalloc(&d_D, sizeof(int));
    cudaMalloc(&d_E, sizeof(int));
    cudaMalloc(&d_F, sizeof(unsigned int));
    cudaMalloc(&d_G, sizeof(unsigned int));
    cudaMalloc(&d_H, sizeof(int));

    // Initialize values
    int b_init = 32640;
    int d_init = 10;
    int e_init = 10;
    unsigned int g_init = 3;
    int h_init = 10;

    cudaMemset(d_A, 0, sizeof(int));  // x for add
    cudaMemcpy(d_B, &b_init, sizeof(int), cudaMemcpyHostToDevice); // y for sub
    cudaMemset(d_C, 0, sizeof(int));  // result for exchange
    cudaMemcpy(d_D, &d_init, sizeof(int), cudaMemcpyHostToDevice); // min
    cudaMemcpy(d_E, &e_init, sizeof(int), cudaMemcpyHostToDevice); // max
    cudaMemset(d_F, 0, sizeof(unsigned int)); // x for inc
    cudaMemcpy(d_G, &g_init, sizeof(unsigned int), cudaMemcpyHostToDevice); // y for dec
    cudaMemcpy(d_H, &h_init, sizeof(int), cudaMemcpyHostToDevice); // value for CAS

    // Launch kernels
    addsub<<<1, 256>>>(d_A, d_B);
    exchange<<<1, 256>>>(d_C);
    minmax<<<1, 256>>>(d_D, d_E);
    incDec<<<1, 7>>>(d_F, d_G);
    CAS<<<1, 1>>>(d_H);

    // Copy back to host
    cudaMemcpy(h_A, d_A, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E, d_E, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F, d_F, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_G, d_G, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_H, d_H, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Sum using atomicAdd: %d\n", *h_A);             // Expect: 0+1+...+255 = 32640
    printf("Sub using atomicSub: %d\n", *h_B);             // 32640 - (0+1+...+255) = 0
    printf("Last threadIdx in atomicExch: %d\n", *h_C);    // Should be 255
    printf("Min from atomicMin: %d\n", *h_D);              // Should be 0
    printf("Max from atomicMax: %d\n", *h_E);              // Should be 255
    printf("atomicAdd + atomicInc result: %u\n", *h_F);    // Wraps to 0 if >= 5
    printf("atomicSub + atomicDec result: %u\n", *h_G);    // Wraps from 0 or > 5 to 5
    printf("Result of atomicCAS (expected 10 -> 77): %d\n", *h_H); // Should be 77

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_E); free(h_F); free(h_G); free(h_H);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
    cudaFree(d_F); cudaFree(d_G); cudaFree(d_H);

    return 0;
}
