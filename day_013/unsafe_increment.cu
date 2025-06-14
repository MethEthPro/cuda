#include<stdio.h>
#include<stdint.h>
#include<cuda_runtime.h>

__global__ void unsafe_increment(int *counter){
    (*counter)++;
}

int main(){

    int N = 1024; // 1024 threads
    size_t size = sizeof(int);

    int *d_a;
    int h_a = 0;
 
    cudaMalloc((void**)&d_a,size);

    for (int i = 0; i < 10; i++) {
        h_a = 0;
        cudaMemcpy(d_a, &h_a, size, cudaMemcpyHostToDevice);
        unsafe_increment<<<4, 256>>>(d_a); // 4 blocks of 256 threads = 1024 total
        cudaDeviceSynchronize();
        cudaMemcpy(&h_a, d_a, size, cudaMemcpyDeviceToHost);
        printf("Run %d: Final counter = %d\n", i, h_a);
    }

    cudaFree(d_a);
    return 0;
}