#include<iostream>
#include<assert.h>
#include<stdio.h>
#include<stdint.h>
#include<cuda_runtime.h>
using namespace std;

// This kernel adds two vectors using shared memory

__global__ void VecAdd(float *a, float *b,float *c, int n){
    extern __shared__ float shared[];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int local_id = threadIdx.x;

    if(idx<n){
        shared[local_id] = a[idx];
        shared[local_id + blockDim.x] = b[idx];

        __syncthreads();

        c[idx] = shared[local_id] + shared[local_id + blockDim.x];
    }
}


int main(){
    int N=10;
    size_t size = sizeof(float) * N;

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for(int i=0;i<N;i++){
        h_a[i] = i;
        h_b[i] = i*i;
    }
    float *d_a,*d_b,*d_c;

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);


    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    int threadsperblock = 256;
    int blockspergrid = (N+threadsperblock-1)/threadsperblock;

    size_t sharedMemorySize = sizeof(float) * threadsperblock * 2;

    VecAdd<<<blockspergrid,threadsperblock,sharedMemorySize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        cout<<h_c[i]<<endl;
    }
    cout<<endl;

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}