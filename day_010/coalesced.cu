l#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define N 1024*1024
#define BLOCK_SIZE 256


__global__ void Coalesced_kernel(int *d_in,int *d_out){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid<N){
        d_out[tid] = d_in[tid];
    }
}

__global__ void Uncoalesced_kernel(int *d_in, int *d_out, int stride){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid*stride<N){
        d_out[tid] = d_in[tid*stride];
    }
}

void benchmark_kernel(int *d_in, int *d_out, int stride=1){
    cudaEvent_t start,stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if(stride==1){
        Coalesced_kernel<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in,d_out);
    }
    else{
        Uncoalesced_kernel<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in,d_out,stride);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time (stride = %d): %.4f ms \n",stride,time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


int main(){
    size_t size = N*(sizeof(int));

    int *h_in = (int *)malloc(size);

    for(int i=0;i<N;i++){
        h_in[i]=i;
    }

    int *d_in;
    int *d_out;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out,size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    benchmark_kernel(d_in,d_out);
    
    benchmark_kernel(d_in,d_out,32);

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);



    return 0;
}