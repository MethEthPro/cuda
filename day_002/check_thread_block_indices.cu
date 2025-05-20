#include<iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void Checking(int *threadIds, int *blockIds, int *values, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<N){
        threadIds[idx] = threadIdx.x;
        blockIds[idx] =  blockIdx.x;
        values[idx] = threadIdx.x + blockIdx.x * blockDim.x;
    }

}


int main(){
    int N = 48;
    size_t size = N * sizeof(int);

    int *h_threadIds = (int*)malloc(size);
    int *h_blockIds = (int*)malloc(size);
    int *h_values = (int*)malloc(size);

    int *d_threadIds, *d_blockIds, *d_values;

    cudaMalloc(&d_threadIds, size);
    cudaMalloc(&d_blockIds, size);
    cudaMalloc(&d_values, size);

    int threadsperblock = 8;
    int blockspergird = (N+threadsperblock-1)/threadsperblock;

    Checking<<<blockspergird, threadsperblock>>>(d_threadIds, d_blockIds, d_values, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_threadIds,d_threadIds,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockIds,d_blockIds,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values,d_values,size,cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        cout<<"Index: "<<i<<", threadIdx: "<<h_threadIds[i]<<", blockIdx: "<<h_blockIds[i]<<", Values: "<<h_values[i]<<endl;
    }

    free(h_threadIds);
    free(h_blockIds);
    free(h_values);

    cudaFree(d_threadIds);
    cudaFree(d_blockIds);
    cudaFree(d_values);






    return 0;
}