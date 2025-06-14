#include<stdio.h>
#include<stdint.h>
#include<cuda_runtime.h>

#define N 1024

__global__ void mykernel(int *A,int *sum){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < N){
        atomicAdd(sum,A[tid]);
    }
}

int main(){
    size_t size = N*sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(sizeof(int));


    for(int i=0;i<N;i++){
        h_A[i]=i;
    }
    int *d_A, *d_B;

    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,sizeof(int));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMemset(d_B, 0, sizeof(int));


    mykernel<<<1,N>>>(d_A,d_B);

    cudaMemcpy(h_B, d_B, sizeof(int), cudaMemcpyDeviceToHost);

    printf("the sum is: %d \n", *h_B);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}