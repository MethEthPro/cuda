// matrix addition program
#include<iostream>
#include<assert.h>
#include<stdint.h>
#include<stdio.h>
#include<cuda_runtime.h>

// in cuda you cant pass matrix as the argument, only pointers can be passed
// so we need to pass the pointer to the first element of the matrix
// and we also flatten our matrix into an array
// so indexing is done like [i][j] = i*N + j

// and we also record the time taken by the kernel to execute
// we can use clock() function to get the time taken by the kernel


// we also learnt that why we used * with elapsed and not with A, B, C
// its because when we do *d_elpased or *h_elapsed we are accessing the value stored at the memory address they point to
// but in cased of d_A, d_B, d_C we dont use * as we use d_A[i] which is synctactic sugar for *(A+i) 

// the start and end time is recorded in the shared memory as they are accessed by all the threads in the block

__global__ void myKernel(float *A,float *B, float *C,clock_t *elapsed,int N){

    __shared__ clock_t start,end;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(tidx == 0 && tidy == 0){
        start = clock();
    }

    int idx = row*N + col;
    if(row < N && col < N){
        C[idx]= A[idx] + B[idx];
    }    

    __syncthreads();

    if(tidx == 0 && tidy == 0){
        end = clock();
        *elapsed = end - start;
        printf("clock cycles: %ld\n",*elapsed);
    }
    

}


// so we dont create a matrix anywhere in the code we think of it as a flattened array
// we also learnt to use cudadevice properties to get the clock rate of the device
// and with the clock rate we can calculate the time taken by the kernel to execute
// time taken = clock cycles / clock rate(KHz), so time is in milliseconds
// and we also learnt that the clock rate is in KHz so we need to multiply it by 1000 to get the time in milliseconds
// but it can overflow int so we use long long int


int main(){
    int N = 16;
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    clock_t *h_elapsed = (clock_t*)malloc(sizeof(clock_t));

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            h_A[i*N + j] = i+j;
            h_B[i*N + j] = i-j;
        }
    }

    float *d_A,*d_B,*d_C;
    clock_t *d_elapsed;
    

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_elapsed,sizeof(clock_t));

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int blockspergrid = 1;
    dim3 threadsperblock(N,N);

    myKernel<<<blockspergrid,threadsperblock>>>(d_A,d_B,d_C,d_elapsed,N);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_elapsed, d_elapsed, sizeof(clock_t),cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%0.1f",h_C[i*N+j]);
            printf("  ");
        }
        printf("\n");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Clock rate: %lld Hz\n", (long long int)prop.clockRate * 1000);
    float time_ms = ((float)(*h_elapsed)) / prop.clockRate;

    printf("time taken: %0.8f milli seconds \n",time_ms);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_elapsed);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_elapsed);

    return 0;
}