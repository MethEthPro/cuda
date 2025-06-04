#include<iostream>
#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<time.h>

__global__ void sq_mat_mul_naive(float* A, float *B, float *C, int N){
    // identifying the thread mapping
    // thread working on C[i][j]

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    // check for edge case
    if(i<N && j < N){
        
        float value = 0;
        for(int k=0;k<N;k++){
            value += A[i*N+k] * B[k*N+j];
        }

        C[i*N+j] = value;
    }
}

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

int main(){

    int N = 64;
    size_t size = N*N*sizeof(float);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // HOST ARRAY POINTERS MEMORY ALLOCATION
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // filling the data
    for(int i=0;i<N*N;i++){
        h_A[i]=(float)i;
        h_B[i]=2.0f * i;
    }

    // DEVICE ARRAY POINTERS
    float *d_A;
    float *d_B;
    float *d_C;


    // DEVICE MEMORY ALLOCATION
    cudaError_t err_A = cudaMalloc((void**) &d_A, size);
    CUDA_CHECK(err_A);

    cudaError_t err_B = cudaMalloc((void**) &d_B, size);
    CUDA_CHECK(err_B);

    cudaError_t err_C = cudaMalloc((void**) &d_C, size);
    CUDA_CHECK(err_C);

    // data transfer from hosst to device

    cudaError_t err_A_ = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);

    cudaError_t err_B_ = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);

    // grid specs(x,y,z)
    dim3 dim_block(4,3,1);
    dim3 dim_grid((N+dim_block.x-1)/dim_block.x, (N+dim_block.y-1)/dim_block.y, 1);

    // kernel execution 

    cudaEventRecord(start, 0);
    sq_mat_mul_naive<<<dim_grid,dim_block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time,start,stop);
    printf("Time taken: %.2f ms\n", time);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // data transfer from device to host

    cudaError_t err_C_ = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%6.1f ", h_C[i * N + j]);

            printf(" ");
            
        }
        printf("\n");
    }
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}