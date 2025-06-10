#include<iostream>
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<cuda_runtime.h>
using namespace std;

#define TILE_WIDTH 2

__global__ void sq_tiled_mat_mul(float *A, float *B, float *C,int N){

    // details regarding this thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // working on C[i][j]
    int i = ty + by*blockDim.y;
    int j = tx + bx*blockDim.x;

    // allocating shared memory 
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    // splitting data into smaller tiles
    for(int phase = 0;phase<N/TILE_WIDTH;phase++){

        // load tiles into shared memory 
        // ensure that the threads lie in the size of matrix
        // otherwise they will not be allowed to load anything
        // and that space in shared memory will be filled as 0
        if((i<N) && ((TILE_WIDTH*phase+tx) < N)){
            sh_A[ty][tx] = A[i*N + TILE_WIDTH*phase + tx];
        }
        else{
            sh_A[ty][tx]=0.0f;
        }

        if(((TILE_WIDTH*phase) < N) && (j<N)){
            sh_B[ty][tx] = B[(TILE_WIDTH*phase + ty)*N+j];
        }
        else{
            sh_B[ty][tx]=0.0f;
        }
        
        

        __syncthreads();

        // dot product with data in shared memory 
        for(int k=0;k<TILE_WIDTH;k++){
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }
    // assigining calculated value by checking location
    i(i<N && j<N){
        C[i*N+j] = value;
    }
    
}

int main(){
    int N = 4;

    size_t size = N*N*sizeof(float);
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            h_A[i*N+j]=i;
            h_B[i*N+j]=j;
        }
    }

    float *d_A,*d_B,*d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(2,2);
    dim3 threadsPerBlock(2,2);

    sq_tiled_mat_mul<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C,d_C,size, cudaMemcpyDeviceToHost);

     printf("\nMatrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.3f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}