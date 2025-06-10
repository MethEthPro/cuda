#include<iostream>
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<cuda_runtime.h>
using namespace std;

#define TILE_WIDTH 2

// A is N1 x N2
// B is N2 x N3
// so as a result 
// C is N1 x N3
__global__ void rectangular_tiled_mat_mul(float *A, float *B, float *C,int N1, int N2, int N3){

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
    // as N2 is the common dimension 
    for(int phase = 0;phase < ceil((float)N2/TILE_WIDTH) ;phase++){

        // load tiles into shared memory 
        // ensure that the threads lie in the size of matrix
        // otherwise they will not be allowed to load anything
        // and that space in shared memory will be filled as 0
        if((i<N1) && ((TILE_WIDTH*phase+tx) < N2)){
            sh_A[ty][tx] = A[i*N2 + TILE_WIDTH*phase + tx];
        }
        else{
            sh_A[ty][tx]=0.0f;
        }

        if(((TILE_WIDTH*phase) < N2) && (j<N3)){
            sh_B[ty][tx] = B[(TILE_WIDTH*phase + ty)*N3+j];
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
    if(i<N1 && j<N3){
        C[i*N3+j] = value;
    }
    
}

int main(){
    int N1 = 3;
    int N2 = 4;
    int N3 = 3;


    size_t size = sizeof(float);
    
    float *h_A = (float *)malloc(size*N1*N2);
    float *h_B = (float *)malloc(size*N2*N3);
    float *h_C = (float *)malloc(size*N1*N3);

    for(int i=0;i<N1;i++){
        for(int j=0;j<N2;j++){
            h_A[i*N2+j]=j;
        }
    }

    for(int i=0;i<N2;i++){
        for(int j=0;j<N3;j++){
            h_B[i*N3+j]=j;
        }
    }

    float *d_A,*d_B,*d_C;

    cudaMalloc((void**)&d_A, size*N1*N2);
    cudaMalloc((void**)&d_B, size*N2*N3);
    cudaMalloc((void**)&d_C, size*N1*N3);

    cudaMemcpy(d_A,h_A,size*N1*N2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size*N2*N3,cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(2,2);
    dim3 threadsPerBlock(2,2);

    rectangular_tiled_mat_mul<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N1,N2,N3);

    cudaMemcpy(h_C,d_C,size*N1*N3, cudaMemcpyDeviceToHost);

     printf("\nMatrix C (Result):\n");
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {
            printf("%8.3f ", h_C[i * N1 + j]);
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