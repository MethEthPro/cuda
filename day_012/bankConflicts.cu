#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 32


__global__ void myKernel(float *A, float *B, int stride) {
    __shared__ float sh[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load data from global to shared memory
    sh[tid] = A[tid];

    __syncthreads();  // Make sure all threads have written

    float val = 0.0f;

    


    // Amplify memory access to exaggerate timing differences
    for (int i = 0; i < 10000; ++i) {
        int index = (tid * stride) % BLOCK_SIZE;

        val += sh[index];
    }

    // Write result to global memory (to avoid compiler optimization)
    B[tid] = val;
}


void Launching(float *A, float *B, int stride){
    cudaEvent_t start,stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    myKernel<<<GRID_SIZE,BLOCK_SIZE>>>(A,B,stride);
    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time,start,stop);
    printf("Stride: %2d → Time taken: %0.4f ms\n", stride, time);

    

}

int main(){
    int N = BLOCK_SIZE*GRID_SIZE ;
    
    size_t size = N*sizeof(float);

    float *h_A;
    cudaMallocHost(&h_A,size);

    for(int i=0;i<N;i++){
        h_A[i]=i*i;
    }

    float *d_A, *d_B;

    cudaMalloc((void**) &d_A, size);
    cudaMalloc((void**) &d_B, size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);

    for(int j=1;j<=32;j*=2){
        Launching(d_A,d_B,j);
    }

    cudaFreeHost(h_A);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}