#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>



__global__ void parallel_atomic_red(float* A,float* B, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid<N){
        atomicAdd(B,A[tid]);
    }

}



int main(){

    int N =1024*1024;
    size_t size = sizeof(float) * N;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* h_A = (float*)malloc(size);
    float h_B = 0;

    for(int i=0;i<N;i++){
        h_A[i] = 1.0f;
    }

    float *d_A,*d_B;

    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,sizeof(float));

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemset(d_B, 0, sizeof(float));


    int threadsperblock = 256;
    int blockspergrid = (N+threadsperblock-1)/threadsperblock;

    cudaEventRecord(start);
    parallel_atomic_red<<<blockspergrid,threadsperblock>>>(d_A,d_B,N);
    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time = %f ms\n",milliseconds);

    cudaMemcpy(&h_B,d_B,sizeof(float),cudaMemcpyDeviceToHost);


    printf("Reduced sum = %f\n", h_B);

    free(h_A);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);





    return 0;
}