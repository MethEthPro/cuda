#include<iostream>
#include<cuda_runtime.h>
using namespace std;

// my kernel
__global__ void MyFunc(float *A, float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    // declaring the size of the arrays
    int N = 1000;
    size_t size = N * sizeof(float);

    // creating arrays on host(cpu) and thus alloting the space as well
    // in c++
    // float *h_A = new float[N];
    // float *h_B = new float[N];
    // float *h_C = new float[N];

    // in c
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // filling the arrays with values
    for(int i=0;i<N;i++){
        h_A[i] = i;
        h_B[i] = 2*i;
    }

    // allocating memory on devic(gpu)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    // copying arrays from host do device
    cudaMemcpy(d_A ,h_A ,size ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B ,h_B ,size ,cudaMemcpyHostToDevice);

    // fixing threadsperblock to 256
    // and selecting value of blockspergrid such that its divisible
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    // calling the kernel
    MyFunc<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C ,N);

    // copying the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout<<"A + B = C"<<endl;
    
    // pritning the result
    for(int i=0;i<N;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<h_C[i]<<endl;
    }

    // deallocation in c++
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;

    // in c
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}