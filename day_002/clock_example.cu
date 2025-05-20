#include<assert.h>
#include<stdint.h>
#include<stdio.h>
#include<cuda_runtime.h>


__global__ void Myfunc(float *input, float *output, clock_t *timer){
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // start clock
    if(tid==0){
        timer[bid] = clock();
    }

    // copying data from global memory to shared memory 
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // parallel reduction
    for(int i=blockDim.x;i>0;i /= 2){
        __syncthreads();
        if(tid<i){
            float f0 = shared[tid];
            float f1 = shared[tid+i];

            shared[tid] = f1>f0? f0:f1;
        }
    }

    // storing the ouput
    if(tid==0){
        output[bid] = shared[0];
    }

    __syncthreads();

    // stop clock
    if(tid==0){
        timer[bid+gridDim.x] = clock();
    }


    
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(){
    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for(int i=0;i<NUM_THREADS*2;i++){
        input[i] = (float)i;
    }

    cudaMalloc(&dinput, sizeof(float) * NUM_THREADS*2);
    cudaMalloc(&doutput, sizeof(float)*NUM_BLOCKS);
    cudaMalloc(&dtimer, sizeof(clock_t)*NUM_BLOCKS*2);


    cudaMemcpy(dinput ,input ,sizeof(float)*NUM_THREADS * 2, cudaMemcpyHostToDevice);

    Myfunc<<<NUM_BLOCKS,NUM_THREADS,sizeof(float)*NUM_THREADS*2>>>(dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t)*NUM_BLOCKS*2, cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double Elapsedclocks = 0;
    for(int i=0;i<NUM_BLOCKS;i++){
        Elapsedclocks += (long double)(timer[i+NUM_BLOCKS] - timer[i]);
    }

    long double avgElapsedclocks = Elapsedclocks/NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedclocks);

    return 0;


}