#include<iostream>
#include<stdio.h>
#include<stdint.h>
#include<assert.h>
#include<cuda_runtime.h>


__global__ void warpDivergence(){
    int tid = threadIdx.x;
    int warp_id = tid/32;

    if(tid%2==0){
        printf("EVEN -- thread id: %d , warp id: %d  \n",tid,warp_id);
    }
    else{
        printf("ODD -- thread id: %d , warp id: %d  \n",tid,warp_id);
    }
}


int main(){
    dim3 blockspergrid(1);
    dim3 threadsperblock(64);
    warpDivergence<<<blockspergrid,threadsperblock>>>();
}

