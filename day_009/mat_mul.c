#include<stdio.h>
#include<stdlib.h>
#include<time.h>


void mat_mul_cpu(float *A, float *B, float *C, int N){
    
    for(int i=0;i<N;i++){
        
        for(int j=0;j<N;j++){
            float val = 0;
            for(int k=0;k<N;k++){
                val += A[i*N+k] * B[k*N+j];
            }
            // assigning calculated value
            C[i*N + j] = val;
        }

        
    }

}


int main(){

    clock_t start,end;
    double cpu_time_used;

    int N = 64;
    float *A =  (float*)malloc(N*N*sizeof(float));
    float *B =  (float*)malloc(N*N*sizeof(float));
    float *C =  (float*)malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        A[i]=(float)i;
        B[i]=2.0f * i;
    }

    start = clock();
    mat_mul_cpu(A,B,C,N);
    end = clock();

    cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;

    printf("Matrix multiplication of %d size matrix took %f seconds\n", N,cpu_time_used);

    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         printf("%f",C[i*N+j]);
    //     }
    //     printf("\n");
    // }

    free(A);
    free(B);        
    free(C);
    return 0;

}