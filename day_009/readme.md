definitely watch this video 

https://youtu.be/oQT7IC0x254?si=J5CRHd8tTTRgoIA8

mat mul is slow on cpu as there is only a single thread going sequentially 

whereas on gpu we have threads for each element of C

we can see which thread will be responsible for which element of C

C[i][j] or C[i*N + j], is responsible by thread with 

i = threadIdx.y + (blockDim.y * blockIdx.y)

similarly , j = threadIdx.x + (blockDim.x * blockIdx.x)

C[i][j] will be formed by i th row of A and j th col of B

so we can traverse ith row of A using A[i*N + k] , 

and we can traverse jth col of B using B[k*N + j]

we also implemented a cuda check function to catch any error while copying data or allocating memory on device