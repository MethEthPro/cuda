TODAY WE LEARN TO FIND THE MIN USING PARALLEL REDUCTION

first we did it for N=512 elmenets , then 
we extended it to any N



in the naive_min.cu file

we have 512 elements 
so an input array of size 512

we have 1 block and 256 threads per block

we have shared memory of size 512 floats

so sh[0] = A[0] and sh[511] = A[511]

this can be done by 
sh[tid] = A[tid]

sh[tid+blockDim.x] = A[tid+blocDim.x]

as blockDim.x is 256 as we256 threads per block

so thread 0 loads , sh[0]=A[0] and sh[256]=A[256]

thread 1 loads , sh[1] = A[1] and sh[257]=A[257]

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

thread 255 loads, sh[255]=A[255] and sh[511] = A[511]

then we just compute minimum by doing parallel reduction
for first iteration i=255
sh[0] to sh[511]

sh[0] = min(sh[0],sh[255]),   basically (sh[tid],sh[tid+i]) 

sh[1] = min(sh[1],sh[256])

::::::::::::::::::::::::::

::::::::::::::::::::::::::

sh[255] = min(sh[255],sh[511])

now only first 255 elements are of interest
in the next iteration
i=256/2=128
i=128

sh[0] to sh[255]


sh[0]=min(sh[0],sh[128])  ,   basically min(sh[tid],sh[tid+i])

sh[1]=min(sh[1],sh[129])

::::::::::::::::::::::::

::::::::::::::::::::::::

sh[127]=min(sh[127],sh[255])


then 

sh[0] to sh[127]

then sh[0] to sh[63]



till we get 

sh[0] to sh[1]

and then sh[0] = min(sh[0],sh[1]) we would get the minimum element in sh[0]


we can then store this in our array



NOW FOR ANY N , we would obviously need more blocks 

and the result we get may need another pass

eg if we have 1024 elements , and we have 256 threads per block

and we know each thread handles 2 elements

so 1 block will handle first 512 elements and the next block will handle next 512 elements

and thus we would get 2 elements , min of first 512 , min of next 512

we would need to iterate over these 2 elements again , to get the overall minimum

2 elements can be easily handled by 1 block , and thus after this step we would stop

remember no of blocks can be caluclated in each step by

blocks = (currentsize + NUM_THREADS*2 -1) /(NUM_THREADS*2)


so for first iteration 

blocks = (1024+511)/512 = 2,

this tells we would need to execute our kernel with 2 blocks

and the array we get in the end would also be 2 elements long

so for next iteration our array has reduced to size 2

for 2nd iteration

blocks=(2+511)/512 = 1

1 block can easily give us our result 

and that would be our answer

Memory Layout with 2 Blocks

Block 0: processes elements 0-511 (512 elements)

Block 1: processes elements 512-1023 (512 elements)

Total: 1024 elements ✓

Thread Indexing

Block 0 (blockIdx.x = 0):

Thread 0: gid = 0 + 0*256*2 = 0, loads A[0] and A[256]

Thread 1: gid = 1 + 0*256*2 = 1, loads A[1] and A[257]

...

Thread 255: gid = 255 + 0*256*2 = 255, loads A[255] and A[511]


Block 1 (blockIdx.x = 1):


Thread 0: gid = 0 + 1*256*2 = 512, loads A[512] and A[768]

Thread 1: gid = 1 + 1*256*2 = 513, loads A[513] and A[769]

...

Thread 255: gid = 255 + 1*256*2 = 767, loads A[767] and A[1023]



BLOCK[0] would give us the min of first 512 elements 

and it is stored in B[0] 

BLOCK[1] would give us the min of next 512 elements 

and it is stored in B[1] 

we then do 

d_A=d_B 

so we can pass d_A as our input for next kernel cal

so that parallel reduction happens on these 2 elements next



now lets take a bigger example , N=1024*1024

this is done in the file min_parallel_reduction





## 🔁 Dry Run (Step-by-Step)

Let’s walk through this with small numbers for clarity.

---

### 🔢 Input Setup:

```cpp
int N = 1024 * 1024;       // 1,048,576 elements
h_A[i] = 2*N - i           // i.e., 2097152 - i → decreasing
// so min value = 2097152 - (N-1) = 1048577
```

---

### 🚀 First Iteration:

* `NUM_THREADS = 256`
* Each block handles **2 \* 256 = 512 elements**
* Number of blocks = `(1048576 + 511) / 512 = 2048`
* Each block reduces 512 values → writes 1 min to `d_B[blockIdx.x]`
* `currentSize` becomes 2048

---

### 🔁 Second Iteration:

* New input size = 2048
* Threads per block = 256
* Elements per block = 512
* Blocks = `(2048 + 511) / 512 = 4`
* Output: `d_B` has 4 elements (1 min per block)
* `currentSize` = 4

---

### 🔁 Third Iteration:

* Input size = 4
* Blocks = `(4 + 511) / 512 = 1`
* One block reduces 4 values to 1
* `currentSize = 1`, loop ends

---

### 🎯 Final Result:

* `d_A[0]` has global minimum: **1048577**
* Printed to screen

---

## Memory Layout Visualization:


### CUDA Global Memory Flow: Multi-Iteration Reduction

This diagram shows how memory pointers (`d_A`, `d_B`, and `d_original`) evolve across iterations of a multi-pass parallel reduction.

---

### 🌀 Iteration 0:

```
🔳 d_A = d_original = 0x1000

┌─────────────┐  
│  Original   │  ← d_A (0x1000), d_original (0x1000)  
│   Input     │  
└─────────────┘  
```

---

### 🔁 Iteration 1:

```
🔳 d_B = 0x2000

┌─────────────┐    ┌─────────────┐  
│  Original   │    │  Result 1   │  ← d_B (0x2000)  
│   Input     │    │             │  
└─────────────┘    └─────────────┘  

Pointers:
- d_original still points to Original (0x1000)
- d_A becomes d_B (0x2000)
```

---

### 🔁 Iteration 2:

```
🔳 d_B = 0x3000

┌─────────────┐    ┌─────────────┐    ┌─────────────┐  
│  Original   │    │  Result 1   │    │  Result 2   │  ← d_B (0x3000)  
│   Input     │    │ (can free)  │    │             │  
└─────────────┘    └─────────────┘    └─────────────┘  

Pointers:
- d_original still points to Original (0x1000)
- d_A (was 0x2000) is now freed
- d_A becomes d_B (0x3000)
```

---

You can repeat this until `currentSize == 1`, after which you copy the final minimum result from `d_A` back to host.

This memory flow ensures:

* No leaks
* No double frees
* Minimal device allocations


so in the end of the program

we delete d_original 
as it was the space that we took to copy h_A into d_A

then we clear d_A which would be storing the final answer
no need to clear d_B as , both d_A and d_B both point to same address
