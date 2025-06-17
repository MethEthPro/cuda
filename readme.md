# CUDA Learning Journal

A day-by-day breakdown of concepts, code experiments, and learnings while exploring CUDA programming.

---

## Day 1: CUDA Basics and Vector Addition

**Resources:** PDF, Code

* Learnt how to write a kernel using `__global__`:

  * Always returns `void` (no return value)
  * Called from host (CPU), executed on device (GPU)
* Understood CUDA thread/block indexing: `threadIdx`, `blockIdx`, `blockDim`
* How indexing works in kernel
* Passing pointers as parameters to kernels
* Bounds checking using `if (idx < N)`
* Host code concepts:

  * Declaring pointers and arrays
  * Allocating memory on GPU (`cudaMalloc`)
  * Copying data between host and device (`cudaMemcpy`)
  * Defining execution configuration: `<<<(N+t-1)/t, t>>>`
  * Freeing GPU memory (`cudaFree`)
* ✅ Implemented a **simple vector addition** program

---

## Day 2: Shared Memory and Timing

**Resources:** PDF, Code

* Explored `threadIdx` and `blockIdx` deeply
* Implemented `clock_example`
* Learnt about **shared memory**:

  * Fast memory accessible by all threads in a block
  * Size is passed via execution config
* Performed **parallel reduction** to compute minimum
* Timed kernel using `clock()` and checked `idx == 0` for start/stop
* Importance of `__syncthreads()` for divergence-safe behavior

---

## Day 3: GPU Architecture and Compute Capabilities

**Resources:** Docs, Readme

* [Cornell GPU architecture guide](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/index)
* [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus#compute)
* [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
* [Volta Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

Explored GPU internals, compute capabilities, and architectural design.

---

## Day 4: Matrix Addition & Multiplication (Naive)

**Resources:** Code

* Matrix is treated as 1D array → Row-major indexing: `A[i*N + j]`
* Implemented matrix addition
* Launched 2D thread blocks for matrix multiplication:

  ```cpp
  dim3 threadsPerBlock(N, N);
  ```
* Row/Col Indexing:

  ```cpp
  row = blockDim.y * blockIdx.y + threadIdx.y;
  col = blockDim.x * blockIdx.x + threadIdx.x;
  idx = row * N + col;
  ```
* Used `cudaDeviceProp` to get `clockRate` and calculated time:

  ```
  time = clock_cycles / clockRate (in KHz)
  ```

---

## Day 5: Warp Divergence

**Resources:** PDF, Code

* A **warp** = 32 threads executed together (SIMT model)
* Divergence happens when threads in a warp take different control paths (e.g., `if/else`)
* Causes performance degradation (threads are serialized)
* CUDA handles this via execution masks
* Use `__syncthreads()` after branching to prevent undefined behavior

---

## Day 6: RGB to Grayscale, Blur & Convolution

**Resources:** PDF, Code, [YouTube](https://youtu.be/C_zFhWdM4ic?si=nLzxQu5o-k3esM6i)

* ✅ Wrote kernel for RGB to Grayscale:

  ```
  gray[i] = red[i] * 0.299 + green[i] * 0.587 + blue[i] * 0.114
  ```
* Treated image as 1D RGB array → `rgb_idx = (row * width + col) * 3`
* ✅ Implemented box blur (mean filter)
* ✅ Wrote convolution kernel with custom kernel matrix (with boundary checks)

---

## Day 7: Shared Memory Vector Addition

**Resources:** Code

* Used shared memory for fast access:

  ```cpp
  shared[local_id] = A[idx];
  shared[local_id + blockDim.x] = B[idx];
  C[idx] = shared[local_id] + shared[local_id + blockDim.x];
  ```
* Allocated 2×threadsPerBlock×sizeof(float) shared memory
* Explained shared memory scope per block

---

## Day 8: GPU History & Evolution

**Resources:** Code

* Read about GPU history and early days
* Programming GPUs pre-CUDA was limited to graphics APIs
* CUDA revolutionized GPGPU programming by offering C-like syntax and abstractions

---

## Day 9: CPU vs GPU Matrix Multiplication

**Resources:** PDF, Code, Readme, [YouTube](https://youtu.be/oQT7IC0x254?si=J5CRHd8tTTRgoIA8)

* ✅ Implemented naive matrix multiplication on CPU
* ✅ Implemented naive matrix multiplication on GPU
* Compared performance difference

---

## Day 10: Coalesced vs Uncoalesced Memory Access

**Resources:** Code, Readme, [YouTube](https://www.youtube.com/watch?v=QmKNE3viwIE)

* **Coalesced Access:**

  ```
  out[id] = in[id];  // FAST
  ```
* **Uncoalesced Access:**

  ```
  out[id] = in[id * stride];  // SLOW
  ```
* Explained memory transaction alignment & warp-level optimization

---

## Day 11: Tiled Matrix Multiplication with Shared Memory

**Resources:** PDF, Code, Readme, [YouTube](https://youtu.be/Q3GgbfGTnVc?si=eenPBkMHABhtWAjg)

* Learnt how to tile matrices and load sub-blocks into shared memory
* Reduces global memory accesses and boosts performance

---

## Day 12: Shared Memory Banks and Conflicts

**Resources:** Code, Readme, Docs

* [Shared Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)
* [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
* Learnt about memory banks

  * Bank conflicts occur when multiple threads access same bank
  * Can serialize accesses → reduce performance

---

## Day 13: Race Conditions and atomicAdd

**Resources:** Code, Readme

* Race conditions occur when multiple threads write to the same memory location
* Used `atomicAdd` to safely increment shared/global variables
* But `atomicAdd` is slower — so should be avoided when possible

---

## ✅ Summary:

We haqve covered:

* Kernels, execution configuration, thread indexing
* Shared memory and synchronization
* Timing using `clock()`
* Matrix/vector addition, convolution, blur
* Warp divergence and coalesced memory
* Tiled matrix multiplication
* Bank conflicts and race conditions


