
---
learnt about some new atomic functions, also struggled with basic conecpts like pointers and memory allocation and had to revisit them
atomicInc and atomicDec did not work as expected
the image attached shows the output we got

## 📘 README: CUDA Atomic Operations – What I Learned

### ✅ Topics Covered Today

1. **Atomic Functions in CUDA**

   * `atomicAdd()`
   * `atomicSub()`
   * `atomicExch()`
   * `atomicMin()`, `atomicMax()`
   * `atomicInc()`, `atomicDec()`
   * `atomicCAS()` (Compare-And-Swap)

2. **Pointers and CUDA Kernels**

   * Device functions work with **pointers to global memory**
   * Always pass **device pointers** into kernels
   * Access the value using `*ptr` only when needed (e.g., for `printf`, not for `atomicX` calls)

3. **Key CUDA Concepts**

   * `cudaMalloc()` allocates device memory; returns a pointer → use `(void**)&` in C for type casting
   * `cudaMemcpy()` copies between host and device
   * `cudaMemset()` sets device memory, byte-wise
   * Use correct format specifier when printing (`%u` for `unsigned int`)

---

### 🔧 Atomic Operation Behaviors

| Function                 | Behavior                                                              |
| ------------------------ | --------------------------------------------------------------------- |
| `atomicAdd(x, val)`      | Adds `val` to `*x` atomically                                         |
| `atomicSub(x, val)`      | Subtracts `val` from `*x` atomically                                  |
| `atomicExch(x, val)`     | Sets `*x = val`                                                       |
| `atomicMin(x, val)`      | Sets `*x = min(*x, val)`                                              |
| `atomicMax(x, val)`      | Sets `*x = max(*x, val)`                                              |
| `atomicCAS(x, cmp, val)` | If `*x == cmp`, sets `*x = val`                                       |
| `atomicInc(x, limit)`    | If `*x >= limit`, sets `*x = 0`; else increments `*x`                 |
| `atomicDec(x, limit)`    | If `*x == 0` or `*x > limit`, sets `*x = limit`; else decrements `*x` |

---

### ⚠️ Notes on `atomicInc()` and `atomicDec()`

* **Did not behave as expected initially**

  * Also: If the initial value is large (e.g., due to unsigned wrap-around), `atomicDec()` resets to the `limit` and can yield confusing results.
* Fixed by:

  * Initializing the values properly (e.g., starting `y = 3`)
  * Using correct data types (`unsigned int`)
  * Using correct format specifier (`%u`) when printing results

---

### 🔍 Example Takeaways

* Use `atomicInc(x, 5)` only when `x` is `unsigned int` and you want it to wrap to `0` after reaching `5`.
* `atomicDec(y, 5)` wraps from `0` → `5`.
* Use `cudaMemcpy()` with care when initializing and printing results.
* For thread-by-thread debugging, device-side `printf()` can be used inside kernels.

---

### ✅ Final Result

You wrote and understood a complete CUDA program that demonstrates:

* Correct use of atomic operations
* Proper memory management and synchronization
* Identified and fixed subtle bugs in wrap-around logic

---

