Although CUDA kernel launches are asynchronous, all GPU-related tasks placed in one stream (which is the default behavior) are executed sequentially.

So, for example,

kernel1<<<X,Y>>>(...); // kernel start execution, CPU continues to next statement

kernel2<<<X,Y>>>(...); // kernel is placed in queue and will start after kernel1 finishes, CPU continues to next statement

cudaMemcpy(...); // CPU blocks until memory is copied, memory copy starts only after kernel2 finishes



13

When you want your GPU to start processing some data, you typically do a kernal invocation.

 When you do so, your device (The GPU) will start to doing whatever it is you told it to do. However, unlike a normal sequential program on your host (The CPU) will continue to execute the next lines of code in your program. cudaDeviceSynchronize makes the host (The CPU) wait until the device (The GPU) have finished executing ALL the threads you have started, and thus your program will continue as if it was a normal sequential program.

In small simple programs you would typically use cudaDeviceSynchronize, when you use the GPU to make computations, to avoid timing mismatches between the CPU requesting the result and the GPU finising the computation. 

To use cudaDeviceSynchronize makes it alot easier to code your program, but there is one major drawback: Your CPU is idle all the time, while the GPU makes the computation. 

Therefore, in high-performance computing, you often strive towards having your CPU making computations while it wait for the GPU to finish.


Kernel launches and host <-> device memory copies that do not specify any stream parameter, or equivalently that set the stream parameter to zero, are issued to the default stream. They are therefore executed in order.


STREAMS--

 A stream is a sequence of commands (possibly issued by different host threads) that execute in order.
 
  Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (for example, inter-kernel communication is undefined).
  
   The commands issued on a stream may execute when all the dependencies of the command are met. The dependencies could be previously launched commands on same stream or dependencies from other streams. The successful completion of synchronize call guarantees that all the commands launched are completed.

   so streams need not be executed sequentially 
   
   rather different host threads can launch streams

   which may overlap

   The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream and whether or not the device supports overlap of data transfer and kernel execution (see Overlap of Data Transfer and Kernel Execution),
   
    concurrent kernel execution (see Concurrent Kernel Execution), and/or concurrent data transfers (see Concurrent Data Transfers).

    Streams allow:
Overlapping kernel execution

Concurrent memory transfers

Concurrent execution on separate CUDA cores (if supported)


  There are various ways to explicitly synchronize streams with each other.

cudaDeviceSynchronize() waits until all preceding commands in all streams of all host threads have completed.

cudaStreamSynchronize()takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device. 


cudaDeviceSynchronize() is a global sync.

It’s not ideal for fine-grained control when using multiple streams — for that, you'd use cudaStreamSynchronize().



so in our code we wrote a simple program of addition using paralle reduction

how is it parallel reduction??

✔️ Multiple Threads Read in Parallel:
Each thread independently reads input[tid].

✔️ Multiple Threads Attempt to Write in Parallel:
All of them try to atomicAdd their value to the same output variable.

✅ So the reduction is parallel in intent, because many threads attempt to contribute to the result simultaneously.

then we compared the time with and without synchronization