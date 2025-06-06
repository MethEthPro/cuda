Coalesced(contiguous) access vs Uncoalesced(strided) access

out[id] = in[id]  -- coalesced

out[id] = in[id*stride] -- uncoalesced

CUDA memory coalescing is very much based on the concept of cache lines and memory transaction granularity.

When threads in a warp (32 threads) access consecutive addresses, these accesses can be coalesced into a single transaction, maximizing memory throughput.
All threads access adjacent memory -> FAST

If all 32 threads in the warp access memory within the same aligned segment, CUDA coalesces these accesses into just 1 or 2 memory transactions.

Non-coalesced (strided) access results in inefficient, scattered memory transactions.
Threads access memory far apart -> SLOW