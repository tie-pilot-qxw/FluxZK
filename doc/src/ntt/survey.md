# Survey
## Algorithms
- Cooley-Tukey
Pros: in-place updates; only `n` space needed to access data.  
Cons: requires a rearrange step before starting, which adds overhead.
- Stockham
Pros: integrates the reordering into the algorithm itself, avoiding an extra rearrange step.  
Cons: read/write strides differ each pass, so it cannot be in-place; requires double buffering (2n space).
- SELF-SORTING IN-PLACE FAST FOURIER TRANSFORMS  CLIVE TEMPERTON
Advantages:  
- Time: compared to C‑T, no rearrange step is needed.  
- Space: compared to Stockham, only `n` space is needed instead of `2n`, leaving room to store precomputed twiddle factors.
## GPU Implementations
Using shared memory (Govindaraju et al., 2008) and warp shuffle (Durrani et al., 2021) to optimize NTT access to global memory has been explored, but prior work treats them as separate optimizations rather than a unified multi-level memory structure that uses both simultaneously.

By fusing butterfly operations across threads (Wang et al., 2023), block synchronization can be reduced. Our kernel does not use this technique for two reasons: first, with warp shuffles the required synchronization is already very low, potentially lower than after fusion. Second, for ZKP big integers each thread already uses many registers (64+). Fusing multiple butterflies increases register usage, reducing the maximum threads per SM and the number of warps per SM, which makes it harder to hide memory latency with warp switching. This also undermines the approach of increasing radix to improve compute utilization (Kim et al., 2020).

In GZKP (Ma et al., 2023), the authors group butterfly operations to coalesce global memory access. This approach is limited: first, it requires data to already be stored in column-major order in global memory, which differs from typical layouts and likely requires a transpose after transfer to the GPU, increasing overhead. Second, grouping butterflies reduces the maximum number of butterfly rounds per block, which reduces the number of rounds each kernel can handle and increases the number of kernel launches and global memory accesses.
