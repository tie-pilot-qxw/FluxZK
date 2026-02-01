# Implement

The arithmetic complexity of NTT is essentially fixed, so optimization focuses on memory access patterns and counts, which are tightly coupled to the target architecture. Our implementation therefore concentrates on optimizing memory access patterns. In short, we use a self-sort-in-place algorithm to eliminate the extra shuffle in Cooley-Tukey, and unlike Stockham it does not require an extra 2x memory overhead. To reduce memory overhead, we fully leverage warp-level shuffle operations and block-level shared memory, optimize access to global and shared memory, and avoid imposing special input data format requirements.

## Precompute Twiddle Factors

Step1: Use `cub::DeviceCopy::Batched` to get `[1, w, w, …, w]`

Step2: Use `cub::DeviceScan::InclusiveScan` to do a prefix product, getting `[1, w, w^2, …, w^n]`

## Split Work

Since stage1 and stage2 each process `deg/2` layers, and their per-launch layer limits differ, we partition `deg` so that each launch handles a similar number of layers and avoid kernels that only process a single layer.
## Stage1 Shared Memory Layout

Stage1 processes a maximum degree of 8–11 per launch, so it uses shared memory. To reduce bank conflicts, shared memory is laid out in column-major order:

a0_word0 a1_word0 a2_word0 a3_word0 ... an_word0 [empty] a0_word1 a1_word1 a2_word1 a3_word1 ...

With this layout, the lane-to-lane stride within a warp equals the index stride, rather than index stride × 8 as in row-major, which reduces bank conflicts.

The padding after `an_word0` is inserted because global loads read in the order `a0_word 0,1,2,…`, and `n` is `2^k`. The padding reduces bank conflicts during loads.

## Stage1: Load from Global to Shared

To coalesce global access, we exploit the fact that big integers span many bytes.

Threads 0–7 read `a0_word0, 1, 2 … 7` respectively, so global access is contiguous and coalesced.

## Further Reduce Bank Conflicts During Loads
With the current read pattern, assume `deg = 5`; then threads 0–7 access shared memory as:

0, 65, 130…, 1, 66, 131. …, …, 7, …

Threads 16–23 access:

`32, 97…`, so we reverse the access order for lanes 16–32.

## Use Warp Shuffle to Reduce Shared Reads

Warp shuffle XOR can simulate butterfly operations, so butterflies within a warp can be done with shuffles, further reducing shared memory reads. The lane mask controls the shuffle stride, which matches the butterfly stride.

## Stage2 Without Shared Memory

Since stage2’s max `deg` is 6, one warp can complete all butterfly operations, so shared memory is unnecessary. To coalesce global access, an explicit data reordering is performed.

## Stage2: Remove __syncthreads

Because shared memory is not used, we only need to ensure the first write happens after all data is read. A barrier can be used instead of `__syncthreads` to reduce synchronization overhead.

## Fixed Thread Count per Block

Since stage2’s max `deg` is 6, one warp can complete all butterfly operations, so shared memory is unnecessary. To coalesce global access, an explicit data reordering is performed.
