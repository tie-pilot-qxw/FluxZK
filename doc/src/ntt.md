# NTT

## Overview

The arithmetic complexity of NTT is essentially fixed, so optimization focuses on memory access patterns and counts, which are highly tied to the target architecture. Our implementation therefore concentrates on optimizing memory access patterns.

In short, our NTT uses a self-sort-in-place algorithm to eliminate the extra shuffle step in Cooley-Tukey, and unlike Stockham it does not require an extra 2x memory overhead.

To reduce memory overhead, we fully leverage warp-level shuffle operations and block-level shared memory, and optimize access to global and shared memory. We also avoid imposing any special input data format requirements.
