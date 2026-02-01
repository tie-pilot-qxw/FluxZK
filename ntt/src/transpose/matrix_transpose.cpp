#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include "matrix_transpose.h"

// Choose block size based on element size
template<size_t N>
constexpr size_t getBlockSize() {
    if (N == 32) return 32;        // 32×32×32 = 32KB
    else if (N == 64) return 24;   // 24×24×64 = 36KB
    else return 16;                // 16×16×96 = 24KB
}

// Blocked transpose
template<size_t N>
void transposeBlock(const LargeInteger<N>* src, LargeInteger<N>* dst, 
                    size_t srcRows, size_t srcCols,
                    size_t blockRow, size_t blockCol) {
    constexpr size_t BLOCK_SIZE = getBlockSize<N>();
    
    const size_t blockRowEnd = std::min(blockRow + BLOCK_SIZE, srcRows);
    const size_t blockColEnd = std::min(blockCol + BLOCK_SIZE, srcCols);
    
    for (size_t i = blockRow; i < blockRowEnd; i++) {
        for (size_t j = blockCol; j < blockColEnd; j++) {
            size_t srcIdx = i * srcCols + j;
            size_t dstIdx = j * srcRows + i;
            
            if (srcIdx >= srcRows * srcCols || dstIdx >= srcRows * srcCols) {
                std::cerr << "Index out of bounds: src=" << srcIdx << " dst=" << dstIdx 
                         << " max=" << srcRows * srcCols << std::endl;
                continue;
            }
            
            // Use AVX-optimized memory operations
            AVXHelper<N>::copy(src[srcIdx].data, dst[dstIdx].data);
        }
    }
}

// Thread worker function
template<size_t N>
void transposeThreadWork(const LargeInteger<N>* src, LargeInteger<N>* dst,
                        size_t rows, size_t cols,
                        size_t startColBlock, size_t endColBlock) {
    constexpr size_t BLOCK_SIZE = getBlockSize<N>();
    
    for (size_t rowBlock = 0; rowBlock < rows; rowBlock += BLOCK_SIZE) {
        for (size_t colBlock = startColBlock; colBlock < endColBlock; colBlock += BLOCK_SIZE) {
            transposeBlock<N>(src, dst, rows, cols, rowBlock, colBlock);
        }
    }
}

// Main transpose function
template<size_t N>
void transpose(const LargeInteger<N>* src, LargeInteger<N>* dst, size_t rows, size_t cols, 
               size_t maxThreads) {  // maxThreads=0 means auto-select
    assert((rows == 16 || rows == 32 || rows == 64) && "Rows must be 16, 32, or 64");
    assert((cols & (cols - 1)) == 0 && "Cols must be a power of 2");
    
    constexpr size_t BLOCK_SIZE = getBlockSize<N>();
    
    // Compute thread count and work per thread
    const size_t hardwareThreads = std::thread::hardware_concurrency();
    const size_t defaultThreads = std::max(hardwareThreads / 2, size_t(1));  // Default: cores of a single CPU
    const size_t userThreads = maxThreads > 0 ? maxThreads : defaultThreads;
    const size_t numThreads = std::min(userThreads, cols / BLOCK_SIZE);
    const size_t colBlocksPerThread = (cols / BLOCK_SIZE + numThreads - 1) / numThreads;
    
    std::cout << "Starting transpose with size " << rows << "x" << cols 
              << " (element size: " << N << " bytes, block size: " << BLOCK_SIZE << ")" << std::endl;
    std::cout << "Hardware threads: " << hardwareThreads 
              << ", Using " << numThreads << " threads (NUMA optimized)" << std::endl;
    std::cout << "Block memory usage: " << (BLOCK_SIZE * BLOCK_SIZE * N / 1024.0) << " KB" << std::endl;
    
    // Create and start threads
    std::vector<std::thread> threads;
    for (size_t t = 0; t < numThreads; ++t) {
        size_t startColBlock = t * colBlocksPerThread * BLOCK_SIZE;
        size_t endColBlock = std::min((t + 1) * colBlocksPerThread * BLOCK_SIZE, cols);
        
        threads.emplace_back(transposeThreadWork<N>,
                           src, dst, rows, cols,
                           startColBlock, endColBlock);
    }
    
    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}

// Explicit instantiations
template void transpose<32>(const LargeInteger<32>*, LargeInteger<32>*, size_t, size_t, size_t);
template void transpose<64>(const LargeInteger<64>*, LargeInteger<64>*, size_t, size_t, size_t);
template void transpose<96>(const LargeInteger<96>*, LargeInteger<96>*, size_t, size_t, size_t);
