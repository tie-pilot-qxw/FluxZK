#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h>

// Big integer representation, size N bytes (N must be a multiple of 32)
template <size_t N>
struct alignas(32) LargeInteger {
    static_assert(N % 32 == 0, "Size must be a multiple of 32 bytes");
    uint8_t data[N];
};

// Helper class for AVX memory operations
template <size_t N>
struct AVXHelper {};

// Specialization for 32 bytes
template <>
struct AVXHelper<32> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
    }
};

// Specialization for 64 bytes
template <>
struct AVXHelper<64> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 32));
        
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 32), v2);
    }
};

// Specialization for 96 bytes
template <>
struct AVXHelper<96> {
    static inline void copy(const uint8_t* src, uint8_t* dst) {
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 32));
        __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + 64));
        
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 32), v2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 64), v3);
    }
};
