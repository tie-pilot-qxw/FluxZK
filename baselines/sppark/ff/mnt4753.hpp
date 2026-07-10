// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_MNT4753_HPP__
#define __SPPARK_FF_MNT4753_HPP__

#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)

namespace device {
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_P[24] = {
        0x245e8001, 0x5e9063de, 0x2cdd119f, 0xe39d5452, 0x9ac425f0, 0x63881071, 0x767254a4, 0x685acce9, 0xcb537e38, 0xb80f0da5, 0xf218059d, 0xb117e776, 0xa15af79d, 0x99d124d9, 0xe8a0ed8d, 0x07fdb925, 0x6c97d873, 0x5eb7e8f9, 0x5b8fafed, 0xb7f99750, 0xeee2cdad, 0x10229022, 0x2d92c411, 0x1c4c6
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_RR[24] = { /* (1<<512)%P */
        0xcfd190c8, 0x84717088, 0x7df03c0a, 0xc7d9ff8e, 0x242b3507, 0xa24bea56, 0xa0714c7d, 0xa896a656, 0xff6f3ddf, 0x80a46659, 0xf88d7ce8, 0x2f47839e, 0x04a3b597, 0xa8c86d46, 0xc4f7ef07, 0xe03c79ca, 0xf4a81245, 0x2505daf1, 0x4c381723, 0x8e460575, 0xcbfdacaf, 0xb081f15b, 0xe89cb485, 0x2a33
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_one[24] = { /* (1<<256)%P */
        0xd9dc6f42, 0x98a8ecab, 0x5a034686, 0x91cd31c6, 0xcd14572e, 0x97c3e4a0, 0xc788b601, 0x79589819, 0x2108976f, 0xed269c94, 0xcf031d68, 0x1e0f4d8a, 0x13338559, 0x320c3bb7, 0xd2f00a62, 0x598b4302, 0xfd8ca621, 0x4074c9cb, 0x3865e88c, 0x0fa47edb, 0x1ff9a195, 0x95455fb3, 0x9ec8e242, 0x7b47
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_Px4[24] = { /* left-aligned value of the modulus */
        0x917a0004, 0x7a418f78, 0xb374467d, 0x8e755148, 0x6b1097c3, 0x8e2041c6, 0xd9c95291, 0xa16b33a5, 0x2d4df8e1, 0xe03c3697, 0xc8601676, 0xc45f9ddb, 0x856bde76, 0x67449366, 0xa283b636, 0x1ff6e497, 0xb25f61cc, 0x7adfa3e5, 0x6e3ebfb5, 0xdfe65d41, 0xbb8b36b6, 0x408a408b, 0xb64b1044, 0x71318
    };
    static __device__ __constant__ const uint32_t MNT4573_M0 = 3831398399;

    static __device__ __constant__ __align__(16) const uint32_t MNT4573_r[24] = {
        0x40000001, 0xd90776e2, 0x0fa13a4f, 0x4ea09917, 0x3f005797, 0xd6c381bc, 0x34993aa4, 0xb9dff976, 0x29212636, 0x3eebca94, 0xc859a99b, 0xb26c5c28, 0xa15af79d, 0x99d124d9, 0xe8a0ed8d, 0x07fdb925, 0x6c97d873, 0x5eb7e8f9, 0x5b8fafed, 0xb7f99750, 0xeee2cdad, 0x10229022, 0x2d92c411, 0x1c4c6
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_rRR[24] = { /* (1<<512)%P */
        0xb7f4c8d1, 0x3f9c69c7, 0xee48d127, 0x70a50fa9, 0x009569cb, 0xcdbe6702, 0xc49edc38, 0x6bd8c6c6, 0xc35ee94e, 0x7955876c, 0xbe54a3f4, 0xc7285529, 0xecec77cf, 0xded52121, 0xee12ee8e, 0x99be80f2, 0x493bdcef, 0xc8a0ff01, 0xf3d9a316, 0xacc27988, 0xfb44b3c9, 0xd9e817a8, 0x8037e0e4, 0x5b5
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_rone[24] = { /* (1<<256)%P */
        0x7fff6f42, 0xb9968014, 0xb589cea8, 0x4eb16817, 0x0c79e179, 0xa1ebd2d9, 0xc549c0da, 0x0f725cae, 0xd3e6dad4, 0xab0c4ee6, 0xde0ccb62, 0x9fbca908, 0x13338498, 0x320c3bb7, 0xd2f00a62, 0x598b4302, 0xfd8ca621, 0x4074c9cb, 0x3865e88c, 0x0fa47edb, 0x1ff9a195, 0x95455fb3, 0x9ec8e242, 0x7b47
    };
    static __device__ __constant__ __align__(16) const uint32_t MNT4573_rx4[24] = { /* left-aligned value of the modulus */
        0x00000004, 0x641ddb89, 0x3e84e93f, 0x3a82645c, 0xfc015e5d, 0x5b0e06f0, 0xd264ea93, 0xe77fe5d8, 0xa48498da, 0xfbaf2a50, 0x2166a66c, 0xc9b170a3, 0x856bde76, 0x67449366, 0xa283b636, 0x1ff6e497, 0xb25f61cc, 0x7adfa3e5, 0x6e3ebfb5, 0xdfe65d41, 0xbb8b36b6, 0x408a408b, 0xb64b1044, 0x71318
    };
    static __device__ __constant__ const uint32_t MNT4573_m0 = 1073741823;
}
# if defined(__CUDA_ARCH__) || defined(__HIPCC__)   // device-side field types
#  if defined(__CUDA_ARCH__)
#   include "mont_t.cuh"
#  elif defined(__HIPCC__)
#   include "mont_t.hip"
typedef uint64_t vec768[12];
#  endif

namespace mnt4753 {

typedef mont_t<753, device::MNT4573_P, device::MNT4573_M0,
                    device::MNT4573_RR, device::MNT4573_one,
                    device::MNT4573_Px4> fp_mont;
struct fp_t : public fp_mont {
    using mem_t = fp_t;
    __device__ __forceinline__ fp_t() {}
    __device__ __forceinline__ fp_t(const fp_mont& a) : fp_mont(a) {}
};
typedef mont_t<753, device::MNT4573_r, device::MNT4573_m0,
                    device::MNT4573_rRR, device::MNT4573_rone,
                    device::MNT4573_rx4> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    __device__ __forceinline__ fr_t() {}
    __device__ __forceinline__ fr_t(const fr_mont& a) : fr_mont(a) {}
#  ifdef __HIPCC__
    __host__   __forceinline__ fr_t(vec768 a)         : fr_mont(a) {}
#  endif
};

}

# endif
#endif


#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)  // host-side field types
# include <blst_t.hpp>

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsubobject-linkage"
# endif

namespace mnt4753 {

struct fp_t {
    using mem_t = vec768;
    mem_t mem;
    inline fp_t() {}
    inline fp_t(const vec768& a) {
      memcpy(mem, a, sizeof(uint64_t) * 12);
    };
};

struct fr_t {
    using mem_t = vec768;
    mem_t mem;
    inline fr_t() {}
    
    inline fr_t(const vec768& a) {
      memcpy(mem, a, sizeof(uint64_t) * 12);
    };
};



}

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
# endif
#endif

#ifdef FEATURE_MNT4753
using namespace mnt4753;
#endif

#endif
