#pragma once

#include <cuda_bf16.h>

namespace quda
{

  namespace mma
  {

    using bfloat162 = __nv_bfloat162;
    using bfloat16 = __nv_bfloat16;

    using half2 = __nv_half2;
    using half = __nv_half;

    struct tfloat32 {
    };

    template <int inst_m, int inst_n, int inst_k, class AB, class CD> struct mma_instruction_t {
    };

    template <> struct mma_instruction_t<16, 8, 16, bfloat16, float> {
      __device__ void operator()(float c[4], const unsigned a[4], const unsigned b[2])
      {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    };

    template <> struct mma_instruction_t<16, 8, 8, bfloat16, float> {
      __device__ void operator()(float c[4], const unsigned a[2], const unsigned b[1])
      {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
      }
    };

    template <> struct mma_instruction_t<16, 8, 16, half, float> {
      __device__ void operator()(float c[4], const unsigned a[4], const unsigned b[2])
      {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    };

    template <> struct mma_instruction_t<16, 8, 8, half, float> {
      __device__ void operator()(float c[4], const unsigned a[2], const unsigned b[1])
      {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
      }
    };

    template <> struct mma_instruction_t<16, 8, 8, tfloat32, float> {
      __device__ void operator()(float c[4], const unsigned a[4], const unsigned b[2])
      {
        asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    };

    template <> struct mma_instruction_t<16, 8, 4, tfloat32, float> {
      __device__ void operator()(float c[4], const unsigned a[2], const unsigned b[1])
      {
        asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
      }
    };

    template <> struct mma_instruction_t<16, 16, 4, half, float> {
      __device__ void operator()(float c[8], const unsigned a[2], const unsigned b[2])
      {
        asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                     "{%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
      }
    };

  } // namespace mma

} // namespace quda
