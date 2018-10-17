#pragma once

#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <inline_ptx.h>
#include <shared_memory_cache_helper.cuh>
#include <math_helper.cuh>

#if (__COMPUTE_CAPABILITY__ >= 700)
#include <cublas_v2.h>
#include <mma.h>
#endif

namespace quda {

#if defined (GPU_DOMAIN_WALL_DIRAC) && (__COMPUTE_CAPABILITY__ >= 700)
  
  template<class T>
  struct TensorCoreSharedMemory
  {
    __device__ inline operator T*()
    {
      extern __shared__ int __smem[];
      return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
      extern __shared__ int __smem[];
      return (T*)__smem;
    }
  };

  // matrix a for m5inv: column major, M/M_sm(size/padded size) by k
  // (spin,Ls) by (spin,Ls), where left most index is the fastest changing one(spin).
  // x by y
  template<int block_dim_x, int Ls_, int M_sm, bool dagger, class Arg>
  __device__ inline void construct_matrix_a_m5inv(Arg& arg, half* sm_a){
    const float k = arg.kappa;
    
    // if we rescale, then the actual matrix is alpha*m5inv+beta.
    // Otherwise a = 1., b = 0.;
    const float a = arg.alpha; 
    const float b = arg.beta; 
    
    const float inv = a*arg.fac_inv;

    int offset_k = threadIdx.y*4;
    int x = threadIdx.x;
    
    while(x < Ls_){
      int offset_m = x*4;
      int exp;
      if(dagger){
        exp = x>threadIdx.y ? Ls_-x+threadIdx.y : threadIdx.y-x;
      }else{
        exp = x<threadIdx.y ? Ls_-threadIdx.y+x : x-threadIdx.y;
      }
      float factorR = inv*__fast_pow(k, exp) * ( x>threadIdx.y ? -arg.m_f : static_cast<float>(1.0) );
      if(dagger){
        exp = x<threadIdx.y ? Ls_-threadIdx.y+x : x-threadIdx.y;
      }else{
        exp = x>threadIdx.y ? Ls_-x+threadIdx.y : threadIdx.y-x;
      }
      float factorL = inv*__fast_pow(k, exp) * ( x<threadIdx.y ? -arg.m_f : static_cast<float>(1.0) );
      
      if(exp){
      sm_a[ (offset_k+0)*(M_sm)+(offset_m+0) ] = factorR + factorL;
      sm_a[ (offset_k+1)*(M_sm)+(offset_m+1) ] = factorR + factorL;
      sm_a[ (offset_k+2)*(M_sm)+(offset_m+2) ] = factorR + factorL;
      sm_a[ (offset_k+3)*(M_sm)+(offset_m+3) ] = factorR + factorL;
      }else{ // exp = 0 means we are on the diagonal.
      sm_a[ (offset_k+0)*(M_sm)+(offset_m+0) ] = factorR + factorL + b;
      sm_a[ (offset_k+1)*(M_sm)+(offset_m+1) ] = factorR + factorL + b;
      sm_a[ (offset_k+2)*(M_sm)+(offset_m+2) ] = factorR + factorL + b;
      sm_a[ (offset_k+3)*(M_sm)+(offset_m+3) ] = factorR + factorL + b;
      }
      // sm_a[ (offset_k+0)*(M_sm)+(offset_m+0) ] = factorR + factorL;
      sm_a[ (offset_k+0)*(M_sm)+(offset_m+1) ] = static_cast<half>(0.0f);
      sm_a[ (offset_k+0)*(M_sm)+(offset_m+2) ] = factorR - factorL;
      sm_a[ (offset_k+0)*(M_sm)+(offset_m+3) ] = static_cast<half>(0.0f);
      
      sm_a[ (offset_k+1)*(M_sm)+(offset_m+0) ] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+1)*(M_sm)+(offset_m+1) ] = factorR + factorL;
      sm_a[ (offset_k+1)*(M_sm)+(offset_m+2) ] = static_cast<half>(0.0f);
      sm_a[ (offset_k+1)*(M_sm)+(offset_m+3) ] = factorR - factorL;
      
      sm_a[ (offset_k+2)*(M_sm)+(offset_m+0) ] = factorR - factorL;
      sm_a[ (offset_k+2)*(M_sm)+(offset_m+1) ] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+2)*(M_sm)+(offset_m+2) ] = factorR + factorL;
      sm_a[ (offset_k+2)*(M_sm)+(offset_m+3) ] = static_cast<half>(0.0f);
      
      sm_a[ (offset_k+3)*(M_sm)+(offset_m+0) ] = static_cast<half>(0.0f);
      sm_a[ (offset_k+3)*(M_sm)+(offset_m+1) ] = factorR - factorL;
      sm_a[ (offset_k+3)*(M_sm)+(offset_m+2) ] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+3)*(M_sm)+(offset_m+3) ] = factorR + factorL; 
    
      x += block_dim_x;
    }
  } 

  // Load data(scaled short values and scale) from global memory to shared memroy.
  // (spin,Ls) by (complex,color,4d), where left most index is the fastest changing one(spin and complex).
  template<int N_sm, class Arg>
  __device__ inline float load_matrix_b_tex(Arg& arg, half2* sm_b, int sid){
    constexpr int N_sm_d2 = N_sm/2;
    float4 in_tex;
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 0*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+0 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+1 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 1*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+2 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+0 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 2*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+1 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+2 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 3*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+0 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+1 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 4*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+2 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+0 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    in_tex = tex1Dfetch<float4>(arg.in.tex, 5*arg.volume_cb + sid); 
    sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+1 ] = __floats2half2_rn(in_tex.x, in_tex.y);
    sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+2 ] = __floats2half2_rn(in_tex.z, in_tex.w);
    
    return tex1Dfetch<float>(arg.in.texNorm, sid);
  } 

  __device__ inline short2 __half22short2_rn(const half2 input){
    return make_short2(__half2short_rn(input.x), __half2short_rn(input.y));
  }
  __device__ inline half __half_max_abs_half2__(half max, const half2 input){
    static constexpr uint32_t mask = ~((0x1u<<31) + (0x1u<<15)); // 01111111 11111111 01111111 11111111 
    // Set the fisrt bit of the half to 0.
    uint32_t input_masked = *reinterpret_cast<const uint32_t*>(&input) & mask;
    max = __hgt(max, reinterpret_cast<half2*>(&input_masked)->x) ? max : reinterpret_cast<half2*>(&input_masked)->x;
    max = __hgt(max, reinterpret_cast<half2*>(&input_masked)->y) ? max : reinterpret_cast<half2*>(&input_masked)->y;
    return max;
  }

  // Store results(scaled short values and scale) in shared memroy to global memroy.
  template<int N_sm, class Arg>
  __device__ inline void store_matrix_c(Arg& arg, half2* sm_b, int sid, float scale){
    half max_ = (half)0.0f;
    constexpr int N_sm_d2 = N_sm/2;
    
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+0 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+1 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+2 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+0 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+1 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+2 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+0 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+1 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+2 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+0 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+1 ]);
    max_ = __half_max_abs_half2__(max_, sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+2 ]);

    arg.out.norm[sid] = __half2float(max_)*scale;
    
    const half2 max_short_div_max2_ = __half2half2( __hdiv(fixedMaxValue<short>::value, max_) );

    short4* out = reinterpret_cast<short4*>(arg.out.field);
    short2 a, b;
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_));
    out[sid + 0*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y); 
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_));
    out[sid + 1*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y); 
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_));
    out[sid + 2*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y); 
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_));
    out[sid + 3*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y); 
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_));
    out[sid + 4*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y); 
    
    a = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_));
    b = __half22short2_rn(__hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_));
    out[sid + 5*arg.volume_cb] = make_short4(a.x, a.y, b.x, b.y);  
  } 

  // "reload" version of wmma gemm. Matrix a is loaded when needed.
  // It is a waste of time but save register usage.
  template<int block_dim_x, int Ls_, int M, int N, int M_sm, int N_sm>
  __device__ inline void wmma_gemm_reload(half* sm_a, half* sm_b, half* sm_c){
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    constexpr int tm_dim = M/WMMA_M;
    constexpr int tn_dim = N/WMMA_N;
    
    constexpr int total_warp = block_dim_x*Ls_/32;
    
    static_assert( (tm_dim*tn_dim)%total_warp==0, "(tm_dim*tn_dim)%%total_warp==0\n" );
    static_assert( tn_dim%(tm_dim*tn_dim/total_warp)==0, "tn_dim%%(tm_dim*tn_dim/total_warp)==0\n" );
    
    const int this_warp = (threadIdx.y*block_dim_x+threadIdx.x)/32;
    
    constexpr int total_tile = tm_dim*tn_dim;
    
    constexpr int warp_cycle = total_tile/total_warp;
    const int warp_m = this_warp*warp_cycle/tn_dim;
    #pragma unroll
    for(int c = 0; c < warp_cycle; c++){
      // Set up the wmma stuff
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

      // The logical warp assigned to each part of the matrix.
      const int phys_warp_index = this_warp*warp_cycle+c;
      const int warp_n = phys_warp_index-warp_m*tn_dim;
      // eg. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111
      
      // Zero the initial acc.
      nvcuda::wmma::fill_fragment(c_frag, static_cast<half>(0.0f));
      
      #pragma unroll
      for( int k = 0; k < tm_dim; k++ ){
        const int a_row = warp_m*WMMA_M;
        const int a_col = k*WMMA_K;
        const int b_row = k*WMMA_K;
        const int b_col = warp_n*WMMA_N;
    
        // Load Matrix
        nvcuda::wmma::load_matrix_sync(a_frag, sm_a+a_row+a_col*M_sm, M_sm);
        nvcuda::wmma::load_matrix_sync(b_frag, sm_c+b_col+b_row*N_sm, N_sm);
        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      } 
    
        int c_row = warp_m*WMMA_M;
        int c_col = warp_n*WMMA_N;
    
        nvcuda::wmma::store_matrix_sync(sm_c+c_col+c_row*N_sm, c_frag, N_sm, nvcuda::wmma::mem_row_major);
    }
  } 
  
  // "preload" version of wmma gemm. Matrix a is preloaded before hand.
  // It saves time but uses more registers.
  template<int block_dim_x, int Ls_, int M, int N, int M_sm, int N_sm, class T>
  __device__ inline void wmma_gemm_preload(T* a_frag, half* sm_b, half* sm_c){
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    constexpr int tm_dim = M/WMMA_M;
    constexpr int tn_dim = N/WMMA_N;
    
    constexpr int total_warp = block_dim_x*Ls_/32;
    
    static_assert( (tm_dim*tn_dim)%total_warp==0, "(tm_dim*tn_dim)%%total_warp==0\n" );
    static_assert( tn_dim%(tm_dim*tn_dim/total_warp)==0, "tn_dim%%(tm_dim*tn_dim/total_warp)==0\n" );
    
    const int this_warp = (threadIdx.y*block_dim_x+threadIdx.x)/32;
    
    constexpr int total_tile = tm_dim*tn_dim;
    
    constexpr int warp_cycle = total_tile/total_warp;
    const int warp_m = this_warp*warp_cycle/tn_dim;
    #pragma unroll
    for(int c = 0; c < warp_cycle; c++){
      // Set up the wmma stuff
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

      // The logical warp assigned to each part of the matrix.
      const int phys_warp_index = this_warp*warp_cycle+c;
      const int warp_n = phys_warp_index-warp_m*tn_dim;
      // eg. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111
      
      // Zero the initial acc.
      nvcuda::wmma::fill_fragment(c_frag, static_cast<half>(0.0f));
      
      #pragma unroll
      for( int k = 0; k < tm_dim; k++ ){
        const int b_row = k*WMMA_K;
        const int b_col = warp_n*WMMA_N;
    
        // Load Matrix
        nvcuda::wmma::load_matrix_sync(b_frag, sm_c+b_col+b_row*N_sm, N_sm);
        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag[k], b_frag, c_frag);
      } 
    
        int c_row = warp_m*WMMA_M;
        int c_col = warp_n*WMMA_N;
    
        nvcuda::wmma::store_matrix_sync(sm_c+c_col+c_row*N_sm, c_frag, N_sm, nvcuda::wmma::mem_row_major);
    }
  } 

#else
#error "Domain wall dslash WITH tensor cores has not been built"
#endif // defined (GPU_DOMAIN_WALL_DIRAC) && (__COMPUTE_CAPABILITY__ >= 700)

} // namespace quda

