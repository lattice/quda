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

#ifdef GPU_DOMAIN_WALL_DIRAC

  static void set_shared_memory_on_volta(const void* f, const char* name){
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties( &device_prop, 0 );
    if(device_prop.major < 7) return;

    auto found = qudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
    printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));

    found = qudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));

    cudaFuncAttributes cfa;
    found = cudaFuncGetAttributes(&cfa, f);
    printfQuda("Found %s: %s\n", name, cudaGetErrorString(found));

    printfQuda("Actual maximum:         %d\n", (int)cfa.maxDynamicSharedSizeBytes);
    printfQuda("Actual maximum percent: %d\n", (int)cfa.preferredShmemCarveout);
  }
  
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

  /**
    @brief Structure containing zMobius / Zolotarev coefficients

    FIXME
    - fix flops counters
    - use kappa notation and not b/c for consistency with other codes and sanity
  */
  template <typename real>
    struct coeff_5 {
      complex<real> a[QUDA_MAX_DWF_LS]; // xpay coefficients
      complex<real> b[QUDA_MAX_DWF_LS];
      complex<real> c[QUDA_MAX_DWF_LS];
    };

  constexpr int size = 4096;
  static __constant__ char mobius_d[size]; // constant buffer used for Mobius coefficients for GPU kernel
  static char mobius_h[size];              // constant buffer used for Mobius coefficients for CPU kernel

  /**
    @brief Helper function for grabbing the constant struct, whether
    we are on the GPU or CPU.
   */
  template <typename real>
    inline __device__ __host__ const coeff_5<real>* coeff() {
#ifdef __CUDA_ARCH__
      return reinterpret_cast<const coeff_5<real>*>(mobius_d);
#else
      return reinterpret_cast<const coeff_5<real>*>(mobius_h);
#endif
    }

  template <typename real, Dslash5Type, typename Arg> struct coeff_type {
    typedef real type;
    const Arg &arg;
    __device__ __host__ coeff_type(const Arg &arg) : arg(arg) { }
    __device__ __host__ real a(int s) { return arg.a; }
    __device__ __host__ real b(int s) { return arg.b; }
    __device__ __host__ real c(int s) { return arg.c; }
  };

  template <typename real, typename Arg> struct coeff_type<real,M5_INV_ZMOBIUS,Arg> {
    typedef complex<real> type;
    __device__ __host__ coeff_type(const Arg &arg) { }
    __device__ __host__ complex<real> a(int s) { return coeff<real>()->a[s]; }
    __device__ __host__ complex<real> b(int s) { return coeff<real>()->b[s]; }
    __device__ __host__ complex<real> c(int s) { return coeff<real>()->c[s]; }
  };

  /**
    @brief Parameter structure for applying the Dslash
  */
  template<int Ls_>
  struct Dslash5TensorCoreArg {
    typedef typename colorspinor_mapper<short, 4, 3>::type F;
    typedef typename mapper<short>::type real;

    F out;                  // output vector field
    const F in;             // input vector field
    const F x;              // auxiliary input vector field
    const int nParity;      // number of parities we're working on
    const int volume_cb;    // checkerboarded volume
    const int volume_4d_cb; // 4-d checkerboarded volume
    const int_fastdiv Ls;   // length of 5th dimension

    const real m_f;         // fermion mass parameter
    const real m_5;         // Wilson mass shift

    const bool dagger;      // dagger
    const bool xpay;        // whether we are doing xpay or not

    real b;                 // real constant Mobius coefficient
    real c;                 // real constant Mobius coefficient
    real a;                 // real xpay coefficient

    Dslash5Type type;

    Dslash5TensorCoreArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
        double m_f, double m_5, const Complex *b_5_, const Complex *c_5_, double a_, bool dagger, Dslash5Type type)
      : out(out), in(in), x(x), nParity(in.SiteSubset()),
      volume_cb(in.VolumeCB()), volume_4d_cb(volume_cb/Ls_), Ls(Ls_),
      m_f(m_f), m_5(m_5), a(a_), dagger(dagger), xpay(a_ == 0.0 ? false : true), type(type)
    {
      if(in.Nspin() != 4){
        errorQuda("nSpin = %d not support", in.Nspin());
      }
      
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      if (sizeof(coeff_5<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(coeff_5<real>));
      coeff_5<real> *coeff = reinterpret_cast<coeff_5<real>*>(&mobius_h);
      auto *a_5 =  coeff->a;
      auto *b_5 =  coeff->b;
      auto *c_5 =  coeff->c;

      switch(type){
        case M5_INV_MOBIUS:
          b = -(c_5_[0].real() * (4.0 + m_5) - 1.0) / (b_5_[0].real() * (4.0 + m_5) + 1.0);
          c = 0.5 / ( 1.0 + std::pow(b,(int)Ls) * m_f );
          a *= pow(0.5 / (b_5_[0].real() * (m_5 + 4.0) + 1.0), 2);
          break;
        default:
          errorQuda("Unknown Dslash5Type %d", type);
      }

      cudaMemcpyToSymbolAsync(mobius_d, mobius_h, sizeof(coeff_5<real>), 0, cudaMemcpyHostToDevice, streams[Nstream-1]);

    }
  };

  __device__ inline half __half_max_abs_half2__(half max, const half2 input){
    static constexpr uint32_t mask = ~((0x1u<<31) + (0x1u<<15)); // 01111111 11111111 01111111 11111111 
    // Set the fisrt bit of the half to 0.
    uint32_t input_masked = *reinterpret_cast<const uint32_t*>(&input) & mask;
    
    max = __hgt(max, reinterpret_cast<half2*>(&input_masked)->x) ? max : reinterpret_cast<half2*>(&input_masked)->x;
    max = __hgt(max, reinterpret_cast<half2*>(&input_masked)->y) ? max : reinterpret_cast<half2*>(&input_masked)->y;
//    max = __hgt(max, __hgt(input.x, (half)0.0f)?input.x:__hneg(input.x)) ? max : __hgt(input.x, (half)0.0f)?input.x:__hneg(input.x);
//    max = __hgt(max, __hgt(input.x, (half)0.0f)?input.y:__hneg(input.y)) ? max : __hgt(input.y, (half)0.0f)?input.y:__hneg(input.y);
    return max;
  }

  /**
    @brief Tensor core kernel for applying the M5inv operator
    @param[in] arg Argument struct containing any meta data and accessors
  */
  template<int block_dim_x, int Ls_, bool dagger, bool xpay, class Arg>
//  __global__ __launch_bounds__(1280, 4) void dslash5inv_tensor_core(Arg arg)
  __global__ void dslash5inv_tensor_core(Arg arg)
  {
    typedef ColorSpinor<float,3,4> Vector;
    
    float scale;

    TensorCoreSharedMemory<half2> shared_memory_data;
    
    constexpr int M = 4*Ls_;
    constexpr int N = 6*block_dim_x;
    
    constexpr int sm_m_pad_size = 0;
    constexpr int sm_n_pad_size = 16;
    
    constexpr int N_sm = N + sm_n_pad_size;
    constexpr int M_sm = M + sm_m_pad_size;
    
    half2* sm_b = shared_memory_data;
    half*  sm_c = (half*)sm_b;
    half*  sm_a = sm_c+M*N_sm;

    { // Construct matrix A
      const auto k = arg.b;
      const auto inv = arg.c;
   
      int offset_k = threadIdx.y*4;
      int offset_m = threadIdx.x*4;
    
      if(threadIdx.x < Ls_){
        int exp;
        exp = threadIdx.x>threadIdx.y ? Ls_-threadIdx.x+threadIdx.y : threadIdx.y-threadIdx.x;
        float factorR = inv*__fast_pow(k, exp) * ( threadIdx.x>threadIdx.y ? -arg.m_f : static_cast<float>(1.0) );
        exp = threadIdx.x<threadIdx.y ? Ls_-threadIdx.y+threadIdx.x : threadIdx.x-threadIdx.y;
        float factorL = inv*__fast_pow(k, exp) * ( threadIdx.x<threadIdx.y ? -arg.m_f : static_cast<float>(1.0) );
   
        sm_a[ (offset_k+0)*(M_sm)+(offset_m+0) ] = factorR + factorL;
        sm_a[ (offset_k+0)*(M_sm)+(offset_m+1) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+0)*(M_sm)+(offset_m+2) ] = factorR - factorL;
        sm_a[ (offset_k+0)*(M_sm)+(offset_m+3) ] = static_cast<half>(0.0f);
        
        sm_a[ (offset_k+1)*(M_sm)+(offset_m+0) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+1)*(M_sm)+(offset_m+1) ] = factorR + factorL;
        sm_a[ (offset_k+1)*(M_sm)+(offset_m+2) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+1)*(M_sm)+(offset_m+3) ] = factorR - factorL;
        
        sm_a[ (offset_k+2)*(M_sm)+(offset_m+0) ] = factorR - factorL;
        sm_a[ (offset_k+2)*(M_sm)+(offset_m+1) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+2)*(M_sm)+(offset_m+2) ] = factorR + factorL;
        sm_a[ (offset_k+2)*(M_sm)+(offset_m+3) ] = static_cast<half>(0.0f);
        
        sm_a[ (offset_k+3)*(M_sm)+(offset_m+0) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+3)*(M_sm)+(offset_m+1) ] = factorR - factorL;
        sm_a[ (offset_k+3)*(M_sm)+(offset_m+2) ] = static_cast<half>(0.0f);
        sm_a[ (offset_k+3)*(M_sm)+(offset_m+3) ] = factorR + factorL; 
      }
    
    } // Construct matrix A
    
    __syncthreads();
   
    bool idle = false;
    int s4_base = blockIdx.x*blockDim.x; // base.
    int s4, sid;
   
//    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag[tm_dim];

//    #pragma unroll
//    for( int k = 0; k < tm_dim; k++ ){
//      const int a_row = warp_m*WMMA_M;
//      const int a_col = k*WMMA_K;
//      // Load Matrix
//      nvcuda::wmma::load_matrix_sync(a_frag[k], sm_a+a_row+a_col*M_sm, M_sm);
//    } 
   
    while(s4_base < arg.volume_4d_cb){
      
      s4 = s4_base + threadIdx.x;
      sid = threadIdx.y*arg.volume_4d_cb + s4;
      
      if (s4 >= arg.volume_4d_cb){
        idle = true;
      }
    
      if(!idle){
        // Vector in = arg.in(sid, 0);  
        
        scale = tex1Dfetch<float>(arg.in.texNorm, sid);
        
//        const half2 fmvinv2_ = __half2half2( __hdiv((half)1.0f, fixedMaxValue<short>::value) );

//        int offset_pre_m;
//        int offset_pre_n = 6*threadIdx.x;
        
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
   
//        offset_pre_m = (threadIdx.y*4+0)*N_sm;
//        sm_b[ (offset_pre_m+offset_pre_n)/2+0 ] = __floats2half2_rn(in(0, 0).real()*scale, in(0, 0).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+1 ] = __floats2half2_rn(in(0, 1).real()*scale, in(0, 1).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+2 ] = __floats2half2_rn(in(0, 2).real()*scale, in(0, 2).imag()*scale);
//        offset_pre_m = (threadIdx.y*4+1)*N_sm;
//        sm_b[ (offset_pre_m+offset_pre_n)/2+0 ] = __floats2half2_rn(in(1, 0).real()*scale, in(1, 0).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+1 ] = __floats2half2_rn(in(1, 1).real()*scale, in(1, 1).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+2 ] = __floats2half2_rn(in(1, 2).real()*scale, in(1, 2).imag()*scale);
//        offset_pre_m = (threadIdx.y*4+2)*N_sm;
//        sm_b[ (offset_pre_m+offset_pre_n)/2+0 ] = __floats2half2_rn(in(2, 0).real()*scale, in(2, 0).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+1 ] = __floats2half2_rn(in(2, 1).real()*scale, in(2, 1).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+2 ] = __floats2half2_rn(in(2, 2).real()*scale, in(2, 2).imag()*scale);
//        offset_pre_m = (threadIdx.y*4+3)*N_sm;
//        sm_b[ (offset_pre_m+offset_pre_n)/2+0 ] = __floats2half2_rn(in(3, 0).real()*scale, in(3, 0).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+1 ] = __floats2half2_rn(in(3, 1).real()*scale, in(3, 1).imag()*scale);
//        sm_b[ (offset_pre_m+offset_pre_n)/2+2 ] = __floats2half2_rn(in(3, 2).real()*scale, in(3, 2).imag()*scale);
      }
      
      __syncthreads();
    
      { // wmma.h
        using namespace nvcuda;
 
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        
        constexpr int tm_dim = M/WMMA_M;
        constexpr int tn_dim = N/WMMA_N;
        
        constexpr int total_warp = block_dim_x*Ls_/32;
        const int this_warp = (threadIdx.y*block_dim_x+threadIdx.x)/32;
        
        constexpr int total_tile = tm_dim*tn_dim;
        
        constexpr int warp_cycle = total_tile/total_warp;
        const int warp_m = this_warp*warp_cycle/tn_dim;
   
        for(int c = 0; c < warp_cycle; c++){
          // Set up the wmma stuff
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
          // The logical warp assigned to each part of the matrix.
          const int phys_warp_index = this_warp*warp_cycle+c;
          const int warp_n = phys_warp_index-warp_m*tn_dim;
          
          // Zero the initial acc.
          wmma::fill_fragment(c_frag, static_cast<half>(0.0f));
          
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
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
          } 
    
          int c_row = warp_m*WMMA_M;
          int c_col = warp_n*WMMA_N;
    
          nvcuda::wmma::store_matrix_sync(sm_c+c_col+c_row*N_sm, c_frag, N_sm, wmma::mem_row_major);
        }
        
      } // wmma.h
      
      __syncthreads();
    
      if(!idle){
#if 1        
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
        
        sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+0 ] = __hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+1 ] = __hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+2 ] = __hmul2(sm_b[ (threadIdx.y*4+0)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+0 ] = __hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+1 ] = __hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+2 ] = __hmul2(sm_b[ (threadIdx.y*4+1)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+0 ] = __hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+1 ] = __hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+2 ] = __hmul2(sm_b[ (threadIdx.y*4+2)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+0 ] = __hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+0 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+1 ] = __hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+1 ], max_short_div_max2_); 
        sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+2 ] = __hmul2(sm_b[ (threadIdx.y*4+3)*N_sm_d2+3*threadIdx.x+2 ], max_short_div_max2_); 
        
        out[sid + 0*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 0 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 1 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 2 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 3 ]) );
        out[sid + 1*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 4 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+0)*N_sm + 6*threadIdx.x + 5 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 0 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 1 ]) );
        out[sid + 2*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 2 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 3 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 4 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+1)*N_sm + 6*threadIdx.x + 5 ]) );
        out[sid + 3*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 0 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 1 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 2 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 3 ]) );
        out[sid + 4*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 4 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+2)*N_sm + 6*threadIdx.x + 5 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 0 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 1 ]) );
        out[sid + 5*arg.volume_cb] = make_short4( __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 2 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 3 ]), 
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 4 ]),
                                                  __half2short_rn(sm_c[ (threadIdx.y*4+3)*N_sm + 6*threadIdx.x + 5 ]) );
#else        
        Vector out;
        int offset_pre_m;
        int offset_pre_n = 6*threadIdx.x;

        offset_pre_m = (threadIdx.y*4+0)*N_sm;
        out(0, 0) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+0 ]) )/scale;
        out(0, 1) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+1 ]) )/scale;
        out(0, 2) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+2 ]) )/scale;
        offset_pre_m = (threadIdx.y*4+1)*N_sm;
        out(1, 0) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+0 ]) )/scale;
        out(1, 1) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+1 ]) )/scale;
        out(1, 2) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+2 ]) )/scale;
        offset_pre_m = (threadIdx.y*4+2)*N_sm;
        out(2, 0) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+0 ]) )/scale;
        out(2, 1) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+1 ]) )/scale;
        out(2, 2) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+2 ]) )/scale;
        offset_pre_m = (threadIdx.y*4+3)*N_sm;
        out(3, 0) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+0 ]) )/scale;
        out(3, 1) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+1 ]) )/scale;
        out(3, 2) = *reinterpret_cast<complex<float>*>(&__half22float2(sm_b[ (offset_pre_m+offset_pre_n)/2+2 ]) )/scale;
        
        // write spinor field back to device memory
        arg.out(sid, 0) = out;
#endif
      }
    
      s4_base += gridDim.x*blockDim.x;
    
    } // while

  }

  template<int Ls_, class Arg>
  class Dslash5TensorCore : public TunableVectorYZ {

    protected:
      Arg &arg;
      const ColorSpinorField &meta;
      static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

      /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
      static constexpr bool var_inverse = true;

      long long flops() const {
        long long Ls = Ls_;
        long long bulk = (Ls-2)*(meta.Volume()/Ls);
        long long wall = 2*meta.Volume()/Ls;
        long long n = meta.Ncolor() * meta.Nspin();

        long long flops_ = 0;
        switch (arg.type) {
          case M5_INV_MOBIUS: // FIXME flops
            //flops_ = ((2 + 8 * n) * Ls + (arg.xpay ? 4ll : 0)) * meta.Volume();
            flops_ = (144 * Ls + (arg.xpay ? 4ll : 0)) * meta.Volume();
            break;
          default:
            errorQuda("Unknown Dslash5Type %d", arg.type);
        }

        return flops_;
      }

      long long bytes() const {
        long long Ls = meta.X(4);
        switch (arg.type) {
          case M5_INV_MOBIUS:
            return arg.out.Bytes() + Ls*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
          default: 
            errorQuda("Unknown Dslash5Type %d", arg.type);
        }
        return 0ll;
      }

      virtual bool tuneGridDim() const { return true; }
//      virtual bool tuneSharedBytes() const { return false; }
      unsigned int minThreads() const { return arg.volume_4d_cb; }
  
      unsigned int shared_bytes_per_block(int x, int y) const { 
        // (Ls*4) by (Ls*4), (Ls*4) by (volume_4d*6 + 16)
        return ( (y*4)*(y*4+0)+(y*4)*(x*6+16) )*2; // 4*4*2 TODO: fix this!
      }
   
      virtual bool advanceBlockDim(TuneParam &param) const
      {
        if(param.block.x < 64){
          param.block.x += 8;
          param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
          return true;
        }else{
          return false;
        }
      }
      
      virtual bool advanceGridDim(TuneParam &param) const
      {
        const unsigned int max_blocks = maxGridSize();
        const int step = deviceProp.multiProcessorCount;
        param.grid.x += step;
        if (param.grid.x > max_blocks) {
          param.grid.x = minGridSize();
          return false;
        } else {
          param.block.x = 16;
          param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
          return true;
        }
      }

      virtual unsigned int maxGridSize() const { return 32*deviceProp.multiProcessorCount; }

      // overloaded to return max dynamic shared memory if doing shared-memory inverse
      unsigned int maxSharedBytesPerBlock() const {
        if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS) ) {
          return maxDynamicSharedBytesPerBlock();
        } else {
          return TunableVectorYZ::maxSharedBytesPerBlock();
        }
      }

    public:
      Dslash5TensorCore(Arg &arg, const ColorSpinorField &meta)
        : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
      {
        strcpy(aux, meta.AuxString());
        if (arg.dagger) strcat(aux, ",Dagger");
        if (arg.xpay) strcat(aux,",xpay");
        switch (arg.type) {
          case M5_INV_MOBIUS:
            strcat(aux, ",m5inv_mobius_tensor_core");
            break;
          default: 
            errorQuda("Unknown Dslash5Type %d", arg.type);
        }
      }
      virtual ~Dslash5TensorCore() { }

      template<typename T>
      inline void launch(T *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
        if ( shared && arg.type == M5_INV_MOBIUS ) {
          // if inverse kernel uses shared memory then maximize total shared memory pool
          setMaxDynamicSharedBytesPerBlock(f);
        }
        void *args[] = { &arg };
        qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
      }

      void apply(const cudaStream_t &stream) {
        // By its name we ONLY have a GPU version
        // TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_DEBUG_VERBOSE);
        switch(arg.type){
          case M5_INV_MOBIUS:
            switch(tp.block.x){
              case 16:
                if (arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<16, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<16, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{          
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<16, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<16, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 24:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<24, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<24, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<24, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<24, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 32:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<32, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<32, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<32, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<32, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 40:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<40, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<40, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<40, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<40, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 48:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<48, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<48, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<48, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<48, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 56:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<56, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<56, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<56, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<56, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              case 64:
                if(arg.xpay){ 
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<64, Ls_, true, true, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<64, Ls_,false, true, Arg>, tp, arg, stream) ;
                }else{
                  arg.dagger ?
                  launch(dslash5inv_tensor_core<64, Ls_, true,false, Arg>, tp, arg, stream) :
                  launch(dslash5inv_tensor_core<64, Ls_,false,false, Arg>, tp, arg, stream) ;
                }
                break;
              default:
                errorQuda("NOT valid blockDim.x(=%d)\n", tp.block.x);
            }
          break;
          default: 
            errorQuda("Unknown Dslash5Type %d", arg.type);
        }
      }

      void initTuneParam(TuneParam &param) const {
        TunableVectorYZ::initTuneParam(param);
        param.block = dim3(32, arg.Ls, 1); // Ls must be contained in the block
        param.grid = dim3(80, 1, 1);
        param.shared_bytes = shared_bytes_per_block(param.block.x, param.block.y); 
      }

      void defaultTuneParam(TuneParam &param) const {
        initTuneParam(param);
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

#endif

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5 * in
  
  void apply_dslash5_tensor_core(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
      double m_f, double m_5, const Complex* b_5, const Complex* c_5, double a, bool dagger, Dslash5Type type)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.DWFPCtype() != QUDA_4D_PC) errorQuda("ONLY 4D preconditioned fields are supported");
    checkLocation(out, in);     // check all locations match
  
    if( checkPrecision(out, in) == QUDA_HALF_PRECISION && in.Ncolor() == 3){
      // switch for Ls
      switch(in.X(4)){
        case 12:
          {
            Dslash5TensorCoreArg<12> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
            Dslash5TensorCore<12, Dslash5TensorCoreArg<12> > dslash(arg, in);
            dslash.apply(streams[Nstream-1]);
          }
        break;
        default: 
          errorQuda("Ls = %d is NOT supported.\n", in.X(4));
      }
    }else{
      errorQuda("Tensor core implemtation ONLY supports HALF precision and n_color = 3.\n");
    }
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

} // namespace quda

