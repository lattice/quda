#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <inline_ptx.h>

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

  /**
     @brief Structure containing zMobius / Zolotarev coefficients
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
     @brief Parameter structure for applying the Dslash
   */
  template <typename Float, int nColor>
  struct Dslash5Arg {
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;
    typedef typename mapper<Float>::type real;

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

    Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
               double m_f, double m_5, const Complex *b_5_, const Complex *c_5_,
               double a, bool dagger, Dslash5Type type)
      : out(out), in(in), x(x), nParity(in.SiteSubset()),
	volume_cb(in.VolumeCB()), volume_4d_cb(volume_cb/in.X(4)), Ls(in.X(4)),
	m_f(m_f), m_5(m_5), a(a), dagger(dagger), xpay(a == 0.0 /*in.V() == x.V()*/ ? false : true), type(type)
    {
      if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      if (sizeof(coeff_5<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(coeff_5<real>));
      coeff_5<real> *coeff = reinterpret_cast<coeff_5<real>*>(&mobius_h);
      auto *a_5 =  coeff->a;
      auto *b_5 =  coeff->b;
      auto *c_5 =  coeff->c;

      switch(type) {
      case DSLASH5_DWF:
	break;
      case DSLASH5_MOBIUS_PRE:
	for (int s=0; s<Ls; s++) {
	  b_5[s] = b_5_[s];
	  c_5[s] = 0.5*c_5_[s];

	  // xpay
	  a_5[s] = 0.5/(b_5_[s]*(m_5+4.0) + 1.0);
	  a_5[s] *= a_5[s] * static_cast<real>(a);
        }
	break;
      case DSLASH5_MOBIUS:
	for (int s=0; s<Ls; s++) {
	  b_5[s] = 1.0;
	  c_5[s] = 0.5 * (c_5_[s] * (m_5 + 4.0) - 1.0) / (b_5_[s] * (m_5 + 4.0) + 1.0);

	  // axpy
	  a_5[s] = 0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0);
	  a_5[s] *= a_5[s] * static_cast<real>(a);
	}
	break;
      case M5_INV_DWF:
        b = 2.0 * (0.5/(5.0 + m_5)); // 2  * kappa_5
        c = 0.5 / ( 1.0 + std::pow(b,(int)Ls) * m_f );
        break;
      case M5_INV_MOBIUS:
        {
          complex<double> k = 1.0;
          for (int s=0; s<Ls; s++) {
            b_5[s] = -(c_5_[s] * (4.0 + m_5) - 1.0) / (b_5_[s] * (4.0 + m_5) + 1.0);
            k *= b_5[s];
          }

          for (int s=0; s<Ls; s++) {
            c_5[s] = 0.5 / ( 1.0 + k * m_f );

            a_5[s] = 0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0);
            a_5[s] *= a_5[s] * static_cast<real>(a);
          }
        }
        break;
      default:
	errorQuda("Unknown Dslash5Type %d", type);
      }

      cudaMemcpyToSymbolAsync(mobius_d, mobius_h, sizeof(coeff_5<real>), 0, cudaMemcpyHostToDevice, streams[Nstream-1]);

    }
  };

  template <typename real>
  inline __device__ __host__ const coeff_5<real>* coeff() {
#ifdef __CUDA_ARCH__
    return reinterpret_cast<const coeff_5<real>*>(mobius_d);
#else
    return reinterpret_cast<const coeff_5<real>*>(mobius_h);
#endif
  }

  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __device__ __host__ inline void dslash5(Arg &arg, int parity, int x_cb, int s) {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,4> Vector;

    Vector out;

    { // forwards direction
      const int fwd_idx = ((s + 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(fwd_idx, parity);
      constexpr int proj_dir = dagger ? +1 : -1;
      if (s == arg.Ls-1) {
	out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
	out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    { // backwards direction
      const int back_idx = ((s + arg.Ls - 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(back_idx, parity);
      constexpr int proj_dir = dagger ? -1 : +1;
      if (s == 0) {
	out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
	out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    if (type == DSLASH5_DWF && xpay) {
      Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
      out = x + arg.a*out;
    } else if (type == DSLASH5_MOBIUS_PRE) {
      Vector diagonal = arg.in(s*arg.volume_4d_cb + x_cb, parity);
      auto *z = coeff<real>();
      out = z->c[s] * out + z->b[s] * diagonal;

      if (xpay) {
	Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
	out = x + z->a[s] * out;
      }
    } else if (type == DSLASH5_MOBIUS) {
      Vector diagonal = arg.in(s*arg.volume_4d_cb + x_cb, parity);
      auto *z = coeff<real>();
      out = z->c[s] * out + diagonal;

      if (xpay) { // really axpy
	Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
	out = z->a[s] * x + out;
      }
    }

    arg.out(s*arg.volume_4d_cb + x_cb, parity) = out;
  }

  // CPU kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  void dslash5CPU(Arg &arg)
  {
    for (int parity= 0; parity < arg.nParity; parity++) {
      for (int s=0; s < arg.Ls; s++) {
	for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) { // 4-d volume
	  dslash5<Float,nColor,dagger,xpay,type>(arg, parity, x_cb, s);
	}  // 4-d volumeCB
      } // ls
    } // parity

  }

  // GPU Kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __global__ void dslash5GPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    int parity = blockIdx.z*blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5<Float,nColor,dagger,xpay,type>(arg, parity, x_cb, s);
  }

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template<typename real>
  __device__ __host__ inline real __fast_pow(real a, int b) {
#ifdef __CUDA_ARCH__
    if (sizeof(real) == sizeof(double)) {
      return pow(a, b);
    } else {
      float sign = signbit(a) ? -1.0f : 1.0f;
      float power = __powf(fabsf(a), b);
      return b&1 ? sign * power : power;
    }
#else
    return std::pow(a, b);
#endif
  }

  /**
     @brief Class which wraps around a shared memory cache for a Vector type,
     where each thread in the thread block stores a unique
     Vector in the cache which any other thread can access.
   */
  template <typename real, typename Vector>
  class VectorCache {

    /**
       @brief This is the handle to the shared memory
       @return Shared memory pointer
     */
    __device__ inline real* cache() {
      extern __shared__ int cache_[];
      return reinterpret_cast<real*>(cache_);
    }

  public:

    /**
       @brief Save the vector into the 3-d shared memory cache.
       Implicitly store the vector at coordinates given by threadIdx.
       @param[in] a The vector to store in the shared memory cache
     */
    __device__ inline void save(const Vector &a) {
#pragma unroll
      for (int i=0; i<2*a.n; i++) {
        int j = ((i*blockDim.z + threadIdx.z)*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
        cache()[j] = *(reinterpret_cast<const real*>(a.data) + i);
      }
      __syncthreads();
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The Vector at coordinates (x,y,z)
     */
    __device__ inline Vector load(int x, int y, int z) {
      Vector a;
#pragma unroll
      for (int i=0; i<2*a.n; i++) {
        int j = ((i*blockDim.z + z)*blockDim.y + y)*blockDim.x + x;
        *(reinterpret_cast<real*>(a.data) + i) = cache()[j];
      }
      return a;
    }

  };

  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, typename Arg>
  __device__ __host__ inline void dslash5inv(Arg &arg, int parity, int x_cb, int s_) {
    constexpr int nSpin = 4;
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,nSpin> Vector;

    Vector out;
    auto *z = coeff<real>();
    const auto k = type == M5_INV_DWF ? arg.b : z->b[s_].real();
    const auto inv_d_n = type == M5_INV_DWF ? arg.c : z->c[s_].real();

    // if using shared-memory caching then load spinor field for my site into cache
    VectorCache<real,Vector> cache;
    if (shared) cache.save(arg.in(s_*arg.volume_4d_cb + x_cb, parity));

#if 0
    int s = s_;
    //real R = inv_d_n * 1.0;
    //real L = inv_d_n * 1.0;
    for (int s_count = 0; s_count<arg.Ls; s_count++) {

      Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s*arg.volume_4d_cb + x_cb, parity);

      {
        int exp = (arg.Ls - s_count) % arg.Ls; //s_ < s ? arg.Ls-s+s_ : s_-s;
        real factorR = inv_d_n * __fast_pow(k,exp) * ( s_ < s ? -arg.m_f : static_cast<real>(1.0) );
        constexpr int proj_dir = dagger ? -1 : +1;
        out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

      {
        int exp = s_count;//s_ > s ? arg.Ls-s_+s : s-s_;
        real factorL = inv_d_n * __fast_pow(k,exp) * ( s_ > s ? -arg.m_f : static_cast<real>(1.0));
        constexpr int proj_dir = dagger ? +1 : -1;
        out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

      s = (s + 1)%arg.Ls;
    }
#else
    for (int s=0; s<arg.Ls; s++) {

      Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s*arg.volume_4d_cb + x_cb, parity);

      {
        int exp = s_ < s ? arg.Ls-s+s_ : s_-s;
        real factorR = inv_d_n * __fast_pow(k,exp) * ( s_ < s ? -arg.m_f : static_cast<real>(1.0) );
        constexpr int proj_dir = dagger ? -1 : +1;
        out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

      {
        int exp = s_ > s ? arg.Ls-s_+s : s-s_;
        real factorL = inv_d_n * __fast_pow(k,exp) * ( s_ > s ? -arg.m_f : static_cast<real>(1.0));
        constexpr int proj_dir = dagger ? +1 : -1;
        out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

    }
#endif

    if (xpay) {
      Vector x = arg.x(s_*arg.volume_4d_cb + x_cb, parity);
      if (type == M5_INV_DWF) {
        out = x + arg.a*out;
      } else if (type == M5_INV_MOBIUS) {
        auto *z = coeff<real>();
        out = x + z->a[s_] * out;
      }
    }

    arg.out(s_*arg.volume_4d_cb + x_cb, parity) = out;
  }

  // CPU kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  void dslash5invCPU(Arg &arg)
  {
    constexpr bool shared = false; // shared memory doesn't apply here
    for (int parity= 0; parity < arg.nParity; parity++) {
      for (int s=0; s < arg.Ls; s++) {
	for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) { // 4-d volume
	  dslash5inv<Float,nColor,dagger,xpay,type,shared>(arg, parity, x_cb, s);
	}  // 4-d volumeCB
      } // ls
    } // parity

  }

  // GPU Kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, typename Arg>
  __global__ void dslash5invGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    int parity = blockIdx.z*blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5inv<Float,nColor,dagger,xpay,type,shared>(arg, parity, x_cb, s);
  }
/*  
  // Tensor core kernel for applying the dslash operator
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, typename Arg, int Ls>
  __global__ void dslash5inv_tensor_core(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    int parity = blockIdx.z*blockDim.z + threadIdx.z;
    
    float scale = 1e4f;
    
    MDWFSharedMemory<float4> sm_data;
    
    constexpr int M = Ls*4;
    constexpr int K = Ls*4;
    const int N = 6*blockDim.x;
    
    constexpr int sm_m_pad_size = 8;
    constexpr int sm_n_pad_size = 8;
    
    const int N_sm = N + sm_n_pad_size;
    constexpr int M_sm = M + sm_m_pad_size;
    
    half* sm_b = (half*)((void*)sm_data);
    half* sm_c = (half*)(sm_b + K*(N_sm));
    half* sm_a = (half*)(sm_c + M*(N_sm));
    
    { // Construct matrix A
    
      int offset_k = threadIdx.y*4;
      int one_or_zero = threadIdx.x >> 4; // Assumming blockDim.x = 32
      int xxx = threadIdx.x-one_or_zero*16;
      int offset_m = xxx*4;
    
      // threadIdx.x should not be idle(?).
      // With Ls=12 and blockDim.x=32 the following gives a 2-way bank conflict.  
      if(xxx < Ls && threadIdx.x < 32){
    
    #ifdef MDWF_mode   // Check whether MDWF option is enabled
        half kappa = -(static_cast<half>(mdwf_c5[ xxx ])*(static_cast<half>(4.0) + static_cast<half>(m5)) - static_cast<half>(1.0))/(static_cast<half>(mdwf_b5[ xxx ])*(static_cast<half>(4.0) + static_cast<half>(m5)) + static_cast<half>(1.0));
    #else
        half kappa = static_cast<half>(2.0)*static_cast<half>(a);
    #endif  // select MDWF mode
    
        half inv_d_n = static_cast<half>(0.5) / ( static_cast<half>(1.0) + static_cast<half>(POW(kappa,param.dc.Ls))*static_cast<half>(mferm) );
        half factorR;
        half factorL;
    
        int exponent = xxx  > threadIdx.y ? param.dc.Ls-xxx+threadIdx.y : threadIdx.y-xxx;
        factorR = inv_d_n * static_cast<half>(POW(kappa,exponent))  * ( xxx > threadIdx.y ? static_cast<half>(-mferm) : static_cast<half>(1.0) );
        int exponent2 = xxx < threadIdx.y ? param.dc.Ls-threadIdx.y+xxx : xxx-threadIdx.y;
        factorL = inv_d_n * static_cast<half>(POW(kappa,exponent2)) * ( xxx < threadIdx.y ? static_cast<half>(-mferm) : static_cast<half>(1.0) );
        // (mu, s) by (nu, t). column-major.
    
        sm_a[ (offset_k+0)*M_sm+(offset_m+0+one_or_zero*2) ] = factorR + static_cast<half>(-2*one_or_zero+1)*factorL;
        //    sm_a[ (offset_k+0)*M_sm+(offset_m+2) ] = factorR - factorL;
    
        sm_a[ (offset_k+1)*M_sm+(offset_m+1+one_or_zero*2) ] = factorR + static_cast<half>(-2*one_or_zero+1)*factorL;
        //    sm_a[ (offset_k+1)*M_sm+(offset_m+3) ] = factorR - factorL;
    
        sm_a[ (offset_k+2)*M_sm+(offset_m+0+one_or_zero*2) ] = factorR + static_cast<half>(+2*one_or_zero-1)*factorL;
        //    sm_a[ (offset_k+2)*M_sm+(offset_m+2) ] = factorR + factorL;
    
        sm_a[ (offset_k+3)*M_sm+(offset_m+1+one_or_zero*2) ] = factorR + static_cast<half>(+2*one_or_zero-1)*factorL;
        //    sm_a[ (offset_k+3)*M_sm+(offset_m+3) ] = factorR + factorL;
    
        sm_a[ (offset_k+0)*M_sm+(offset_m+1+one_or_zero*2) ] = static_cast<half>(0.);
        //    sm_a[ (offset_k+0)*M_sm+(offset_m+3) ] = static_cast<half>(0.0f);
    
        sm_a[ (offset_k+1)*M_sm+(offset_m+0+one_or_zero*2) ] = static_cast<half>(0.);
        //    sm_a[ (offset_k+1)*M_sm+(offset_m+2) ] = static_cast<half>(0.0f);
    
        sm_a[ (offset_k+2)*M_sm+(offset_m+1+one_or_zero*2) ] = static_cast<half>(0.);
        //    sm_a[ (offset_k+2)*M_sm+(offset_m+3) ] = static_cast<half>(0.0f);
    
        sm_a[ (offset_k+3)*M_sm+(offset_m+0+one_or_zero*2) ] = static_cast<half>(0.);
        //    sm_a[ (offset_k+3)*M_sm+(offset_m+2+one_or_zero*2) ] = static_cast<half>(0.0f);
    
      }
    
    } // Construct matrix A
    
    __syncthreads();
       
    int s4_base = blockIdx.x*blockDim.x; // base.
    int s5 = blockIdx.y*blockDim.y+threadIdx.y;
    int s4_increment = gridDim.x*blockDim.x;
    
    int s4, sid;
    int X, coord[5], boundaryCrossing;
    
    // wmma.h
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    constexpr int tm_dim = M / WMMA_M;
    constexpr int tk_dim = K / WMMA_K;
    const int tn_dim = N >> 4;
    
    // Set up the wmma stuff
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag[tk_dim];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
    // The actual/physical warp assigned to each thread in this block
    //int phys_warp_n_dim = float(blockDim.x)/float(warpSize); // TODO: should make sure blockDim.x is AT LEAST 32.
    //int phys_warp_m_dim = blockDim.y;
    
    const int total_warp = (blockDim.x*blockDim.y) >> 5;
    const int this_warp = (threadIdx.y*blockDim.x+threadIdx.x) >> 5;
    
    //int phys_warp_n = float(threadIdx.x)/float(warpSize);
    //int phys_warp_m = threadIdx.y; 
    
    //int total_num_warp = phys_warp_n_dim*phys_warp_m_dim;
    const int total_tile = tm_dim*tn_dim;
    
    const int warp_cycle = float(total_tile)/total_warp;
    
    const int warp_m = float(this_warp)*warp_cycle/tn_dim;
    
    #pragma unroll
    for(int k = 0; k < tk_dim; k++){
      const int a_row = warp_m*WMMA_M;
      const int a_col = k*WMMA_K;
      nvcuda::wmma::load_matrix_sync(a_frag[k], sm_a+a_row+a_col*M_sm, M_sm);
    }
    
    while(s4_base < param.threads){
    
      s4 = s4_base+threadIdx.x;
      sid = s5*param.threads+s4;
      
      if (s4 >= param.threads){
        idle = true;
      }
    
      if(!idle){
    
        if( param.partial_length ){
          coordsFromIndexShrinked<5,QUDA_4D_PC,EVEN_X>(X, coord, sid, param);
        }else{
    
          boundaryCrossing = sid/param.dc.Xh[0] + sid/(param.dc.X[1]*param.dc.Xh[0]) + sid/(param.dc.X[2]*param.dc.X[1]*param.dc.Xh[0]);
    
          X = 2*sid + (boundaryCrossing + param.parity) % 2;
          coord[4] = X/(param.dc.X[0]*param.dc.X[1]*param.dc.X[2]*param.dc.X[3]);
    
        }
    
        //  boundaryCrossing = sid/param.dc.Xh[0] + sid/(param.dc.X[1]*param.dc.Xh[0]) + sid/(param.dc.X[2]*param.dc.X[1]*param.dc.Xh[0]);
        //  
        //  X = 2*sid + (boundaryCrossing + param.parity) % 2;
        //  coord[4] = X/(param.dc.X[0]*param.dc.X[1]*param.dc.X[2]*param.dc.X[3]);
    
        READ_SPINOR( SPINORTEX, param.sp_stride, X/2, X/2 );
    
        int offset_pre_m;
        int offset_pre_n = threadIdx.x*6;
    
        // data layout for tensor core B and C: s, spin, spatial, color, complex; Lsx4 by Lsx4 @ Lsx4 by 6xblockDim.x.
        // lda = Lsx4, column-major
        // ldb = blockDim.x x 6, row-major
        // total number of halves = Ls*24*blockDim.x
        offset_pre_m = (coord[4]*4+0)*N_sm;
        sm_b[ offset_pre_m+offset_pre_n+0 ] = i00_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+1 ] = i00_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+2 ] = i01_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+3 ] = i01_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+4 ] = i02_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+5 ] = i02_im*scale;
        offset_pre_m = (coord[4]*4+1)*N_sm;
        sm_b[ offset_pre_m+offset_pre_n+0 ] = i10_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+1 ] = i10_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+2 ] = i11_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+3 ] = i11_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+4 ] = i12_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+5 ] = i12_im*scale;
        offset_pre_m = (coord[4]*4+2)*N_sm;
        sm_b[ offset_pre_m+offset_pre_n+0 ] = i20_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+1 ] = i20_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+2 ] = i21_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+3 ] = i21_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+4 ] = i22_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+5 ] = i22_im*scale;
        offset_pre_m = (coord[4]*4+3)*N_sm;
        sm_b[ offset_pre_m+offset_pre_n+0 ] = i30_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+1 ] = i30_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+2 ] = i31_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+3 ] = i31_im*scale;
        sm_b[ offset_pre_m+offset_pre_n+4 ] = i32_re*scale;
        sm_b[ offset_pre_m+offset_pre_n+5 ] = i32_im*scale;
    
      }

  __syncthreads();

  // wmma.h
  {
    using namespace nvcuda;

    for(int c = 0; c < warp_cycle; c++){
      int phys_warp_index = this_warp*warp_cycle+c;
      // The logical warp assigned to each part of the matrix.
// TODO: This is moved to other places
//      int warp_m = float(phys_warp_index)/tn_dim;
      const int warp_n = phys_warp_index-warp_m*tn_dim;
      // Zero the initial acc.
      wmma::fill_fragment(c_frag, (half)0.0f);
      #pragma unroll
      for( int k = 0; k < 3; k++ ){

        const int a_row = warp_m*WMMA_M;
        const int a_col = k*WMMA_K;
        const int b_row = k*WMMA_K;
        const int b_col = warp_n*WMMA_N;

        //    if( a_row < M && a_col < K && b_row < K && b_col < N ){    
        // Load Matrix
        // wmma::load_matrix_sync(a_frag, sm_a+a_row+a_col*M_sm, M_sm);
        wmma::load_matrix_sync(b_frag, sm_b+b_col+b_row*N_sm, N_sm);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag[k], b_frag, c_frag);
        //    }
        //    __syncthreads();
      } 

      //  __syncthreads();

      int c_row = warp_m*WMMA_M;
      int c_col = warp_n*WMMA_N;

      if(c_row < M && c_col < N){ 
        wmma::store_matrix_sync(sm_c+c_col+c_row*N_sm, c_frag, N_sm, wmma::mem_row_major);
      }
      //  __syncthreads();
    }
    __syncthreads();

    o00_re = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+0 ])/scale;
    o00_im = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+1 ])/scale;
    o01_re = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+2 ])/scale;
    o01_im = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+3 ])/scale;
    o02_re = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+4 ])/scale;
    o02_im = float(sm_c[ (coord[4]*4+0)*(N_sm)+threadIdx.x*6+5 ])/scale;
    o10_re = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+0 ])/scale;
    o10_im = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+1 ])/scale;
    o11_re = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+2 ])/scale;
    o11_im = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+3 ])/scale;
    o12_re = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+4 ])/scale;
    o12_im = float(sm_c[ (coord[4]*4+1)*(N_sm)+threadIdx.x*6+5 ])/scale;
    o20_re = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+0 ])/scale;
    o20_im = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+1 ])/scale;
    o21_re = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+2 ])/scale;
    o21_im = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+3 ])/scale;
    o22_re = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+4 ])/scale;
    o22_im = float(sm_c[ (coord[4]*4+2)*(N_sm)+threadIdx.x*6+5 ])/scale;
    o30_re = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+0 ])/scale;
    o30_im = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+1 ])/scale;
    o31_re = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+2 ])/scale;
    o31_im = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+3 ])/scale;
    o32_re = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+4 ])/scale;
    o32_im = float(sm_c[ (coord[4]*4+3)*(N_sm)+threadIdx.x*6+5 ])/scale;

  } // wmma.h

  if(!idle){
    // write spinor field back to device memory
    WRITE_SPINOR(param.sp_stride);
  }

  s4_base += s4_increment;

} // while


    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5inv<Float,nColor,dagger,xpay,type,shared>(arg, parity, x_cb, s);
  }
*/
  template <typename Float, int nColor, typename Arg>
  class Dslash5 : public TunableVectorYZ {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

    long long flops() const {
      long long Ls = meta.X(4);
      long long bulk = (Ls-2)*(meta.Volume()/Ls);
      long long wall = 2*meta.Volume()/Ls;
      int n = meta.Ncolor() * meta.Nspin();
      bool zMobius = true; // set to true when we have complexity

      long long flops_ = 0;
      switch (arg.type) {
      case DSLASH5_DWF:
        flops_ = n * (8ll*bulk + 10ll*wall + (arg.xpay ? 4ll * meta.Volume() : 0) );
        break;
      case DSLASH5_MOBIUS_PRE:
        flops_ = n * (8ll*bulk + 10ll*wall + (zMobius ? 14ll : 6ll) * meta.Volume() +
                      (arg.xpay ? (zMobius ? 8ll : 4ll) * meta.Volume() : 0) );
        break;
      case DSLASH5_MOBIUS:
        flops_ = n * (8ll*bulk + 10ll*wall + (zMobius ? 8ll : 4ll) * meta.Volume() +
                      (arg.xpay ? (zMobius ? 8ll : 4ll) * meta.Volume() : 0) );
        break;
      case M5_INV_DWF:
        flops_ = (n * n * Ls + (arg.xpay ? 4 : 0)) * meta.Volume();
        break;
      case M5_INV_MOBIUS:
        flops_ = (n * n * Ls + (arg.xpay ? 4 : 0)) * meta.Volume();
        break;
      default:
	errorQuda("Unknown Dslash5Type %d", arg.type);
      }

      return flops_;
    }

    long long bytes() const {
      long long Ls = meta.X(4);
      switch (arg.type) {
      case DSLASH5_DWF:        return arg.out.Bytes() + 2*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS_PRE: return arg.out.Bytes() + 3*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS:     return arg.out.Bytes() + 3*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case M5_INV_DWF:         return arg.out.Bytes() + Ls*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case M5_INV_MOBIUS:      return arg.out.Bytes() + Ls*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }
      return 0ll;
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }
    unsigned int sharedBytesPerThread() const {
      if (shared && arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS) {
        return 2*4*nColor*sizeof(typename mapper<Float>::type);
      } else {
        return 0;
      }
    }

  public:
    Dslash5(Arg &arg, const ColorSpinorField &meta)
      : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      if (arg.dagger) strcat(aux, ",Dagger");
      if (arg.xpay) strcat(aux,",xpay");
      switch (arg.type) {
      case DSLASH5_DWF:        strcat(aux, ",DSLASH5_DWF"); break;
      case DSLASH5_MOBIUS_PRE: strcat(aux, ",DSLASH5_MOBIUS_PRE"); break;
      case DSLASH5_MOBIUS:     strcat(aux, ",DSLASH5_MOBIUS"); break;
      case M5_INV_DWF:         strcat(aux, ",M5_INV_DWF"); break;
      case M5_INV_MOBIUS:      strcat(aux, ",M5_INV_MOBIUS"); break;
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }
    }
    virtual ~Dslash5() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.type == DSLASH5_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true,true,DSLASH5_DWF>(arg) :
			  dslash5CPU<Float,nColor,false,true,DSLASH5_DWF>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_DWF>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_DWF>(arg);
	} else if (arg.type == DSLASH5_MOBIUS_PRE) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true, true,DSLASH5_MOBIUS_PRE>(arg) :
			  dslash5CPU<Float,nColor,false, true,DSLASH5_MOBIUS_PRE>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_MOBIUS_PRE>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_MOBIUS_PRE>(arg);
	} else if (arg.type == DSLASH5_MOBIUS) {
	  if (arg.xpay) arg.dagger ?
			  dslash5CPU<Float,nColor, true, true,DSLASH5_MOBIUS>(arg) :
			  dslash5CPU<Float,nColor,false, true,DSLASH5_MOBIUS>(arg);
	  else          arg.dagger ?
			  dslash5CPU<Float,nColor, true,false,DSLASH5_MOBIUS>(arg) :
			  dslash5CPU<Float,nColor,false,false,DSLASH5_MOBIUS>(arg);
	} else if (arg.type == M5_INV_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5invCPU<Float,nColor, true, true,M5_INV_DWF>(arg) :
			  dslash5invCPU<Float,nColor,false, true,M5_INV_DWF>(arg);
	  else          arg.dagger ?
			  dslash5invCPU<Float,nColor, true,false,M5_INV_DWF>(arg) :
			  dslash5invCPU<Float,nColor,false,false,M5_INV_DWF>(arg);
	} else if (arg.type == M5_INV_MOBIUS) {
	  if (arg.xpay) arg.dagger ?
			  dslash5invCPU<Float,nColor, true, true,M5_INV_MOBIUS>(arg) :
			  dslash5invCPU<Float,nColor,false, true,M5_INV_MOBIUS>(arg);
	  else          arg.dagger ?
			  dslash5invCPU<Float,nColor, true,false,M5_INV_MOBIUS>(arg) :
			  dslash5invCPU<Float,nColor,false,false,M5_INV_MOBIUS>(arg);
	}
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.type == DSLASH5_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_DWF> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == DSLASH5_MOBIUS_PRE) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_MOBIUS_PRE> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == DSLASH5_MOBIUS) {
	  if (arg.xpay) arg.dagger ?
			  dslash5GPU<Float,nColor, true, true,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false, true,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5GPU<Float,nColor, true,false,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5GPU<Float,nColor,false,false,DSLASH5_MOBIUS> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == M5_INV_DWF) {
	  if (arg.xpay) arg.dagger ?
			  dslash5invGPU<Float,nColor, true, true,M5_INV_DWF,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5invGPU<Float,nColor,false, true,M5_INV_DWF,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5invGPU<Float,nColor, true,false,M5_INV_DWF,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5invGPU<Float,nColor,false,false,M5_INV_DWF,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	} else if (arg.type == M5_INV_MOBIUS) {
	  // To set the shared memory stuff.
    static bool init = false; 
    if(!init){
			set_shared_memory_on_volta((const void*)dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared, Arg>, 
				"dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared>");
      set_shared_memory_on_volta((const void*)dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared, Arg>, 
				"dslash5invGPU<Float,nColor,false, true,M5_INV_MOBIUS,shared>");
			set_shared_memory_on_volta((const void*)dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared, Arg>, 
				"dslash5invGPU<Float,nColor, true,false,M5_INV_MOBIUS,shared>");
      set_shared_memory_on_volta((const void*)dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared, Arg>, 
				"dslash5invGPU<Float,nColor,false,false,M5_INV_MOBIUS,shared>");
      init = true;
    }
    
    if (arg.xpay) arg.dagger ?
			  dslash5invGPU<Float,nColor, true, true,M5_INV_MOBIUS,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5invGPU<Float,nColor,false, true,M5_INV_MOBIUS,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else          arg.dagger ?
			  dslash5invGPU<Float,nColor, true,false,M5_INV_MOBIUS,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
			  dslash5invGPU<Float,nColor,false,false,M5_INV_MOBIUS,shared> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	}
      }
    }

    void initTuneParam(TuneParam &param) const {
      TunableVectorYZ::initTuneParam(param);
      if ( shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS) ) {
        param.block.y = arg.Ls; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z;
      }
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableVectorYZ::defaultTuneParam(param);
      if ( shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS) ) {
        param.block.y = arg.Ls; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z;
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
    Dslash5Arg<Float,nColor> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
    Dslash5<Float,nColor,Dslash5Arg<Float,nColor> > dslash(arg, in);
    dslash.apply(streams[Nstream-1]);
  }

  // template on the number of colors
  template <typename Float>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
    switch(in.Ncolor()) {
    case 3: ApplyDslash5<Float,3>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

#endif

  //Apply the 5th dimension dslash operator to a colorspinor field
  //out = Dslash5*in
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
		    double m_f, double m_5, const Complex *b_5, const Complex *c_5,
		    double a, bool dagger, Dslash5Type type)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.DWFPCtype() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
    checkLocation(out, in);     // check all locations match

    switch(checkPrecision(out,in)) {
    case QUDA_DOUBLE_PRECISION: ApplyDslash5<double>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_SINGLE_PRECISION: ApplyDslash5<float> (out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_HALF_PRECISION:   ApplyDslash5<short> (out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

} // namespace quda

