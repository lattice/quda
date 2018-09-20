#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <inline_ptx.h>

namespace quda {

#ifdef GPU_DOMAIN_WALL_DIRAC

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

            a_5[s] = 0.5 * (b_5_[s] * (m_5 + 4.0) + 1.0);
            a_5[s]*= a_5[s] * static_cast<real>(a);
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

