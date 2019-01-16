#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <inline_ptx.h>
#include <shared_memory_cache_helper.cuh>
#include <math_helper.cuh>

namespace quda {
  namespace mobius_eofa {

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
    static __constant__ char m5_shift_d[size];
    static __constant__ char m5inv_shift_lc_d[size];
    static __constant__ char m5inv_shift_norm_d[size];
    static __constant__ char m5inv_shift_dagger_lc_d[size];
    static __constant__ char m5inv_shift_dagger_norm_d[size];

    static char m5_shift_h[size];
    static char m5inv_shift_lc_h[size];
    static char m5inv_shift_norm_h[size];
    static char m5inv_shift_dagger_lc_h[size];
    static char m5inv_shift_dagger_norm_h[size];

    /**
      @brief Helper function for grabbing the constant struct, whether
      we are on the GPU or CPU.
     */
    // FIXME
//    template <typename real>
//      inline __device__ __host__ const coeff_5<real>* coeff() {
//#ifdef __CUDA_ARCH__
//        return reinterpret_cast<const coeff_5<real>*>(mobius_d);
//#else
//        return reinterpret_cast<const coeff_5<real>*>(mobius_h);
//#endif
//      }
//
//    template <typename real, Dslash5Type, typename Arg> struct coeff_type {
//      typedef real type;
//      const Arg &arg;
//      __device__ __host__ coeff_type(const Arg &arg) : arg(arg) { }
//      __device__ __host__ real a(int s) { return arg.a; }
//      __device__ __host__ real b(int s) { return arg.b; }
//      __device__ __host__ real c(int s) { return arg.c; }
//    };
//
//    template <typename real, typename Arg> struct coeff_type<real,M5_INV_ZMOBIUS,Arg> {
//      typedef complex<real> type;
//      __device__ __host__ coeff_type(const Arg &arg) { }
//      __device__ __host__ complex<real> a(int s) { return coeff<real>()->a[s]; }
//      __device__ __host__ complex<real> b(int s) { return coeff<real>()->b[s]; }
//      __device__ __host__ complex<real> c(int s) { return coeff<real>()->c[s]; }
//    };

    /**
      @brief Parameter structure for applying the Dslash
     */
    template <typename storage_type, int nColor>
      struct Dslash5Arg {
        typedef typename colorspinor_mapper<storage_type,4,nColor>::type F;
        typedef typename mapper<storage_type>::type real;

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

        real b = 0.;                 // real constant Mobius coefficient
        real c = 0.;                 // real constant Mobius coefficient
        real a = 0.;                 // real xpay coefficient

        real kappa = 0.;

        int  eofa_pm;
        real eofa_norm; // k in Grid implementation. (A12)
        real eofa_shift; // \beta in (B16), or the "shift" in Grid implementation

        real mq1, mq2, mq3; // mq1, mq2 and mq3 in Grid implementation

        Dslash5Type type;

        Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
            const double m_f_, const double m_5_, const Complex *b_5_, const Complex *c_5_, double a_, 
            const double mq1_, const double mq2_, const double mq3_, 
            const int eofa_pm_, const double eofa_norm_, const double eofa_shift_,
            bool dagger, Dslash5Type type_)
          : out(out), in(in), x(x), nParity(in.SiteSubset()),
          volume_cb(in.VolumeCB()), volume_4d_cb(volume_cb/in.X(4)), Ls(in.X(4)),
          m_f(m_f_), m_5(m_5_), a(a_), dagger(dagger), xpay(a_ == 0.0 ? false : true), type(type_),
          mq1(mq1_), mq2(mq2_), mq3(mq3_), eofa_pm(eofa_pm_), eofa_norm(eofa_norm_), eofa_shift(eofa_shift_)
          {
            if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
            if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

//            if (sizeof(coeff_5<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(coeff_5<real>));
//            coeff_5<real> *coeff = reinterpret_cast<coeff_5<real>*>(&mobius_h);
//            auto *a_5 =  coeff->a;
//            auto *b_5 =  coeff->b;
//            auto *c_5 =  coeff->c;

            double alpha, N;
            int idx = 0;
            switch(type) {
              case m5_eofa: // For Mobius the b's and c's are constants and real
                b = b_5_[0].real();
                c = c_5_[0].real();
                kappa = 0.5 * (c*(m_5+4.0)-1.0) / (b*(m_5+4.0)+1.0);
                alpha = b+c;
                // Following the Grid implementation of MobiusEOFAFermion<Impl>::SetCoefficientsPrecondShiftOps()
                N = ( (eofa_pm == 1) ? 1.0 : -1.0 ) * (2.0*eofa_shift*eofa_norm) * ( std::pow(alpha+1.0,Ls) + mq1*std::pow(alpha-1.0,Ls) );
                for(int s=0; s<Ls; ++s){
                  idx = (eofa_pm == 1) ? (s) : (Ls-1-s);
                  m5_shift_h[idx] = N * std::pow(-1.0,s) * std::pow(alpha-1.0,s) / std::pow(alpha+1.0,Ls+s+1);
                } 
                cudaMemcpyToSymbolAsync(m5_shift_d, m5_shift_h, sizeof(real)*Ls, 0, cudaMemcpyHostToDevice, streams[Nstream-1]);
                break;
                //              case m5inv_eofa:
                //                for (int s=0; s<Ls; s++) {
                //                  b_5[s] = b_5_[s];
                //                  c_5[s] = 0.5*c_5_[s];
                //
                //                  // xpay
                //                  a_5[s] = 0.5/(b_5_[s]*(m_5+4.0) + 1.0);
                //                  a_5[s] *= a_5[s] * static_cast<real>(a);
                //                }
                //                break;
              default:
                errorQuda("Unknown EOFA Dslash5Type %d", type);
            }
          }
      };

    /**
      @brief Apply the D5 operator at given site
      @param[in] arg Argument struct containing any meta data and accessors
      @param[in] parity Parity we are on
      @param[in] x_b Checkerboarded 4-d space-time index
      @param[in] s Ls dimension coordinate
     */
    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
      __device__ inline void dslash5(Arg &arg, int parity, int x_cb, int s) {
        typedef typename mapper<storage_type>::type real;
        typedef ColorSpinor<real,nColor,4> Vector;

        VectorCache<real,Vector> cache;

        Vector out;
        cache.save(arg.in(s*arg.volume_4d_cb + x_cb, parity));
        cache.sync();

        auto Ls = arg.Ls;

        { // forwards direction
          const Vector in = cache.load(threadIdx.x, (s+1)%Ls, parity);
          constexpr int proj_dir = dagger ? +1 : -1;
          if (s == Ls-1) {
            out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
          } else {
            out += in.project(4, proj_dir).reconstruct(4, proj_dir);
          }
        }

        { // backwards direction
          const Vector in = cache.load(threadIdx.x, (s+Ls-1)%Ls, parity);
          constexpr int proj_dir = dagger ? -1 : +1;
          if (s == 0) {
            out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
          } else {
            out += in.project(4, proj_dir).reconstruct(4, proj_dir);
          }
        }

        if (type == m5_eofa) {
          Vector diagonal = cache.load(threadIdx.x, s, parity);
          out = arg.kappa*out + diagonal; // 1 + kappa*D5

          if( dagger &&  pm){ // in Grid: axpby_ssp_pplus(chi, one, chi, shift_coeffs[s], psi, s, Ls-1);
            if(s == Ls-1)
              for(int s_ = 0; s_ < Ls; s_++){
                out += m5_shift_d[s_] * cache.load(threadIdx.x, s, parity).project(4,+1).reconstruct(4,+1);  
              }
          }else if( dagger && !pm){ // in Grid: axpby_ssp_pminus(chi, one, chi, shift_coeffs[s], psi, s, 0);
            if(s ==   0 )
              for(int s_ = 0; s_ < Ls; s_++){
                out += m5_shift_d[s_] * cache.load(threadIdx.x, s, parity).project(4,+1).reconstruct(4,+1);  
              }
          }else if(!dagger &&  pm){ // in Grid: axpby_ssp_pplus(chi, one, chi, shift_coeffs[s], psi, Ls-1, s);

            out += m5_shift_d[s] * cache.load(threadIdx.x, Ls-1, parity).project(4,-1).reconstruct(4,-1);  


          }else if(!dagger && !pm){ // in Grid: axpby_ssp_pminus(chi, one, chi, shift_coeffs[s], psi, 0, s);

            out += m5_shift_d[s] * cache.load(threadIdx.x,    0, parity).project(4,-1).reconstruct(4,-1);  


          }

          if (xpay) { // really axpy
            Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
            out = arg.a*x + out;
          }
        }

        arg.out(s*arg.volume_4d_cb + x_cb, parity) = out;
      }

    /**
      @brief GPU kernel for applying the D5 operator
      @param[in] arg Argument struct containing any meta data and accessors
     */
    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
      __global__ void dslash5GPU(Arg arg)
      {
        int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
        int s = blockIdx.y*blockDim.y + threadIdx.y;
        int parity = blockIdx.z*blockDim.z + threadIdx.z;

        if (x_cb >= arg.volume_4d_cb) return;
        if (s >= arg.Ls) return;
        if (parity >= arg.nParity) return;

        dslash5<storage_type,nColor,dagger,pm,xpay,type>(arg, parity, x_cb, s);
      }

    template <typename storage_type, int nColor, typename Arg>
      class Dslash5 : public TunableVectorYZ {

        protected:
          Arg &arg;
          const ColorSpinorField &meta;
          static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

          /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
          static constexpr bool var_inverse = true;

          long long flops() const {
            long long Ls = meta.X(4);
            long long bulk = (Ls-2)*(meta.Volume()/Ls);
            long long wall = 2*meta.Volume()/Ls;
            long long n = meta.Ncolor() * meta.Nspin();

            long long flops_ = 0;
            switch (arg.type) {
              case m5_eofa:
                flops_ = n * (8ll*bulk + 10ll*wall + (arg.xpay ? 4ll * meta.Volume() : 0) );
                break;
              default:
                errorQuda("Unknown Dslash5Type %d", arg.type);
            }

            return flops_;
          }

          long long bytes() const {
            long long Ls = meta.X(4);
            switch (arg.type) {
              case m5_eofa:
                return arg.out.Bytes() + 2*arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
              default: errorQuda("Unknown Dslash5Type %d", arg.type);
            }
            return 0ll;
          }

          bool tuneGridDim() const { return false; }
          unsigned int minThreads() const { return arg.volume_4d_cb; }
          int blockStep() const { return 4; }
          int blockMin() const { return 4; }
          unsigned int sharedBytesPerThread() const {
              // spin components in shared depend on inversion algorithm
              int nSpin = meta.Nspin();
              return 2*nSpin*nColor*sizeof(typename mapper<storage_type>::type);
          }

          // overloaded to return max dynamic shared memory if doing shared-memory inverse
          unsigned int maxSharedBytesPerBlock() const {
            if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS) ) {
              return maxDynamicSharedBytesPerBlock();
            } else {
              return TunableVectorYZ::maxSharedBytesPerBlock();
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
              case m5_eofa:        strcat(aux, ",mobius_m5_eofa");        break;
              default: errorQuda("Unknown Dslash5Type %d", arg.type);
            }
          }
          virtual ~Dslash5() { }

          template <typename T>
            inline void launch(T *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
              if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS) ) {
                // if inverse kernel uses shared memory then maximize total shared memory pool
                setMaxDynamicSharedBytesPerBlock(f);
              }
              void *args[] = { &arg };
              qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
            }

          void apply(const cudaStream_t &stream) {
              TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
              if (arg.type == m5_eofa) {
                if(arg.eofa_pm == 1){
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true , true,m5_eofa, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true , true,m5_eofa, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true ,false,m5_eofa, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true ,false,m5_eofa, Arg>, tp, arg, stream);
                  }
                }else{
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false, true,m5_eofa, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false, true,m5_eofa, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false,false,m5_eofa, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false,false,m5_eofa, Arg>, tp, arg, stream);
                  }
                }
              }else{
                errorQuda("Unknown Dslash5Type %d", arg.type);
              }            
          }

          void initTuneParam(TuneParam &param) const {
            TunableVectorYZ::initTuneParam(param);
            param.block.y = arg.Ls; // Ls must be contained in the block
            param.grid.y = 1;
            param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z;
          }

          void defaultTuneParam(TuneParam &param) const {
            TunableVectorYZ::defaultTuneParam(param);
            param.block.y = arg.Ls; // Ls must be contained in the block
            param.grid.y = 1;
            param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z;
          }

          TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
      };


    template <typename storage_type, int nColor>
    void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
        double m_f, double m_5, const Complex *b_5, const Complex *c_5, double a,
        const double mq1, const double mq2, const double mq3, 
        const int eofa_pm, const double eofa_norm, const double eofa_shift,
        bool dagger, Dslash5Type type)
    {
      Dslash5Arg<storage_type,nColor> arg(out, in, x, m_f, m_5, b_5, c_5, a, mq1, mq2, mq3, eofa_pm, eofa_norm, eofa_shift, dagger, type);
      Dslash5<storage_type,nColor,Dslash5Arg<storage_type,nColor> > dslash(arg, in);
      dslash.apply(streams[Nstream-1]);
    }

    // template on the number of colors
    template <typename storage_type>
    void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
        double m_f, double m_5, const Complex *b_5, const Complex *c_5, double a,
        const double mq1, const double mq2, const double mq3, 
        const int eofa_pm, const double eofa_norm, const double eofa_shift,
        bool dagger, Dslash5Type type)
   {
     switch(in.Ncolor()) {
       case 3: ApplyDslash5<storage_type,3>(out, in, x, m_f, m_5, b_5, c_5, a, mq1, mq2, mq3, eofa_pm, eofa_norm, eofa_shift, dagger, type); break;
       default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
     }
   }

#endif
    //Apply the 5th dimension dslash operator to a colorspinor field
    //out = Dslash5*in
    void apply_dslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
        double m_f, double m_5, const Complex *b_5, const Complex *c_5, double a,
        const double mq1, const double mq2, const double mq3, 
        const int eofa_pm, const double eofa_norm, const double eofa_shift,
        bool dagger, Dslash5Type type)
    {
#ifdef GPU_DOMAIN_WALL_DIRAC
      if (in.DWFPCtype() != QUDA_4D_PC) errorQuda("Only 4d preconditioned fields are supported");
      checkLocation(out, in);     // check all locations match

      switch(checkPrecision(out,in)) {
        case QUDA_DOUBLE_PRECISION: ApplyDslash5<double>(out, in, x, m_f, m_5, b_5, c_5, a, mq1, mq2, mq3, eofa_pm, eofa_norm, eofa_shift, dagger, type); break;
        case QUDA_SINGLE_PRECISION: ApplyDslash5<float >(out, in, x, m_f, m_5, b_5, c_5, a, mq1, mq2, mq3, eofa_pm, eofa_norm, eofa_shift, dagger, type); break;
        case QUDA_HALF_PRECISION:   ApplyDslash5<short >(out, in, x, m_f, m_5, b_5, c_5, a, mq1, mq2, mq3, eofa_pm, eofa_norm, eofa_shift, dagger, type); break;
        default: errorQuda("Unsupported precision %d\n", in.Precision());
      }
#else
      errorQuda("Mobius EOFA dslash has not been built");
#endif
    }
  } // namespace mobius_eofa
} // namespace quda

