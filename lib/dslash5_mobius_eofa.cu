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
    struct eofa_coeff {
      real u[QUDA_MAX_DWF_LS]; // xpay coefficients
      real x[QUDA_MAX_DWF_LS];
      real y[QUDA_MAX_DWF_LS];
    };

    constexpr int size = 4096;
    static __constant__ char mobius_eofa_d[size];
    static              char mobius_eofa_h[size];
    
    /**
      @brief Helper function for grabbing the constant struct, whether
      we are on the GPU or CPU.
     */
    template <typename real>
    inline __device__ __host__ const eofa_coeff<real>* get_eofa_coeff() {
#ifdef __CUDA_ARCH__
        return reinterpret_cast<const eofa_coeff<real>*>(mobius_eofa_d);
#else
        return reinterpret_cast<const eofa_coeff<real>*>(mobius_eofa_h);
#endif
    }
    
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
        real a;                      // real xpay coefficient

        real kappa = 0.;
        real inv = 0.;

        int  eofa_pm;
        real eofa_norm; // k in Grid implementation. (A12)
        real eofa_shift; // \beta in (B16), or the "shift" in Grid implementation

        real sherman_morrison;

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
          mq1(mq1_), mq2(mq2_), mq3(mq3_), eofa_pm(eofa_pm_), eofa_norm(eofa_norm_), eofa_shift(eofa_shift_), sherman_morrison(0.)
          {
            if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
            if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
            if (sizeof(eofa_coeff<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(eofa_coeff<real>));
            
            eofa_coeff<real>* eofa_coeffs = reinterpret_cast<eofa_coeff<real>*>(mobius_eofa_h);
            // For Mobius the b's and c's are constants and real
            b = b_5_[0].real();
            c = c_5_[0].real();
            kappa = (c*(m_5+4.)-1.) / (b*(m_5+4.)+1.);
            double alpha = b+c;
            
            // Following the Grid implementation of MobiusEOFAFermion<Impl>::SetCoefficientsPrecondShiftOps()
            double N = ( (eofa_pm == 1)?1.:-1. ) * (2.*this->eofa_shift*this->eofa_norm) * ( std::pow(alpha+1.,Ls) + this->mq1*std::pow(alpha-1.,Ls) );
            // QUDA uses the kappa preconditioning: there is a (2.*kappa_b)^-1 difference here.
            N *= 1./(b*(m_5+4.)+1.);
            
            // Here the signs are somewhat mixed:
            // There is one -1 from N for eofa_pm = 0/minus, thus the u_- here is actually -u_- in the document
            // It turns out this actually simplies things.
            for(int s = 0; s < Ls; s++){
              int idx = (eofa_pm == 1) ? (s) : (Ls-1-s);
              eofa_coeffs->u[idx] = N * std::pow(-1.,s) * std::pow(alpha-1.,s) / std::pow(alpha+1.,Ls+s+1);
            } 

            real factor = -kappa*m_f;
            switch(type) {
              case M5_EOFA:
                cudaMemcpyToSymbolAsync(mobius_eofa_d, mobius_eofa_h, sizeof(eofa_coeff<real>)/3, 0, cudaMemcpyHostToDevice, streams[Nstream-1]);
                break;
              case M5INV_EOFA:
                /** Here we want to solve 
                 * M_pm^-1      * u_pm = x_pm
                 * M_pm^-dagger * v_pm = y_pm
                 */
                if(eofa_pm){
                  // eofa_pm = plus
                  // Computing x
                  eofa_coeffs->x[0] = eofa_coeffs->u[0]; 
                  for(int s = Ls-1; s > 0; s--){
                    eofa_coeffs->x[0] -= factor*eofa_coeffs->u[s];
                    factor *= -kappa;
                  }
                  eofa_coeffs->x[0] /= 1.+factor;
                  for(int s = 1; s < Ls; s++){
                    eofa_coeffs->x[s] = eofa_coeffs->x[s-1]*(-kappa) + eofa_coeffs->u[s];
                  }
                  // Computing y
                  eofa_coeffs->y[Ls-1] = 1./(1.+factor);
                  sherman_morrison = eofa_coeffs->x[Ls-1];
                  for(int s = Ls-1; s > 0; s--){
                    eofa_coeffs->y[s-1] = eofa_coeffs->y[s]*(-kappa);
                  }
                }else{
                  // eofa_pm = minus
                  // Computing x
                  eofa_coeffs->x[Ls-1] = eofa_coeffs->u[Ls-1]; 
                  for(int s = 0; s < Ls; s++){
                    eofa_coeffs->x[Ls-1] -= factor*eofa_coeffs->u[s];
                    factor *= -kappa;
                  }
                  eofa_coeffs->x[Ls-1] /= 1.+factor;
                  for(int s = Ls-1; s > 0; s--){
                    eofa_coeffs->x[s-1] = eofa_coeffs->x[s]*(-kappa) + eofa_coeffs->u[s-1];
                  }
                  // Computing y
                  eofa_coeffs->y[0] = 1./(1.+factor);
                  sherman_morrison = eofa_coeffs->x[0];
                  for(int s = 1; s < Ls; s++){
                    eofa_coeffs->y[s] = eofa_coeffs->y[s-1]*(-kappa); 
                  }
                }
                inv = 0.5/(1.+factor); // 0.5 for the spin project factor
                sherman_morrison = -0.5/(1.+sherman_morrison); // 0.5 for the spin project factor
                cudaMemcpyToSymbolAsync(mobius_eofa_d, mobius_eofa_h, sizeof(eofa_coeff<real>), 0, cudaMemcpyHostToDevice, streams[Nstream-1]);
                break;
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

        if (type == M5_EOFA) {
          const eofa_coeff<real>* eofa_coeffs = get_eofa_coeff<real>();
          Vector diagonal = cache.load(threadIdx.x, s, parity);
          out = (static_cast<real>(0.5)*arg.kappa)*out + diagonal; // 1 + kappa*D5; the 0.5 for spin projection
          
          constexpr int proj_dir = pm ? +1 : -1;
          
          if(dagger){
            // in Grid: 
            // axpby_ssp_pplus(chi, one, chi, shift_coeffs[s], psi, Ls-1, s);
            // axpby_ssp_pminus(chi, one, chi, shift_coeffs[s], psi, 0, s);
            if(s == (pm?Ls-1:0)){
              for(int sp = 0; sp < Ls; sp++){
                out += (static_cast<real>(0.5)*eofa_coeffs->u[sp]) * cache.load(threadIdx.x, sp, parity).project(4,proj_dir).reconstruct(4,proj_dir);  
              }
            }
          }else{
            // in Grid: 
            // axpby_ssp_pplus(chi, one, chi, shift_coeffs[s], psi, s, Ls-1);
            // axpby_ssp_pminus(chi, one, chi, shift_coeffs[s], psi, s, 0);
            out += (static_cast<real>(0.5)*eofa_coeffs->u[s]) * cache.load(threadIdx.x, pm?Ls-1:0, parity).project(4,proj_dir).reconstruct(4,proj_dir);  
          }

          if (xpay) { // really axpy
            Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
            out = arg.a*x + out;
          }
        }
        arg.out(s*arg.volume_4d_cb + x_cb, parity) = out;
      }
  
    /**
      @brief Apply the M5 inverse operator at a given site on the
      lattice.  This is the original algorithm as described in Kim and
      Izubushi (LATTICE 2013_033), where the b and c coefficients are
      constant along the Ls dimension, so is suitable for Shamir and
      Mobius domain-wall fermions.

      @tparam shared Whether to use a shared memory scratch pad to
      store the input field acroos the Ls dimension to minimize global
      memory reads.
      @param[in] arg Argument struct containing any meta data and accessors
      @param[in] parity Parity we are on
      @param[in] x_b Checkerboarded 4-d space-time index
      @param[in] s_ Ls dimension coordinate
     */

    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
      __device__ __host__ inline void dslash5inv(Arg &arg, int parity, int x_cb, int s) {
        typedef typename mapper<storage_type>::type real;
        typedef ColorSpinor<real,nColor,4> Vector;

        const real k = -arg.kappa; // k is -kappa
        const real inv = arg.inv;
        const real sherman_morrison = arg.sherman_morrison;
        VectorCache<real,Vector> cache;
        cache.save(arg.in(s*arg.volume_4d_cb + x_cb, parity));
        cache.sync();

        Vector out;
        const eofa_coeff<real>* eofa_coeffs = get_eofa_coeff<real>(); 

        for (int sp = 0; sp < arg.Ls; sp++) {
          Vector in = cache.load(threadIdx.x, sp, parity);
          {
            int exp = s < sp ? arg.Ls-sp+s : s-sp;
            real factorR = inv * __fast_pow(k,exp) * ( s < sp ? -arg.m_f : static_cast<real>(1.0) );
            constexpr int proj_dir = dagger ? -1 : +1;
            out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
          }
          {
            int exp = s > sp ? arg.Ls-s+sp : sp-s;
            real factorL = inv * __fast_pow(k,exp) * ( s > sp ? -arg.m_f : static_cast<real>(1.0) );
            constexpr int proj_dir = dagger ? +1 : -1;
            out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
          }
          // The EOFA stuff
          {
            constexpr int proj_dir = pm ? +1 : -1;
            real t = dagger ? eofa_coeffs->y[s]*eofa_coeffs->x[sp] : eofa_coeffs->x[s]*eofa_coeffs->y[sp];
            out += (t*sherman_morrison) * (in.project(4,proj_dir)).reconstruct(4,proj_dir);
          }
        }
        if (xpay) { // really axpy
          Vector x = arg.x(s*arg.volume_4d_cb + x_cb, parity);
          out = arg.a*x + out;
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

        if(type == M5_EOFA){
          dslash5<storage_type,nColor,dagger,pm,xpay,type>(arg, parity, x_cb, s);
        }else if(type == M5INV_EOFA){
          dslash5inv<storage_type,nColor,dagger,pm,xpay,type>(arg, parity, x_cb, s);
        }
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
              case M5_EOFA:
              case M5INV_EOFA:
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
              case M5_EOFA:
              case M5INV_EOFA:
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
              case M5_EOFA:        strcat(aux, ",mobius_M5_EOFA");        break;
              case M5INV_EOFA:     strcat(aux, ",mobius_M5INV_EOFA");     break;
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
              if (arg.type == M5_EOFA) {
                if(arg.eofa_pm == 1){
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true , true,M5_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true , true,M5_EOFA, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true ,false,M5_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true ,false,M5_EOFA, Arg>, tp, arg, stream);
                  }
                }else{
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false, true,M5_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false, true,M5_EOFA, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false,false,M5_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false,false,M5_EOFA, Arg>, tp, arg, stream);
                  }
                }
              }else if(arg.type == M5INV_EOFA){
                if(arg.eofa_pm == 1){
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true , true,M5INV_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true , true,M5INV_EOFA, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,true ,false,M5INV_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,true ,false,M5INV_EOFA, Arg>, tp, arg, stream);
                  }
                }else{
                  if (arg.xpay) {
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false, true,M5INV_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false, true,M5INV_EOFA, Arg>, tp, arg, stream);
                  }else{
                    arg.dagger ?
                    launch(dslash5GPU<storage_type, nColor, true,false,false,M5INV_EOFA, Arg>, tp, arg, stream) :
                    launch(dslash5GPU<storage_type, nColor,false,false,false,M5INV_EOFA, Arg>, tp, arg, stream);
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

