#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <clover_field.h>
#include <clover_field_order.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>

namespace quda {
#include <dslash_events.cuh>
#include <dslash_policy.cuh>
}

#include <kernels/dslash_wilson_clover_hasenbusch_twist_preconditioned.cuh>
//#include <kernels/dslash_wilson_clover_hasenbusch_twist.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct WilsonCloverHasenbuschTwistClovInvLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(xpay == true, "wilsonCloverHasenbuschTwistClovInv operator only defined for xpay");
      dslash.launch(wilsonCloverHasenbuschTwistClovInvGPU<Float,nDim,nColor,nParity,dagger,xpay,kernel_type,Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
    struct WilsonCloverHasenbuschTwistNoClovInvLaunch {
      template <typename Dslash>
      inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
        static_assert(xpay == true, "wilsonCloverHasenbuschNoTwistClovInv operator only defined for xpay");
        dslash.launch(wilsonCloverHasenbuschTwistNoClovInvGPU<Float,nDim,nColor,nParity,dagger,xpay,kernel_type,Arg>, tp, arg, stream);
      }
    };

    /* ***************************
  	  * No Clov Inv
  	  * **************************/
  template <typename Float, int nDim, int nColor, typename Arg>
    class WilsonCloverHasenbuschTwistNoClovInv : public Dslash<Float> {

    protected:
      Arg &arg;
      const ColorSpinorField &in;

    public:

      WilsonCloverHasenbuschTwistNoClovInv(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
        : Dslash<Float>(arg, out, in), arg(arg), in(in)
      {  }

      virtual ~WilsonCloverHasenbuschTwistNoClovInv() { }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash<Float>::setParam(arg);
        if (arg.xpay) Dslash<Float>::template instantiate<WilsonCloverHasenbuschTwistNoClovInvLaunch,nDim,nColor,true>(tp, arg, stream);
        else errorQuda("Wilson-clover hasenbusch twist no clover operator only defined for xpay=true");
      }

      // Fixme get this right
      long long flops() const {
        int clover_flops = 504;
        long long flops = Dslash<Float>::flops();
        switch(arg.kernel_type) {
        case EXTERIOR_KERNEL_X:
        case EXTERIOR_KERNEL_Y:
        case EXTERIOR_KERNEL_Z:
        case EXTERIOR_KERNEL_T:
        case EXTERIOR_KERNEL_ALL:
  	break; // all clover flops are in the interior kernel
        case INTERIOR_KERNEL:
        case KERNEL_POLICY:
  	flops += clover_flops * in.Volume();
  	break;
        }
        return flops;
      }

      // Fixme: Get this right
      long long bytes() const {
        bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
        int clover_bytes = 72 * in.Precision() + (isFixed ? 2*sizeof(float) : 0);

        long long bytes = Dslash<Float>::bytes();
        switch(arg.kernel_type) {
        case EXTERIOR_KERNEL_X:
        case EXTERIOR_KERNEL_Y:
        case EXTERIOR_KERNEL_Z:
        case EXTERIOR_KERNEL_T:
        case EXTERIOR_KERNEL_ALL:
  	break;
        case INTERIOR_KERNEL:
        case KERNEL_POLICY:
  	bytes += clover_bytes*in.Volume();
  	break;
        }

        return bytes;
      }

      TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
    };

    template <typename Float, int nColor, QudaReconstructType recon>
    void ApplyWilsonCloverHasenbuschTwistNoClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
  					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
  					const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
  #ifdef DYNAMIC_CLOVER
      constexpr bool dynamic_clover = true;
  #else
      constexpr bool dynamic_clover = false;
  #endif

      using ArgType = WilsonCloverHasenbuschTwistArg<Float,nColor,recon,dynamic_clover>;

      ArgType arg(out, in, U, A, kappa, mu, x, parity, dagger, comm_override);
      WilsonCloverHasenbuschTwistNoClovInv<Float,nDim,nColor,ArgType > wilson(arg, out, in);

      DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                                in.VolumeCB(), in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }

    // template on the gauge reconstruction
    template <typename Float, int nColor>
    void ApplyWilsonCloverHasenbuschTwistNoClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
  					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
                           const int *comm_override, TimeProfile &profile)
    {
      if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, A, kappa, mu, x, parity, dagger, comm_override, profile);
      } else {
        errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
      }
    }

    // template on the number of colors
    template <typename Float>
    void ApplyWilsonCloverHasenbuschTwistNoClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
  					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
                           const int *comm_override, TimeProfile &profile)
    {
      if (in.Ncolor() == 3) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<Float,3>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else {
        errorQuda("Unsupported number of colors %d\n", U.Ncolor());
      }
    }

  #endif // GPU_CLOVER_DIRAC

    // Apply the Wilson-clover operator
    // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
    // Uses the kappa normalization for the Wilson operator.
    void ApplyWilsonCloverHasenbuschTwistNoClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
  					const CloverField &A, double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
  			 const int *comm_override, TimeProfile &profile)
    {
  #ifdef GPU_CLOVER_DIRAC
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      if (in.FieldOrder() != out.FieldOrder())
        errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

      // check all precisions match
      checkPrecision(out, in, U, A);

      // check all locations match
      checkLocation(out, in, U, A);

      if (U.Precision() == QUDA_DOUBLE_PRECISION) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<double>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<float>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else if (U.Precision() == QUDA_HALF_PRECISION) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<short>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
        ApplyWilsonCloverHasenbuschTwistNoClovInv<char>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
      } else {
        errorQuda("Unsupported precision %d\n", U.Precision());
      }
  #else
      errorQuda("Clover dslash has not been built");
  #endif
    }

#ifdef GPU_CLOVER_DIRAC
    /* ***************************
     * Clov Inv
     * **************************/
    template <typename Float, int nDim, int nColor, typename Arg>
      class WilsonCloverHasenbuschTwistClovInv : public Dslash<Float> {

      protected:
        Arg &arg;
        const ColorSpinorField &in;

      public:

        WilsonCloverHasenbuschTwistClovInv(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
          : Dslash<Float>(arg, out, in), arg(arg), in(in)
        {}

        virtual ~WilsonCloverHasenbuschTwistClovInv() { }

        void apply(const cudaStream_t &stream) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          Dslash<Float>::setParam(arg);
          if (arg.xpay) Dslash<Float>::template instantiate<WilsonCloverHasenbuschTwistClovInvLaunch,nDim,nColor,true>(tp, arg, stream);
          else errorQuda("Wilson-clover hasenbusch twist no clover operator only defined for xpay=true");
        }

        // Fixme get this right
        long long flops() const {
          int clover_flops = 504;
          long long flops = Dslash<Float>::flops();
          switch(arg.kernel_type) {
          case EXTERIOR_KERNEL_X:
          case EXTERIOR_KERNEL_Y:
          case EXTERIOR_KERNEL_Z:
          case EXTERIOR_KERNEL_T:
          case EXTERIOR_KERNEL_ALL:
    	break; // all clover flops are in the interior kernel
          case INTERIOR_KERNEL:
          case KERNEL_POLICY:
    	flops += clover_flops * in.Volume();
    	break;
          }
          return flops;
        }

        // Fixme: Get this right
        long long bytes() const {
          bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
          int clover_bytes = 72 * in.Precision() + (isFixed ? 2*sizeof(float) : 0);

          long long bytes = Dslash<Float>::bytes();
          switch(arg.kernel_type) {
          case EXTERIOR_KERNEL_X:
          case EXTERIOR_KERNEL_Y:
          case EXTERIOR_KERNEL_Z:
          case EXTERIOR_KERNEL_T:
          case EXTERIOR_KERNEL_ALL:
    	break;
          case INTERIOR_KERNEL:
          case KERNEL_POLICY:
    	bytes += clover_bytes*in.Volume();
    	break;
          }

          return bytes;
        }

        TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
      };

      template <typename Float, int nColor, QudaReconstructType recon>
      void ApplyWilsonCloverHasenbuschTwistClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
    					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
    					const int *comm_override, TimeProfile &profile)
      {
        constexpr int nDim = 4;
    #ifdef DYNAMIC_CLOVER
        constexpr bool dynamic_clover = true;
    #else
        constexpr bool dynamic_clover = false;
    #endif

        using ArgType = WilsonCloverHasenbuschTwistArg<Float,nColor,recon,dynamic_clover>;

        ArgType arg(out, in, U, A, kappa, mu, x, parity, dagger, comm_override);
        WilsonCloverHasenbuschTwistClovInv<Float,nDim,nColor,ArgType > wilson(arg, out, in);

        DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                                  in.VolumeCB(), in.GhostFaceCB(), profile);
        policy.apply(0);

        checkCudaError();
      }

      // template on the gauge reconstruction
      template <typename Float, int nColor>
      void ApplyWilsonCloverHasenbuschTwistClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
    					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
                             const int *comm_override, TimeProfile &profile)
      {
        if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
          ApplyWilsonCloverHasenbuschTwistClovInv<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
          ApplyWilsonCloverHasenbuschTwistClovInv<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
          ApplyWilsonCloverHasenbuschTwistClovInv<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, A, kappa, mu, x, parity, dagger, comm_override, profile);
        } else {
          errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
        }
      }

      // template on the number of colors
      template <typename Float>
      void ApplyWilsonCloverHasenbuschTwistClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
    					double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
                             const int *comm_override, TimeProfile &profile)
      {
        if (in.Ncolor() == 3) {
          ApplyWilsonCloverHasenbuschTwistClovInv<Float,3>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else {
          errorQuda("Unsupported number of colors %d\n", U.Ncolor());
        }
      }

    #endif // GPU_CLOVER_DIRAC

      // Apply the Wilson-clover operator
      // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
      // Uses the kappa normalization for the Wilson operator.
      void ApplyWilsonCloverHasenbuschTwistClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
    					const CloverField &A, double kappa, double mu, const ColorSpinorField &x, int parity, bool dagger,
    			 const int *comm_override, TimeProfile &profile)
      {
    #ifdef GPU_CLOVER_DIRAC
        if (in.V() == out.V()) errorQuda("Aliasing pointers");
        if (in.FieldOrder() != out.FieldOrder())
          errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

        // check all precisions match
        checkPrecision(out, in, U, A);

        // check all locations match
        checkLocation(out, in, U, A);

        if (U.Precision() == QUDA_DOUBLE_PRECISION) {
          ApplyWilsonCloverHasenbuschTwistClovInv<double>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
          ApplyWilsonCloverHasenbuschTwistClovInv<float>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else if (U.Precision() == QUDA_HALF_PRECISION) {
          ApplyWilsonCloverHasenbuschTwistClovInv<short>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
          ApplyWilsonCloverHasenbuschTwistClovInv<char>(out, in, U, A, kappa,mu, x, parity, dagger, comm_override, profile);
        } else {
          errorQuda("Unsupported precision %d\n", U.Precision());
        }
    #else
        errorQuda("Clover dslash has not been built");
    #endif
      }

} // namespace quda

#endif
