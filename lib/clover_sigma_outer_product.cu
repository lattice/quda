#include <cstdio>
#include <cstdlib>

#include <tune_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>

#include <jitify_helper.cuh>
#include <kernels/clover_sigma_outer_product.cuh>

namespace quda {

#ifdef GPU_CLOVER_DIRAC

  template <typename Float, typename Arg> class CloverSigmaOprod : public TunableVectorYZ
  {
    Arg &arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.length; }
    bool tuneGridDim() const { return false; }

  public:
      CloverSigmaOprod(Arg &arg, const GaugeField &meta) : TunableVectorYZ(2, 6), arg(arg), meta(meta)
      {
        writeAuxString("%s,nvector=%d", meta.AuxString(), arg.nvector);
        // this sets the communications pattern for the packing kernel
#ifdef JITIFY
        create_jitify_program("kernels/clover_sigma_outer_product.cuh");
#endif
      }

      virtual ~CloverSigmaOprod() {}

      void apply(const qudaStream_t &stream)
      {
        if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::sigmaOprodKernel")
                             .instantiate(arg.nvector, Type<Float>(), Type<Arg>())
                             .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                             .launch(arg);
#else
          switch (arg.nvector) {
          case 1: sigmaOprodKernel<1, Float><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;
          default: errorQuda("Unsupported nvector = %d\n", arg.nvector);
          }
#endif
        } else { // run the CPU code
          errorQuda("No CPU support for staggered outer-product calculation\n");
        }
      } // apply

      void preTune() { this->arg.oprod.save(); }
      void postTune() { this->arg.oprod.load(); }

      long long flops() const
      {
        return (2 * (long long)arg.length) * 6
            * ((0 + 144 + 18) * arg.nvector + 18); // spin_mu_nu + spin trace + multiply-add
      }
      long long bytes() const
      {
        return (2 * (long long)arg.length) * 6
            * ((arg.inA[0].Bytes() + arg.inB[0].Bytes()) * arg.nvector + 2 * arg.oprod.Bytes());
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), "CloverSigmaOprod", aux); }
  }; // CloverSigmaOprod

  template<typename Float>
  void computeCloverSigmaOprod(GaugeField& oprod, const std::vector<ColorSpinorField*> &x,
			       const std::vector<ColorSpinorField*> &p, const std::vector<std::vector<double> > &coeff, int nvector)
  {
    // Create the arguments
    CloverSigmaOprodArg<Float, 3> arg(oprod, x, p, coeff, nvector);
    CloverSigmaOprod<Float, decltype(arg)> sigma_oprod(arg, oprod);
    sigma_oprod.apply(0);
  } // computeCloverSigmaOprod

#endif // GPU_CLOVER_FORCE

  void computeCloverSigmaOprod(GaugeField& oprod,
			       std::vector<ColorSpinorField*> &x,
			       std::vector<ColorSpinorField*> &p,
			       std::vector<std::vector<double> > &coeff)
  {

#ifdef GPU_CLOVER_DIRAC
    if (x.size() > MAX_NVECTOR) {
      // divide and conquer
      std::vector<ColorSpinorField*> x0(x.begin(), x.begin()+x.size()/2);
      std::vector<ColorSpinorField*> p0(p.begin(), p.begin()+p.size()/2);
      std::vector<std::vector<double> > coeff0(coeff.begin(), coeff.begin()+coeff.size()/2);
      for (unsigned int i=0; i<coeff0.size(); i++) {
	coeff0[i].reserve(2); coeff0[i][0] = coeff[i][0]; coeff0[i][1] = coeff[i][1];
      }
      computeCloverSigmaOprod(oprod, x0, p0, coeff0);

      std::vector<ColorSpinorField*> x1(x.begin()+x.size()/2, x.end());
      std::vector<ColorSpinorField*> p1(p.begin()+p.size()/2, p.end());
      std::vector<std::vector<double> > coeff1(coeff.begin()+coeff.size()/2, coeff.end());
      for (unsigned int i=0; i<coeff1.size(); i++) {
	coeff1[i].reserve(2); coeff1[i][0] = coeff[coeff.size()/2 + i][0]; coeff1[i][1] = coeff[coeff.size()/2 + i][1];
      }
      computeCloverSigmaOprod(oprod, x1, p1, coeff1);

      return;
    }

    if (oprod.Order() != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Unsupported output ordering: %d\n", oprod.Order());

    if (checkPrecision(*x[0], *p[0], oprod) == QUDA_DOUBLE_PRECISION) {
      computeCloverSigmaOprod<double>(oprod, x, p, coeff, x.size());
    } else {
      errorQuda("Unsupported precision: %d\n", oprod.Precision());
    }
#else // GPU_CLOVER_DIRAC not defined
    errorQuda("Clover Dirac operator has not been built!");
#endif

    checkCudaError();
  } // computeCloverForce

} // namespace quda
