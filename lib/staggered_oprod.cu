#include <staggered_oprod.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/staggered_outer_product.cuh>

namespace quda {

  enum OprodKernelType { INTERIOR, EXTERIOR };

  template <typename Float, int nColor, QudaReconstructType recon>
  class StaggeredOprod : public TunableKernel1D {
    using real = typename mapper<Float>::type;
    template <int dim = -1> using Arg = StaggeredOprodArg<Float, nColor, dim>;
    GaugeField &U;
    GaugeField &L;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const int parity;
    const real coeff[2];
    const int nFace;
    OprodKernelType kernel;
    int dir;
    int displacement;
    unsigned int minThreads() const
    {
      return kernel == INTERIOR ? inB.VolumeCB() : displacement * inB.GhostFaceCB()[dir];
    }

  public:
    StaggeredOprod(GaugeField &U, GaugeField &L, ColorSpinorField &inA, ColorSpinorField &inB,
                   int parity, const double coeff[2], int nFace) :
      TunableKernel1D(U),
      U(U),
      L(L),
      inA(inA),
      inB(inB),
      parity(parity),
      coeff{static_cast<real>(coeff[0]), static_cast<real>(coeff[1])},
      nFace(nFace)
    {
      char aux2[TuneKey::aux_n];
      strcpy(aux2, aux);
      kernel = INTERIOR;
      apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) {
        if (commDimPartitioned(i)) {
          // update parameters for this exterior kernel
          kernel = EXTERIOR;
          dir = i;

          // one and three hop terms
          for (auto hop : std::array<int, 2>{1, 3}) {
            if (hop == 3 && nFace != 3) continue;
            displacement = hop;
            strcpy(aux, aux2);
            strcat(aux, ",dir=");
            u32toa(aux + strlen(aux), dir);
            strcat(aux, ",displacement=");
            u32toa(aux + strlen(aux), displacement);
            apply(device::get_default_stream());
          }
        }
      } // i=3,..,0
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (kernel == INTERIOR) {
        launch<Interior>(tp, stream, Arg<>(U, L, inA, inB, parity, displacement, nFace, coeff));
      } else if (kernel == EXTERIOR) {
        switch (dir) {
        case 0: launch<Exterior>(tp, stream, Arg<0>(U, L, inA, inB, parity, displacement, nFace, coeff)); break;
        case 1: launch<Exterior>(tp, stream, Arg<1>(U, L, inA, inB, parity, displacement, nFace, coeff)); break;
        case 2: launch<Exterior>(tp, stream, Arg<2>(U, L, inA, inB, parity, displacement, nFace, coeff)); break;
        case 3: launch<Exterior>(tp, stream, Arg<3>(U, L, inA, inB, parity, displacement, nFace, coeff)); break;
        default: errorQuda("Unexpected direction %d", dir);
        }
      } else {
        errorQuda("Kernel type not supported\n");
      }
    } // apply

    void preTune() { U.backup(); if (U.Gauge_p() != L.Gauge_p()) L.backup(); }
    void postTune() { U.restore(); if (U.Gauge_p() != L.Gauge_p()) L.restore(); }

    long long flops() const { return 0; } // FIXME
    long long bytes() const { return 0; } // FIXME
  }; // StaggeredOprod

  void computeStaggeredOprod(GaugeField &U, GaugeField &L, ColorSpinorField &inEven, ColorSpinorField &inOdd,
                             int parity, const double coeff[2], int nFace)
  {
    checkNative(U, L);
    ColorSpinorField &inA = (parity & 1) ? inOdd : inEven;
    ColorSpinorField &inB = (parity & 1) ? inEven : inOdd;

    inB.exchangeGhost((QudaParity)(1-parity), nFace, 0);
    instantiate<StaggeredOprod, ReconstructNone>(U, L, inA, inB, parity, coeff, nFace);

    inB.bufferIndex = (1 - inB.bufferIndex);
  }

#ifdef GPU_STAGGERED_DIRAC
  void computeStaggeredOprod(GaugeField *out[], ColorSpinorField& in, const double coeff[], int nFace)
  {
    if (nFace == 1) {
      computeStaggeredOprod(*out[0], *out[0], in.Even(), in.Odd(), 0, coeff, nFace);
      double coeff_[2] = {-coeff[0],0.0}; // need to multiply by -1 on odd sites
      computeStaggeredOprod(*out[0], *out[0], in.Even(), in.Odd(), 1, coeff_, nFace);
    } else if (nFace == 3) {
      computeStaggeredOprod(*out[0], *out[1], in.Even(), in.Odd(), 0, coeff, nFace);
      computeStaggeredOprod(*out[0], *out[1], in.Even(), in.Odd(), 1, coeff, nFace);
    } else {
      errorQuda("Invalid nFace=%d", nFace);
    }
  }
#else // GPU_STAGGERED_DIRAC not defined
  void computeStaggeredOprod(GaugeField *[], ColorSpinorField &, const double [], int)
  {
    errorQuda("Staggered Outer Product has not been built!");
  }
#endif

} // namespace quda
