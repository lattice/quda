#include <cstdio>
#include <cstdlib>

#include <staggered_oprod.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>

namespace quda {

  enum OprodKernelType { OPROD_INTERIOR_KERNEL, OPROD_EXTERIOR_KERNEL };

  template <typename Float, int nColor_> struct StaggeredOprodArg {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    using F = typename colorspinor_mapper<Float, nSpin, nColor>::type;
    using GU = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using GL = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;

    const F inA; /** input vector field */
    const F inB; /** input vector field */
    GU U;        /** output one-hop field */
    GL L;        /** output three-hop field */

    unsigned int length;
    const int parity;
    int dir;
    int displacement;
    OprodKernelType kernelType;
    const int nFace;
    Float coeff[2];
    int X[4];
    bool partitioned[4];

    StaggeredOprodArg(GaugeField &U, GaugeField &L, const ColorSpinorField &inA, const ColorSpinorField &inB,
                      int parity, int dir, int displacement, const OprodKernelType &kernelType, int nFace, const double coeff[2]) :
      inA(inA),
      inB(inB, nFace),
      U(U),
      L(L),
      length(U.VolumeCB()),
      parity(parity),
      dir(dir),
      displacement(displacement),
      kernelType(kernelType),
      nFace(nFace)
    {
      this->coeff[0] = coeff[0];
      this->coeff[1] = coeff[1];
      for (int i = 0; i < 4; ++i) this->X[i] = U.X()[i];
      for (int i = 0; i < 4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
    }
  };

  template<typename real, typename Arg> __global__ void interiorOprodKernel(Arg arg)
  {
    using complex = complex<real>;
    using matrix = Matrix<complex, Arg::nColor>;
    using vector = ColorSpinor<real, Arg::nColor, 1>;

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int gridSize = gridDim.x*blockDim.x;

    matrix result;

    while (idx < arg.length) {
      const vector x = arg.inA(idx, 0);

#pragma unroll
      for (int dim=0; dim<4; ++dim) {
        int shift[4] = {0,0,0,0};
        shift[dim] = 1;
        const int first_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
        if (first_nbr_idx >= 0) {
          const vector y = arg.inB(first_nbr_idx, 0);
          result = outerProduct(y, x);
          matrix tempA = arg.U(dim, idx, arg.parity);
          result = tempA + result*arg.coeff[0];

          arg.U(dim, idx, arg.parity) = result;

          if (arg.nFace == 3) {
            shift[dim] = 3;
            const int third_nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
            if (third_nbr_idx >= 0) {
              const vector z = arg.inB(third_nbr_idx, 0);
              result = outerProduct(z, x);
              matrix tempB = arg.L(dim, idx, arg.parity);
              result = tempB + result*arg.coeff[1];
              arg.L(dim, idx, arg.parity) = result;
            }
          }
        }
      } // dim

      idx += gridSize;
    }
  } // interiorOprodKernel

  template<int dim, typename real, typename Arg> __global__ void exteriorOprodKernel(Arg arg)
  {
    using complex = complex<real>;
    using matrix = Matrix<complex, Arg::nColor>;
    using vector = ColorSpinor<real, Arg::nColor, 1>;

    unsigned int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int gridSize = gridDim.x*blockDim.x;

    matrix result;

    auto &out = (arg.displacement == 1) ? arg.U : arg.L;
    real coeff = (arg.displacement == 1) ? arg.coeff[0] : arg.coeff[1];

    int x[4];
    while (cb_idx < arg.length) {
      coordsFromIndexExterior(x, cb_idx, arg.X, arg.dir, arg.displacement, arg.parity);
      const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);

      matrix inmatrix = out(arg.dir, bulk_cb_idx, arg.parity);
      const vector a = arg.inA(bulk_cb_idx, 0);
      const vector b = arg.inB.Ghost(arg.dir, 1, cb_idx, 0);

      result = outerProduct(b, a);
      result = inmatrix + result*coeff;
      out(arg.dir, bulk_cb_idx, arg.parity) = result;

      cb_idx += gridSize;
    }
  }

  template<typename Float, typename Arg>
  class StaggeredOprodField : public Tunable {

    Arg &arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.U.volumeCB; }
    bool tunedGridDim() const { return false; }

  public:
    StaggeredOprodField(Arg &arg, const GaugeField &meta)
      : arg(arg), meta(meta) {
      writeAuxString(meta.AuxString());
    }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	// Disable tuning for the time being
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	  interiorOprodKernel<Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	} else if (arg.kernelType == OPROD_EXTERIOR_KERNEL) {
          if (arg.dir == 0) exteriorOprodKernel<0,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 1) exteriorOprodKernel<1,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 2) exteriorOprodKernel<2,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 3) exteriorOprodKernel<3,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	} else {
	  errorQuda("Kernel type not supported\n");
	}
      } else { // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply

    void preTune() { arg.U.save(); arg.L.save(); }
    void postTune() { arg.U.load(); arg.L.load(); }

    long long flops() const { return 0; } // FIXME
    long long bytes() const { return 0; } // FIXME
    TuneKey tuneKey() const {
      char aux[TuneKey::aux_n];
      strcpy(aux, this->aux);
      if (arg.kernelType == OPROD_EXTERIOR_KERNEL) {
        strcat(aux, ",dir=");
        char tmp[2];
        u32toa(tmp, arg.dir);
        strcat(aux, tmp);
        strcat(aux, ",displacement=");
        u32toa(tmp, arg.displacement);
        strcat(aux, tmp);
      }
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }
  }; // StaggeredOprodField

  template <typename Float>
  void computeStaggeredOprod(GaugeField &U, GaugeField &L, ColorSpinorField &inA, ColorSpinorField &inB, int parity, const double coeff[2], int nFace)
  {
    // Create the arguments for the interior kernel
    StaggeredOprodArg<Float, 3> arg(U, L, inA, inB, parity, 0, 1, OPROD_INTERIOR_KERNEL, nFace, coeff);
    StaggeredOprodField<Float, decltype(arg)> oprod(arg, U);

    arg.kernelType = OPROD_INTERIOR_KERNEL;
    arg.length = U.VolumeCB();
    oprod.apply(0);

    for (int i = 3; i >= 0; i--) {
      if (commDimPartitioned(i)) {
        // update parameters for this exterior kernel
        arg.kernelType = OPROD_EXTERIOR_KERNEL;
        arg.dir = i;

        // First, do the one hop term
        {
          arg.displacement = 1;
          arg.length = inB.GhostFaceCB()[i];
          oprod.apply(0);
        }

        // Now do the 3 hop term
        if (nFace == 3) {
          arg.displacement = 3;
          arg.length = arg.displacement * inB.GhostFaceCB()[i];
          oprod.apply(0);
        }
      }
    } // i=3,..,0

    checkCudaError();
  } // computeStaggeredOprod

  void computeStaggeredOprod(GaugeField &U, GaugeField &L, ColorSpinorField &inEven, ColorSpinorField &inOdd,
                             int parity, const double coeff[2], int nFace)
  {
    if (U.Order() != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Unsupported output ordering: %d\n", U.Order());
    if (L.Order() != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Unsupported output ordering: %d\n", L.Order());

    ColorSpinorField &inA = (parity & 1) ? inOdd : inEven;
    ColorSpinorField &inB = (parity & 1) ? inEven : inOdd;

    inB.exchangeGhost((QudaParity)(1-parity), nFace, 0);

    auto prec = checkPrecision(inEven, inOdd, U, L);
    if (prec == QUDA_DOUBLE_PRECISION) {
      computeStaggeredOprod<double>(U, L, inA, inB, parity, coeff, nFace);
    } else if (prec == QUDA_SINGLE_PRECISION) {
      computeStaggeredOprod<float>(U, L, inA, inB, parity, coeff, nFace);
    } else {
      errorQuda("Unsupported precision: %d", prec);
    }

    inB.bufferIndex = (1 - inB.bufferIndex);
  }

  void computeStaggeredOprod(GaugeField *out[], ColorSpinorField& in, const double coeff[], int nFace)
  {
#ifdef GPU_STAGGERED_DIRAC
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
#else // GPU_STAGGERED_DIRAC not defined
    errorQuda("Staggered Outer Product has not been built!");
#endif
  }

} // namespace quda
