#include <cstdio>
#include <cstdlib>

#include <tune_quda.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <dslash_quda.h>

namespace quda {

  enum OprodKernelType { OPROD_INTERIOR_KERNEL, OPROD_EXTERIOR_KERNEL };

  template<typename Float, int nColor_, QudaReconstructType recon>
  struct CloverForceArg {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr int spin_project = true;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project>::type;
    using Gauge = typename gauge_mapper<Float, recon, 18>::type;
    using Force = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;

    const F inA;
    const F inB;
    const F inC;
    const F inD;
    Gauge  gauge;
    Force force;
    unsigned int length;
    int X[4];
    unsigned int parity;
    unsigned int dir;
    unsigned int displacement;
    OprodKernelType kernelType;
    bool partitioned[4];
    Float coeff;

    CloverForceArg(GaugeField &force, const GaugeField &gauge, const ColorSpinorField &inA, const ColorSpinorField &inB,
                   const ColorSpinorField &inC, const ColorSpinorField &inD, const unsigned int parity, const double coeff) :
      inA(inA),
      inB(inB),
      inC(inC),
      inD(inD),
      gauge(gauge),
      force(force),
      length(gauge.VolumeCB()),
      parity(parity),
      dir(5),
      displacement(1),
      kernelType(OPROD_INTERIOR_KERNEL),
      coeff(coeff)
    {
      for (int i=0; i<4; ++i) this->X[i] = gauge.X()[i];
      for (int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
    }
  };

  template<typename real, typename Arg> __global__ void interiorOprodKernel(Arg arg)
  {
    typedef complex<real> Complex;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    ColorSpinor<real, Arg::nColor, 4> A, B_shift, C, D_shift;
    Matrix<Complex, Arg::nColor> U, result, temp;

    while (idx<arg.length) {
      A = arg.inA(idx, 0);
      C = arg.inC(idx, 0);

#pragma unroll
      for (int dim=0; dim<4; ++dim) {
	int shift[4] = {0,0,0,0};
	shift[dim] = 1;
	const int nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);

	if (nbr_idx >= 0) {
	  B_shift = arg.inB(nbr_idx, 0);
	  D_shift = arg.inD(nbr_idx, 0);

	  B_shift = (B_shift.project(dim,1)).reconstruct(dim,1);
	  result = outerProdSpinTrace(B_shift,A);

	  D_shift = (D_shift.project(dim,-1)).reconstruct(dim,-1);
	  result += outerProdSpinTrace(D_shift,C);

	  temp = arg.force(dim, idx, arg.parity);
	  U = arg.gauge(dim, idx, arg.parity);
	  result = temp + U*result*arg.coeff;
	  arg.force(dim, idx, arg.parity) = result;
	}
      } // dim

      idx += gridDim.x*blockDim.x;
    }
  } // interiorOprodKernel

  template<int dim, typename real, typename Arg> __global__ void exteriorOprodKernel(Arg arg)
  {
    typedef complex<real> Complex;
    int cb_idx = blockIdx.x*blockDim.x + threadIdx.x;

    ColorSpinor<real, Arg::nColor, 4> A, B_shift, C, D_shift;
    ColorSpinor<real, Arg::nColor, 2> projected_tmp;
    Matrix<Complex, Arg::nColor> U, result, temp;

    int x[4];
    while (cb_idx<arg.length) {
      coordsFromIndexExterior(x, cb_idx, arg.X, dim, arg.displacement, arg.parity);
      const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);
      A = arg.inA(bulk_cb_idx, 0);
      C = arg.inC(bulk_cb_idx, 0);

      projected_tmp = arg.inB.Ghost(dim, 1, cb_idx, 0);
      B_shift = projected_tmp.reconstruct(dim, 1);
      result = outerProdSpinTrace(B_shift,A);

      projected_tmp = arg.inD.Ghost(dim, 1, cb_idx, 0);
      D_shift = projected_tmp.reconstruct(dim,-1);
      result += outerProdSpinTrace(D_shift,C);

      temp = arg.force(dim, bulk_cb_idx, arg.parity);
      U = arg.gauge(dim, bulk_cb_idx, arg.parity);
      result = temp + U*result*arg.coeff;
      arg.force(dim, bulk_cb_idx, arg.parity) = result;

      cb_idx += gridDim.x*blockDim.x;
    }
  } // exteriorOprodKernel

  template<typename Float, typename Arg>
  class CloverForce : public Tunable {
    Arg &arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.length; }
    bool tuneGridDim() const { return false; }

  public:
    CloverForce(Arg &arg, GaugeField &meta) :
      arg(arg), meta(meta) {
      writeAuxString(meta.AuxString());
      // this sets the communications pattern for the packing kernel
      int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1), commDimPartitioned(2), commDimPartitioned(3) };
      setPackComms(comms);
    }

    void apply(const qudaStream_t &stream) {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	// Disable tuning for the time being
	TuneParam tp = tuneLaunch(*this,getTuning(),getVerbosity());

	if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	  interiorOprodKernel<Float><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
        } else if (arg.kernelType == OPROD_EXTERIOR_KERNEL) {
          if (arg.dir == 0)
            exteriorOprodKernel<0,Float><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 1) exteriorOprodKernel<1,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 2) exteriorOprodKernel<2,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
	  else if (arg.dir == 3) exteriorOprodKernel<3,Float><<<tp.grid,tp.block,tp.shared_bytes, stream>>>(arg);
        } else {
          errorQuda("Kernel type not supported\n");
        }
      }else{ // run the CPU code
	errorQuda("No CPU support for staggered outer-product calculation\n");
      }
    } // apply

    void preTune() {
      this->arg.force.save();
    }
    void postTune() {
      this->arg.force.load();
    }

    long long flops() const {
      if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	return ((long long)arg.length)*4*(24 + 144 + 234); // spin project + spin trace + multiply-add
      } else {
	return ((long long)arg.length)*(144 + 234); // spin trace + multiply-add
      }
    }
    long long bytes() const {
      if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	return arg.length*(arg.inA.Bytes() + arg.inC.Bytes() + 4*(arg.inB.Bytes() + arg.inD.Bytes() + 2*arg.force.Bytes() + arg.gauge.Bytes()));
      } else {
	return arg.length*(arg.inA.Bytes() + arg.inB.Bytes()/2 + arg.inC.Bytes() + arg.inD.Bytes()/2 + 2*arg.force.Bytes() + arg.gauge.Bytes());
      }
    }

    TuneKey tuneKey() const {
      char new_aux[TuneKey::aux_n];
      strcpy(new_aux, aux);
      if (arg.kernelType == OPROD_INTERIOR_KERNEL) {
	strcat(new_aux, ",interior");
      } else {
	strcat(new_aux, ",exterior");
	if (arg.dir==0) strcat(new_aux, ",dir=0");
	else if (arg.dir==1) strcat(new_aux, ",dir=1");
	else if (arg.dir==2) strcat(new_aux, ",dir=2");
	else if (arg.dir==3) strcat(new_aux, ",dir=3");
      }
      return TuneKey(meta.VolString(), "CloverForce", new_aux);
    }
  }; // CloverForce

  void exchangeGhost(cudaColorSpinorField &a, int parity, int dag) {
    // need to enable packing in temporal direction to get spin-projector correct
    pushKernelPackT(true);

    // first transfer src1
    qudaDeviceSynchronize();

    MemoryLocation location[2*QUDA_MAX_DIM] = {Device, Device, Device, Device, Device, Device, Device, Device};
    a.pack(1, 1-parity, dag, Nstream-1, location, Device);

    qudaDeviceSynchronize();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	// Initialize the host transfer from the source spinor
	a.gather(1, dag, 2*i);
      } // commDim(i)
    } // i=3,..,0

    qudaDeviceSynchronize(); comm_barrier();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	a.commsStart(1, 2*i, dag);
      }
    }

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	a.commsWait(1, 2*i, dag);
	a.scatter(1, dag, 2*i);
      }
    }

    qudaDeviceSynchronize();
    popKernelPackT(); // restore packing state

    a.bufferIndex = (1 - a.bufferIndex);
    comm_barrier();
  }

  template <typename Float, QudaReconstructType recon>
  void computeCloverForce(GaugeField &force, const GaugeField &gauge, const ColorSpinorField& inA, const ColorSpinorField& inB,
                          const ColorSpinorField& inC, const ColorSpinorField& inD, int parity, const double coeff)
  {
    // Create the arguments for the interior kernel
    CloverForceArg<Float, 3, recon> arg(force, gauge, inA, inB, inC, inD, parity, coeff);
    CloverForce<Float,decltype(arg)> oprod(arg, force);

    arg.kernelType = OPROD_INTERIOR_KERNEL;
    arg.length = inA.VolumeCB();
    oprod.apply(0);

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
        // update parameters for this exterior kernel
        arg.kernelType = OPROD_EXTERIOR_KERNEL;
        arg.dir = i;
        arg.length = inA.GhostFaceCB()[i];
        arg.displacement = 1; // forwards displacement
        oprod.apply(0);
      }
    } // i=3,..,0
  } // computeCloverForce

  void computeCloverForce(GaugeField &force, const GaugeField &U, std::vector<ColorSpinorField *> &x,
                          std::vector<ColorSpinorField *> &p, std::vector<double> &coeff)
  {
#ifdef GPU_CLOVER_DIRAC
    if (force.Order() != QUDA_FLOAT2_GAUGE_ORDER) errorQuda("Unsupported output ordering: %d\n", force.Order());
    checkPrecision(*x[0], *p[0], force, U);

    int dag = 1;

    for (unsigned int i=0; i<x.size(); i++) {
      static_cast<cudaColorSpinorField&>(x[i]->Even()).allocateGhostBuffer(1);
      static_cast<cudaColorSpinorField&>(x[i]->Odd()).allocateGhostBuffer(1);
      static_cast<cudaColorSpinorField&>(p[i]->Even()).allocateGhostBuffer(1);
      static_cast<cudaColorSpinorField&>(p[i]->Odd()).allocateGhostBuffer(1);

      for (int parity=0; parity<2; parity++) {

	ColorSpinorField& inA = (parity&1) ? p[i]->Odd() : p[i]->Even();
	ColorSpinorField& inB = (parity&1) ? x[i]->Even(): x[i]->Odd();
	ColorSpinorField& inC = (parity&1) ? x[i]->Odd() : x[i]->Even();
	ColorSpinorField& inD = (parity&1) ? p[i]->Even(): p[i]->Odd();

	if (x[0]->Precision() == QUDA_DOUBLE_PRECISION) {
          exchangeGhost(static_cast<cudaColorSpinorField&>(inB), parity, dag);
          exchangeGhost(static_cast<cudaColorSpinorField&>(inD), parity, 1-dag);

	  if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	    computeCloverForce<double, QUDA_RECONSTRUCT_NO>(force, U, inA, inB, inC, inD, parity, coeff[i]);
	  } else if (U.Reconstruct() == QUDA_RECONSTRUCT_12) {
	    computeCloverForce<double, QUDA_RECONSTRUCT_12>(force, U, inA, inB, inC, inD, parity, coeff[i]);
	  } else {
	    errorQuda("Unsupported recontruction type");
	  }
	} else {
	  errorQuda("Unsupported precision: %d\n", x[0]->Precision());
	}
      }
    }
#else // GPU_CLOVER_DIRAC not defined
   errorQuda("Clover Dirac operator has not been built!");
#endif

   checkCudaError();
  } // computeCloverForce



} // namespace quda
