#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/clover_outer_product.cuh>

namespace quda {

  enum OprodKernelType { INTERIOR, INTERIOR_DOUBLET, EXTERIOR, EXTERIOR_DOUBLET };

  template <typename Float, int nColor, QudaReconstructType recon> class CloverForce : public TunableKernel1D {
    using real = typename mapper<Float>::type;
    template <int dim = -1> using Arg = CloverForceArg<Float, nColor, recon, dim>;
    GaugeField &force;
    const GaugeField &U;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const ColorSpinorField &inC;
    const ColorSpinorField &inD;
    const int parity;
    const real coeff;
    const bool doublet;         // whether we applying the operator to a doublet
    OprodKernelType kernel;
    int dir;
    unsigned int minThreads() const override { return kernel <= INTERIOR_DOUBLET ? inB.VolumeCB() : inB.GhostFaceCB()[dir]; }

  public:
    CloverForce(const GaugeField &U, GaugeField &force, const ColorSpinorField& inA,
                const ColorSpinorField& inB, const ColorSpinorField& inC, const ColorSpinorField& inD,
                int parity, double coeff) :
      TunableKernel1D(force),
      force(force),
      U(U),
      inA(inA),
      inB(inB),
      inC(inC),
      inD(inD),
      parity(parity),
      coeff(static_cast<real>(coeff)),
      doublet(inA.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET)
    {
      char aux2[TuneKey::aux_n];
      strcpy(aux2, aux);
      strcat(aux, ",interior,doublet=");
      strcat(aux, doublet == 0 ? "0" : "1");
      if (doublet) kernel = INTERIOR_DOUBLET;
      else kernel = INTERIOR;
      apply(device::get_default_stream());

      for (int i=3; i>=0; i--) {
        dir = i;
        if (!commDimPartitioned(i)) continue;
        strcpy(aux, aux2);
        strcat(aux, ",exterior,dir=");
        strcat(aux, dir == 0 ? "0" : dir == 1 ? "1" : dir == 2 ? "2" : "3");
        strcat(aux, ",doublet=");
        strcat(aux, doublet == 0 ? "0" : "1");
        if (doublet) kernel = EXTERIOR_DOUBLET;
        else kernel = EXTERIOR;
        apply(device::get_default_stream());
      }
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (kernel == INTERIOR) {
        launch<Interior>(tp, stream, Arg<>(force, U, inA, inB, inC, inD, parity, coeff));
      } else if (kernel == INTERIOR_DOUBLET) {
        launch<InteriorDoublet>(tp, stream, Arg<>(force, U, inA, inB, inC, inD, parity, coeff));
      } else if (kernel == EXTERIOR) {
        switch (dir) {
        case 0: launch<Exterior>(tp, stream, Arg<0>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 1: launch<Exterior>(tp, stream, Arg<1>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 2: launch<Exterior>(tp, stream, Arg<2>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 3: launch<Exterior>(tp, stream, Arg<3>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        default: errorQuda("Unexpected direction %d", dir);
        }
      } else if (kernel == EXTERIOR_DOUBLET) {
        switch (dir) {
        case 0: launch<ExteriorDoublet>(tp, stream, Arg<0>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 1: launch<ExteriorDoublet>(tp, stream, Arg<1>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 2: launch<ExteriorDoublet>(tp, stream, Arg<2>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        case 3: launch<ExteriorDoublet>(tp, stream, Arg<3>(force, U, inA, inB, inC, inD, parity, coeff)); break;
        default: errorQuda("Unexpected direction %d", dir);
        }
      }
    }

    void preTune() override { force.backup(); }
    void postTune() override { force.restore(); }

    // spin trace + multiply-add (ignore spin-project)
    long long flops() const override { return minThreads() * (144 + 234) * (kernel <= INTERIOR_DOUBLET ? 4 : 1); }

    long long bytes() const override
    {
      if (kernel <= INTERIOR_DOUBLET) {
	return inA.Bytes() + inC.Bytes() + 4*(inB.Bytes() + inD.Bytes()) + force.Bytes() + U.Bytes() / 2;
      } else {
	return minThreads() * (nColor * (4 * 2 + 2 * 2) + 2 * force.Reconstruct() + U.Reconstruct())
          * sizeof(Float);
      }
    }
  }; // CloverForce

  void exchangeGhost(const ColorSpinorField &a, int parity, int dag)
  {
    // this sets the communications pattern for the packing kernel
    int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1),
                                commDimPartitioned(2), commDimPartitioned(3) };

    setPackComms(comms);

    // first transfer src1
    qudaDeviceSynchronize();

    MemoryLocation location[2*QUDA_MAX_DIM] = {Device, Device, Device, Device, Device, Device, Device, Device};
    a.pack(1, 1-parity, dag, device::get_default_stream(), location, Device);

    qudaDeviceSynchronize();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	// Initialize the host transfer from the source spinor
	a.gather(2*i, device::get_stream(2*i));
      } // commDim(i)
    } // i=3,..,0

    qudaDeviceSynchronize(); comm_barrier();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	a.commsStart(2*i, device::get_stream(2 * i));
      }
    }

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	a.commsWait(2*i, device::get_stream(2*i));
	a.scatter(2*i, device::get_stream(2*i));
      }
    }

    qudaDeviceSynchronize();

    a.bufferIndex = (1 - a.bufferIndex);
    comm_barrier();
  }

  void computeCloverForce(GaugeField &force, const GaugeField &U, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<const ColorSpinorField> &p, const std::vector<double> &coeff)
  {
    if constexpr (is_enabled_clover()) {
      checkNative(x[0], p[0], force, U);
      checkPrecision(x[0], p[0], force, U);

      int dag = 1;

      for (auto i = 0u; i < x.size(); i++) {
        for (int parity = 0; parity < 2; parity++) {
          const ColorSpinorField &inA = (parity & 1) ? p[i].Odd() : p[i].Even();
          const ColorSpinorField &inB = (parity & 1) ? x[i].Even() : x[i].Odd();
          const ColorSpinorField &inC = (parity & 1) ? x[i].Odd() : x[i].Even();
          const ColorSpinorField &inD = (parity & 1) ? p[i].Even() : p[i].Odd();

          getProfile().TPSTART(QUDA_PROFILE_COMMS);
          exchangeGhost(inB, parity, dag);
          exchangeGhost(inD, parity, 1 - dag);
          getProfile().TPSTOP(QUDA_PROFILE_COMMS);

          getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
          instantiate<CloverForce, ReconstructNo12>(U, force, inA, inB, inC, inD, parity, coeff[i]);
          getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
        }
      }
    } else {
      errorQuda("Clover Dirac operator has not been built!");
    }
  }

} // namespace quda
