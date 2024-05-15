#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/clover_outer_product.cuh>

namespace quda {

  enum OprodKernelType { INTERIOR, EXTERIOR };

  template <typename Float, int nColor, QudaReconstructType recon> class CloverForce : public TunableKernel2D
  {
    using real = typename mapper<Float>::type;
    template <int dim = -1, bool doublet = false> using Arg = CloverForceArg<Float, nColor, recon, dim, doublet>;
    GaugeField &force;
    const GaugeField &U;
    const ColorSpinorField &p;
    const ColorSpinorField &x;
    const real coeff;
    const bool doublet; // whether we applying the operator to a doublet
    const int n_flavor;
    OprodKernelType kernel;
    int dir;
    unsigned int minThreads() const override
    {
      return (kernel == INTERIOR ? x.VolumeCB() : x.GhostFaceCB()[dir]) / n_flavor;
    }

  public:
    CloverForce(const GaugeField &U, GaugeField &force, const ColorSpinorField &p, const ColorSpinorField &x,
                double coeff) :
      TunableKernel2D(force, x.SiteSubset()),
      force(force),
      U(U),
      p(p),
      x(x),
      coeff(static_cast<real>(coeff)),
      doublet(x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      n_flavor(doublet ? 2 : 1)
    {
      if (doublet) strcat(aux, ",doublet");
      char aux2[TuneKey::aux_n];
      strcpy(aux2, aux);
      strcat(aux, ",interior");
      kernel = INTERIOR;
      apply(device::get_default_stream());

      for (int i=3; i>=0; i--) {
        dir = i;
        if (!commDimPartitioned(i)) continue;
        strcpy(aux, aux2);
        strcat(aux, ",exterior,dir=");
        strcat(aux, dir == 0 ? "0" : dir == 1 ? "1" : dir == 2 ? "2" : "3");
        kernel = EXTERIOR;
        apply(device::get_default_stream());
      }
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (kernel == INTERIOR) {
        if (doublet)
          launch<Interior>(tp, stream, Arg<-1, true>(force, U, p, x, coeff));
        else
          launch<Interior>(tp, stream, Arg<>(force, U, p, x, coeff));
      } else if (kernel == EXTERIOR) {
        switch (dir) {
        case 0: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<0, true>(force, U, p, x, coeff));
          else
            launch<Exterior>(tp, stream, Arg<0>(force, U, p, x, coeff));
          break;
        }
        case 1: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<1, true>(force, U, p, x, coeff));
          else
            launch<Exterior>(tp, stream, Arg<1>(force, U, p, x, coeff));
          break;
        }
        case 2: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<2, true>(force, U, p, x, coeff));
          else
            launch<Exterior>(tp, stream, Arg<2>(force, U, p, x, coeff));
          break;
        }
        case 3: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<3, true>(force, U, p, x, coeff));
          else
            launch<Exterior>(tp, stream, Arg<3>(force, U, p, x, coeff));
          break;
        }
        default: errorQuda("Unexpected direction %d", dir);
        }
      }
    }

    void preTune() override { force.backup(); }
    void postTune() override { force.restore(); }

    // spin trace + multiply-add (ignore spin-project)
    long long flops() const override
    {
      int oprod_flops = nColor * nColor * (8 * x.Nspin() - 2);
      int gemm_flops = nColor * nColor * (8 * nColor - 2);
      int mat_size = 2 * nColor * nColor;

      return 2 * minThreads() * n_flavor * (2 * oprod_flops + gemm_flops + 3 * mat_size) * (kernel == INTERIOR ? 4 : 1);
    }

    long long bytes() const override
    {
      if (kernel == INTERIOR) {
        return x.Bytes() + p.Bytes() + 4 * (x.Bytes() + p.Bytes()) + 2 * force.Bytes() + U.Bytes();
      } else {
        return minThreads() * n_flavor
          * (nColor * (2 * x.Nspin() + 2 * x.Nspin() / 2) * 2 + 2 * force.Reconstruct() + U.Reconstruct())
          * sizeof(Float);
      }
    }
  }; // CloverForce

  void exchangeGhost(const ColorSpinorField &a, int dag)
  {
    // this sets the communications pattern for the packing kernel
    int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1),
                                commDimPartitioned(2), commDimPartitioned(3) };

    setPackComms(comms);

    // first transfer src1
    qudaDeviceSynchronize();

    MemoryLocation location[2*QUDA_MAX_DIM] = {Device, Device, Device, Device, Device, Device, Device, Device};
    a.pack(1, 0, dag, device::get_default_stream(), location, Device);

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
        getProfile().TPSTART(QUDA_PROFILE_COMMS);
        exchangeGhost(x[i], dag);
        exchangeGhost(p[i], 1 - dag);
        getProfile().TPSTOP(QUDA_PROFILE_COMMS);

        getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
        instantiate<CloverForce, ReconstructNo12>(U, force, p[i], x[i], coeff[i]);
        getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      }
    } else {
      errorQuda("Clover Dirac operator has not been built");
    }
  }

} // namespace quda
