#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/clover_outer_product.cuh>

namespace quda {

  enum OprodKernelType { INTERIOR, EXTERIOR };

  template <typename Float, int nColor, QudaReconstructType recon> class CloverOprod : public TunableKernel3D
  {
    using real = typename mapper<Float>::type;
    template <int dim = -1, bool doublet = false> using Arg = CloverOprodArg<Float, nColor, recon, dim, doublet>;
    GaugeField &force;
    const GaugeField &U;
    cvector_ref<const ColorSpinorField> &p;
    const ColorSpinorField &p_halo;
    cvector_ref<const ColorSpinorField> &x;
    const ColorSpinorField &x_halo;
    const std::vector<double> &coeff;
    const bool doublet; // whether we applying the operator to a doublet
    const int n_flavor;
    OprodKernelType kernel;
    int dir;
    unsigned int minThreads() const override
    {
      return (kernel == INTERIOR ? (int)x_halo.getDslashConstant().volume_4d_cb :
                                   x_halo.getDslashConstant().ghostFaceCB[dir]);
    }

  public:
    CloverOprod(const GaugeField &U, GaugeField &force, cvector_ref<const ColorSpinorField> &p,
                const ColorSpinorField &p_halo, cvector_ref<const ColorSpinorField> &x, const ColorSpinorField &x_halo,
                const std::vector<double> &coeff) :
      TunableKernel3D(force, x.SiteSubset(), 4),
      force(force),
      U(U),
      p(p),
      p_halo(p_halo),
      x(x),
      x_halo(x_halo),
      coeff(coeff),
      doublet(x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      n_flavor(doublet ? 2 : 1)
    {
      if (doublet) strcat(aux, ",doublet");
      setRHSstring(aux, p.size());
      char aux2[TuneKey::aux_n];
      strcpy(aux2, aux);
      strcat(aux, ",interior");
      kernel = INTERIOR;
      apply(device::get_default_stream());

      for (int i = 3; i >= 0; i--) {
        resizeVector(x.SiteSubset(), 1);
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
          launch<Interior>(tp, stream, Arg<-1, true>(force, U, p, p_halo, x, x_halo, coeff));
        else
          launch<Interior>(tp, stream, Arg<>(force, U, p, p_halo, x, x_halo, coeff));
      } else if (kernel == EXTERIOR) {
        switch (dir) {
        case 0: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<0, true>(force, U, p, p_halo, x, x_halo, coeff));
          else
            launch<Exterior>(tp, stream, Arg<0>(force, U, p, p_halo, x, x_halo, coeff));
          break;
        }
        case 1: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<1, true>(force, U, p, p_halo, x, x_halo, coeff));
          else
            launch<Exterior>(tp, stream, Arg<1>(force, U, p, p_halo, x, x_halo, coeff));
          break;
        }
        case 2: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<2, true>(force, U, p, p_halo, x, x_halo, coeff));
          else
            launch<Exterior>(tp, stream, Arg<2>(force, U, p, p_halo, x, x_halo, coeff));
          break;
        }
        case 3: {
          if (doublet)
            launch<Exterior>(tp, stream, Arg<3, true>(force, U, p, p_halo, x, x_halo, coeff));
          else
            launch<Exterior>(tp, stream, Arg<3>(force, U, p, p_halo, x, x_halo, coeff));
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

      return 2 * minThreads() * n_flavor * p.size() * (2 * oprod_flops + gemm_flops + 3 * mat_size)
        * (kernel == INTERIOR ? 4 : 1);
    }

    long long bytes() const override
    {
      if (kernel == INTERIOR) {
        return 8 * (x.Bytes() + p.Bytes()) + 2 * force.Bytes() + U.Bytes();
      } else {
        return 2 * minThreads()
          * (n_flavor * p.size() * nColor * (2 * x.Nspin() + 2 * x.Nspin() / 2) * 2 + 2 * force.Reconstruct()
             + U.Reconstruct())
          * sizeof(Float);
      }
    }
  }; // CloverOprod

  void exchangeGhost(const ColorSpinorField &halo, cvector_ref<const ColorSpinorField> &v, int dag)
  {
    // this sets the communications pattern for the packing kernel
    int comms[QUDA_MAX_DIM] = { commDimPartitioned(0), commDimPartitioned(1),
                                commDimPartitioned(2), commDimPartitioned(3) };

    setPackComms(comms);

    // first transfer src1
    qudaDeviceSynchronize();

    MemoryLocation location[2*QUDA_MAX_DIM] = {Device, Device, Device, Device, Device, Device, Device, Device};
    halo.pack(1, 0, dag, device::get_default_stream(), location, Device, true, 0.0, 0.0, 0.0, 0, v);

    qudaDeviceSynchronize();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
	// Initialize the host transfer from the source spinor
        halo.gather(2 * i, device::get_stream(2 * i));
      } // commDim(i)
    } // i=3,..,0

    qudaDeviceSynchronize(); comm_barrier();

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) { halo.commsStart(2 * i, device::get_stream(2 * i)); }
    }

    for (int i=3; i>=0; i--) {
      if (commDimPartitioned(i)) {
        halo.commsWait(2 * i, device::get_stream(2 * i));
        halo.scatter(2 * i, device::get_stream(2 * i));
      }
    }

    qudaDeviceSynchronize();

    halo.bufferIndex = (1 - halo.bufferIndex);
    comm_barrier();
  }

  void computeCloverOprod(GaugeField &force, const GaugeField &U, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<const ColorSpinorField> &p, const std::vector<double> &coeff)
  {
    if constexpr (is_enabled_clover()) {
      if (x.size() > get_max_multi_rhs()) {
        computeCloverOprod(force, U, {x.begin(), x.begin() + x.size() / 2}, {p.begin(), p.begin() + p.size() / 2},
                           {coeff.begin(), coeff.begin() + coeff.size() / 2});
        computeCloverOprod(force, U, {x.begin() + x.size() / 2, x.end()}, {p.begin() + p.size() / 2, p.end()},
                           {coeff.begin() + coeff.size() / 2, coeff.end()});
        return;
      }

      checkNative(x, p, force, U);
      checkPrecision(x, p, force, U);

      int dag = 1;
      getProfile().TPSTART(QUDA_PROFILE_COMMS);
      auto x_halo = ColorSpinorField::create_comms_batch(x);
      auto p_halo = ColorSpinorField::create_comms_batch(p);
      exchangeGhost(x_halo, x, dag);
      exchangeGhost(p_halo, p, 1 - dag);
      getProfile().TPSTOP(QUDA_PROFILE_COMMS);

      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      instantiate<CloverOprod, ReconstructNo12>(U, force, p, p_halo, x, x_halo, coeff);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Clover Dirac operator has not been built");
    }
  }

} // namespace quda
