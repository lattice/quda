#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon, int dim_ = -1, bool doublet_ = false>
  struct CloverForceArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr int dim = dim_;
    static constexpr int spin_project = true;
    static constexpr bool doublet = doublet_; // whether we applying the operator to a doublet
    static constexpr int n_flavor = doublet ? 2 : 1;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project>::type;
    using Gauge = typename gauge_mapper<Float, recon, 18>::type;
    using Force = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;

    Force force;
    const F x;
    const F p;
    const Gauge U;
    int X[4];
    int displacement;
    bool partitioned[4];
    real coeff;
    const unsigned int volume_4d_cb;
    const unsigned int ghost_face_4d_cb;

    CloverForceArg(GaugeField &force, const GaugeField &U, const ColorSpinorField &p, const ColorSpinorField &x,
                   const double coeff) :
      kernel_param(dim3(
        dim == -1 ? (x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? x.VolumeCB() / 2 : x.VolumeCB()) :
                    (x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? x.GhostFaceCB()[dim] / 2 : x.GhostFaceCB()[dim]),
        x.SiteSubset(), dim == -1 ? 4 : 1)),
      force(force),
      x(x),
      p(p),
      U(U),
      displacement(1),
      coeff(coeff),
      volume_4d_cb(x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? x.VolumeCB() / 2 : x.VolumeCB()),
      ghost_face_4d_cb(x.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? x.GhostFaceCB()[dim] / 2 : x.GhostFaceCB()[dim])
    {
      for (int i=0; i<4; ++i) this->X[i] = U.X()[i];
      for (int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;

      // need to reset the ghost pointers since default ghost_offset
      // (Ghost() method) not set (this is temporary work around)
      void *ghost[8] = {};
      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) { ghost[2 * dim + dir] = (char *)x.Ghost2() + x.GhostOffset(dim, dir); }
      }
      this->x.resetGhost(ghost);
      p.bufferIndex = (1 - p.bufferIndex);

      for (int dim = 0; dim < 4; dim++) {
        for (int dir = 0; dir < 2; dir++) { ghost[2 * dim + dir] = (char *)p.Ghost2() + p.GhostOffset(dim, dir); }
      }
      this->p.resetGhost(ghost);
      p.bufferIndex = (1 - p.bufferIndex);
    }
  };

  template <typename Arg> struct Interior {
    const Arg &arg;
    constexpr Interior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <int mu> __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using Complex = complex<typename Arg::real>;
      using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      using Link = Matrix<Complex, Arg::nColor>;

      int shift[4] = {0, 0, 0, 0};
      shift[mu] = 1;
      const int nbr_idx = neighborIndex(x_cb, shift, arg.partitioned, parity, arg.X);

      if (nbr_idx >= 0) {
        Link result = {};
#pragma unroll
        for (int flavor = 0; flavor < Arg::n_flavor; ++flavor) {
          const int flavor_offset_idx = flavor * arg.volume_4d_cb;
          Spinor A = arg.p(x_cb + flavor_offset_idx, parity);
          Spinor C = arg.x(x_cb + flavor_offset_idx, parity);
          Spinor B_shift = arg.x(nbr_idx + flavor_offset_idx, 1 - parity);
          Spinor D_shift = arg.p(nbr_idx + flavor_offset_idx, 1 - parity);

          B_shift = (B_shift.project(mu, 1)).reconstruct(mu, 1);
          result += outerProdSpinTrace(B_shift, A);

          D_shift = (D_shift.project(mu, -1)).reconstruct(mu, -1);
          result += outerProdSpinTrace(D_shift, C);
        } // flavor

        Link force = arg.force(mu, x_cb, parity);
        Link U = arg.U(mu, x_cb, parity);
        force = force + U * result * arg.coeff;
        arg.force(mu, x_cb, parity) = force;
      }
    }

    __device__ __host__ inline void operator()(int x_cb, int parity, int mu)
    {
      switch (mu) {
      case 0: this->operator()<0>(x_cb, parity); break;
      case 1: this->operator()<1>(x_cb, parity); break;
      case 2: this->operator()<2>(x_cb, parity); break;
      case 3: this->operator()<3>(x_cb, parity); break;
      }
    }
  };

  template <typename Arg> struct Exterior {
    const Arg &arg;
    constexpr Exterior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int)
    {
      using Complex = complex<typename Arg::real>;
      using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      using HalfSpinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
      using Link = Matrix<Complex, Arg::nColor>;

      int x[4];
#pragma unroll
      for (int flavor = 0; flavor < Arg::n_flavor; ++flavor) {
        const int flavor_offset_bulk_idx = flavor * arg.volume_4d_cb;
        const int flavor_offset_ghost_idx = flavor * arg.ghost_face_4d_cb;
        coordsFromIndexExterior(x, x_cb, arg.X, Arg::dim, arg.displacement, parity);
        const unsigned int bulk_cb_idx = ((((x[3] * arg.X[2] + x[2]) * arg.X[1] + x[1]) * arg.X[0] + x[0]) >> 1);
        Spinor A = arg.p(bulk_cb_idx + flavor_offset_bulk_idx, parity);
        Spinor C = arg.x(bulk_cb_idx + flavor_offset_bulk_idx, parity);

        HalfSpinor projected_tmp = arg.x.Ghost(Arg::dim, 1, x_cb + flavor_offset_ghost_idx, 1 - parity);
        Spinor B_shift = projected_tmp.reconstruct(Arg::dim, 1);
        Link result = outerProdSpinTrace(B_shift, A);

        projected_tmp = arg.p.Ghost(Arg::dim, 1, x_cb + flavor_offset_ghost_idx, 1 - parity);
        Spinor D_shift = projected_tmp.reconstruct(Arg::dim, -1);
        result += outerProdSpinTrace(D_shift, C);

        Link temp = arg.force(Arg::dim, bulk_cb_idx, parity);
        Link U = arg.U(Arg::dim, bulk_cb_idx, parity);
        result = temp + U * result * arg.coeff;
        arg.force(Arg::dim, bulk_cb_idx, parity) = result;
      }
    }
  };

}
