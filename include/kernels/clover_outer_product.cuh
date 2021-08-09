#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon, int dim_ = -1>
  struct CloverForceArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr int dim = dim_;
    static constexpr int spin_project = true;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project>::type;
    using Gauge = typename gauge_mapper<Float, recon, 18>::type;
    using Force = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;

    Force force;
    const F inA;
    const F inB;
    const F inC;
    const F inD;
    const Gauge U;
    int X[4];
    int parity;
    int displacement;
    bool partitioned[4];
    real coeff;

    CloverForceArg(GaugeField &force, const GaugeField &U, const ColorSpinorField &inA, const ColorSpinorField &inB,
                   const ColorSpinorField &inC, const ColorSpinorField &inD, const unsigned int parity, const double coeff) :
      kernel_param(dim3(dim == -1 ? inA.VolumeCB() : inA.GhostFaceCB()[dim])),
      force(force),
      inA(inA),
      inB(inB),
      inC(inC),
      inD(inD),
      U(U),
      parity(parity),
      displacement(1),
      coeff(coeff)
    {
      for (int i=0; i<4; ++i) this->X[i] = U.X()[i];
      for (int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
    }
  };

  template <typename Arg> struct Interior {
    const Arg &arg;
    constexpr Interior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using Complex = complex<typename Arg::real>;
      using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      using Link = Matrix<Complex, Arg::nColor>;

      Spinor A = arg.inA(x_cb, 0);
      Spinor C = arg.inC(x_cb, 0);

#pragma unroll
      for (int dim=0; dim<4; ++dim) {
        int shift[4] = {0, 0, 0, 0};
        shift[dim] = 1;
        const int nbr_idx = neighborIndex(x_cb, shift, arg.partitioned, arg.parity, arg.X);

        if (nbr_idx >= 0) {
          Spinor B_shift = arg.inB(nbr_idx, 0);
          Spinor D_shift = arg.inD(nbr_idx, 0);

          B_shift = (B_shift.project(dim,1)).reconstruct(dim,1);
          Link result = outerProdSpinTrace(B_shift,A);

          D_shift = (D_shift.project(dim,-1)).reconstruct(dim,-1);
          result += outerProdSpinTrace(D_shift,C);

          Link temp = arg.force(dim, x_cb, arg.parity);
          Link U = arg.U(dim, x_cb, arg.parity);
          result = temp + U*result*arg.coeff;
          arg.force(dim, x_cb, arg.parity) = result;
        }
      } // dim
    }
  };

  template <typename Arg> struct Exterior {
    const Arg &arg;
    constexpr Exterior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using Complex = complex<typename Arg::real>;
      using Spinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      using HalfSpinor = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
      using Link = Matrix<Complex, Arg::nColor>;

      int x[4];
      coordsFromIndexExterior(x, x_cb, arg.X, Arg::dim, arg.displacement, arg.parity);
      const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);
      Spinor A = arg.inA(bulk_cb_idx, 0);
      Spinor C = arg.inC(bulk_cb_idx, 0);

      HalfSpinor projected_tmp = arg.inB.Ghost(Arg::dim, 1, x_cb, 0);
      Spinor B_shift = projected_tmp.reconstruct(Arg::dim, 1);
      Link result = outerProdSpinTrace(B_shift,A);

      projected_tmp = arg.inD.Ghost(Arg::dim, 1, x_cb, 0);
      Spinor D_shift = projected_tmp.reconstruct(Arg::dim,-1);
      result += outerProdSpinTrace(D_shift,C);

      Link temp = arg.force(Arg::dim, bulk_cb_idx, arg.parity);
      Link U = arg.U(Arg::dim, bulk_cb_idx, arg.parity);
      result = temp + U*result*arg.coeff;
      arg.force(Arg::dim, bulk_cb_idx, arg.parity) = result;
    }
  };

}
