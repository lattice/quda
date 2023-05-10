#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, int dim_ = -1>
  struct StaggeredOprodArg : kernel_param<> {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr int dim = dim_;
    using F = typename colorspinor_mapper<Float, nSpin, nColor>::type;
    using GU = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using GL = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;

    GU U;        /** output one-hop field */
    GL L;        /** output three-hop field */
    const F inA; /** input vector field */
    const F inB; /** input vector field */

    const int parity;
    int displacement;
    const int nFace;
    real coeff[2];
    int X[4];
    bool partitioned[4];

    StaggeredOprodArg(GaugeField &U, GaugeField &L, const ColorSpinorField &inA, const ColorSpinorField &inB,
                      int parity, int displacement, int nFace, const real coeff[2]) :
      kernel_param(dim3(dim == -1 ? inB.VolumeCB() : displacement * inB.GhostFaceCB()[dim])),
      U(U),
      L(L),
      inA(inA),
      inB(inB, nFace),
      parity(parity),
      displacement(displacement),
      nFace(nFace),
      coeff{coeff[0], coeff[1]}
    {
      for (int i = 0; i < 4; ++i) this->X[i] = U.X()[i];
      for (int i = 0; i < 4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
    }
  };

  template <typename Arg> struct Interior
  {
    const Arg &arg;
    constexpr Interior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using matrix = Matrix<complex<typename Arg::real>, Arg::nColor>;
      using vector = ColorSpinor<typename Arg::real, Arg::nColor, 1>;

      const vector x = arg.inA(x_cb, 0);

#pragma unroll
      for (int dim=0; dim<4; ++dim) {
        int shift[4] = {0,0,0,0};
        shift[dim] = 1;
        const int first_nbr_idx = neighborIndex(x_cb, shift, arg.partitioned, arg.parity, arg.X);
        if (first_nbr_idx >= 0) {
          const vector y = arg.inB(first_nbr_idx, 0);
          matrix result = outerProduct(y, x);
          matrix tempA = arg.U(dim, x_cb, arg.parity);
          result = tempA + result * arg.coeff[0];

          arg.U(dim, x_cb, arg.parity) = result;

          if (arg.nFace == 3) {
            shift[dim] = 3;
            const int third_nbr_idx = neighborIndex(x_cb, shift, arg.partitioned, arg.parity, arg.X);
            if (third_nbr_idx >= 0) {
              const vector z = arg.inB(third_nbr_idx, 0);
              matrix result = outerProduct(z, x);
              matrix tempB = arg.L(dim, x_cb, arg.parity);
              result = tempB + result * arg.coeff[1];
              arg.L(dim, x_cb, arg.parity) = result;
            }
          }
        }
      }
    }
  };

  template <typename Arg> struct Exterior {
    const Arg &arg;
    constexpr Exterior(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using matrix = Matrix<complex<typename Arg::real>, Arg::nColor>;
      using vector = ColorSpinor<typename Arg::real, Arg::nColor, 1>;

      auto coeff = (arg.displacement == 1) ? arg.coeff[0] : arg.coeff[1];

      int x[4];
      coordsFromIndexExterior(x, x_cb, arg.X, Arg::dim, arg.displacement, arg.parity);
      const unsigned int bulk_cb_idx = ((((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] + x[0]) >> 1);

      const vector a = arg.inA(bulk_cb_idx, 0);
      const vector b = arg.inB.Ghost(Arg::dim, 1, x_cb, 0);

      if (arg.displacement == 1) {
        matrix inmatrix = arg.U(Arg::dim, bulk_cb_idx, arg.parity);
        matrix result = outerProduct(b, a);
        result = inmatrix + result * coeff;
        arg.U(Arg::dim, bulk_cb_idx, arg.parity) = result;
      } else {
        matrix inmatrix = arg.L(Arg::dim, bulk_cb_idx, arg.parity);
        matrix result = outerProduct(b, a);
        result = inmatrix + result * coeff;
        arg.L(Arg::dim, bulk_cb_idx, arg.parity) = result;
      }
    }
  };

}
