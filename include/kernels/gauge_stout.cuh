#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int stoutDim_>
  struct STOUTArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int stoutDim = stoutDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    const Gauge in;

    int X[4];    // grid dimensions
    int border[4];
    const Float rho;
    const Float staple_coeff;
    const Float rectangle_coeff;

    STOUTArg(GaugeField &out, const GaugeField &in, Float rho, Float epsilon = 0) :
      kernel_param(dim3(1, 2, stoutDim)),
      out(out),
      in(in),
      rho(rho),
      staple_coeff(rho * (5.0 - 2.0 * epsilon) / 3.0),
      rectangle_coeff(rho * (1.0 - epsilon) / 12.0)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        this->threads.x *= X[dir];
      }
      this->threads.x /= 2;
    }
  };

  template <typename Arg> struct STOUT
  {
    using real = typename Arg::Float;
    using Complex = complex<real>;
    using Link = Matrix<complex<real>, Arg::nColor>;

    const Arg &arg;
    constexpr STOUT(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      // Compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link U, Stap, Q;

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, X, parity, dir, Stap, Arg::stoutDim);

      // Get link U
      U = arg.in(dir, linkIndex(x, X), parity);

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //--------------------------------------------------------------------
      // Compute \Omega = \rho * S * U^{\dagger}
      Q = (arg.rho * Stap) * conj(U);
      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
      makeHerm(Q);
      // Q is now defined.

      Link exp_iQ = exponentiate_iQ(Q);
      U = exp_iQ * U;
      arg.out(dir, linkIndex(x, X), parity) = U;

      // Debug tools
#if 0
      //Test for Traceless:
      double error = getTrace(Q).real();
      printf("Trace test %d %d %.15e\n", x_cb, dir, error);
      //Test for hermiticity:
      Link Q_diff = conj(Q) - Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
      Q_diff *= Q_diff;
      error = getTrace(Q_diff).real();
      printf("Herm test %d %d %.15e\n", x_cb, dir, error);
      //Test for expiQ unitarity:
      error = ErrorSU3(exp_iQ);
      printf("expiQ test %d %d %.15e\n", x_cb, dir, error);
      //Test for expiQ*U unitarity:
      error = ErrorSU3(U);
      printf("expiQ*u test %d %d %.15e\n", x_cb, dir, error);
#endif
    }
  };

  //------------------------//
  // Over-Improved routines //
  //------------------------//
  template <typename Arg> struct OvrImpSTOUT
  {
    using real = typename Arg::Float;
    using Complex = complex<real>;
    using Link = Matrix<complex<real>, Arg::nColor>;

    const Arg &arg;
    constexpr OvrImpSTOUT(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      // Compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link U, Q;
      SharedMemoryCache<Link> Stap(target::block_dim());
      SharedMemoryCache<Link> Rect(target::block_dim(), sizeof(Link));

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, x, X, parity, dir, Stap, Rect, Arg::stoutDim);

      // Get link U
      U = arg.in(dir, linkIndex(x, X), parity);

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //-------------------------------------------------------------------
      // Compute \rho * staple_coeff * S - \rho * rectangle_coeff * R
      Q = ((arg.staple_coeff * static_cast<const Link &>(Stap)) - (arg.rectangle_coeff * static_cast<const Link &>(Rect))) * conj(U);
      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
      makeHerm(Q);
      // Q is now defined.

      Link exp_iQ = exponentiate_iQ(Q);
      U = exp_iQ * U;
      arg.out(dir, linkIndex(x, X), parity) = U;

      // Debug tools
#if 0
      //Test for Traceless:
      double error = getTrace(Q).real();
      printf("Trace test %d %d %.15e\n", x_cb, dir, error);
      //Test for hermiticity:
      Link Q_diff = conj(Q) - Q; //This should be the zero matrix. Test by ReTr(Q_diff^2);
      Q_diff *= Q_diff;
      error = getTrace(Q_diff).real();
      printf("Herm test %d %d %.15e\n", x_cb, dir, error);
      //Test for expiQ unitarity:
      error = ErrorSU3(exp_iQ);
      printf("expiQ test %d %d %.15e\n", x_cb, dir, error);
      //Test for expiQ*U unitarity:
      error = ErrorSU3(U);
      printf("expiQ*u test %d %d %.15e\n", x_cb, dir, error);
#endif
    }
  };

} // namespace quda
