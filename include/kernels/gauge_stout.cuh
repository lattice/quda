#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>
#include <thread_local_cache.h>

namespace quda
{

  template <typename store_t, int nColor_, QudaReconstructType recon_, int stoutDim_> struct STOUTArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int stoutDim = stoutDim_;
    typedef typename gauge_mapper<store_t, recon>::type Gauge;

    Gauge out;
    const Gauge in;

    int_fastdiv X[4]; // regular grid dims
    int_fastdiv E[4]; // extended grid dims
    int border[4];
    const real rho;
    const real staple_coeff;
    const real rectangle_coeff;
    const int dir_ignore;
    const real anisotropy;

    STOUTArg(GaugeField &out, const GaugeField &in, real rho, real epsilon, int dir_ignore, real anisotropy) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, stoutDim)),
      out(out),
      in(in),
      rho(rho),
      staple_coeff(rho * (5.0 - 2.0 * epsilon) / 3.0),
      rectangle_coeff(rho * (1.0 - epsilon) / 12.0),
      dir_ignore(dir_ignore),
      anisotropy(anisotropy)
    {
      for (int dir = 0; dir < 4; ++dir) {
        E[dir] = in.X()[dir];
        border[dir] = in.R()[dir];
        X[dir] = E[dir] - 2 * border[dir];
      }
    }
  };

  template <typename Arg> struct STOUT {

    const Arg &arg;
    constexpr STOUT(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // Compute spacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr)
        x[dr] += arg.border[dr];
      dir = dir + (dir >= arg.dir_ignore);

      Link U, Stap, Q;

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, arg.E, parity, dir, Stap, arg.dir_ignore);

      // Get link U
      U = arg.in(dir, linkIndex(x, arg.E), parity);

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
      arg.out(dir, linkIndex(x, arg.E), parity) = U;

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
  template <typename Arg> struct OvrImpSTOUTOps {
    using real = typename Arg::real;
    using Complex = complex<real>;
    using Link = Matrix<complex<real>, Arg::nColor>;
    using StapCacheT = ThreadLocalCache<Link>;                               // zero offset
    using RectCacheT = ThreadLocalCache<Link, 0, StapCacheT>;                // offset by StapCacheT
    using Ops = KernelOps<StapCacheT, RectCacheT>;
  };

  template <typename Arg> struct OvrImpSTOUT : OvrImpSTOUTOps<Arg>::Ops {
    using typename OvrImpSTOUTOps<Arg>::Ops::KernelOpsT;

    const Arg &arg;
    template <typename... OpsArgs>
    constexpr OvrImpSTOUT(const Arg &arg, const OpsArgs &...ops) : KernelOpsT(ops...), arg(arg)
    {
    }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // Compute spacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr)
        x[dr] += arg.border[dr];
      dir = dir + (dir >= arg.dir_ignore);

      Link U, Q;
      typename OvrImpSTOUTOps<Arg>::StapCacheT Stap {*this};
      typename OvrImpSTOUTOps<Arg>::RectCacheT Rect {*this};

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, x, arg.E, parity, dir, Stap, Rect, arg.dir_ignore);

      // Get link U
      U = arg.in(dir, linkIndex(x, arg.E), parity);

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //-------------------------------------------------------------------
      // Compute \rho * staple_coeff * S - \rho * rectangle_coeff * R
      Q = ((arg.staple_coeff * static_cast<const Link &>(Stap)) - (arg.rectangle_coeff * static_cast<const Link &>(Rect)))
        * conj(U);
      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
      makeHerm(Q);
      // Q is now defined.

      Link exp_iQ = exponentiate_iQ(Q);
      U = exp_iQ * U;
      arg.out(dir, linkIndex(x, arg.E), parity) = U;

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
