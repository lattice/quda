#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <fast_intdiv.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int level_, int hypDim_>
  struct GaugeHYPArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int hypDim = hypDim_;
    static constexpr int level = level_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    Gauge out;
    Gauge tmp[4];
    const Gauge in;

    int_fastdiv E[4]; // extended grid dimensions
    int_fastdiv X[4]; // grid dimensions
    int border[4];
    const Float alpha;
    const int dir_ignore;
    const Float tolerance;

    GaugeHYPArg(GaugeField &out, GaugeField* tmp[4], const GaugeField &in, double alpha, int dir_ignore) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, hypDim)),
      out(out),
      tmp {*tmp[0], *tmp[1], *tmp[2], *tmp[3]},
      in(in),
      alpha(alpha),
      dir_ignore(dir_ignore),
      tolerance(in.toleranceSU3())
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        E[dir] = in.X()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
      }
    }
  };

  /**
     @brief Calculates a staple as part of the HYP calculation, returning the product

     @param[in,out] staple The accumulated staple
     @param[in] arg Kernel argument
     @param[in] gauge_mu Gauge/tensor field used for the parallel direction
     @param[in] gauge_nu Gauge/tensor field used for the perpendicular direction
     @param[in] x Full index array
     @param[in] parity Parity index
     @param[in] tensor_arg The {parallel, perpendicular} indices into the gauge/tensor fields
     @tparam The {parallel, perpendicular} pair of directions for coordinate offsets
  */
  template <int mu, int nu, typename Arg>
  __host__ __device__ inline void accumulateStaple(Matrix<complex<typename Arg::Float>, Arg::nColor> &staple, const Arg &arg,
    const typename Arg::Gauge &gauge_mu, const typename Arg::Gauge &gauge_nu,
    int x[], int parity, int2 tensor_arg)
  {
    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
    int dx[4] = {0, 0, 0, 0};

    /* Computes the upper staple :
     *                 mu (B)
     *               +-------+
     *         nu    |       |
     *         (A)   |       |(C)
     *               X       X
     */
    {
      /* load matrix A*/
      Link a = gauge_nu(tensor_arg.y, linkIndex(x, arg.E), parity);

      /* load matrix B*/
      dx[nu]++;
      Link b = gauge_mu(tensor_arg.x, linkIndexShift(x, dx, arg.E), 1-parity);
      dx[nu]--;

      /* load matrix C*/
      dx[mu]++;
      Link c = gauge_nu(tensor_arg.y, linkIndexShift(x, dx, arg.E), 1-parity);
      dx[mu]--;

      staple = staple + a * b * conj(c);
    }

    /* Computes the lower staple :
     *                 X       X
     *           nu    |       |
     *           (A)   |       | (C)
     *                 +-------+
     *                   mu (B)
     */
    {
      /* load matrix A*/
      dx[nu]--;
      Link a = gauge_nu(tensor_arg.y, linkIndexShift(x, dx, arg.E), 1-parity);

      /* load matrix B*/
      Link b = gauge_mu(tensor_arg.x, linkIndexShift(x, dx, arg.E), 1-parity);

      /* load matrix C*/
      dx[mu]++;
      Link c = gauge_nu(tensor_arg.y, linkIndexShift(x, dx, arg.E), parity);
      dx[mu]--;
      dx[nu]++;

      staple = staple + conj(a)*b*c;
    }
  }

  /**
     @brief Calculates a staple as part of the HYP calculation, returning the product

     @param[in,out] staple The accumulated staple
     @param[in] arg Kernel argument
     @param[in] gauge_mu Gauge/tensor field used for the parallel direction
     @param[in] gauge_nu Gauge/tensor field used for the perpendicular direction
     @param[in] x Full index array
     @param[in] parity Parity index
     @param[in] tensor_arg The {parallel, perpendicular} indices into the gauge/tensor fields
     @param[in] shifts The {parallel, perpendicular} pair of directions for coordinate offsets
  */
  template <typename Arg>
  __host__ __device__ inline void accumulateStaple(Matrix<complex<typename Arg::Float>, Arg::nColor> &staple, const Arg &arg,
          const typename Arg::Gauge &gauge_mu, const typename Arg::Gauge &gauge_nu, int x[], int parity, int2 tensor_arg, int2 shifts)
  {
    // for readability
    const int mu = shifts.x;
    const int nu = shifts.y;

    switch (mu) {
      case 0:
        switch (nu) {
        case 1: accumulateStaple<0,1>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 2: accumulateStaple<0,2>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 3: accumulateStaple<0,3>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        } break;
      case 1:
        switch (nu) {
        case 0: accumulateStaple<1,0>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 2: accumulateStaple<1,2>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 3: accumulateStaple<1,3>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        } break;
      case 2:
        switch (nu) {
        case 0: accumulateStaple<2,0>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 1: accumulateStaple<2,1>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 3: accumulateStaple<2,3>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
      } break;
      case 3:
        switch (nu) {
        case 0: accumulateStaple<3,0>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 1: accumulateStaple<3,1>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        case 2: accumulateStaple<3,2>(staple, arg, gauge_mu, gauge_nu, x, parity, tensor_arg); break;
        } break;
      }
  }

  template <typename Arg>
  __host__ __device__ inline void computeStapleLevel1(const Arg &arg, int x[], int parity,
                                                      int mu, Matrix<complex<typename Arg::Float>, Arg::nColor> staple[3])
  {
    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
    for (int i = 0; i < 3; ++i) staple[i] = Link();

    thread_array<int, 4> dx = {};
    int cnt = -1;
#pragma unroll
    for (int nu = 0; nu < 4; ++nu) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      cnt += 1;

      accumulateStaple(staple[cnt], arg, arg.in, arg.in, x, parity, { mu, nu }, { mu, nu });

    }
  }

  template <typename Arg>
  __host__ __device__ inline void computeStapleLevel2(const Arg &arg, int x[], int parity,
                                                      int mu, Matrix<complex<typename Arg::Float>, Arg::nColor> staple[3])
  {
    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
    for (int i = 0; i < 3; ++i) staple[i] = Link();

    thread_array<int, 4> dx = {};
    int cnt = -1;
#pragma unroll
    for (int nu = 0; nu < 4; nu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      cnt += 1;

      for (int rho = 0; rho < 4; ++rho) {
        if (rho == mu || rho == nu) continue;
        int sigma = 0;
        while (sigma == mu || sigma == nu || sigma == rho) sigma += 1;

        const int sigma_with_rho = rho % 2 * 3 + sigma - (sigma > rho);
        const int sigma_with_mu = mu % 2 * 3 + sigma - (sigma > mu);

        accumulateStaple(staple[cnt], arg, arg.tmp[mu / 2], arg.tmp[rho / 2], x, parity, {sigma_with_mu, sigma_with_rho}, {mu, rho});
      }
    }
  }

  template <typename Arg>
  __host__ __device__ inline void computeStapleLevel3(const Arg &arg, int x[], int parity,
                                                      int mu, Matrix<complex<typename Arg::Float>, Arg::nColor> staple[3])
  {
    using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;

    thread_array<int, 4> dx = {};
#pragma unroll
    for (int nu = 0; nu < 4; nu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      const int mu_with_nu = nu % 2 * 3 + mu - (mu > nu);
      const int nu_with_mu = mu % 2 * 3 + nu - (nu > mu);

      accumulateStaple(staple[0], arg, arg.tmp[mu / 2 + 2], arg.tmp[nu / 2 + 2], x, parity, {nu_with_mu, mu_with_nu}, {mu, nu});
    }
  }

  template <typename Arg> struct HYP {
    const Arg &arg;
    constexpr HYP(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;

      // compute spacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap[3], TestU, I;

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, arg.E), parity);
      setIdentity(&I);

      if constexpr (Arg::level == 1) {
        computeStapleLevel1(arg, x, parity, dir, Stap);

        for (int i = 0; i < 3; ++i) {
          TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap[i] * conj(U) * (arg.alpha / ((real)2.));
          polarSu3<real>(TestU, arg.tolerance);
          arg.tmp[dir / 2](dir % 2 * 3 + i, linkIndexShift(x, dx, arg.E), parity) = TestU * U;
        }
      } else if constexpr (Arg::level == 2) {
        computeStapleLevel2(arg, x, parity, dir, Stap);

        for (int i = 0; i < 3; ++i) {
          TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap[i] * conj(U) * (arg.alpha / ((real)4.));
          polarSu3<real>(TestU, arg.tolerance);
          arg.tmp[dir / 2 + 2](dir % 2 * 3 + i, linkIndexShift(x, dx, arg.E), parity) = TestU * U;
        }
      } else if constexpr (Arg::level == 3) {
        computeStapleLevel3(arg, x, parity, dir, Stap);

        TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap[0] * conj(U) * (arg.alpha / ((real)6.));
        polarSu3<real>(TestU, arg.tolerance);
        arg.out(dir, linkIndexShift(x, dx, arg.E), parity) = TestU * U;
      }
    }
  };

  template <typename Arg, typename Staple>
  __host__ __device__ inline void computeStaple3DLevel1(const Arg &arg, int x[], int parity,
                                                        int mu, Staple staple[2], const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    for (int i = 0; i < 2; ++i) staple[i] = Link();

    thread_array<int, 4> dx = {};
    int cnt = -1;
#pragma unroll
    for (int nu = 0; nu < 4; ++nu) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu || nu == dir_ignore) continue;

      cnt += 1;

      accumulateStaple(staple[cnt], arg, arg.in, arg.in, x, parity, {mu, nu}, {mu, nu});
    }
  }

  template <typename Arg, typename Staple>
  __host__ __device__ inline void computeStaple3DLevel2(const Arg &arg, int x[], int parity,
                                                        int mu, Staple staple[2], int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    staple[0] = Link();

    thread_array<int, 4> dx = {};
#pragma unroll
    for (int nu = 0; nu < 4; nu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu || nu == dir_ignore) continue;

      for (int rho = 0; rho < 4; ++rho) {
        if (rho == mu || rho == nu || rho == dir_ignore) continue;

        const int rho_with_nu = (nu - (nu > dir_ignore)) * 2 + rho - (rho > nu) - (rho > dir_ignore);
        const int rho_with_mu = (mu - (mu > dir_ignore)) * 2 + rho - (rho > mu) - (rho > dir_ignore);

        accumulateStaple(staple[0], arg, arg.tmp[0], arg.tmp[0], x, parity, {rho_with_mu, rho_with_nu}, {mu, nu});
      }
    }
  }

  template <typename Arg> struct HYP3D {
    const Arg &arg;
    constexpr HYP3D(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      int dir_ = dir;
      dir = dir + (dir >= arg.dir_ignore);

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap[2], TestU, I;

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, arg.E), parity);
      setIdentity(&I);

      if constexpr (Arg::level == 1) {
        computeStaple3DLevel1(arg, x, parity, dir, Stap, arg.dir_ignore);

        for (int i = 0; i < 2; ++i) {
          TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap[i] * conj(U) * (arg.alpha / ((real)2.));
          polarSu3<real>(TestU, arg.tolerance);
          arg.tmp[0](dir_ * 2 + i, linkIndexShift(x, dx, arg.E), parity) = TestU * U;
        }
      } else if constexpr (Arg::level == 2) {
        computeStaple3DLevel2(arg, x, parity, dir, Stap, arg.dir_ignore);

        TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap[0] * conj(U) * (arg.alpha / ((real)4.));
        polarSu3<real>(TestU, arg.tolerance);
        arg.out(dir, linkIndexShift(x, dx, arg.E), parity) = TestU * U;
      }
    }
  };
} // namespace quda