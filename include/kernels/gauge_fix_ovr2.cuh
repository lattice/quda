#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int parity_, bool over_relaxation_>
  struct FixGaugeArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int parity = parity_;
    static constexpr bool over_relaxation = over_relaxation_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    Gauge rot;
    const Gauge u;

    int X[4]; // grid dimensions
    int border[4];
    const Float omega;
    const int dir_ignore;
    const Float tolerance;

    FixGaugeArg(GaugeField &rot, const GaugeField &u, double omega, int dir_ignore) :
      kernel_param(dim3(u.LocalVolumeCB())),
      rot(rot),
      u(u),
      omega(omega),
      dir_ignore(dir_ignore),
      tolerance(u.toleranceSU3())
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = u.R()[dir];
        X[dir] = u.X()[dir] - border[dir] * 2;
      }
    }
  };

  /**
   * @brief Maximize the real trace of UW in SU(2) subgroups and apply
   * the result to U. Note it's equivalent to a closest unitary matrix
   * problem in SU(2) subgroups.
   * U' = \argmax_{U}\mathfrak{Re}\mathrm{Tr}(UV) = \argmin_{U}||U-V^\dagger||_F^2
   * Specifically, for GL(2,C) matrices, we have
   * U = \frac{V^\dagger}{\sqrt{\det(V^\dagger)}}
   * Also accepts the over-relaxation parameter for gauge fixing
   */
  template <int su2_index, typename Float, typename Arg>
  __host__ __device__ inline void argmaxReTrUW(Matrix<complex<Float>, 3> &U, Matrix<complex<Float>, 3> &W, const Arg &arg)
  {
    int i1, i2;
    switch (su2_index) {
    case 0: i1 = 0, i2 = 1; break;
    case 1: i1 = 1, i2 = 2; break;
    case 2: i1 = 0, i2 = 2; break;
    default: break;
    }

    Matrix<complex<Float>, 3> V = U * W;
    double versors[4]; // use double to avoid precision issues

    versors[0] = static_cast<double>(V(i1, i1).real() + V(i2, i2).real());
    versors[1] = static_cast<double>(V(i1, i1).imag() - V(i2, i2).imag());
    versors[2] = static_cast<double>(V(i1, i2).real() - V(i2, i1).real());
    versors[3] = static_cast<double>(V(i1, i2).imag() + V(i2, i1).imag());

    double norm
      = sqrt(versors[0] * versors[0] + versors[1] * versors[1] + versors[2] * versors[2] + versors[3] * versors[3]);
    if (norm > arg.tolerance) {
      double inv_norm = 1.0 / norm;
      versors[0] *= inv_norm;
#pragma unroll
      for (int i = 1; i < 4; ++i) { versors[i] *= -inv_norm; }
    } else {
      versors[0] = 1.0;
#pragma unroll
      for (int i = 1; i < 4; ++i) { versors[i] = 0.0; }
    }

    if constexpr (Arg::over_relaxation) {
      // a workaround for fp32 numerical issues
      // clamp versors[0] to [-1, 1] to avoid NaN in acos
      // if (versors[0] > 1.0) {
      //   versors[0] = 1.0;
      // } else if (versors[0] < -1.0) {
      //   versors[0] = -1.0;
      // }
      double angle = acos(versors[0]);
      double sin_angle = sin(angle);
      double sin_omega_angle, cos_omega_angle;
      sincos(arg.omega * angle, &sin_omega_angle, &cos_omega_angle);
      versors[0] = cos_omega_angle;
      if (sin_angle > arg.tolerance) {
        double coeff = sin_omega_angle / sin_angle;
#pragma unroll
        for (int i = 1; i < 4; ++i) { versors[i] *= coeff; }
      } else {
#pragma unroll
        for (int i = 1; i < 4; ++i) { versors[i] = 0.0; }
      }
    }

    setIdentity(&V);
    V(i1, i1) = complex(static_cast<Float>(versors[0]), static_cast<Float>(versors[1]));
    V(i2, i2) = complex(static_cast<Float>(versors[0]), static_cast<Float>(-versors[1]));
    V(i1, i2) = complex(static_cast<Float>(versors[2]), static_cast<Float>(versors[3]));
    V(i2, i1) = complex(static_cast<Float>(-versors[2]), static_cast<Float>(versors[3]));

    U = V * U;
  }

  // /**
  //  * @brief Solving the closest unitary matrix in SU(2) subgroups.
  //  * This is another way to project the input matrix on the SU(3) group.
  //  */
  // template <typename Float, typename Arg>
  // __host__ __device__ inline void closestSu3(Matrix<complex<Float>,3> &in, Float tol)
  // {
  //   Matrix<complex<Float>, 3> out;
  //   setIdentity(&out);

  //   constexpr int max_iter = 100;
  //   int i = 0;
  //   Float old_retr, retr = getTrace(in).real();
  //   do { // iterate until matrix is unitary
  //     // loop over SU(2) subgroup indices
  //     argmaxReTrUW<0, Float>(out, in);
  //     argmaxReTrUW<1, Float>(out, in);
  //     argmaxReTrUW<2, Float>(out, in);
  //     old_retr = retr;
  //     retr = getTrace(out * in).real();
  //   } while (abs(retr - old_retr) / old_retr > tol && ++i < max_iter);

  //   in = out;
  // }

  template <typename Arg> struct FixGauge {
    const Arg &arg;
    constexpr FixGauge(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;
      constexpr int parity = Arg::parity;

      // compute spacetime and local coords
      int x[4], X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) { X[dr] = arg.X[dr]; }
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link g, K, tmp, tmp2;
      g = arg.rot(0, linkIndex(x, X), parity);
#pragma unroll
      for (int dir = 0; dir < 4; ++dir) {
        if (dir != arg.dir_ignore) {
          tmp = arg.u(dir, linkIndex(x, X), parity);
          tmp2 = arg.rot(0, linkIndexP1(x, X, dir), 1 - parity);
          K += tmp * conj(tmp2);
          tmp = arg.u(dir, linkIndexM1(x, X, dir), 1 - parity);
          tmp2 = arg.rot(0, linkIndexM1(x, X, dir), 1 - parity);
          K += conj(tmp2 * tmp);
        }
      }

      // loop over SU(2) subgroup indices
      argmaxReTrUW<0, real>(g, K, arg);
      argmaxReTrUW<1, real>(g, K, arg);
      argmaxReTrUW<2, real>(g, K, arg);

      arg.rot(0, linkIndex(x, X), parity) = g;
    }
  };
} // namespace quda
