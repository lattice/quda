#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernels/gauge_utils.cuh>
#include <su3_project.cuh>
#include <kernel.h>
#include <thread_local_cache.h>

namespace quda
{

  template <typename store_t, int nColor_, QudaReconstructType recon_, int wflow_dim_, QudaGaugeSmearType wflow_type_,
            QudaWFlowStepType step_type_>
  struct GaugeWFlowArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int wflow_dim = wflow_dim_;
    static constexpr QudaGaugeSmearType wflow_type = wflow_type_;
    static constexpr QudaWFlowStepType step_type = step_type_;
    typedef typename gauge_mapper<store_t, recon>::type Gauge;
    typedef typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type Matrix; // temp field not on the manifold

    Gauge out;
    Matrix temp;
    const Gauge in;

    int_fastdiv X[4]; // grid dimensions
    int border[4];
    int_fastdiv E[4];
    const real epsilon;
    const real anisotropy;
    const real coeff1x1;
    const real coeff2x1;

    GaugeWFlowArg(GaugeField &out, GaugeField &temp, const GaugeField &in, real epsilon, real anisotropy) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, wflow_dim)),
      out(out),
      temp(temp),
      in(in),
      epsilon(epsilon),
      anisotropy(anisotropy),
      coeff1x1(5.0 / 3.0),
      coeff2x1(-1.0 / 12.0)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        E[dir] = in.X()[dir];
      }
    }
  };

  template <typename Arg> struct computeStapleOpsWF {
    using real = typename Arg::real;
    using Link = Matrix<complex<real>, Arg::nColor>;
    using WilsonOps = NoKernelOps;                    // Ops for case of QUDA_GAUGE_SMEAR_WILSON_FLOW
    using StapOp = ThreadLocalCache<Link>;            // zero offset
    using RectOp = ThreadLocalCache<Link, 0, StapOp>; // offset by StapOp
    using SymanzikOps = KernelOps<StapOp, RectOp>;    // GAUGE_SMEAR_SYMANZIK_FLOW
    using Ops = std::conditional_t<Arg::wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, SymanzikOps, WilsonOps>;
  };

  template <typename Ftor>
  __host__ __device__ inline auto computeStaple(const Ftor &ftor, const int *x, int parity, int dir)
  {
    using Arg = typename Ftor::Arg;
    const Arg &arg = ftor.arg;
    using real = typename Arg::real;
    using Link = Matrix<complex<real>, Arg::nColor>;
    Link Z;
    // Compute staples and Z factor
    static_assert(Arg::wflow_type == QUDA_GAUGE_SMEAR_WILSON_FLOW || Arg::wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW);
    if constexpr (Arg::wflow_type == QUDA_GAUGE_SMEAR_WILSON_FLOW) {
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, arg.E, parity, dir, Z, Arg::wflow_dim);
    } else if constexpr (Arg::wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW) {
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      typename computeStapleOpsWF<Arg>::StapOp Stap {ftor};
      typename computeStapleOpsWF<Arg>::RectOp Rect {ftor};
      computeStapleRectangle(arg, x, arg.E, parity, dir, Stap, Rect, Arg::wflow_dim);
      Z = arg.coeff1x1 * static_cast<const Link &>(Stap) + arg.coeff2x1 * static_cast<const Link &>(Rect);
    }
    return Z;
  }
  
    template <typename Link, typename Ftor>
  __host__ __device__ inline auto computeW1Step(const Ftor &ftor, Link &U, const int *x, const int parity,
                                                const int x_cb, const int dir)
  {
    using Arg = typename Ftor::Arg;
    const Arg &arg = ftor.arg;
    // Compute staples and Z0
    Link Z0 = computeStaple(ftor, x, parity, dir);
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    Z0 *= conj(U);
    arg.temp(dir, x_cb, parity) = Z0;
    Z0 *= static_cast<typename Arg::real>(1.0 / 4.0) * arg.epsilon;
    return Z0;
  }

  template <typename Link, typename Ftor>
  __host__ __device__ inline auto computeW2Step(const Ftor &ftor, Link &U, const int *x, const int parity,
                                                const int x_cb, const int dir)
  {
    using Arg = typename Ftor::Arg;
    const Arg &arg = ftor.arg;
    // Compute staples and Z1
    Link Z1 = static_cast<typename Arg::real>(8.0 / 9.0) * computeStaple(ftor, x, parity, dir);
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    Z1 *= conj(U);

    // Retrieve Z0, (8/9 Z1 - 17/36 Z0) stored in temp
    Link Z0 = arg.temp(dir, x_cb, parity);
    Z0 *= static_cast<typename Arg::real>(17.0 / 36.0);
    Z1 = Z1 - Z0;
    arg.temp(dir, x_cb, parity) = Z1;
    Z1 *= arg.epsilon;
    return Z1;
  }

  template <typename Link, typename Ftor>
  __host__ __device__ inline auto computeVtStep(const Ftor &ftor, Link &U, const int *x, const int parity,
                                                const int x_cb, const int dir)
  {
    using Arg = typename Ftor::Arg;
    const Arg &arg = ftor.arg;
    // Compute staples and Z2
    Link Z2 = static_cast<typename Arg::real>(3.0 / 4.0) * computeStaple(ftor, x, parity, dir);
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    Z2 *= conj(U);

    // Use (8/9 Z1 - 17/36 Z0) computed from W2 step
    Link Z1 = arg.temp(dir, x_cb, parity);
    Z2 = Z2 - Z1;
    Z2 *= arg.epsilon;
    return Z2;
  }
  
  template <typename Link, typename Ftor>
  __host__ __device__ inline auto computeWStep(const Ftor &ftor, Link &U, const int *x, const int parity,
                                        const int x_cb, const int dir, const double coeff_a, const double coeff_b,
                                        const bool get_stored, const bool do_store)
  {
    using Arg = typename Ftor::Arg;
    const Arg &arg = ftor.arg;
    // Compute SU^\dagger
    Link Z = computeStaple(ftor, x, parity, dir);
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    Z *= conj(U);
    if( get_stored ) {
      // Retrieve previous Z stored in temp and multiply by coeff_a
      Link Z0 = arg.temp(dir, x_cb, parity);
      Z0 *= static_cast<typename Arg::real>(coeff_a);
      Z = Z - Z0;
    }
    // Store Z in temp for next stage
    if( do_store ) arg.temp(dir, x_cb, parity) = Z;
    // Return coeff_b*epsilon*Z
    Z *= static_cast<typename Arg::real>(coeff_b) * arg.epsilon;
    return Z;
  }

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3
  template <typename Arg_> struct WFlow : computeStapleOpsWF<Arg_>::Ops {
    using Arg = Arg_;
    using real = typename Arg_::real;
    using typename computeStapleOpsWF<Arg>::Ops::KernelOpsT;
    const Arg &arg;
    template <typename... OpsArgs> constexpr WFlow(const Arg &arg, const OpsArgs &...ops) : KernelOpsT(ops...), arg(arg)
    {
    }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using Link = Matrix<complex<real>, Arg::nColor>;
      complex<real> im(0.0, -1.0);

      // Get spacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr];

      // Coefficients for fourth order integrator
      double coeff_a[] = {0.0, 0.737101392796, 1.634740794341, 0.744739003780, 1.469897351522, 2.813971388035};
      double coeff_b[] = {0.032918605146, 0.823256998200, 0.381530948900, 0.200092213184, 1.718581042715, 0.27};

      Link U, Z;
      switch (arg.step_type) {
      case WFLOW_STEP_W1: Z = computeW1Step(*this, U, x, parity, x_cb, dir); break;
      case WFLOW_STEP_W2: Z = computeW2Step(*this, U, x, parity, x_cb, dir); break;
      case WFLOW_STEP_VT: Z = computeVtStep(*this, U, x, parity, x_cb, dir); break;
      case WFLOW_FOURTH_ORDER_STEP_1:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[0], coeff_b[0], false, true);
        break;
      case WFLOW_FOURTH_ORDER_STEP_2:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[1], coeff_b[1], true, true);
        break;
      case WFLOW_FOURTH_ORDER_STEP_3:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[2], coeff_b[2], true, true);
        break;
      case WFLOW_FOURTH_ORDER_STEP_4:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[3], coeff_b[3], true, true);
        break;
      case WFLOW_FOURTH_ORDER_STEP_5:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[4], coeff_b[4], true, true);
        break;
      case WFLOW_FOURTH_ORDER_STEP_6:
        Z = computeWStep(*this, U, x, parity, x_cb, dir, coeff_a[5], coeff_b[5], true, false);
        break;
      }

      // Compute anti-hermitian projection of Z, exponentiate, update U
      makeAntiHerm(Z);
      Z = im * Z;
      U = exponentiate_iQ(Z) * U;
      arg.out(dir, linkIndex(x, arg.E), parity) = U;
    }
  };

} // namespace quda
