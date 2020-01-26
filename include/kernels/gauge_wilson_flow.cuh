#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  enum WFlowStepType {
    WFLOW_STEP_W1,
    WFLOW_STEP_W2,
    WFLOW_STEP_VT,
  };

  template <typename Float_, int nColor_, QudaReconstructType recon_, int wflow_dim_>
  struct GaugeWFlowArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int wflow_dim = wflow_dim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    Gauge temp;
    const Gauge in;

    int threads; // number of active threads required
    int_fastdiv X[4];    // grid dimensions
    int border[4];
    int_fastdiv E[4];
    const Float epsilon;
    const QudaWFlowType wflow_type;
    const WFlowStepType step_type;

    GaugeWFlowArg(GaugeField &out, GaugeField &temp, const GaugeField &in, const Float epsilon, const QudaWFlowType wflow_type, const WFlowStepType step_type) :
      out(out),
      in(in),
      temp(temp),
      threads(1),
      epsilon(epsilon),
      wflow_type(wflow_type),
      step_type(step_type)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        threads *= X[dir];
        E[dir] = in.X()[dir];
      }
      threads /= 2;
    }
  };

  template <QudaWFlowType wflow_type, typename Arg>
  __host__ __device__ void computeW1Step(Arg &arg, const int *x, const int parity, const int x_cb, const int dir)
  {
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    Link U, Stap, Rect, Z0, exp_Z0, Id, temp1, temp2;
    Complex im(0.0,-1.0);
    const real coeff1x1 = 5.0/3.0;
    const real coeff2x1 = -1.0/12.0;

    // Get link U
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    // Compute staples and Z0
    switch(wflow_type) {
    case QUDA_WFLOW_TYPE_WILSON :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, arg.E, parity, dir, Stap, Arg::wflow_dim);
      Z0 = Stap * conj(U);
      break;
    case QUDA_WFLOW_TYPE_SYMANZIK :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, x, arg.E, parity, dir, Stap, Rect, Arg::wflow_dim);
      Z0 = (coeff1x1 * Stap + coeff2x1 * Rect) * conj(U);
      break;
    }

    arg.temp(dir, x_cb, parity) = Z0;
    Z0 *= (1.0 / 4.0) * arg.epsilon;

    // Compute anti-hermitian projection of Z0, exponentiate, update U
    makeAntiHerm(Z0);
    Z0 = im * Z0;
    exponentiate_iQ(Z0, &exp_Z0);
    U = exp_Z0 * U;
    arg.out(dir, linkIndex(x, arg.E), parity) = U;
  }

  template <QudaWFlowType wflow_type, typename Arg>
  __host__ __device__ void computeW2Step(Arg &arg, const int *x, const int parity, const int x_cb, const int dir)
  {
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    Link U, Stap, Rect, Z1, Z0, exp_Z1, Id, temp1, temp2;
    Complex im(0.0,-1.0);
    const real coeff1x1 = 5.0/3.0;
    const real coeff2x1 = -1.0/12.0;

    // Get link U
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    // Compute staples and Z1
    switch(wflow_type) {
    case QUDA_WFLOW_TYPE_WILSON :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, arg.E, parity, dir, Stap, Arg::wflow_dim);
      Z1 = (8.0/9.0) * Stap * conj(U);
      break;
    case QUDA_WFLOW_TYPE_SYMANZIK :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, x, arg.E, parity, dir, Stap, Rect, Arg::wflow_dim);
      Z1 = (8.0/9.0) * (coeff1x1 * Stap + coeff2x1 * Rect) * conj(U);
      break;
    }

    // Retrieve Z0, (8/9 Z1 - 17/36 Z0) stored in temp
    Z0 = arg.temp(dir, x_cb, parity);
    Z0 *= (17.0 / 36.0);
    Z1 = Z1 - Z0;
    arg.temp(dir, x_cb, parity) = Z1;
    Z1 *= arg.epsilon;

    // Compute anti-hermitian projection of Z1, exponentiate, update U
    makeAntiHerm(Z1);
    Z1 = im * Z1;
    exponentiate_iQ(Z1, &exp_Z1);
    U = exp_Z1 * U;
    arg.out(dir, linkIndex(x, arg.E), parity) = U;
  }

  template <QudaWFlowType wflow_type, typename Arg>
  __host__ __device__ void computeVtStep(Arg &arg, const int *x, const int parity, const int x_cb, const int dir)
  {
    using real = typename Arg::Float;
    typedef complex<real> Complex;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    Link U, Stap, Rect, Z2, Z1, exp_Z2, Id, temp1, temp2;
    Complex im(0.0,-1.0);
    const real coeff1x1 = 5.0/3.0;
    const real coeff2x1 = -1.0/12.0;

    // Get link U
    U = arg.in(dir, linkIndex(x, arg.E), parity);
    // Compute staples and Z1
    switch(wflow_type) {
    case QUDA_WFLOW_TYPE_WILSON :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, arg.E, parity, dir, Stap, Arg::wflow_dim);
      Z2 = (3.0/4.0) * Stap * conj(U);
      break;
    case QUDA_WFLOW_TYPE_SYMANZIK :
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      // and the 1x2 and 2x1 rectangles of length 5. From the following paper:
      // https://arxiv.org/abs/0801.1165
      computeStapleRectangle(arg, x, arg.E, parity, dir, Stap, Rect, Arg::wflow_dim);
      Z2 = (3.0/4.0) * (coeff1x1 * Stap + coeff2x1 * Rect) * conj(U);
      break;
    }

    // Use (8/9 Z1 - 17/36 Z0) computed from W2 step
    Z1 = arg.temp(dir, x_cb, parity);
    Z2 = Z2 - Z1;
    Z2 *= arg.epsilon;

    // Compute anti-hermitian projection of Z2, exponentiate, update U
    makeAntiHerm(Z2);
    Z2 = im * Z2;
    exponentiate_iQ(Z2, &exp_Z2);
    U = exp_Z2 * U;
    arg.out(dir, linkIndex(x, arg.E), parity) = U;
  }

  // Wilson Flow as defined in https://arxiv.org/abs/1006.4518v3
  template <QudaWFlowType wflow_type, typename Arg> __global__ void computeWFlowStep(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    int dir = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_cb >= arg.threads) return;
    if (dir >= Arg::wflow_dim) return;

    //Get stacetime and local coords
    int x[4];
    getCoords(x, x_cb, arg.X, parity);
    for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr];

    switch(arg.step_type) {
    case WFLOW_STEP_W1: computeW1Step<wflow_type>(arg, x, parity, x_cb, dir); break;
    case WFLOW_STEP_W2: computeW2Step<wflow_type>(arg, x, parity, x_cb, dir); break;
    case WFLOW_STEP_VT: computeVtStep<wflow_type>(arg, x, parity, x_cb, dir); break;
    }
  }

} // namespace quda
