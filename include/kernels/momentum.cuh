#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <array.h>
#include <kernel.h>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct MomActionArg : ReduceArg<double> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    const typename gauge_mapper<Float, recon>::type mom;

    MomActionArg(const GaugeField &mom) :
      ReduceArg<double>(dim3(mom.VolumeCB(), 2, 1)),
      mom(mom) { }
  };

  template <typename Arg> struct MomAction : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr MomAction(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // calculate the momentum contribution to the action.  This uses the
    // MILC convention where we subtract 4.0 from each matrix norm in
    // order to increase stability
    __device__ __host__ inline reduce_t operator()(reduce_t &action, int x_cb, int parity)
    {
      using matrix = Matrix<complex<typename Arg::Float>, Arg::nColor>;

      // loop over direction
      for (int mu=0; mu<4; mu++) {
	const matrix mom = arg.mom(mu, x_cb, parity);

        reduce_t local_sum;
        local_sum  = 0.5 * mom(0,0).imag() * mom(0,0).imag();
        local_sum += 0.5 * mom(1,1).imag() * mom(1,1).imag();
        local_sum += 0.5 * mom(2,2).imag() * mom(2,2).imag();
        local_sum += mom(0,1).real() * mom(0,1).real();
        local_sum += mom(0,1).imag() * mom(0,1).imag();
        local_sum += mom(0,2).real() * mom(0,2).real();
        local_sum += mom(0,2).imag() * mom(0,2).imag();
        local_sum += mom(1,2).real() * mom(1,2).real();
        local_sum += mom(1,2).imag() * mom(1,2).imag();
	local_sum -= 4.0;

	action = operator()(action, local_sum);
      }
      return action;
    }
  };

  template<typename Float_, int nColor_, QudaReconstructType recon_>
  struct UpdateMomArg : ReduceArg<array<double, 2>>
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typename gauge_mapper<Float, QUDA_RECONSTRUCT_10>::type mom;
    typename gauge_mapper<Float, recon>::type force;
    Float coeff;
    int X[4]; // grid dimensions on mom
    int E[4]; // grid dimensions on force (possibly extended)
    int border[4]; //

    UpdateMomArg(GaugeField &mom, const Float &coeff, const GaugeField &force) :
      ReduceArg<reduce_t>(dim3(mom.VolumeCB(), 2, 1)),
      mom(mom),
      force(force),
      coeff(coeff)
    {
      for (int dir=0; dir<4; ++dir) {
        X[dir] = mom.X()[dir];
        E[dir] = force.X()[dir];
        border[dir] = force.R()[dir];
      }
    }
  };

  template <typename Arg> struct MomUpdate : maximum<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using maximum<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr MomUpdate(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // calculate the momentum contribution to the action.  This uses the
    // MILC convention where we subtract 4.0 from each matrix norm in
    // order to increase stability
    __device__ __host__ inline reduce_t operator()(reduce_t &norm, int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x, arg.E);

#pragma unroll
      for (int d=0; d<4; d++) {
        Matrix<complex<typename Arg::Float>, Arg::nColor> m = arg.mom(d, x_cb, parity);
        Matrix<complex<typename Arg::Float>, Arg::nColor> f = arg.force(d, e_cb, parity);

        // project to traceless anti-hermitian prior to taking norm
        makeAntiHerm(f);

        // compute force norms
        norm = operator()(reduce_t{f.L1(), f.L2()}, norm);

        m = m + arg.coeff * f;

        // strictly speaking this shouldn't be needed since the
        // momentum should already be traceless anti-hermitian but at
        // present the unit test will fail without this
        makeAntiHerm(m);
        arg.mom(d, x_cb, parity) = m;
      }
      return norm;
    }
  };

  template <typename Float_, int nColor_, QudaReconstructType recon>
  struct ApplyUArg : kernel_param<>
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type force;
    const typename gauge_mapper<Float, recon>::type U;
    int X[4]; // grid dimensions
    ApplyUArg(GaugeField &force, const GaugeField &U) :
      kernel_param(dim3(U.VolumeCB(), 2, 1)),
      force(force),
      U(U)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = U.X()[dir];
    }
  };

  template <typename Arg> struct ApplyU
  {
    const Arg &arg;
    constexpr ApplyU(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using mat = Matrix<complex<typename Arg::Float>, Arg::nColor>;

      for (int d=0; d<4; d++) {
        mat f = arg.force(d, x_cb, parity);
        mat u = arg.U(d, x_cb, parity);

        f = u * f;

        arg.force(d, x_cb, parity) = f;
      }
    }
  };

}
