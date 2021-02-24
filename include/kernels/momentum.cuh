#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <kernel.h>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct BaseArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    dim3 threads; // number of active threads required
    BaseArg(const GaugeField &meta) :
      threads(meta.VolumeCB(), 2, 1) {}
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct MomActionArg : ReduceArg<double>, BaseArg<Float, nColor, recon> {
    typedef typename gauge_mapper<Float, recon>::type Mom;
    const Mom mom;

    MomActionArg(const GaugeField &mom) :
      BaseArg<Float, nColor, recon>(mom),
      mom(mom) {}

    __device__ __host__ double init() const { return 0.0; }
  };

  template <typename Arg> struct MomAction : plus<double> {
    using reduce_t = double;
    using plus<reduce_t>::operator();
    Arg &arg;
    constexpr MomAction(Arg &arg) : arg(arg) {}
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

	action = plus::operator()(action, local_sum);
      }
      return action;
    }
  };

  template<typename Float_, int nColor, QudaReconstructType recon>
  struct UpdateMomArg : ReduceArg<vector_type<double, 2>>, BaseArg<Float_, nColor, recon>
  {
    using reduce_t = vector_type<double, 2>;
    using Float = Float_;
    typename gauge_mapper<Float, QUDA_RECONSTRUCT_10>::type mom;
    typename gauge_mapper<Float, recon>::type force;
    Float coeff;
    int X[4]; // grid dimensions on mom
    int E[4]; // grid dimensions on force (possibly extended)
    int border[4]; //

    UpdateMomArg(GaugeField &mom, const Float &coeff, const GaugeField &force) :
      BaseArg<Float, nColor, recon>(mom),
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

    __device__ __host__ reduce_t init() const{ return reduce_t(); }
  };

  template <typename Arg> struct MomUpdate {
    using reduce_t = vector_type<double, 2>;
    Arg &arg;
    constexpr MomUpdate(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    static constexpr bool do_sum = false;

    /**
       @brief Functor for finding the maximum over a vec2 field.  Each
       lane is evaluated separately.
    */
    __device__ __host__ inline reduce_t operator()(const reduce_t &a, const reduce_t &b) const
    {
      auto c = a;
      if (b[0] > a[0]) c[0] = b[0];
      if (b[1] > a[1]) c[1] = b[1];
      return c;
    }

    // calculate the momentum contribution to the action.  This uses the
    // MILC convention where we subtract 4.0 from each matrix norm in
    // order to increase stability
    __device__ __host__ inline reduce_t operator()(reduce_t &norm, int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x,arg.E);

#pragma unroll
      for (int d=0; d<4; d++) {
        Matrix<complex<typename Arg::Float>,3> m = arg.mom(d, x_cb, parity);
        Matrix<complex<typename Arg::Float>,3> f = arg.force(d, e_cb, parity);

        // project to traceless anti-hermitian prior to taking norm
        makeAntiHerm(f);

        // compute force norms
        norm = operator()(reduce_t(f.L1(), f.L2()), norm);

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

  template <typename Float, int nColor, QudaReconstructType recon>
  struct ApplyUArg : BaseArg<Float, nColor, recon>
  {
    typedef typename gauge_mapper<Float,recon>::type G;
    typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type F;
    F force;
    const G U;
    int X[4]; // grid dimensions
    ApplyUArg(GaugeField &force, const GaugeField &U) :
      BaseArg<Float, nColor, recon>(U),
      force(force),
      U(U)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = U.X()[dir];
    }
  };

  template <typename Arg> struct ApplyU
  {
    Arg &arg;
    constexpr ApplyU(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using mat = Matrix<complex<typename Arg::Float>,Arg::nColor>;

      for (int d=0; d<4; d++) {
        mat f = arg.force(d, x_cb, parity);
        mat u = arg.U(d, x_cb, parity);

        f = u * f;

        arg.force(d, x_cb, parity) = f;
      }
    }
  };

}
