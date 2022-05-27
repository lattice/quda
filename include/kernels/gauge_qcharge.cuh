#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool density_ = false> struct QChargeArg :
    public ReduceArg<array<double, 3>>
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool density = density_;
    typedef typename gauge_mapper<Float,recon>::type F;

    F f;
    Float *qDensity;

    QChargeArg(const GaugeField &Fmunu, Float *qDensity = nullptr) :
      ReduceArg<reduce_t>(dim3(Fmunu.VolumeCB(), 2, 1)),
      f(Fmunu),
      qDensity(qDensity) { }
  };

  // Core routine for computing the topological charge from the field strength
  template <typename Arg> struct qCharge : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr qCharge(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the qcharge and field strength at site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &E, int x_cb, int parity)
    {
      using real = typename Arg::Float;
      using Link = Matrix<complex<real>, Arg::nColor>;
      constexpr real q_norm = static_cast<real>(-1.0 / (4*M_PI*M_PI));
      constexpr real n_inv = static_cast<real>(1.0 / Arg::nColor);

      reduce_t E_local{0, 0, 0};
      double &Q = E_local[2];

      // Load the field-strength tensor from global memory
      //F0 = F[Y,X], F1 = F[Z,X], F2 = F[Z,Y],
      //F3 = F[T,X], F4 = F[T,Y], F5 = F[T,Z]
      Link F[] = {arg.f(0, x_cb, parity), arg.f(1, x_cb, parity), arg.f(2, x_cb, parity),
                  arg.f(3, x_cb, parity), arg.f(4, x_cb, parity), arg.f(5, x_cb, parity)};

      // first compute the field energy
      Link iden;
      setIdentity(&iden);
#pragma unroll
      for (int i=0; i<6; i++) {
        // Make traceless
        auto tmp = F[i] - n_inv * getTrace(F[i]) * iden;

        // Sum trace of square, normalise in .cu
        if (i<3) E_local[0] -= getTrace(tmp * tmp).real(); //spatial
        else     E_local[1] -= getTrace(tmp * tmp).real(); //temporal
      }

      // now compute topological charge
      double Q_idx = 0.0;
      double Qi[3] = {0.0,0.0,0.0};
      // unroll computation
#pragma unroll
      for (int i=0; i<3; i++) {
        Qi[i] = getTrace(F[i] * F[5 - i]).real();
      }

      // apply correct levi-civita symbol
      for (int i=0; i<3; i++) i % 2 == 0 ? Q_idx += Qi[i]: Q_idx -= Qi[i];
      Q = Q_idx * q_norm;
      if (Arg::density) arg.qDensity[x_cb + parity * arg.threads.x] = Q;

      return operator()(E, E_local);
    }

  };

} // namespace quda
