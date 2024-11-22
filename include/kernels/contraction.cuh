#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <gamma.cuh>

namespace quda
{
  static constexpr int max_contract_results = 16; // sized for nSpin**2 = 16

  using spinor_array = array<array<double, 2>, max_contract_results>;
  using staggered_spinor_array = array<double, 2>;

  template <int reduction_dim, class T> __device__ void sink_from_t_xyz(int sink[4], int t, int xyz, T X[4])
  {
#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != reduction_dim) {
        sink[d] = xyz % X[d];
        xyz /= X[d];
      }
    }
    sink[reduction_dim] = t;
    return;
  }

  template <class T> __device__ int idx_from_sink(T X[4], int *sink)
  {
    return ((sink[3] * X[2] + sink[2]) * X[1] + sink[1]) * X[0] + sink[0];
  }

  template <typename Float, int nColor_, int nSpin_ = 4, int reduction_dim_ = 3, typename contract_t = spinor_array>
  struct ContractionSummedArg : public ReduceArg<contract_t> {
    using reduce_t = contract_t;
    // This the direction we are performing reduction on. default to 3.
    static constexpr int reduction_dim = reduction_dim_;

    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr bool spin_project = nSpin_ == 1 ? false : true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    int s1, b1;
    int mom_mode[4];
    QudaFFTSymmType fft_type[4];
    int source_position[4];
    int NxNyNzNt[4];
    int t_offset;
    int offsets[4];

    int_fastdiv X[4]; // grid dimensions

    ContractionSummedArg(const ColorSpinorField &x, const ColorSpinorField &y, const int source_position_in[4],
                         const int mom_mode_in[4], const QudaFFTSymmType fft_type_in[4], const int s1, const int b1) :
      ReduceArg<reduce_t>(dim3(x.Volume() / x.X()[reduction_dim], 1, x.X()[reduction_dim]), x.X()[reduction_dim]),
      x(x),
      y(y),
      s1(s1),
      b1(b1)
    // Launch xyz threads per t, t times.
    {
      for (int i = 0; i < 4; i++) {
        X[i] = x.X()[i];
        source_position[i] = source_position_in[i];
        mom_mode[i] = mom_mode_in[i];
        fft_type[i] = fft_type_in[i];
        offsets[i] = comm_coord(i) * x.X()[i];
        NxNyNzNt[i] = comm_dim(i) * x.X()[i];
      }
    }
  };

  template <typename Arg> struct DegrandRossiContractFT : plus<spinor_array> {

    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 1; //

    const Arg &arg;
    constexpr DegrandRossiContractFT(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    // overload comm_reduce to defer until the entire "tile" is complete
    template <typename U> static inline void comm_reduce(U &) { }

    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int, int t)
    {
      constexpr int nSpin = Arg::nSpin;
      constexpr int nColor = Arg::nColor;

      using real = typename Arg::real;
      using Vector = ColorSpinor<real, nColor, nSpin>;

      constexpr array<array<int, nSpin>, nSpin *nSpin> gm_i = get_dr_gm_i();
      constexpr array<array<complex<real>, nSpin>, nSpin *nSpin> g5gm_z = get_dr_g5gm_z<real>();

      int s1 = arg.s1;
      int b1 = arg.b1;

      // The coordinate of the sink
      int sink[4];
      sink_from_t_xyz<Arg::reduction_dim>(sink, t, xyz, arg.X);

      // Calculate exp(-i * [x dot p])
      double Sum_dXi_dot_Pi = 0.0;
      for (int i = 0; i < 4; i++)
        Sum_dXi_dot_Pi += (arg.source_position[i] - sink[i] - arg.offsets[i]) * arg.mom_mode[i] * 1. / arg.NxNyNzNt[i];

      complex<double> phase = {cospi(Sum_dXi_dot_Pi * 2.), -sinpi(Sum_dXi_dot_Pi * 2.)};

      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz<Arg::reduction_dim>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

      // loop over channels
      reduce_t result_all_channels = {};
      for (int G_idx = 0; G_idx < 16; G_idx++) {
        for (int s2 = 0; s2 < nSpin; s2++) {

          // We compute the contribution from s1,b1 and s2,b2 from props x and y respectively.
          int b2 = gm_i[G_idx][s2];
          // get non-zero column index for current s1
          int b1_tmp = gm_i[G_idx][s1];

          // only contributes if we're at the correct b1 from the outer loop FIXME
          if (b1_tmp == b1) {
            // use tr[ Gamma * Prop * Gamma * g5 * conj(Prop) * g5] = tr[g5*Gamma*Prop*g5*Gamma*(-1)^{?}*conj(Prop)].
            // gamma_5 * gamma_i <phi | phi > gamma_5 * gamma_idx
            auto prop_product = g5gm_z[G_idx][b2] * innerProduct(x, y, b2, s2) * g5gm_z[G_idx][b1];
            result_all_channels[G_idx][0] += prop_product.real() * phase.real() - prop_product.imag() * phase.imag();
            result_all_channels[G_idx][1] += prop_product.imag() * phase.real() + prop_product.real() * phase.imag();
          }
        }
      }

      return operator()(result_all_channels, result);
    }
  };

  template <typename Arg> struct StaggeredContractFT : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();

    static constexpr int reduce_block_dim = 1;

    const Arg &arg;
    constexpr StaggeredContractFT(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    // overload comm_reduce to defer until the entire "tile" is complete
    template <typename U> static inline void comm_reduce(U &) { }

    // y index param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int, int t)
    {
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // The coordinate of the sink
      int sink[4];
      sink_from_t_xyz<Arg::reduction_dim>(sink, t, xyz, arg.X);

      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz<Arg::reduction_dim>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

      // Color inner product: <\phi(x)_{\mu} | \phi(y)_{\nu}> ; The Bra is conjugated
      complex<real> prop_prod = innerProduct(x, y);

      // Fourier phase
      complex<double> ph;
      complex<double> phase(1.0, 0.0);
      // Phase factor for each direction is either the cos, sin, or exp Fourier phase
#pragma unroll
      for (int dir = 0; dir < 4; dir++) {
        auto dXi_dot_Pi
          = 2.0 * (sink[dir] + arg.offsets[dir] - arg.source_position[dir]) * arg.mom_mode[dir] / arg.NxNyNzNt[dir];
        if (arg.fft_type[dir] == QUDA_FFT_SYMM_EO) {
          // exp(+i k.x) case
          ph = {cospi(dXi_dot_Pi), sinpi(dXi_dot_Pi)};
        } else if (arg.fft_type[dir] == QUDA_FFT_SYMM_EVEN) {
          // cos(k.x) case
          ph = {cospi(dXi_dot_Pi), 0.0};
        } else if (arg.fft_type[dir] == QUDA_FFT_SYMM_ODD) {
          // sin(k.x) case
          ph = {0.0, sinpi(dXi_dot_Pi)};
        }
        phase *= ph;
      }

      complex<double> result_all_channels = phase * complex<double> {prop_prod.real(), prop_prod.imag()};
      return operator()({result_all_channels.real(), result_all_channels.imag()}, result);
    }
  };

  template <typename Float, int nSpin_, int nColor_, bool spin_project_> struct ContractionArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    int X[4];    // grid dimensions

    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr bool spin_project = spin_project_;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorField (F for fermion)
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;

    F x;
    F y;
    matrix_field<complex<Float>, nSpin> s;

    ContractionArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<Float> *s) :
      kernel_param(dim3(x.VolumeCB(), 2, 1)),
      x(x),
      y(y),
      s(s, x.VolumeCB())
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename Arg> struct ColorContract {
    const Arg &arg;
    constexpr ColorContract(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      constexpr int nSpin = Arg::nSpin;
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      Matrix<complex<real>, nSpin> A;
#pragma unroll
      for (int mu = 0; mu < nSpin; mu++) {
#pragma unroll
        for (int nu = 0; nu < nSpin; nu++) {
          // Color inner product: <\phi(x)_{\mu} | \phi(y)_{\nu}>
          // The Bra is conjugated
          A(mu, nu) = innerProduct(x, y, mu, nu);
        }
      }

      arg.s.save(A, x_cb, parity);
    }
  };

  template <typename Arg> struct DegrandRossiContract {
    const Arg &arg;
    constexpr DegrandRossiContract(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      constexpr int nSpin = Arg::nSpin;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, nColor, nSpin>;

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      complex<real> I(0.0, 1.0);
      complex<real> spin_elem[nSpin][nSpin];
      complex<real> result_local(0.0, 0.0);

      // Color contract: <\phi(x)_{\mu} | \phi(y)_{\nu}>
      // The Bra is conjugated
      for (int mu = 0; mu < nSpin; mu++) {
        for (int nu = 0; nu < nSpin; nu++) { spin_elem[mu][nu] = innerProduct(x, y, mu, nu); }
      }

      Matrix<complex<real>, nSpin> A_;
      auto A = A_.data;

      // Spin contract: <\phi(x)_{\mu} \Gamma_{mu,nu}^{rho,tau} \phi(y)_{\nu}>
      // The rho index runs slowest.
      // Layout is defined in enum_quda.h: G_idx = 4*rho + tau
      // DMH: Hardcoded to Degrand-Rossi. Need a template on Gamma basis.

      int G_idx = 0;

      // SCALAR
      // G_idx = 0: I
      result_local = 0.0;
      result_local += spin_elem[0][0];
      result_local += spin_elem[1][1];
      result_local += spin_elem[2][2];
      result_local += spin_elem[3][3];
      A[G_idx++] = result_local;

      // VECTORS
      // G_idx = 1: \gamma_1
      result_local = 0.0;
      result_local += I * spin_elem[0][3];
      result_local += I * spin_elem[1][2];
      result_local -= I * spin_elem[2][1];
      result_local -= I * spin_elem[3][0];
      A[G_idx++] = result_local;

      // G_idx = 2: \gamma_2
      result_local = 0.0;
      result_local -= spin_elem[0][3];
      result_local += spin_elem[1][2];
      result_local += spin_elem[2][1];
      result_local -= spin_elem[3][0];
      A[G_idx++] = result_local;

      // G_idx = 3: \gamma_3
      result_local = 0.0;
      result_local += I * spin_elem[0][2];
      result_local -= I * spin_elem[1][3];
      result_local -= I * spin_elem[2][0];
      result_local += I * spin_elem[3][1];
      A[G_idx++] = result_local;

      // G_idx = 4: \gamma_4
      result_local = 0.0;
      result_local += spin_elem[0][2];
      result_local += spin_elem[1][3];
      result_local += spin_elem[2][0];
      result_local += spin_elem[3][1];
      A[G_idx++] = result_local;

      // PSEUDO-SCALAR
      // G_idx = 5: \gamma_5
      result_local = 0.0;
      result_local += spin_elem[0][0];
      result_local += spin_elem[1][1];
      result_local -= spin_elem[2][2];
      result_local -= spin_elem[3][3];
      A[G_idx++] = result_local;

      // PSEUDO-VECTORS
      // DMH: Careful here... we may wish to use  \gamma_1,2,3,4\gamma_5 for pseudovectors
      // G_idx = 6: \gamma_5\gamma_1
      result_local = 0.0;
      result_local += I * spin_elem[0][3];
      result_local += I * spin_elem[1][2];
      result_local += I * spin_elem[2][1];
      result_local += I * spin_elem[3][0];
      A[G_idx++] = result_local;

      // G_idx = 7: \gamma_5\gamma_2
      result_local = 0.0;
      result_local -= spin_elem[0][3];
      result_local += spin_elem[1][2];
      result_local -= spin_elem[2][1];
      result_local += spin_elem[3][0];
      A[G_idx++] = result_local;

      // G_idx = 8: \gamma_5\gamma_3
      result_local = 0.0;
      result_local += I * spin_elem[0][2];
      result_local -= I * spin_elem[1][3];
      result_local += I * spin_elem[2][0];
      result_local -= I * spin_elem[3][1];
      A[G_idx++] = result_local;

      // G_idx = 9: \gamma_5\gamma_4
      result_local = 0.0;
      result_local += spin_elem[0][2];
      result_local += spin_elem[1][3];
      result_local -= spin_elem[2][0];
      result_local -= spin_elem[3][1];
      A[G_idx++] = result_local;

      // TENSORS
      // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
      result_local = 0.0;
      result_local += spin_elem[0][0];
      result_local -= spin_elem[1][1];
      result_local += spin_elem[2][2];
      result_local -= spin_elem[3][3];
      A[G_idx++] = result_local;

      // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]
      result_local = 0.0;
      result_local -= I * spin_elem[0][2];
      result_local -= I * spin_elem[1][3];
      result_local += I * spin_elem[2][0];
      result_local += I * spin_elem[3][1];
      A[G_idx++] = result_local;

      // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
      result_local = 0.0;
      result_local -= spin_elem[0][1];
      result_local -= spin_elem[1][0];
      result_local += spin_elem[2][3];
      result_local += spin_elem[3][2];
      A[G_idx++] = result_local;

      // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
      result_local = 0.0;
      result_local += spin_elem[0][1];
      result_local += spin_elem[1][0];
      result_local += spin_elem[2][3];
      result_local += spin_elem[3][2];
      A[G_idx++] = result_local;

      // G_idx = 14: (i/2) * [\gamma_2, \gamma_4]
      result_local = 0.0;
      result_local -= I * spin_elem[0][1];
      result_local += I * spin_elem[1][0];
      result_local += I * spin_elem[2][3];
      result_local -= I * spin_elem[3][2];
      A[G_idx++] = result_local;

      // G_idx = 15: (i/2) * [\gamma_3, \gamma_4]
      result_local = 0.0;
      result_local -= spin_elem[0][0];
      result_local -= spin_elem[1][1];
      result_local += spin_elem[2][2];
      result_local += spin_elem[3][3];
      A[G_idx++] = result_local;

      arg.s.save(A_, x_cb, parity);
    }
  };

  template <typename Arg> struct StaggeredContract {
    const Arg &arg;
    constexpr StaggeredContract(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      constexpr int nSpin = Arg::nSpin;
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      Vector x = arg.x(x_cb, parity);
      Vector y = arg.y(x_cb, parity);

      Matrix<complex<real>, nSpin> A;
      // Color inner product: <\phi(x)_{\mu} | \phi(y)_{\nu}> ; The Bra is conjugated
      A(0, 0) = innerProduct(x, y);

      arg.s.save(A, x_cb, parity);
    }
  };
} // namespace quda
