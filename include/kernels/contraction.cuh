#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>

namespace quda {

  using spinor_array = vector_type<double2, 4>;

  template <class T> __device__ int idx_from_t_xyz(int t, int xyz, T X[4])
  {
    int x[4];
#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != 3) {
	x[d] = xyz % X[d];
	xyz /= X[d];
      }
    }    
    x[3] = t;    
    return (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);
  }

  template <typename Float, int nColor_, int red = 3>
  struct EvecProjectionArg : public ReduceArg<spinor_array>
  {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpinX = 4;
    static constexpr int nSpinY = 1;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    typedef typename colorspinor_mapper<Float, nSpinX, nColor, spin_project, spinor_direct_load>::type F4;

    typedef typename colorspinor_mapper<Float, nSpinY, nColor, spin_project, spinor_direct_load>::type F1;

    F4 x;
    F1 y;

    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    
    EvecProjectionArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<spinor_array>(x.X()[3]),
      x(x),
      y(y),
      // Launch xyz threads per t, t times.
      threads(x.Volume()/x.X()[3], x.X()[3])
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
      }
    }
    __device__ __host__ spinor_array init() const { return spinor_array(); }
  };
  
  
  template <typename Arg> struct EvecProjection : plus<spinor_array> {
    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();    
    Arg &arg;
    constexpr EvecProjection(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      constexpr int nSpinX = Arg::nSpinX;
      constexpr int nSpinY = Arg::nSpinY;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector4 = ColorSpinor<real, nColor, nSpinX>;
      using Vector4 = ColorSpinor<real, nColor, nSpinY>;

      reduce_t result_all_channels = spinor_array();

      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz(t, xyz, arg.X);
      // This helper will change the value of 'parity' to the correct one, 
      // and return the checkerboard index.
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
      Vector4 x = arg.x(idx_cb, parity);
      Vector1 y = arg.y(idx_cb, parity);

      // Compute the inner product over colour
      complex<real> prod;
      for (int mu = 0; mu < nSpinX; mu++) {
        prod = innerProduct(y, x, 0, mu);
        result_all_channels[mu].x += prod.real();
        result_all_channels[mu].y += prod.imag();
      }
      
      return plus::operator()(result_all_channels, result);
    }
  };


  template <typename Float, int nColor_> struct ContractionArg {
    using real = typename mapper<Float>::type;
    int X[4];    // grid dimensions

    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorField (F for fermion)
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;

    F x;
    F y;
    matrix_field<complex<Float>, nSpin> s;
    dim3 threads;

    ContractionArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<Float> *s) :
      x(x),
      y(y),
      s(s, x.VolumeCB()),
      threads(x.VolumeCB(), 2)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename Arg> struct ColorContract {
    Arg &arg;
    constexpr ColorContract(Arg &arg) : arg(arg) {}
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
    Arg &arg;
    constexpr DegrandRossiContract(Arg &arg) : arg(arg) {}
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
} // namespace quda
