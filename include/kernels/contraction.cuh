#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <su3_project.cuh>

namespace quda
{

  template <typename real> struct ContractionArg {
    int threads; // number of active threads required
    int X[4];    // grid dimensions

    static constexpr int nSpin = 4;
    static constexpr int nColor = 3;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorField (F for fermion)
    typedef typename colorspinor_mapper<real, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    F x;
    F y;
    matrix_field<complex<real>, nSpin> s;

    ContractionArg(const ColorSpinorField &x, const ColorSpinorField &y, complex<real> *s) :
      threads(x.VolumeCB()),
      x(x),
      y(y),
      s(s, x.VolumeCB())
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename real, typename Arg> __global__ void computeColorContraction(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (x_cb >= arg.threads) return;

    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpin> Vector;

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

  template <typename real, typename Arg> __global__ void computeDegrandRossiContraction(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    const int nSpin = arg.nSpin;
    const int nColor = arg.nColor;

    if (x_cb >= arg.threads) return;

    typedef ColorSpinor<real, nColor, nSpin> Vector;

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

    complex<real> A[nSpin * nSpin];

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

    arg.s.save(A, x_cb, parity);
  }
} // namespace quda
