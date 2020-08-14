#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <su3_project.cuh>

namespace quda
{
  template <typename real> class DRGammaMatrix
  {

  public:
    //FIXME make these private?
    int gm_i[16][4] {};        // stores gamma matrix column index for non-zero complex value
    complex<real> gm_z[16][4]; // stores gamma matrix non-zero complex value for the corresponding gm_i
    //! Constructor
    DRGammaMatrix()
    {

      const complex<real> i(0., 1.);

      // SCALAR
      // G_idx = 0: I
      gm_i[0][0] = 0;
      gm_i[0][1] = 1;
      gm_i[0][2] = 2;
      gm_i[0][3] = 3;
      gm_z[0][0] = 1.;
      gm_z[0][1] = 1.;
      gm_z[0][2] = 1.;
      gm_z[0][3] = 1.;

      // VECTORS
      // G_idx = 1: \gamma_1
      gm_i[1][0] = 3;
      gm_i[1][1] = 2;
      gm_i[1][2] = 1;
      gm_i[1][3] = 0;
      gm_z[1][0] = i;
      gm_z[1][1] = i;
      gm_z[1][2] = -i;
      gm_z[1][3] = -i;

      // G_idx = 2: \gamma_2
      gm_i[2][0] = 3;
      gm_i[2][1] = 2;
      gm_i[2][2] = 1;
      gm_i[2][3] = 0;
      gm_z[2][0] = -1.;
      gm_z[2][1] = 1.;
      gm_z[2][2] = 1.;
      gm_z[2][3] = -1.;

      // G_idx = 3: \gamma_3
      gm_i[3][0] = 2;
      gm_i[3][1] = 3;
      gm_i[3][2] = 0;
      gm_i[3][3] = 1;
      gm_z[3][0] = i;
      gm_z[3][1] = -i;
      gm_z[3][2] = -i;
      gm_z[3][3] = i;

      // G_idx = 4: \gamma_4
      gm_i[4][0] = 2;
      gm_i[4][1] = 3;
      gm_i[4][2] = 0;
      gm_i[4][3] = 1;
      gm_z[4][0] = 1.;
      gm_z[4][1] = 1.;
      gm_z[4][2] = 1.;
      gm_z[4][3] = 1.;

      // PSEUDO-SCALAR
      // G_idx = 5: \gamma_5
      gm_i[5][0] = 0;
      gm_i[5][1] = 1;
      gm_i[5][2] = 2;
      gm_i[5][3] = 3;
      gm_z[5][0] = 1.;
      gm_z[5][1] = 1.;
      gm_z[5][2] = -1.;
      gm_z[5][3] = -1.;

      // PSEUDO-VECTORS
      // DMH: Careful here... we may wish to use  \gamma_1,2,3,4\gamma_5 for pseudovectors
      // G_idx = 6: \gamma_5\gamma_1
      gm_i[6][0] = 3;
      gm_i[6][1] = 2;
      gm_i[6][2] = 1;
      gm_i[6][3] = 0;
      gm_z[6][0] = i;
      gm_z[6][1] = i;
      gm_z[6][2] = i;
      gm_z[6][3] = i;

      // G_idx = 7: \gamma_5\gamma_2
      gm_i[7][0] = 3;
      gm_i[7][1] = 2;
      gm_i[7][2] = 1;
      gm_i[7][3] = 0;
      gm_z[7][0] = -1.;
      gm_z[7][1] = 1.;
      gm_z[7][2] = -1.;
      gm_z[7][3] = 1.;

      // G_idx = 8: \gamma_5\gamma_3
      gm_i[8][0] = 2;
      gm_i[8][1] = 3;
      gm_i[8][2] = 0;
      gm_i[8][3] = 1;
      gm_z[8][0] = i;
      gm_z[8][1] = -i;
      gm_z[8][2] = i;
      gm_z[8][3] = -i;

      // G_idx = 9: \gamma_5\gamma_4
      gm_i[9][0] = 2;
      gm_i[9][1] = 3;
      gm_i[9][2] = 0;
      gm_i[9][3] = 1;
      gm_z[9][0] = 1.;
      gm_z[9][1] = 1.;
      gm_z[9][2] = -1.;
      gm_z[9][3] = -1.;

      // TENSORS
      // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
      gm_i[10][0] = 0;
      gm_i[10][1] = 1;
      gm_i[10][2] = 2;
      gm_i[10][3] = 3;
      gm_z[10][0] = 1.;
      gm_z[10][1] = -1.;
      gm_z[10][2] = 1.;
      gm_z[10][3] = -1.;

      // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]
      gm_i[11][0] = 2;
      gm_i[11][1] = 3;
      gm_i[11][2] = 0;
      gm_i[11][3] = 1;
      gm_z[11][0] = -i;
      gm_z[11][1] = -i;
      gm_z[11][2] = i;
      gm_z[11][3] = i;

      // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
      gm_i[12][0] = 1;
      gm_i[12][1] = 0;
      gm_i[12][2] = 3;
      gm_i[12][3] = 2;
      gm_z[12][0] = -1.;
      gm_z[12][1] = -1.;
      gm_z[12][2] = 1.;
      gm_z[12][3] = 1.;

      // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
      gm_i[13][0] = 1;
      gm_i[13][1] = 0;
      gm_i[13][2] = 3;
      gm_i[13][3] = 2;
      gm_z[13][0] = 1.;
      gm_z[13][1] = 1.;
      gm_z[13][2] = 1.;
      gm_z[13][3] = 1.;

      // G_idx = 14: (i/2) * [\gamma_2, \gamma_4]
      gm_i[14][0] = 1;
      gm_i[14][1] = 0;
      gm_i[14][2] = 3;
      gm_i[14][3] = 2;
      gm_z[14][0] = -i;
      gm_z[14][1] = i;
      gm_z[14][2] = i;
      gm_z[14][3] = -i;

      // G_idx = 15: (i/2) * [\gamma_3, \gamma_4]
      gm_i[15][0] = 0;
      gm_i[15][1] = 1;
      gm_i[15][2] = 2;
      gm_i[15][3] = 3;
      gm_z[15][0] = -1.;
      gm_z[15][1] = -1.;
      gm_z[15][2] = 1.;
      gm_z[15][3] = 1.;
    };
    //FIXME convert these to device functions?
    //     inline int get_gm_i(const int G_idx, const int row_idx) const {return gm_i[G_idx][row_idx];};
    //     inline complex<real> get_gm_z(const int G_idx, const int col_idx) const {return gm_z[G_idx][col_idx];};
  };

  using spinor_array = vector_type<double2, 16>;

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
      threads(x.VolumeCB()), x(x), y(y), s(s, x.VolumeCB())
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = x.X()[dir];
    }
  };

  template <typename Float_, int reduction_dim_ = 3> struct ContractionSumArg :
    public ReduceArg<spinor_array>  
  {
    static constexpr int reduction_dim = reduction_dim_; // This the direction we are performing reduction on. default to 3.

    //- Welcome back! We first define the variables in the argument structure:
    //- Naturally, this is the number of threads in the block
    int threads; // number of active threads required
    //- This is the number of lattice sites on the MPI node
    int_fastdiv X[4];    // grid dimensions - using int_fastdiv to reduce division overhead on device

    using Float = Float_;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    int s1, b1;

    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;
    F x;
    F y;

    DRGammaMatrix<Float_> Gamma;
    int t_offset;
    ContractionSumArg(const ColorSpinorField &x, const ColorSpinorField &y,
        const int s1, const int b1) :
      ReduceArg<spinor_array>(comm_dim(reduction_dim) * x.X(reduction_dim)), // n_reduce = global_dim_t
      threads(x.VolumeCB() / x.X(reduction_dim)),
      x(x),
      y(y),
      s1(s1),
      b1(b1),
      Gamma(),
      t_offset(comm_coord(reduction_dim) * x.X(reduction_dim)) // offset of the slice we are doing reduction on
    {
      for (int dir = 0; dir < 4; dir++) { X[dir] = x.X()[dir]; }
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

  template <int blockSize, typename Arg> __global__ void computeColorContractionSum(Arg arg)
  {
    int t = blockIdx.z; // map t to z block index
    int xyz = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    using real = typename Arg::Float;
    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;
    typedef ColorSpinor<real, nColor, nSpin> Vector;

    spinor_array res;
    Matrix<complex<real>, nSpin> A;

    // the while loop is restricted to the same time slice
    while (xyz < arg.threads) {

      // arg.threads is the parity timeslice volume
      int idx_cb = t * arg.threads + xyz;

      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

#pragma unroll
      for (int mu = 0; mu < nSpin; mu++) {
#pragma unroll
        for (int nu = 0; nu < nSpin; nu++) {
          // Color inner product: <\phi(x)_{\mu} | \phi(y)_{\nu}>
          // The Bra is conjugated
          A(mu, nu) = innerProduct(x, y, mu, nu);
          res[mu * nSpin + nu].x = A(mu, nu).real();
          res[mu * nSpin + nu].y = A(mu, nu).imag();
        }
      }

      xyz += blockDim.x * gridDim.x;
    }
    reduce2d<blockSize, 2>(arg, res, t);
  }

  //- Welcome! This function will take the open spinor contractions, insert gamma matrices, and
  //- return the gamma matrix contracation. If you've made it this far, you should be brave enough
  //- to negotiate all the lines on your own.
  template <typename real>
  __host__ __device__ inline void computeDRGammaContraction(const complex<real> spin_elem[][4], complex<real> *A)
  {
    complex<real> I(0.0, 1.0);
    complex<real> result_local(0.0, 0.0);

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
    complex<real> A[nSpin * nSpin];

    // Color contract: <\phi(x)_{\mu} | \phi(y)_{\nu}>
    // The Bra is conjugated
    for (int mu = 0; mu < nSpin; mu++) {
      for (int nu = 0; nu < nSpin; nu++) { spin_elem[mu][nu] = innerProduct(x, y, mu, nu); }
    }

    // Compute all gamma matrix insertions
    computeDRGammaContraction(spin_elem, A);

    // Save data to return array
    arg.s.save(A, x_cb, parity);
  }

  template <int t_d, class T> __device__ int x_cb_from_t_xyz_d(int t, int xyz_cb, T X[4], int parity)
  {

    int x[4];
    int xyz = xyz_cb * 2;

#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != t_d) {
        x[d] = xyz % X[d];
        xyz /= X[d];
      }
    }

    x[t_d] = t;

    if (t_d > 0) {
      x[0] += (x[0] + x[1] + x[2] + x[3] + parity) & 1;
    } else {
      x[1] += (x[0] + x[1] + x[2] + x[3] + parity) & 1;
    }

    return (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) / 2;
  }

  template <int blockSize, typename Arg> __global__ void computeDegrandRossiContractionSum(Arg arg)
  {
    int t = blockIdx.z; // map t to z block index
    int xyz = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    int s1 = arg.s1;
    int b1 = arg.b1;

    using real = typename Arg::Float;
    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;

    complex<real> propagator_product;

    // result array needs to be a spinor_array type object because of the reduce function at the end
    vector_type<double2, 16> result_all_channels;

    while (xyz < arg.threads) { // loop over all space-coordinates of one time slice
      // extract current ColorSpinor at xyzt from ColorSpinorField

      // This function calculates the index_cb assuming t is the coordinate in direction reduction_dim,
      // and xyz is the linearized index_cb excluding reduction_dim. So this will work for reduction_dim < 3 as well.
      int idx_cb = x_cb_from_t_xyz_d<Arg::reduction_dim>(t, xyz, arg.X, parity);

      ColorSpinor<real, nColor, nSpin> x = arg.x(idx_cb, parity);
      ColorSpinor<real, nColor, nSpin> y = arg.y(idx_cb, parity);

      // loop over channels
      for (int G_idx = 0; G_idx < 16; G_idx++) {
        for (int s2 = 0; s2 < nSpin; s2++) {
          int b2 = arg.Gamma.gm_i[G_idx][s2];
          //get non-zero column index for current s1
          int b1_tmp = arg.Gamma.gm_i[G_idx][s1];
          //only contributes if were at the correct b1 from the outer loop
          if (b1_tmp == b1) {
            propagator_product = arg.Gamma.gm_z[G_idx][b2] * innerProduct(x, y, b2, s2) * arg.Gamma.gm_z[G_idx][b1];
            result_all_channels[G_idx].x += propagator_product.real();
            result_all_channels[G_idx].y += propagator_product.imag();
          }
        }
      }
      xyz += blockDim.x * gridDim.x;
    }

    // This function reduces the data in result_all_channels in all threads - different threads reduce result to
    // different index t + arg.t_offset
    reduce2d<blockSize, 2>(arg, result_all_channels, t + arg.t_offset);
  }

} // namespace quda
