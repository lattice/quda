#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <su3_project.cuh>

namespace quda
{

  using spinor_array = vector_type<double2, 16>;
  
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
    //inline int get_gm_i(const int G_idx, const int row_idx) const {return gm_i[G_idx][row_idx];};
    //inline complex<real> get_gm_z(const int G_idx, const int col_idx) const {return gm_z[G_idx][col_idx];};
  };

  template <typename Float_, int reduction_dim_ = 3> struct ContractionSpatialSumArg :
    public ReduceArg<spinor_array>  
  {
    static constexpr int reduction_dim = reduction_dim_; // This the direction we are performing reduction on. default to 3.

    int threads; // number of active threads required
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
    ContractionSpatialSumArg(const ColorSpinorField &x, const ColorSpinorField &y,
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

  template <int blockSize, typename Arg> __global__ void computeDegrandRossiContractionSpatialSum(Arg arg)
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

      // This function calculates the index_cb assuming t is the coordinate in
      // direction reduction_dim, and xyz is the linearized index_cb excluding reduction_dim.
      // So this will work for reduction_dim < 3 as well.
      int idx_cb = x_cb_from_t_xyz_d<Arg::reduction_dim>(t, xyz, arg.X, parity);
      
      ColorSpinor<real, nColor, nSpin> x = arg.x(idx_cb, parity);
      ColorSpinor<real, nColor, nSpin> y = arg.y(idx_cb, parity);

      // loop over channels
      for (int G_idx = 0; G_idx < 16; G_idx++) {
        for (int s2 = 0; s2 < nSpin; s2++) {
          int b2 = arg.Gamma.gm_i[G_idx][s2];
          // get non-zero column index for current s1
          int b1_tmp = arg.Gamma.gm_i[G_idx][s1];
          // only contributes if we're at the correct b1 from the outer loop
          if (b1_tmp == b1) {
            propagator_product = arg.Gamma.gm_z[G_idx][b2] * innerProduct(x, y, b2, s2) * arg.Gamma.gm_z[G_idx][b1];
            result_all_channels[G_idx].x += propagator_product.real();
            result_all_channels[G_idx].y += propagator_product.imag();
          }
        }
      }
      xyz += blockDim.x * gridDim.x;
    }

    // This function reduces the data in result_all_channels in all threads -
    // different threads reduce result to different index t + arg.t_offset
    arg.template reduce2d<blockSize, 2>(result_all_channels, t + arg.t_offset);
  }

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

  //- Welcome from contract_quda in quda/lib/contract.cu!
  //- We will now create the `arg` object that will use the input spinors to extract meta data.
  //- This structure is fairly complex, but none of the elements are beyond the comprehension 
  //- of a student of lattice gauge theory with a rudimentary knowledge of GPU computing. 
  //- As advertised, this stuct is templated on precision, and inherits from a reduction. The
  //- The reduction itself is templated on `spinor_array` which is a custom data type for this
  //- struct. It is at the top of the file, go take a look and return here.
  template <typename Float_> struct ContractionSumArg :    
    public ReduceArg<spinor_array>  
  {
    //- Welcome back! We first define the variables in the argument structure: 
    //- Naturally, this is the number of threads in the block
    int threads; // number of active threads required
    //- This is the number of lattice sites on the MPI node
    int X[4];    // grid dimensions

    //- These vales are hardcoded to prevent template explosion. This particular kernel will 
    //- only work for spin=4, color=3 fermion fields. The spin_project boolean is a neat 
    //- feature of QUDA that will automatically convert any fermion into the desired 
    //- gamma matrix order (defined in the ColorSpinorField object) as it is loaded into the 
    //- kernel. QUDA uses some very esotric ordering behind the scenes! 
    using Float = Float_;    
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    //- This object is a QUDA data type that will handle the way in which fermions are passed 
    //- to the kernel. As you can see, it needs to know the precision, color, and spin of the 
    //- fermion, as well as any operations (such as spin_project) that need to be performed 
    //- during the load. We give it a name F...
    // Create a typename F for the ColorSpinorField (F for fermion)
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    //- ...and declare two such objects, one for eack color spinor
    F x;
    F y;

    //- We now come to the constructor of this argument structure. This is what was called 
    //- by the ContractionArgSum<real> arg(x, y); line back in quda/lib/contract.cu. 
    //- `threads` is initialised with the checkerboarded sub-volume of a fermion field, 
    //- the two fermion fields are, naturally, x, and y. The actual body of the argument 
    //- structure is to then populate the X array with the MPI lattice dimensions. 
    //- This argument structure provides the kernel with the pointers it needs to 
    //- locate data, and the parameters it needs to perform it's computation.
    ContractionSumArg(const ColorSpinorField &x, const ColorSpinorField &y) :
      ReduceArg<spinor_array>(),
      threads(x.VolumeCB() / x.X(3)),
      x(x),
      y(y)
    {
      for (int dir = 0; dir < 4; dir++) 
	X[dir] = x.X()[dir];
    }
    //- And that the argument struct done. Please go back to `quda/lib/contract.cu` so see
    //- the computation.
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
    arg.template reduce2d<blockSize, 2>(res, t);
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

  //- Hello traveller! I am very pleased you have made it this far. None but the bravest dare venture
  //- this deep into QUDA! What You are about to see is what an individual thread in QUDA
  //- executes. This is the very core of the computation we set out to do way back in 
  //- quda/lib/interface_quda.cpp.
  //- First, notice the template arguments. These are set by the code in the 
  //- `LAUNCH_KERNEL_LOCAL_PARITY` macro. We have a `blockSize` argument because we will be doing
  //- a reduction on the data. The other argument `Arg` is all the boilerplace preparation
  //- we did leading up to here packaging the data and the parameters. Let's see it in action!
  template <int blockSize, typename Arg> __global__ void computeDegrandRossiContractionSum(Arg arg)
  {
    //- The first this we do is establish 'where` we are in the lattice. Recall that we were going
    //- use the z grid dimension (z block index) to refer to the time slice we are on? We use it 
    //- here
    int t = blockIdx.z; // map t to z block index
    //- Now we deduce which spatial point we are on (independent of time)
    int xyz = threadIdx.x + blockIdx.x * blockDim.x;
    //- And lastly which parity.
    int parity = threadIdx.y;
    
    //- The extremly astute adventurer just asked themself a question: how are all the threadIdx
    //- and blockIdx and blockDim values set so they match up with the spacetime coordinates!?
    //- The answer is that we specifically chose to use the TunableLocalParity class, which 
    //- uses these particular variables for those specific lattice coordinates.

    //- Next we extract some of the variables from the argument structure.
    using real = typename Arg::Float;
    constexpr int nSpin = Arg::nSpin;
    constexpr int nColor = Arg::nColor;
    
    //- This is a data type known throught QUDA. It's a colorSpinor, templetised on data type,
    //- colours, and spins. Let's just call it `Vector` 
    typedef ColorSpinor<real, nColor, nSpin> Vector;

    //- Remember that custom data type we made, the one at the top of the page? We declare one here
    //- called `res` for result.
    spinor_array res;

    //- This is slightly cumbersome, not massively. We use a 2D array and a 1D array of complex
    //- just to be convenient for each use case. spin_elem will hold color contractions, and 
    //- A will hold the gamma matrix contractions.
    complex<real> spin_elem[nSpin][nSpin];
    complex<real> A[nSpin * nSpin];
       
    //- The while loop is restricted to the same time slice. Multiple warps may fit into a timeslice
    //- so we just keep looping until the whole slice is filled up. Any left over threads will 
    //- not excuute because they are out of bounds.
    while (xyz < arg.threads) {
            
      //- arg.threads is the checkerboard volume of a timeslice. We set it in the argument struct.
      //- We use it to define a global checkerboard index.
      int idx_cb = t * arg.threads + xyz;

      //- Now, a litle QUDA magic. We declare two colorSpinor LOCAL field objects x,y, 
      //- and we populate them using a sepecial accessor defined in the ARG STRUCTURE objects x,y.
      //- It was that long `F` typedef we made. The `F` type has these built in accessors that 
      //- allow us to use the local coordinates defined by the block and thread index to access
      //- the relevant data in the fermion fields.
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);
      
      //- Number crunching time. We compute all 16 possible colour contractions, and put the result
      //- in spen_elem.
      // Color contract: <\phi(x)_{\mu} | \phi(y)_{\nu}>
      // The Bra is conjugated      
      for (int mu = 0; mu < nSpin; mu++) {
	for (int nu = 0; nu < nSpin; nu++) { 
	  //- The function `innerProduct` is defined in `quda/include/color_spinor.h` along with
	  //- a multitude of functions that compute operations on colorSpinor objects. The code
	  //- there is quite straightforward as it all it does is compute these simple 
	  //- arithmetic operations. You are encouraged to explore that file and make your
	  //- own custom functions. But remember to document your fucnction with doxygen!!!
	  spin_elem[mu][nu] = innerProduct(x, y, mu, nu); 
	}	
      }
     
      //- Next, we use that spin_elem array to compute all gamma matrix insertions. Let's take a 
      //- quick look, it is defined in this file.
      computeDRGammaContraction(spin_elem, A);
      
      //- Now that the gamma matrix contractions are done, we load result into res array.
      //- This is the array on which we perform the final reduction. Because it's a double2 
      //- data type we must use the .x and .y methods for real and imaginary.
      for (int mu = 0; mu < nSpin; mu++) {
	for (int nu = 0; nu < nSpin; nu++) { 
	  res[mu*nSpin + nu].x += A[mu*nSpin + nu].real();
	  res[mu*nSpin + nu].y += A[mu*nSpin + nu].imag();	  
	}	
      }

      //- Increment spatial index
      xyz += blockDim.x * gridDim.x;
    }
    
    //- The final step... timeslice reduction. This is another part of QUDA which we simply trust
    //- will work for us. It takes blockSize as one template arg, and 2 for parity. It will reduce 
    //- the data in res, accoring to the timeslice t, and place the result in an array which is
    //- defined in the arg structure. Remember, the argument structure iherits from ReduceArg, 
    //- which is why the reduced array already exists.
    arg.template reduce2d<blockSize, 2>(res, t);

    //- We have computed the contraction! We finally done. Let the Eagles of Manwe take us back to 
    //- quda/lib/contract.cu, and the line:
    //- LAUNCH_KERNEL_LOCAL_PARITY(computeDegrandRossiContractionSum, (*this), tp, stream, arg, Arg);
  }
    
} // namespace quda
