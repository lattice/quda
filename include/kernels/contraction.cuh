#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>

namespace quda
{

  // Additions
  //------------------------------------------------------------------------------
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
  
  template <typename Float, int nColor_> struct ContractionSlicedFTArg
  {
    using real = typename mapper<Float>::type;    
    int_fastdiv X[4]; // grid dimensions - using int_fastdiv to reduce division overhead on device
    
    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    // Create a typename F for the ColorSpinorField (F for fermion)
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;
    
    F x;
    F y;
    
    int s1, b1;
    int source_position[4];
    int mom_mode[4];
    int global_lat_dim[4];

    DRGammaMatrix<Float> Gamma;
    int t_offset;
    int offsets[4];
    dim3 threads; // number of active threads required
    int reduction_dim;
    
    ContractionSlicedFTArg(const ColorSpinorField &x, const ColorSpinorField &y,
			   const int _source_position[4], const int _mom_mode[4],
			   const int s1, const int b1, const int reduction_dim) :
      threads(x.VolumeCB(), 2),
      x(x),
      y(y),
      s1(s1),
      b1(b1),
      reduction_dim(reduction_dim),
      Gamma(),
      t_offset(comm_coord(reduction_dim) * x.X(reduction_dim)) // offset of the slice we are doing reduction on
    {
      for (int i = 0; i < 4; i++) {
        X[i] = x.X()[i]; //Local coordinate
        mom_mode[i] = _mom_mode[i]; // momentum mode
        source_position[i] = _source_position[i]; //Global source position
        offsets[i]  = comm_coord(i) * x.X(i); // Positive Distance from 0 MPI node 
        global_lat_dim[i] = comm_dim(i) * x.X(i); // Global lattice dims
      }
    }
  };
  
  template <class T> __device__ int* get_sink(int t, int xyz_cb, T X[4], int parity, int t_d)
  {
    static int sink[4];
    int xyz = xyz_cb * 2;

#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != t_d) {
        sink[d] = xyz % X[d];
        xyz /= X[d];
      }
    }

    sink[t_d] = t;

    if (t_d > 0) {
      sink[0] += (sink[0] + sink[1] + sink[2] + sink[3] + parity) & 1;
    } else {
      sink[1] += (sink[0] + sink[1] + sink[2] + sink[3] + parity) & 1;
    }
    return sink;
  }
  
  template <class T> __device__ int x_cb_from_sink(T X[4], int* sink)
  {
    return (((sink[3] * X[2] + sink[2]) * X[1] + sink[1]) * X[0] + sink[0]) / 2;
  }
  
  template <class T> __device__ int x_cb_from_t_xyz_d(int t, int xyz_cb, T X[4], int parity, int td)
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

  template <typename Arg> struct DegrandRossiContractFT {
    Arg &arg;
    constexpr DegrandRossiContractFT(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      constexpr int nSpin = Arg::nSpin;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      
      int t = blockIdx.z; // map t to z block index
      int xyz = threadIdx.x + blockIdx.x * blockDim.x;
      //int parity = threadIdx.y;
      
      int s1 = arg.s1;
      int b1 = arg.b1;
      
      int mom_mode[4];
      int source_position[4];
      int offsets[4];
      int global_lat_dim[4];
      
      for(int i=0; i<4; i++) {
	source_position[i] = arg.source_position[i]; // Global source position
	offsets[i] = arg.offsets[i];                 // Distance from 0 MPI node 
	mom_mode[i] = arg.mom_mode[i];               // momentum mode
	global_lat_dim[i] = arg.global_lat_dim[i];   // Global lattice dims
      }
      
      
      complex<real> propagator_product;
      
      //the coordinate of the sink
      int *sink;
      // result array needs to be a spinor_array type object because of the reduce function at the end
      vector_type<double2, 16> result_all_channels;
      double phase_real;
      double phase_imag;
      double Sum_dXi_dot_Pi;

      // extract current ColorSpinor at xyzt from ColorSpinorField      
      // This function calculates the index_cb assuming t is the coordinate in
      // direction reduction_dim, and xyz is the linearized index_cb excluding reduction_dim.
      // So this will work for reduction_dim < 3 as well.
      sink = get_sink(t, xyz, arg.X, parity, arg.reduction_dim);
      
      int idx_cb = x_cb_from_sink(arg.X, sink);
      //calculate exp(-i*(x-x_0)*p) relative to the source position
      Sum_dXi_dot_Pi = (double)((source_position[0] - (sink[0] + offsets[0])) * mom_mode[0] * 1.0/global_lat_dim[0]+
				(source_position[1] - (sink[1] + offsets[1])) * mom_mode[1] * 1.0/global_lat_dim[1]+
				(source_position[2] - (sink[2] + offsets[2])) * mom_mode[2] * 1.0/global_lat_dim[2]+
				(source_position[3] - (sink[3] + offsets[3])) * mom_mode[3] * 1.0/global_lat_dim[3]);
      
      phase_real =  cos(Sum_dXi_dot_Pi * 2.0 * M_PI);
      phase_imag = -sin(Sum_dXi_dot_Pi * 2.0 * M_PI);
      
      using Vector = ColorSpinor<real, nColor, nSpin>;
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);

      // loop over channels
      for (int G_idx = 0; G_idx < 16; G_idx++) {
	for (int s2 = 0; s2 < nSpin; s2++) {
	  int b2 = arg.Gamma.gm_i[G_idx][s2];
	  // get non-zero column index for current s1
	  int b1_tmp = arg.Gamma.gm_i[G_idx][s1];
	  // only contributes if we're at the correct b1 from the outer loop
	  if (b1_tmp == b1) {
	    propagator_product = arg.Gamma.gm_z[G_idx][b2] * innerProduct(x, y, b2, s2) * arg.Gamma.gm_z[G_idx][b1];
	    result_all_channels[G_idx].x += propagator_product.real()*phase_real - propagator_product.imag()*phase_imag;
	    result_all_channels[G_idx].y += propagator_product.imag()*phase_real + propagator_product.real()*phase_imag;
	    
	    //result_all_channels[G_idx].x += 1.0 * ((t + arg.t_offset) + 1);
	    //result_all_channels[G_idx].y += 2.0 * ((t + arg.t_offset) + 1);
	    
	    //if(xyz == 0) printf("Yes Comp\n");
	  } else {
	    //if(xyz == 0) printf("No Comp\n");  
	  }
	}
      }
      
      // This function reduces the data in result_all_channels in all threads -
      // different threads reduce result to different index t + arg.t_offset
      //arg.template reduce2d<blockSize, 2>(result_all_channels, (t + arg.t_offset)%global_lat_dim[3]);
    }
  };    
  //--------------------------------------------------------------------------------


  // Generic kernels
  //--------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------
