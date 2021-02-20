#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>
#include <kernels/contraction_helper.cuh>

namespace quda {
   
  template <typename Float, int nColor_, int reduction_dim_ = 3>
  struct ContractionSummedArg :  public ReduceArg<spinor_matrix>
  {
    // This the direction we are performing reduction on. default to 3.
    static constexpr int reduction_dim = reduction_dim_; 

    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load

    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;
    F x;
    F y;
    int s1, b1;
    int mom_mode[4];
    int source_position[4];
    int NxNyNzNt[4];
    DRGammaMatrix<real> Gamma;
    int t_offset;
    int offsets[4];
    
    dim3 threads;     // number of active threads required
    int_fastdiv X[4]; // grid dimensions
    
    ContractionSummedArg(const ColorSpinorField &x, const ColorSpinorField &y,
			 const int source_position_in[4], const int mom_mode_in[4],
			 const int s1, const int b1) :
      ReduceArg<spinor_matrix>(x.X()[reduction_dim]),
      x(x),
      y(y),
      s1(s1),
      b1(b1),
      Gamma(),
      // Launch xyz threads per t, t times.
      threads(x.Volume()/x.X()[reduction_dim], x.X()[reduction_dim])
    {
      for(int i=0; i<4; i++) {
	X[i] = x.X()[i];
	mom_mode[i] = mom_mode_in[i];
        source_position[i] = source_position_in[i];
        offsets[i] = comm_coord(i) * x.X()[i];
        NxNyNzNt[i] = comm_dim(i) * x.X()[i];
      }
    }
    __device__ __host__ spinor_matrix init() const { return spinor_matrix(); }
  };

  
  template <typename Arg> struct DegrandRossiContractFT : plus<spinor_matrix> {
    using reduce_t = spinor_matrix;
    using plus<reduce_t>::operator();    
    Arg &arg;
    constexpr DegrandRossiContractFT(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    // Final param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int t, int)
    {
      constexpr int nSpin = Arg::nSpin;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, nColor, nSpin>;

      reduce_t result_all_channels = spinor_matrix();
      int s1 = arg.s1;
      int b1 = arg.b1;
      int mom_mode[4];
      int source_position[4];
      int offsets[4];
      int NxNyNzNt[4];
      for(int i=0; i<4; i++) {
	source_position[i] = arg.source_position[i];
	offsets[i] = arg.offsets[i];
	mom_mode[i] = arg.mom_mode[i];
	NxNyNzNt[i] = arg.NxNyNzNt[i];
      }
      
      complex<real> propagator_product;
      
      //The coordinate of the sink
      int *sink;
      
      double phase_real;
      double phase_imag;
      double Sum_dXi_dot_Pi;
      
      sink = sink_from_t_xyz<Arg::reduction_dim>(t, xyz, arg.X);
      
      // Calculate exp(-i * [x dot p])
      Sum_dXi_dot_Pi = (double)((source_position[0]-sink[0]-offsets[0])*mom_mode[0]*1./NxNyNzNt[0]+
				(source_position[1]-sink[1]-offsets[1])*mom_mode[1]*1./NxNyNzNt[1]+
				(source_position[2]-sink[2]-offsets[2])*mom_mode[2]*1./NxNyNzNt[2]+
				(source_position[3]-sink[3]-offsets[3])*mom_mode[3]*1./NxNyNzNt[3]);
      
      phase_real =  cos(Sum_dXi_dot_Pi*2.*M_PI);
      phase_imag = -sin(Sum_dXi_dot_Pi*2.*M_PI);
      
      // Collect vector data
      int parity = 0;
      int idx = idx_from_t_xyz<Arg::reduction_dim>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);
      Vector x = arg.x(idx_cb, parity);
      Vector y = arg.y(idx_cb, parity);
      
      // loop over channels
      for (int G_idx = 0; G_idx < 16; G_idx++) {
	for (int s2 = 0; s2 < nSpin; s2++) {

	  // We compute the contribution from s1,b1 and s2,b2 from props x and y respectively.
	  int b2 = arg.Gamma.gm_i[G_idx][s2];	  
	  // get non-zero column index for current s1
	  int b1_tmp = arg.Gamma.gm_i[G_idx][s1];
	  
	  // only contributes if we're at the correct b1 from the outer loop FIXME
	  if (b1_tmp == b1) {
	    // use tr[ Gamma * Prop * Gamma * g5 * conj(Prop) * g5] = tr[g5*Gamma*Prop*g5*Gamma*(-1)^{?}*conj(Prop)].
	    // gamma_5 * gamma_i <phi | phi > gamma_5 * gamma_idx 
	    propagator_product = arg.Gamma.g5gm_z[G_idx][b2] * innerProduct(x, y, b2, s2) * arg.Gamma.g5gm_z[G_idx][b1];
	    result_all_channels[G_idx].x += propagator_product.real()*phase_real-propagator_product.imag()*phase_imag;
	    result_all_channels[G_idx].y += propagator_product.imag()*phase_real+propagator_product.real()*phase_imag;
	  }
	}
      }

      // Debug
      //for (int G_idx = 0; G_idx < arg.nSpin*arg.nSpin; G_idx++) {
      //result_all_channels[G_idx].x += (G_idx+t) + idx;
      //result_all_channels[G_idx].y += (G_idx+t) + idx;
      //}
      
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
} // namespace quda
