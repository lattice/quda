#pragma once

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <kernel.h>

namespace quda {

  template <typename store_t_>
  struct CloverArg {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = 3;
    static constexpr int nSpin = 4;
    static constexpr bool exp_clover = !exponentiated_clover();
    
    using Clover = typename clover_mapper<store_t>::type;
    using Fmunu = typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type;

    Clover clover;
    const Fmunu f;
    dim3 threads; // number of active threads required
    int X[4]; // grid dimensions
    const real kappa;
    const real c_sw;
    
    CloverArg(CloverField &clover, const GaugeField &f, const double kappa, const double c_sw) :
      clover(clover, 0),
      f(f),
      threads(f.VolumeCB(), 2, 1),
      kappa(kappa),
      c_sw(c_sw)
    { 
      for (int dir=0; dir<4; ++dir) X[dir] = f.X()[dir];
    }
  };

  /*
    Put into clover order
    Upper-left block (chirality index 0)
       /                                                                                \
       |  1 + c*I*(F[0,1] - F[2,3]) ,     c*I*(F[1,2] - F[0,3]) + c*(F[0,2] + F[1,3])   |
       |                                                                                |
       |  c*I*(F[1,2] - F[0,3]) - c*(F[0,2] + F[1,3]),   1 - c*I*(F[0,1] - F[2,3])      |
       |                                                                                |
       \                                                                                /

       /
       | 1 - c*I*(F[0] - F[5]),   -c*I*(F[2] - F[3]) - c*(F[1] + F[4])
       |
       |  -c*I*(F[2] -F[3]) + c*(F[1] + F[4]),   1 + c*I*(F[0] - F[5])
       |
       \

     Lower-right block (chirality index 1)

       /                                                                  \
       |  1 - c*I*(F[0] + F[5]),  -c*I*(F[2] + F[3]) - c*(F[1] - F[4])    |
       |                                                                  |
       |  -c*I*(F[2]+F[3]) + c*(F[1]-F[4]),     1 + c*I*(F[0] + F[5])     |
       \                                                                  /
  */
  // Core routine for constructing clover term from field strength
  template <typename Arg> struct CloverCompute {
    Arg &arg;
    constexpr CloverCompute(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      constexpr int N = Arg::nColor*Arg::nSpin / 2;
      using real = typename Arg::real;
      using Complex = complex<real>;
      using Link = Matrix<Complex, Arg::nColor>;
      using HClov = HMatrix<real,N>;
      
      // Load the field-strength tensor from global memory
      Link F[6];
#pragma unroll
      for (int i=0; i<6; ++i) F[i] = arg.f(i, x_cb, parity);

      const real coeff_re = arg.kappa * arg.c_sw;
      const Complex coeff_im(0.0, coeff_re);
      const Complex I(0.0,1.0);
      Link block1[2], block2[2];
      HClov C;

      if(arg.exp_clover) {
	// Just compute the blocks, scale at the exponentiation
	setIdentity(&C);
	C *= 1.0/(2 * arg.kappa);
      }
      
      block1[0] = coeff_im*(F[0]-F[5]); // (18 + 6*9=) 72 floating-point ops
      block1[1] = coeff_im*(F[0]+F[5]); // 72 floating-point ops
      block2[0] = coeff_re*(F[1]+F[4] - I*(F[2]-F[3])); // 126 floating-point ops
      block2[1] = coeff_re*(F[1]-F[4] - I*(F[2]+F[3])); // 126 floating-point ops

      // This uses lots of unnecessary memory
#pragma unroll
      for (int ch=0; ch<2; ++ch) {
        HClov A;
        // c = 0(1) => positive(negative) chiral block
        // Compute real diagonal elements
#pragma unroll
        for (int i=0; i<N/2; ++i) {
          A(i+0,i+0) = 1.0 - block1[ch](i,i).real();
          A(i+3,i+3) = 1.0 + block1[ch](i,i).real();
        }

        // Compute off diagonal components
        // First row
        A(1,0) = -block1[ch](1,0);
        // Second row
        A(2,0) = -block1[ch](2,0);
        A(2,1) = -block1[ch](2,1);
        // Third row
        A(3,0) =  block2[ch](0,0);
        A(3,1) =  block2[ch](0,1);
        A(3,2) =  block2[ch](0,2);
        // Fourth row
        A(4,0) =  block2[ch](1,0);
        A(4,1) =  block2[ch](1,1);
        A(4,2) =  block2[ch](1,2);
        A(4,3) =  block1[ch](1,0);
        // Fifth row
        A(5,0) =  block2[ch](2,0);
        A(5,1) =  block2[ch](2,1);
        A(5,2) =  block2[ch](2,2);
        A(5,3) =  block1[ch](2,0);
        A(5,4) =  block1[ch](2,1);
        A *= static_cast<real>(0.5);

	// The clover mat is now defined and scaled:
	// A = C_sw/(4+m0) * (i/4) * \sigma_{munu} \hat{F}_{munu}(x)
	if(arg.exp_clover) {
	  if(x_cb == 0 && parity == 0 && ch == 0) printf("Exping\n");
	  // Exponentiate this matrix
	  expsuNTaylor(A);
	  // Scale and subtract the constant 1/(2kappa) terms
	  A = A * C;
	  A = A - C;
	}
        arg.clover(x_cb, parity, ch) = A;
      } // ch
      // 84 floating-point ops
    }
  };

}
