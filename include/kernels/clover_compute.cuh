#pragma once

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <kernel.h>

namespace quda {

  template <typename store_t_>
  struct CloverArg : kernel_param<> {
    using store_t = store_t_;
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = N_COLORS;
    static constexpr int nSpin = 4;
    
    using Clover = typename clover_mapper<store_t>::type;
    using Fmunu = typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type;

    Clover clover;
    const Fmunu f;
    int X[4]; // grid dimensions
    real coeff;
    
    CloverArg(CloverField &clover, const GaugeField &f, double coeff) :
      kernel_param(dim3(f.VolumeCB(), 2, 1)),
      clover(clover, 0),
      f(f),
      coeff(coeff)
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
    const Arg &arg;
    constexpr CloverCompute(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      constexpr int N = Arg::nColor*Arg::nSpin / 2;
      using real = typename Arg::real;
      using Complex = complex<real>;
      using Link = Matrix<Complex, Arg::nColor>;

      // Load the field-strength tensor from global memory
      Link F[6];
#pragma unroll
      for (int i=0; i<6; ++i) F[i] = arg.f(i, x_cb, parity);

      Complex I(0.0,1.0);
      Complex coeff(0.0, arg.coeff);
      Link block1[2], block2[2];
      // diagonal blocks
      block1[0] = coeff*(F[0] - F[5]); // (2*Nc*Nc+6*Nc*Nc=) 72 Nc=3 floating-point ops
      block1[1] = coeff*(F[0] + F[5]); // 72 floating-point ops

      // off-diagonal blocks
      block2[0] = arg.coeff*(F[1] + F[4] - I*(F[2] - F[3])); // 126 floating-point ops
      block2[1] = arg.coeff*(F[1] - F[4] - I*(F[2] + F[3])); // 126 floating-point ops

      // This uses lots of unnecessary memory
#pragma unroll
      for (int ch=0; ch<2; ++ch) {
        HMatrix<real,N> A;
        // c = 0(1) => positive(negative) chiral block
        // Compute real diagonal elements
#pragma unroll
        for (int i=0; i<N/2; ++i) {
          A(i,i)                    = 1.0 - block1[ch](i,i).real();
          A(i+N_COLORS, i+N_COLORS) = 1.0 + block1[ch](i,i).real();
        }
	
        // Compute off diagonal components, populating the A matrix
	// row by row

	// First nColor rows
#pragma unroll
	for(int c=1; c<Arg::nColor; c++) {
	  for(int d=0; d<c; d++) {
	    A(c,d) = - block1[ch](c,d);
	  }
	}
	
	// Second nColor rows
#pragma unroll
	for(int c=Arg::nColor; c<2*Arg::nColor; c++) {
	  for(int d=0; d<Arg::nColor; d++) {
	    A(c,d) = block2[ch](c - Arg::nColor,d);	      
	  }
	  for(int d=Arg::nColor; d<c; d++) {
	    A(c,d) = block1[ch](c - Arg::nColor,d - Arg::nColor);
	  }
	}
	
        A *= static_cast<real>(0.5);
	
        arg.clover(x_cb, parity, ch) = A;
      } // ch
      // DMH FIXME 84 floating-point ops
    }
  };

}
