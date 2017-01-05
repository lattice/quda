/**
 * @file    color_spinor.h
 *
 * @section Description
 *
 * The header file defines some helper structs for dealing with
 * ColorSpinors (e.g., a vector with both color and spin degrees of
 * freedom).
 */
namespace quda {

  /**
     This is the generic declaration of ColorSpinor.
   */
  template <typename Float, int Nc, int Ns>
    struct ColorSpinor {

      complex<Float> data[Nc*Ns];

    };

  /**
     This is the specialization for Nspin=4.  For fields with four
     spins we can define a spin projection operation.
   */
  template <typename Float, int Nc>
    struct ColorSpinor<Float, Nc, 4> {
    static const int Ns = 4;
    complex<Float> data[Nc*4];

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>() {
      for (int i=0; i<Nc*Ns; i++) {
	data[i] = 0;
      }      
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>(const ColorSpinor<Float, Nc, 4> &a) {
      for (int i=0; i<Nc*Ns; i++) {
	data[i] = a.data[i];
      }      
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>& operator=(const ColorSpinor<Float, Nc, 4> &a) {
      if (this != &a) {
	for (int i=0; i<Nc*Ns; i++) {
	  data[i] = a.data[i];
	}
      }
      return *this;
    }

    /** 
	Return this spinor spin projected
	@param dim Which dimension projector are we using
	@param sign Positive or negative projector
	@return The spin-projected Spinor
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,2> project(int dim, int sign) {
      ColorSpinor<Float,Nc,2> proj;
      complex<Float> j(0.0,1.0);
      
      switch (dim) {
      case 0: // x dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + j * (*this)(3,i);
	    proj(1,i) = (*this)(1,i) + j * (*this)(2,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - j * (*this)(3,i);
	    proj(1,i) = (*this)(1,i) - j * (*this)(2,i);
	  }
	  break;
	}
	break;
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + (*this)(3,i);
	    proj(1,i) = (*this)(1,i) - (*this)(2,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - (*this)(3,i);
	    proj(1,i) = (*this)(1,i) + (*this)(2,i);
	  }
	  break;
	}
      	break;
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + j * (*this)(2,i);
	    proj(1,i) = (*this)(1,i) - j * (*this)(3,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - j * (*this)(2,i);
	    proj(1,i) = (*this)(1,i) + j * (*this)(3,i);
	  }
	  break;
	}
	break;
      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = 2*(*this)(0,i);
	    proj(1,i) = 2*(*this)(1,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = 2*(*this)(2,i);
	    proj(1,i) = 2*(*this)(3,i);
	  }
	  break;
	}
	break;
      }

      return proj;
    }

    /**
       Return this spinor multiplied by sigma(mu,nu)
       @param mu mu direction
       @param nu nu direction

       sigma(0,1) =  i  0  0  0
                     0 -i  0  0
		     0  0  i  0
		     0  0  0 -i

       sigma(0,2) =  0 -1  0  0
                     1  0  0  0
		     0  0  0 -1
		     0  0  1  0

       sigma(0,3) =  0  0  0 -i
                     0  0 -i  0
		     0 -i  i  0
		    -i  0  0  0

       sigma(1,2) =  0  i  0  0
                     i  0  0  0
		     0  0  0  i
		     0  0  i  0

       sigma(1,3) =  0  0  0 -1
                     0  0  1  0
		     0 -1  0  0
		     1  0  0  0

       sigma(2,3) =  0  0 -i  0
                     0  0  0  i
		    -i  0  0  0
		     0  i  0  0
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,4> sigma(int mu, int nu) {
      ColorSpinor<Float,Nc,4> a;
      ColorSpinor<Float,Nc,4> &b = *this;
      complex<Float> j(0.0,1.0);

      switch(mu) {
      case 0:
	switch(nu) {
	case 1:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  j*b(0,i);
	    a(1,i) = -j*b(1,i);
	    a(2,i) =  j*b(2,i);
	    a(3,i) = -j*b(3,i);
	  }
	  break;
	case 2:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -b(1,i);
	    a(1,i) =  b(0,i);
	    a(2,i) = -b(3,i);
	    a(3,i) =  b(2,i);
	  }
	  break;
	case 3:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(3,i);
	    a(1,i) = -j*b(2,i);
	    a(2,i) = -j*b(1,i);
	    a(3,i) = -j*b(0,i);
	  }
	  break;
	}
	break;
      case 1:
	switch(nu) {
	case 0:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(0,i);
	    a(1,i) =  j*b(1,i);
	    a(2,i) = -j*b(2,i);
	    a(3,i) =  j*b(3,i);
	  }
	  break;
	case 2:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = j*b(1,i);
	    a(1,i) = j*b(0,i);
	    a(2,i) = j*b(3,i);
	    a(3,i) = j*b(2,i);
	  }
	  break;
	case 3:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -b(3,i);
	    a(1,i) =  b(2,i);
	    a(2,i) = -b(1,i);
	    a(3,i) =  b(0,i);
	  }
	  break;
	}
	break;
      case 2:
	switch(nu) {
	case 0:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  b(1,i);
	    a(1,i) = -b(0,i);
	    a(2,i) =  b(3,i);
	    a(3,i) = -b(2,i);
	  }
	  break;
	case 1:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(1,i);
	    a(1,i) = -j*b(0,i);
	    a(2,i) = -j*b(3,i);
	    a(3,i) = -j*b(2,i);
	  }
	  break;
	case 3:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(2,i);
	    a(1,i) =  j*b(3,i);
	    a(2,i) = -j*b(0,i);
	    a(3,i) =  j*b(1,i);
	  }
	  break;
	}
	break;
      case 3:
	switch(nu) {
	case 0:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = j*b(3,i);
	    a(1,i) = j*b(2,i);
	    a(2,i) = j*b(1,i);
	    a(3,i) = j*b(0,i);
	  }
	  break;
	case 1:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  b(3,i);
	    a(1,i) = -b(2,i);
	    a(2,i) =  b(1,i);
	    a(3,i) = -b(0,i);
	  }
	  break;
	case 2:
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  j*b(2,i);
	    a(1,i) = -j*b(3,i);
	    a(2,i) =  j*b(0,i);
	    a(3,i) = -j*b(1,i);
	  }
	  break;
	}
	break;
      }

      return a;
    }


    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline complex<Float>& operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }

    __device__ __host__ void print() {
      for (int s=0; s<4; s++) {
	for (int c=0; c<3; c++) {
	  printf("s=%d c=%d %e %e\n", s, c, data[s*Nc+c].real(), data[s*Nc+c].imag());
	}
      }
    };
  };

  /**
     This is the specialization for Nspin=2.  For fields with two
     spins we can define a spin reconstruction operation.
   */
  template <typename Float, int Nc>
    struct ColorSpinor<Float, Nc, 2> {
    static const int Ns = 2;
    complex<Float> data[Nc*2];
    
    __device__ __host__ inline ColorSpinor<Float, Nc, 2>() {
      for (int i=0; i<Nc*Ns; i++) {
	data[i] = 0;
      }      
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 2>(const ColorSpinor<Float, Nc, 2> &a) {
      for (int i=0; i<Nc*Ns; i++) {
	data[i] = a.data[i];
      }      
    }


    __device__ __host__ inline ColorSpinor<Float, Nc, 2>& operator=(const ColorSpinor<Float, Nc, 2> &a) {
      if (this != &a) {
	for (int i=0; i<Nc*Ns; i++) {
	  data[i] = a.data[i];
	}
      }
      return *this;
    }

    /** 
	Spin reconstruct the full Spinor from the projected spinor
	@param dim Which dimension projector are we using
	@param sign Positive or negative projector
	@return The spin-reconstructed Spinor
    */
      __device__ __host__ inline ColorSpinor<Float,Nc,4> reconstruct(int dim, int sign) {
      ColorSpinor<Float,Nc,4> recon;
      complex<Float> j(0.0,1.0);
      
      switch (dim) {
      case 0: // x dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = -j*(*this)(1,i);
	    recon(3,i) = -j*(*this)(0,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = j*(*this)(1,i);
	    recon(3,i) = j*(*this)(0,i);
	  }
	  break;
	}
	break;
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = -(*this)(1,i);
	    recon(3,i) = (*this)(0,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = (*this)(1,i);
	    recon(3,i) = -(*this)(0,i);
	  }
	  break;
	}
	break;      
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = -j*(*this)(0,i);
	    recon(3,i) = j*(*this)(1,i);
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = j*(*this)(0,i);
	    recon(3,i) = -j*(*this)(1,i);
	  }
	  break;
	}
	break;
      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = 0;
	    recon(3,i) = 0;
	  }
	  break;
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = 0;
	    recon(1,i) = 0;
	    recon(2,i) = (*this)(0,i);
	    recon(3,i) = (*this)(1,i);
	  }
	  break;
	}
	break;
      }
      return recon;
    }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline complex<Float>& operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }

    __device__ __host__ void print() {
      for (int s=0; s<2; s++) {
	for (int c=0; c<3; c++) {
	  printf("s=%d c=%d %e %e\n", s, c, data[s*Nc+c].real(), data[s*Nc+c].imag());
	}
      }
    };
  };


  /**
     Compute the outer product over color and take the spin trace
     out(j,i) = \sum_s a(s,j) * conj (b(s,i))
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The spin traced matrix
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
    Matrix<complex<Float>,Nc> outerProdSpinTrace(const ColorSpinor<Float,Nc,Ns> &a, const ColorSpinor<Float,Nc,Ns> &b) {

    Matrix<complex<Float>,Nc> out;

    // outer product over color
#pragma unroll
    for (int i=0; i<Nc; i++) {
#pragma unroll
      for(int j=0; j<Nc; j++) {
	// trace over spin (manual unroll for perf)
	out(j,i).real(                   a(0,j).real() * b(0,i).real() );
	out(j,i).real( out(j,i).real() + a(0,j).imag() * b(0,i).imag() );
	out(j,i).imag(                   a(0,j).imag() * b(0,i).real() );
	out(j,i).imag( out(j,i).imag() - a(0,j).real() * b(0,i).imag() );
	//out(j,i) = a(0,j) * conj(b(0,i));

#pragma unroll
	for (int s=1; s<Ns; s++) {
	  out(j,i).real( out(j,i).real() + a(s,j).real() * b(s,i).real() );
	  out(j,i).real( out(j,i).real() + a(s,j).imag() * b(s,i).imag() );
	  out(j,i).imag( out(j,i).imag() + a(s,j).imag() * b(s,i).real() );
	  out(j,i).imag( out(j,i).imag() - a(s,j).real() * b(s,i).imag() );
	  // out(j,i) += a(s,j) * conj(b(s,i));
	}
      }
    }
    return out;
  }

  
} // namespace quda
