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
    complex<Float> data[Nc*4];
    
    /** 
	Return this spinor spin projected
	@param dim Which dimension projector are we using
	@param sign Positive or negative projector
	@return The spin-projected Spinor
    */
    __device__ __host__ ColorSpinor<Float,Nc,2> project(int dim, int sign) {
      ColorSpinor<Float,Nc,2> proj;
      complex<Float> j(0.0,1.0);
      
      switch (dim) {
      case 0: // x dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + j * (*this)(3,i);
	    proj(1,i) = (*this)(1,i) + j * (*this)(2,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - j * (*this)(3,i);
	    proj(1,i) = (*this)(1,i) - j * (*this)(2,i);
	    break;
	  }
	}
	
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + (*this)(3,i);
	    proj(1,i) = (*this)(1,i) - (*this)(2,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - (*this)(3,i);
	    proj(1,i) = (*this)(1,i) + (*this)(2,i);
	    break;
	  }
	}
      
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) + j * (*this)(2,i);
	    proj(1,i) = (*this)(1,i) - j * (*this)(3,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = (*this)(0,i) - j * (*this)(2,i);
	    proj(1,i) = (*this)(1,i) + j * (*this)(3,i);
	    break;
	  }
	}

      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = 2*(*this)(0,i);
	    proj(1,i) = 2*(*this)(1,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    proj(0,i) = 2*(*this)(2,i);
	    proj(1,i) = 2*(*this)(3,i);
	    break;
	  }
	}
      }
      return proj;
    }
    
    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Comlex number at this spin and color index
     */
    __device__ __host__ complex<Float> operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }
  };

  /**
     This is the specialization for Nspin=2.  For fields with two
     spins we can define a spin reconstruction operation.
   */
  template <typename Float, int Nc>
    struct ColorSpinor<Float, Nc, 2> {
    complex<Float> data[Nc*2];
    
    /** 
	Spin reconstruct the full Spinor from the projected spinor
	@param dim Which dimension projector are we using
	@param sign Positive or negative projector
	@return The spin-reconstructed Spinor
    */
      __device__ __host__ ColorSpinor<Float,Nc,4> reconstruct(int dim, int sign) {
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
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = j*(*this)(1,i);
	    recon(3,i) = j*(*this)(0,i);
	    break;
	  }
	}
	
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = -(*this)(1,i);
	    recon(3,i) = (*this)(0,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = (*this)(1,i);
	    recon(3,i) = -(*this)(0,i);
	    break;
	  }
	}
      
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = -j*(*this)(0,i);
	    recon(3,i) = j*(*this)(1,i);
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = j*(*this)(0,i);
	    recon(3,i) = -j*(*this)(1,i);
	    break;
	  }
	}

      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = (*this)(0,i);
	    recon(1,i) = (*this)(1,i);
	    recon(2,i) = 0;
	    recon(3,i) = 0;
	    break;
	  }
	case -1: // negative projector
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = 0;
	    recon(1,i) = 0;
	    recon(2,i) = (*this)(0,i);
	    recon(3,i) = (*this)(1,i);
	    break;
	  }
	}
      }
      return recon;
    }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ complex<Float> operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       Accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }
  };


  /**
     Compute the outer product over color and take the spin trace
     out(j,i) = \sum_s a(s,j) * conj (b(s,i))
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The spin traced matrix
  */
  template<typename Float, int Nc, int Ns> __device__ __host__
    Matrix< typename ComplexTypeId<Float>::Type ,Nc> outerProdSpinTrace(const ColorSpinor<Float,Nc,Ns> &a, const ColorSpinor<Float,Nc,Ns> &b) {

    typedef typename ComplexTypeId<Float>::Type complex_type;
    Matrix<complex_type,Nc> out;
    
    // trace over spin
    for (int s=0; s<Ns; s++) {
      // outer product over color
      for (int i=0; i<Nc; i++) {
        for(int j=0; j<Nc; j++) {
	  out(j,i) +=
	    a(s,j) *
	    conj(b(s,i));
	}
      }
    }
    return out;
  }


  
} // namespace quda
