#pragma once

#include <complex_quda.h>
#include <quda_matrix.h>

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

  template<typename Float, typename T> struct colorspinor_wrapper;
  template<typename Float, typename T> struct colorspinor_ghost_wrapper;

  /**
     This is the generic declaration of ColorSpinor.
   */
  template <typename Float, int Nc, int Ns>
    struct ColorSpinor {

    static constexpr int size = Nc * Ns;
    complex<Float> data[size];

    __device__ __host__ inline ColorSpinor<Float, Nc, Ns>()
    {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
      }

      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const ColorSpinor<Float, Nc, Ns> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }

      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>& operator=(const ColorSpinor<Float, Nc, Ns> &a) {
	if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
        }
	return *this;
      }

      __device__ __host__ inline ColorSpinor<Float, Nc, Ns> operator-() const
      {
        ColorSpinor<Float, Nc, Ns> a;
#pragma unroll
        for (int i = 0; i < size; i++) { a.data[i] = -data[i]; }
        return a;
      }

      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>& operator+=(const ColorSpinor<Float, Nc, Ns> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] += a.data[i]; }
        return *this;
      }

      template <typename T> __device__ __host__ inline ColorSpinor<Float, Nc, Ns> &operator*=(const T &a)
      {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] *= a; }
        return *this;
      }

      __device__ __host__ inline ColorSpinor<Float, Nc, Ns> &operator-=(const ColorSpinor<Float, Nc, Ns> &a)
      {
        if (this != &a) {
#pragma unroll
          for (int i = 0; i < Nc * Ns; i++) { data[i] -= a.data[i]; }
        }
        return *this;
      }

      template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_ghost_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_ghost_wrapper<Float, S> &s);

      /**
	 @brief 2-d accessor functor
	 @param[in] s Spin index
	 @param[in] c Color index
	 @return Complex number at this spin and color index
      */
      __device__ __host__ inline complex<Float>& operator()(int s, int c) { return data[s*Nc + c]; }

      /**
	 @brief 2-d accessor functor
	 @param[in] s Spin index
	 @param[in] c Color index
	 @return Complex number at this spin and color index
      */
      __device__ __host__ inline const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline complex<Float>& operator()(int idx) { return data[idx]; }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline const complex<Float>& operator()(int idx) const { return data[idx]; }

      /**
         @brief Prints the NsxNc complex elements of the color spinor
      */
      __device__ __host__ void print() const
      {
        for (int s=0; s<Ns; s++) {
          for (int c=0; c<Nc; c++) {
            printf("s=%d c=%d %e %e\n", s, c, data[s*Nc+c].real(), data[s*Nc+c].imag());
          }
        }
      }
    };

    /**
       This is the specialization for Nspin=4.  For fields with four
       spins we can define a spin projection operation.
    */
    template <typename Float, int Nc> struct ColorSpinor<Float, Nc, 4> {
      static constexpr int Ns = 4;
      static constexpr int size = Nc * Ns;
      complex<Float> data[size];

      __device__ __host__ inline ColorSpinor<Float, Nc, 4>()
      {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
      }

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>(const ColorSpinor<Float, Nc, 4> &a) {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>& operator=(const ColorSpinor<Float, Nc, 4> &a) {
      if (this != &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }
      return *this;
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 4>& operator+=(const ColorSpinor<Float, Nc, 4> &a) {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] += a.data[i]; }
      return *this;
    }

    template <typename T> __device__ __host__ inline ColorSpinor<Float, Nc, 4> &operator*=(const T &a)
    {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] *= a; }
      return *this;
    }

    /**
	Return this application of gamma_dim to this spinor
	@param dim Which dimension gamma matrix we are applying
	@return The new spinor
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,4> gamma(int dim) {
      ColorSpinor<Float,Nc,4> a;
      const auto &t = *this;

      switch (dim) {
      case 0: // x dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = i_(t(3, i));
          a(1, i) = i_(t(2, i));
          a(2, i) = -i_(t(1, i));
          a(3, i) = -i_(t(0, i));
        }
	break;
      case 1: // y dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = t(3, i);
          a(1, i) = -t(2, i);
          a(2, i) = -t(1, i);
          a(3, i) = t(0, i);
        }
	break;
      case 2: // z dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = i_(t(2, i));
          a(1, i) = -i_(t(3, i));
          a(2, i) = -i_(t(0, i));
          a(3, i) = i_(t(1, i));
        }
	break;
      case 3: // t dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = t(0, i);
          a(1, i) = t(1, i);
          a(2, i) = -t(2, i);
          a(3, i) = -t(3, i);
        }
	break;
      case 4: // gamma_5
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = t(2, i);
          a(1, i) = t(3, i);
          a(2, i) = t(0, i);
          a(3, i) = t(1, i);
        }
	break;
      }

      return a;
    }

    /**
	Return this application of gamma_dim to this spinor
	@param dim Which dimension gamma matrix we are applying
	@return The new spinor
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,4> igamma(int dim) {
      ColorSpinor<Float,Nc,4> a;
      const auto &t = *this;

      switch (dim) {
      case 0: // x dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = -t(3, i);
          a(1, i) = -t(2, i);
          a(2, i) = t(1, i);
          a(3, i) = t(0, i);
        }
	break;
      case 1: // y dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = i_(t(3, i));
          a(1, i) = -i_(t(2, i));
          a(2, i) = -i_(t(1, i));
          a(3, i) = i_(t(0, i));
        }
	break;
      case 2: // z dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = -t(2, i);
          a(1, i) = t(3, i);
          a(2, i) = t(0, i);
          a(3, i) = -t(1, i);
        }
	break;
      case 3: // t dimension
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = i_(t(0, i));
          a(1, i) = i_(t(1, i));
          a(2, i) = -i_(t(2, i));
          a(3, i) = -i_(t(3, i));
        }
	break;
      case 4: // gamma_5
#pragma unroll
	for (int i=0; i<Nc; i++) {
          a(0, i) = i_(t(2, i));
          a(1, i) = i_(t(3, i));
          a(2, i) = i_(t(0, i));
          a(3, i) = i_(t(1, i));
        }
	break;
      }

      return a;
    }

    /**
       @brief Project four-component spinor to either chirality
       @param[in] chirality Which chirality
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,2> chiral_project(int chirality) const {
      ColorSpinor<Float,Nc,2> proj;
#pragma unroll
      for (int s=0; s<Ns/2; s++) {
#pragma unroll
	for (int c=0; c<Nc; c++) {
	  proj(s,c) = (*this)(chirality*Ns/2+s,c);
	}
      }
      return proj;
    }

    /**
        Return this spinor spin projected
        @param dim Which dimension projector are we using
        @param sign Positive or negative projector
        @return The spin-projected Spinor
    */
    __device__ __host__ inline ColorSpinor<Float, Nc, 2> project(int dim, int sign) const
    {
      ColorSpinor<Float,Nc,2> proj;
      const auto &t = *this;
      switch (dim) {
      case 0: // x dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) + i_(t(3, i));
            proj(1, i) = t(1, i) + i_(t(2, i));
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) - i_(t(3, i));
            proj(1, i) = t(1, i) - i_(t(2, i));
          }
	  break;
	}
	break;
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) + t(3, i);
            proj(1, i) = t(1, i) - t(2, i);
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) - t(3, i);
            proj(1, i) = t(1, i) + t(2, i);
          }
	  break;
	}
      	break;
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) + i_(t(2, i));
            proj(1, i) = t(1, i) - i_(t(3, i));
          }
          break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = t(0, i) - i_(t(2, i));
            proj(1, i) = t(1, i) + i_(t(3, i));
          }
	  break;
	}
	break;
      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = 2 * t(0, i);
            proj(1, i) = 2 * t(1, i);
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            proj(0, i) = 2 * t(2, i);
            proj(1, i) = 2 * t(3, i);
          }
	  break;
	}
	break;
      case 4:
        switch (sign) {
        case 1: // positive projector
#pragma unroll
          for (int i = 0; i < Nc; i++) {
            proj(0, i) = t(0, i) + t(2, i);
            proj(1, i) = t(1, i) + t(3, i);
          }
          break;
        case -1: // negative projector
#pragma unroll
          for (int i = 0; i < Nc; i++) {
            proj(0, i) = t(0, i) - t(2, i);
            proj(1, i) = t(1, i) - t(3, i);
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
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  j*b(0,i);
	    a(1,i) = -j*b(1,i);
	    a(2,i) =  j*b(2,i);
	    a(3,i) = -j*b(3,i);
	  }
	  break;
	case 2:
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -b(1,i);
	    a(1,i) =  b(0,i);
	    a(2,i) = -b(3,i);
	    a(3,i) =  b(2,i);
	  }
	  break;
	case 3:
#pragma unroll
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
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(0,i);
	    a(1,i) =  j*b(1,i);
	    a(2,i) = -j*b(2,i);
	    a(3,i) =  j*b(3,i);
	  }
	  break;
	case 2:
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = j*b(1,i);
	    a(1,i) = j*b(0,i);
	    a(2,i) = j*b(3,i);
	    a(3,i) = j*b(2,i);
	  }
	  break;
	case 3:
#pragma unroll
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
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  b(1,i);
	    a(1,i) = -b(0,i);
	    a(2,i) =  b(3,i);
	    a(3,i) = -b(2,i);
	  }
	  break;
	case 1:
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = -j*b(1,i);
	    a(1,i) = -j*b(0,i);
	    a(2,i) = -j*b(3,i);
	    a(3,i) = -j*b(2,i);
	  }
	  break;
	case 3:
#pragma unroll
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
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) = j*b(3,i);
	    a(1,i) = j*b(2,i);
	    a(2,i) = j*b(1,i);
	    a(3,i) = j*b(0,i);
	  }
	  break;
	case 1:
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    a(0,i) =  b(3,i);
	    a(1,i) = -b(2,i);
	    a(2,i) =  b(1,i);
	    a(3,i) = -b(0,i);
	  }
	  break;
	case 2:
#pragma unroll
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
       @brief 2-d accessor functor
       @param[in] s Spin index
       @param[in] c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline complex<Float>& operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       @brief 2-d accessor functor
       @param[in] s Spin index
       @param[in] c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }

    /**
       @brief 1-d accessor functor
       @param[in] idx Index
       @return Complex number at this index
     */
    __device__ __host__ inline complex<Float>& operator()(int idx) { return data[idx]; }

    /**
       @brief 1-d accessor functor
       @param[in] idx Index
       @return Complex number at this index
     */
    __device__ __host__ inline const complex<Float>& operator()(int idx) const { return data[idx]; }

    template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_ghost_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_ghost_wrapper<Float, S> &s);

    /**
       @brief Transform from relativistic into non-relavisitic basis
       Required normalization factor of 1/2 included in clover normalization
    */
    __device__ __host__ inline void toNonRel() {
      ColorSpinor<Float,Nc,Ns> a;
#pragma unroll
      for (int c=0; c<Nc; c++) {
	a(0,c) =  (*this)(1,c)+(*this)(3,c);
	a(1,c) = -(*this)(2,c)-(*this)(0,c);
	a(2,c) = -(*this)(3,c)+(*this)(1,c);
	a(3,c) = -(*this)(0,c)+(*this)(2,c);
      }
      *this = a;
    }

    /**
       @brief Transform from non-relativistic into relavisitic basis
    */
    __device__ __host__ inline void toRel() {
      ColorSpinor<Float,Nc,Ns> a;
#pragma unroll
      for (int c=0; c<Nc; c++) {
	a(0,c) = -(*this)(1,c)-(*this)(3,c);
	a(1,c) =  (*this)(2,c)+(*this)(0,c);
	a(2,c) =  (*this)(3,c)-(*this)(1,c);
	a(3,c) =  (*this)(0,c)-(*this)(2,c);
      }
      *this = a;
    }

    __device__ __host__ void print() const
    {
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  printf("s=%d c=%d %e %e\n", s, c, data[s*Nc+c].real(), data[s*Nc+c].imag());
	}
      }
    }
    };

  /**
     This is the specialization for Nspin=2.  For fields with two
     spins we can define a spin reconstruction operation.
   */
  template <typename Float, int Nc>
    struct ColorSpinor<Float, Nc, 2> {
    static constexpr int Ns = 2;
    static constexpr int size = Ns * Nc;
    complex<Float> data[size];

    __device__ __host__ inline ColorSpinor<Float, Nc, 2>() {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 2>(const ColorSpinor<Float, Nc, 2> &a) {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
    }


    __device__ __host__ inline ColorSpinor<Float, Nc, 2>& operator=(const ColorSpinor<Float, Nc, 2> &a) {
      if (this != &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }
      return *this;
    }

    __device__ __host__ inline ColorSpinor<Float, Nc, 2>& operator+=(const ColorSpinor<Float, Nc, 2> &a) {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] += a.data[i]; }
      return *this;
    }

    template <typename T> __device__ __host__ inline ColorSpinor<Float, Nc, 2> &operator*=(const T &a)
    {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] *= a; }
      return *this;
    }

    /**
       @brief Reconstruct two-component spinor to a four-component spinor
       @param[in] chirality Which chirality we assigning to
    */
    __device__ __host__ inline ColorSpinor<Float,Nc,4> chiral_reconstruct(int chirality) const {
      ColorSpinor<Float,Nc,4> recon;
#pragma unroll
      for (int s=0; s<Ns; s++) {
#pragma unroll
	for (int c=0; c<Nc; c++) {
	  recon(chirality*Ns+s,c) = (*this)(s,c);
	}
      }
      return recon;
    }

    /**
        @brief Spin reconstruct the full Spinor from the projected spinor
        @param dim Which dimension projector are we using
        @param sign Positive or negative projector
        @return The spin-reconstructed Spinor
    */
    __device__ __host__ inline ColorSpinor<Float, Nc, 4> reconstruct(int dim, int sign) const
    {
      ColorSpinor<Float, Nc, 4> recon;
      const auto t = *this;

      switch (dim) {
      case 0: // x dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = -i_(t(1, i));
            recon(3, i) = -i_(t(0, i));
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = i_(t(1, i));
            recon(3, i) = i_(t(0, i));
          }
	  break;
	}
	break;
      case 1: // y dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = -t(1, i);
            recon(3, i) = t(0, i);
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = t(1, i);
            recon(3, i) = -t(0, i);
          }
          break;
        }
        break;
      case 2: // z dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = -i_(t(0, i));
            recon(3, i) = i_(t(1, i));
          }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = i_(t(0, i));
            recon(3, i) = -i_(t(1, i));
          }
	  break;
	}
	break;
      case 3: // t dimension
	switch (sign) {
	case 1: // positive projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2,i) = 0;
	    recon(3,i) = 0;
	  }
	  break;
	case -1: // negative projector
#pragma unroll
	  for (int i=0; i<Nc; i++) {
	    recon(0,i) = 0;
	    recon(1,i) = 0;
            recon(2, i) = t(0, i);
            recon(3, i) = t(1, i);
          }
	  break;
	}
	break;
      case 4:
        switch (sign) {
        case 1: // positive projector
#pragma unroll
          for (int i = 0; i < Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = t(0, i);
            recon(3, i) = t(1, i);
          }
          break;
        case -1: // negative projector
#pragma unroll
          for (int i = 0; i < Nc; i++) {
            recon(0, i) = t(0, i);
            recon(1, i) = t(1, i);
            recon(2, i) = -t(0, i);
            recon(3, i) = -t(1, i);
          }
          break;
        }
        break;
      }
      return recon;
    }

    /**
       @brief 2-d accessor functor
       @param s Spin index
       @paran c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline complex<Float>& operator()(int s, int c) { return data[s*Nc + c]; }

    /**
       @brief 2-d accessor functor
       @param s Spin index
       @param c Color index
       @return Complex number at this spin and color index
     */
    __device__ __host__ inline const complex<Float>& operator()(int s, int c) const { return data[s*Nc + c]; }

    /**
       @brief 1-d accessor functor
       @param[in] idx Index
       @return Complex number at this index
     */
    __device__ __host__ inline complex<Float>& operator()(int idx) { return data[idx]; }

    /**
       @brief 1-d accessor functor
       @param[in] idx Index
       @return Complex number at this index
     */
    __device__ __host__ inline const complex<Float>& operator()(int idx) const { return data[idx]; }

    template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline ColorSpinor<Float, Nc, Ns>(const colorspinor_ghost_wrapper<Float, S> &s);

    template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_ghost_wrapper<Float, S> &s);

    __device__ __host__ void print() const
    {
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  printf("s=%d c=%d %e %e\n", s, c, data[s*Nc+c].real(), data[s*Nc+c].imag());
	}
      }
    }
  };

  /**
     @brief Compute the inner product over color and spin
     dot = \sum_s,c conj(a(s,c)) * b(s,c)
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The inner product
  */
  template <typename Float, int Nc, int Ns>
  __device__ __host__ inline complex<Float> innerProduct(const ColorSpinor<Float, Nc, Ns> &a,
                                                         const ColorSpinor<Float, Nc, Ns> &b)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int s = 0; s < Ns; s++) { dot += innerProduct(a, b, s, s); }
    return dot;
  }

  /**
     Compute the inner product over color at spin s between two ColorSpinor fields
     dot = \sum_c conj(a(s,c)) * b(s,c)
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @param s diagonal spin index
     @return The inner product
  */
  template <typename Float, int Nc, int Ns>
  __device__ __host__ inline complex<Float> innerProduct(const ColorSpinor<Float, Nc, Ns> &a,
                                                         const ColorSpinor<Float, Nc, Ns> &b, int s)
  {
    return innerProduct(a, b, s, s);
  }

  /**
     Compute the inner product over color at spin sa and sb  between two ColorSpinor fields
     dot = \sum_c conj(a(s1,c)) * b(s2,c)
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @param sa Left-hand side spin index
     @param sb Right-hand side spin index
     @return The inner product
  */
  template <typename Float, int Nc, int Ns>
  __device__ __host__ inline complex<Float> innerProduct(const ColorSpinor<Float, Nc, Ns> &a,
                                                         const ColorSpinor<Float, Nc, Ns> &b, int sa, int sb)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int c = 0; c < Nc; c++) {
      dot.x += a(sa, c).real() * b(sb, c).real();
      dot.x += a(sa, c).imag() * b(sb, c).imag();
      dot.y += a(sa, c).real() * b(sb, c).imag();
      dot.y -= a(sa, c).imag() * b(sb, c).real();
    }
    return dot;
  }

  /**
     @brief Compute the inner product over color at spin sa and sb between a
     color spinors a and b of different spin length
     dot = \sum_c conj(a(c)) * b(s,c)
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The inner product
  */
  template <typename Float, int Nc, int Nsa, int Nsb>
  __device__ __host__ inline complex<Float> innerProduct(const ColorSpinor<Float, Nc, Nsa> &a,
                                                         const ColorSpinor<Float, Nc, Nsb> &b, int sa, int sb)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int c = 0; c < Nc; c++) {
      dot.x += a(sa, c).real() * b(sb, c).real();
      dot.x += a(sa, c).imag() * b(sb, c).imag();
      dot.y += a(sa, c).real() * b(sb, c).imag();
      dot.y -= a(sa, c).imag() * b(sb, c).real();
    }
    return dot;
  }

  /**
     Compute the outer product over color and take the spin trace
     out(j,i) = \sum_s a(s,j) * conj (b(s,i))
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The spin traced matrix
  */
  template <typename Float, int Nc, int Ns>
  __device__ __host__ inline Matrix<complex<Float>, Nc> outerProdSpinTrace(const ColorSpinor<Float, Nc, Ns> &a,
                                                                           const ColorSpinor<Float, Nc, Ns> &b)
  {
    Matrix<complex<Float>, Nc> out;

    // outer product over color
#pragma unroll
    for (int i = 0; i < Nc; i++) {
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // trace over spin (manual unroll for perf)
        out(j, i).real(a(0, j).real() * b(0, i).real());
        out(j, i).real(out(j, i).real() + a(0, j).imag() * b(0, i).imag());
        out(j, i).imag(a(0, j).imag() * b(0, i).real());
        out(j, i).imag(out(j, i).imag() - a(0, j).real() * b(0, i).imag());
        // out(j,i) = a(0,j) * conj(b(0,i));

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

  /**
     Compute the outer product over color and take the spin trace
     out(j,i) = \sum_s a(s,j) * conj (b(s,i))
     @param a Left-hand side ColorSpinor
     @param b Right-hand side ColorSpinor
     @return The spin traced matrix
  */
  template <typename Float, int Nc>
  __device__ __host__ inline Matrix<complex<Float>, Nc> outerProduct(const ColorSpinor<Float, Nc, 1> &a,
                                                                     const ColorSpinor<Float, Nc, 1> &b)
  {
    Matrix<complex<Float>, Nc> out;

    // outer product over color
#pragma unroll
    for (int i = 0; i < Nc; i++) {
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // trace over spin (manual unroll for perf)
        out(j, i).real(a(0, j).real() * b(0, i).real());
        out(j, i).real(out(j, i).real() + a(0, j).imag() * b(0, i).imag());
        out(j, i).imag(a(0, j).imag() * b(0, i).real());
        out(j, i).imag(out(j, i).imag() - a(0, j).real() * b(0, i).imag());
        // out(j,i) = a(0,j) * conj(b(0,i));
      }
    }
    return out;
  }

  /**
     @brief ColorSpinor addition operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
    ColorSpinor<Float,Nc,Ns> operator+(const ColorSpinor<Float,Nc,Ns> &x, const ColorSpinor<Float,Nc,Ns> &y) {

    ColorSpinor<Float,Nc,Ns> z;

#pragma unroll
    for (int i=0; i<Nc; i++) {
#pragma unroll
      for (int s=0; s<Ns; s++) {
	z.data[s*Nc + i] = x.data[s*Nc + i] + y.data[s*Nc + i];
      }
    }

    return z;
  }

  /**
     @brief ColorSpinor subtraction operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
    ColorSpinor<Float,Nc,Ns> operator-(const ColorSpinor<Float,Nc,Ns> &x, const ColorSpinor<Float,Nc,Ns> &y) {

    ColorSpinor<Float,Nc,Ns> z;

#pragma unroll
    for (int i=0; i<Nc; i++) {
#pragma unroll
      for (int s=0; s<Ns; s++) {
	z.data[s*Nc + i] = x.data[s*Nc + i] - y.data[s*Nc + i];
      }
    }

    return z;
  }

  /**
     @brief Compute the scalar-vector product y = a * x
     @param[in] a Input scalar
     @param[in] x Input vector
     @return The vector a * x
  */
  template<typename Float, int Nc, int Ns, typename S> __device__ __host__ inline
    ColorSpinor<Float,Nc,Ns> operator*(const S &a, const ColorSpinor<Float,Nc,Ns> &x) {

    ColorSpinor<Float,Nc,Ns> y;

#pragma unroll
    for (int i=0; i<Nc; i++) {
#pragma unroll
      for (int s=0; s<Ns; s++) {
	y.data[s*Nc + i] = a * x.data[s*Nc + i];
      }
    }

    return y;
  }

  /**
     @brief Compute the matrix-vector product y = A * x
     @param[in] A Input matrix
     @param[in] x Input vector
     @return The vector A * x
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
    ColorSpinor<Float,Nc,Ns> operator*(const Matrix<complex<Float>,Nc> &A, const ColorSpinor<Float,Nc,Ns> &x) {

    ColorSpinor<Float,Nc,Ns> y;

#pragma unroll
    for (int i=0; i<Nc; i++) {
#pragma unroll
      for (int s=0; s<Ns; s++) {
	y.data[s*Nc + i].x  = A(i,0).real() * x.data[s*Nc + 0].real();
	y.data[s*Nc + i].x -= A(i,0).imag() * x.data[s*Nc + 0].imag();
	y.data[s*Nc + i].y  = A(i,0).real() * x.data[s*Nc + 0].imag();
	y.data[s*Nc + i].y += A(i,0).imag() * x.data[s*Nc + 0].real();
      }
#pragma unroll
      for (int j=1; j<Nc; j++) {
#pragma unroll
	for (int s=0; s<Ns; s++) {
	  y.data[s*Nc + i].x += A(i,j).real() * x.data[s*Nc + j].real();
	  y.data[s*Nc + i].x -= A(i,j).imag() * x.data[s*Nc + j].imag();
	  y.data[s*Nc + i].y += A(i,j).real() * x.data[s*Nc + j].imag();
	  y.data[s*Nc + i].y += A(i,j).imag() * x.data[s*Nc + j].real();
	}
      }
    }

    return y;
  }

  /**
     @brief Compute the matrix-vector product y = A * x
     @param[in] A Input Hermitian matrix with dimensions NcxNs x NcxNs
     @param[in] x Input vector
     @return The vector A * x
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
    ColorSpinor<Float,Nc,Ns> operator*(const HMatrix<Float,Nc*Ns> &A, const ColorSpinor<Float,Nc,Ns> &x) {

    ColorSpinor<Float,Nc,Ns> y;
    constexpr int N = Ns * Nc;

#pragma unroll
    for (int i=0; i<N; i++) {
      if (i==0) {
	y.data[i].x  = A(i,0).real() * x.data[0].real();
	y.data[i].y  = A(i,0).real() * x.data[0].imag();
      } else {
	y.data[i].x  = A(i,0).real() * x.data[0].real();
	y.data[i].x -= A(i,0).imag() * x.data[0].imag();
	y.data[i].y  = A(i,0).real() * x.data[0].imag();
	y.data[i].y += A(i,0).imag() * x.data[0].real();
      }
#pragma unroll
      for (int j=1; j<N; j++) {
	if (i==j) {
	  y.data[i].x += A(i,j).real() * x.data[j].real();
	  y.data[i].y += A(i,j).real() * x.data[j].imag();
	} else {
	  y.data[i].x += A(i,j).real() * x.data[j].real();
	  y.data[i].x -= A(i,j).imag() * x.data[j].imag();
	  y.data[i].y += A(i,j).real() * x.data[j].imag();
	  y.data[i].y += A(i,j).imag() * x.data[j].real();
	}
      }
    }

    return y;
  }

} // namespace quda
