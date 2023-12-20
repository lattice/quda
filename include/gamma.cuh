#pragma once

#include <complex_quda.h>

namespace quda {

  //A simple Euclidean gamma matrix class for use with the Wilson projectors.
  template <typename ValueType, QudaGammaBasis basis, int dir>
  class Gamma {
  private:
    static constexpr int ndim = 4;

  protected:

    //Which gamma matrix (dir = 0,4)
    //dir = 0: gamma^1, dir = 1: gamma^2, dir = 2: gamma^3, dir = 3: gamma^4, dir =4: gamma^5
    //int dir;

    //The basis to be used.
    //QUDA_DEGRAND_ROSSI_GAMMA_BASIS is the chiral basis
    //QUDA_UKQCD_GAMMA_BASIS is the non-relativistic basis.
    //QudaGammaBasis basis;

  public:
    __device__ __host__ inline Gamma() { }
    Gamma(const Gamma &g) = default;

    __device__ __host__ inline int getcol(int row) const {
      if (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	switch(dir) {
	case 0:
	case 1:
	  switch(row) {
	  case 0: return 3;
	  case 1: return 2;
	  case 2: return 1;
	  case 3: return 0;
	  }
	  break;
	case 2:
	case 3:
	  switch(row) {
	  case 0: return 2;
	  case 1: return 3;
	  case 2: return 0;
	  case 3: return 1;
	  }
	  break;
	case 4:
	  switch(row) {
	  case 0: return 0;
	  case 1: return 1;
	  case 2: return 2;
	  case 3: return 3;
	  }
	  break;
	}
      } else {
	switch(dir) {
	case 0:
	case 1:
	  switch(row) {
	  case 0: return 3;
	  case 1: return 2;
	  case 2: return 1;
	  case 3: return 0;
	  }
	  break;
	case 2:
	  switch(row) {
	  case 0: return 2;
	  case 1: return 3;
	  case 2: return 0;
	  case 3: return 1;
	  }
	  break;
	case 3:
	  switch(row) {
	  case 0: return 0;
	  case 1: return 1;
	  case 2: return 2;
	  case 3: return 3;
	  }
	  break;
	case 4:
	  switch(row) {
	  case 0: return 2;
	  case 1: return 3;
	  case 2: return 0;
	  case 3: return 1;
	  }
	  break;
	}
      }
      return 0;
    }

    __device__ __host__ inline complex<ValueType> getelem(int row) const {
      complex<ValueType> I(0,1);
      if (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	switch(dir) {
	case 0:
	  switch(row) {
	  case 0:
	  case 1:
	    return I;
	  case 2:
	  case 3:
	    return -I;
	  }
	  break;
	case 1:
	  switch(row) {
	  case 0:
	  case 3:
	    return -1;
	  case 1:
	  case 2:
	    return 1;
	  }
	  break;
	case 2:
	  switch(row) {
	  case 0:
	  case 3:
	    return I;
	  case 1:
	  case 2:
	    return -I;
	  }
	  break;
	case 3:
	  switch(row) {
	  case 0:
	  case 1:
	  case 2:
	  case 3:
	    return 1;
	  }
	  break;
	case 4:
	  switch(row) {
	  case 0:
	  case 1:
	    return -1;
	  case 2:
	  case 3:
	    return 1;
	  }
	  break;
	}
      } else if (basis == QUDA_UKQCD_GAMMA_BASIS) {
	switch(dir) {
	case 0:
	  switch(row) {
	  case 0:
	  case 1:
	    return I;
	  case 2:
	  case 3:
	    return -I;
	  }
	  break;
	case 1:
	  switch(row) {
	  case 0:
	  case 3:
	    return 1;
	  case 1:
	  case 2:
	    return -1;
	  }
	  break;
	case 2:
	  switch(row) {
	  case 0:
	  case 3:
	    return I;
	  case 1:
	  case 2:
	    return -I;
	  }
	  break;
	case 3:
	  switch(row) {
	  case 0:
	  case 1:
	    return 1;
	  case 2:
	  case 3:
	    return -1;
	  }
	  break;
	case 4:
	  switch(row) {
	  case 0:
	  case 1:
	  case 2:
	  case 3:
	    return 1;
	  }
	  break;
	}
      }
      return 0;
    }

    //Returns the matrix element.
    __device__ __host__ inline complex<ValueType> getelem(int row, int col) const {
      return getcol(row) == col ? getelem(row) : 0;
    }

    //Like getelem, but one only needs to specify the row.
    //The column of the non-zero component is returned via the "col" reference
    __device__ __host__ inline complex<ValueType> getrowelem(int row, int &col) const {
      col = getcol(row);
      return getelem(row);
    }

    // Multiplies a given row of the gamma matrix to a complex number
    __device__ __host__ inline complex<ValueType> apply(int row, const complex<ValueType> &a) const {
      if (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	switch(dir) {
	case 0:
	  switch(row) {
	  case 0: case 1: return complex<ValueType>(-a.imag(), a.real()); //  I
	  case 2: case 3: return complex<ValueType>(a.imag(), -a.real()); // -I
	  }
	  break;
	case 1:
	  switch(row) {
	  case 0: case 3: return -a;
	  case 1: case 2: return a;
          }
          break;
	case 2:
	  switch(row) {
	  case 0: case 3: return complex<ValueType>(-a.imag(), a.real()); //  I
	  case 1: case 2: return complex<ValueType>(a.imag(), -a.real()); // -I
          }
          break;
	case 3:
	  switch(row) {
	  case 0: case 1: case 2: case 3: return a;
          }
          break;
	case 4:
	  switch(row) {
	  case 0: case 1: return complex<ValueType>(-a.real(), -a.imag());
	  case 2: case 3: return a;
	  }
	  break;
	}
      } else if (basis == QUDA_UKQCD_GAMMA_BASIS) {
	switch(dir) {
	case 0:
	  switch(row) {
	  case 0: case 1: return complex<ValueType>(-a.imag(), a.real()); //  I
	  case 2: case 3:return complex<ValueType>(a.imag(), -a.real()); // -I
	  }
	  break;
	case 1:
	  switch(row) {
	  case 0: case 3: return a;
	  case 1: case 2: return -a;
	  }
	  break;
	case 2:
	  switch(row) {
	  case 0: case 3: return complex<ValueType>(-a.imag(), a.real()); //  I
	  case 1: case 2: return complex<ValueType>(a.imag(), -a.real()); // -I
	  }
	  break;
	case 3:
	  switch(row) {
	  case 0: case 1: return a;
	  case 2: case 3: return -a;
	  }
	  break;
	case 4:
	  switch(row) {
	  case 0: case 1: case 2: case 3: return a;
	  }
	  break;
	}
      }
      return a;
    }

    //Returns the type of Gamma matrix
    inline constexpr int Dir() const { return dir;  }
  };
  
  // list of specialized structures used in the contraction kernels: 
  static constexpr int nspin = 4;

  constexpr array<array<int, nspin>, nspin*nspin> get_dr_gm_i() {      
      return {{   
        // VECTORS
        // G_idx = 1: \gamma_1
        {3, 2, 1, 0},	 

        // G_idx = 2: \gamma_2
        {3, 2, 1, 0},

        // G_idx = 3: \gamma_3
        {2, 3, 0, 1},

        // G_idx = 4: \gamma_4
        {2, 3, 0, 1},

        // PSEUDO-VECTORS
        // G_idx = 6: \gamma_5\gamma_1
        {3, 2, 1, 0},

        // G_idx = 7: \gamma_5\gamma_2
        {3, 2, 1, 0},

        // G_idx = 8: \gamma_5\gamma_3
        {2, 3, 0, 1},

        // G_idx = 9: \gamma_5\gamma_4
        {2, 3, 0, 1}, 

        // SCALAR
        // G_idx = 0: I
        {0, 1, 2, 3},

        // PSEUDO-SCALAR
        // G_idx = 5: \gamma_5
        {0, 1, 2, 3},

        // TENSORS
        // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
        {0, 1, 2, 3},

        // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]. this matrix was corrected
        {1, 0, 3, 2},

        // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
        {1, 0, 3, 2},

        // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
        {1, 0, 3, 2},

        // G_idx = 14: (i/2) * [\gamma_2, \gamma_4]
        {1, 0, 3, 2},

        // G_idx = 15: (i/2) * [\gamma_3, \gamma_4]. this matrix was corrected
        {0, 1, 2, 3}
      }};
  }

  template<typename T>
  constexpr array<array<complex<T>, nspin>, nspin*nspin> get_dr_g5gm_z() {  

      constexpr complex<T> p_i = complex<T>( 0., +1.);
      constexpr complex<T> m_i = complex<T>( 0., -1.);
      constexpr complex<T> p_1 = complex<T>(+1.,  0.);
      constexpr complex<T> m_1 = complex<T>(-1.,  0.);      

      return {{
        // VECTORS
        // G_idx = 1: \gamma_1
        {p_i, p_i, p_i, p_i},

        // G_idx = 2: \gamma_2
        {m_1, p_1, m_1, p_1},

        // G_idx = 3: \gamma_3
        {p_i, m_i, p_i, m_i},

        // G_idx = 4: \gamma_4
        {p_1, p_1, m_1, m_1},        

        // PSEUDO-VECTORS
        // G_idx = 6: \gamma_5\gamma_1
        {p_i, p_i, m_i, m_i},        

        // G_idx = 7: \gamma_5\gamma_2
        {m_1, p_1, p_1, m_1},        

        // G_idx = 8: \gamma_5\gamma_3
        {p_i, m_i, m_i, p_i},        

        // G_idx = 9: \gamma_5\gamma_4
        {p_1, p_1, p_1, p_1},      

        // SCALAR
        // G_idx = 0: I
        {p_1, p_1, m_1, m_1},        

        // PSEUDO-SCALAR
        // G_idx = 5: \gamma_5
        {p_1, p_1, p_1, p_1},

        // TENSORS
        // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
        {p_1, m_1, m_1, p_1},        

        // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]. this matrix was corrected
        {m_i, p_i, p_i, m_i},                

        // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
        {m_1, m_1, m_1, m_1},                

        // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
        {p_1, p_1, m_1, m_1},                

        // G_idx = 14: (i/2) * [\gamma_2, \gamma_4]
        {m_i, p_i, m_i, p_i},                

        // G_idx = 15: (i/2) * [\gamma_3, \gamma_4]. this matrix was corrected
        {m_1, p_1, m_1, p_1}
      }};
  }
} // namespace quda
