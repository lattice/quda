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

    //The column with the non-zero element for each row
    int coupling[4];
    //The value of the matrix element, for each row
    complex<ValueType> elem[4];

  public:

    Gamma() {
      complex<ValueType> I(0,1);
      if((dir==0) || (dir==1)) {
	coupling[0] = 3;
	coupling[1] = 2;
	coupling[2] = 1;
	coupling[3] = 0;
      } else if (dir == 2) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else if ((dir == 3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else if ((dir == 3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	coupling[0] = 0;
	coupling[1] = 1;
	coupling[2] = 2;
	coupling[3] = 3;
      } else if ((dir == 4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	coupling[0] = 0;
	coupling[1] = 1;
	coupling[2] = 2;
	coupling[3] = 3;
      } else if ((dir == 4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else {
	printf("Warning: Gamma matrix not defined for dir = %d and basis = %d\n", dir, basis);
	coupling[0] = 0;
	coupling[1] = 0;
	coupling[2] = 0;
	coupling[3] = 0;
      }


      if((dir==0)) {
	elem[0] = I;
	elem[1] = I;
	elem[2] = -I;
	elem[3] = -I;
      } else if((dir==1) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = -1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = -1;
      } else if((dir==1) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = -1;
	elem[2] = -1;
	elem[3] = 1;
      } else if((dir==2)) {
	elem[0] = I;
	elem[1] = -I;
	elem[2] = -I;
	elem[3] = I;
      } else if((dir==3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = 1;
      } else if((dir==3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = -1;
	elem[3] = -1;
      } else if((dir==4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = -1;
	elem[1] = -1;
	elem[2] = 1;
	elem[3] = 1;
      } else if((dir==4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = 1;
      } else {
	elem[0] = 0;
	elem[1] = 0;
	elem[2] = 0;
	elem[3] = 0;
      }
    }

    Gamma(const Gamma &g) {
      for(int i = 0; i < ndim+1; i++) {
	coupling[i] = g.coupling[i];
	elem[i] = g.elem[i];
      }
    }

    //Returns the matrix element.
    __device__ __host__ inline complex<ValueType> getelem(int row, int col) const {
      return coupling[row] == col ? elem[row] : 0;
    }

    //Like getelem, but one only needs to specify the row.
    //The column of the non-zero component is returned via the "col" reference
    __device__ __host__ inline complex<ValueType> getrowelem(int row, int &col) const {
      col = coupling[row];
      return elem[row];
    }

    //Returns the type of Gamma matrix
    inline constexpr int Dir() const { return dir;  }
  };

} // namespace quda
