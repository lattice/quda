#pragma once 

#include <blas_reference.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

#define DOUBLE_TOL 1e-9
#define SINGLE_TOL 2e-5

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;
using namespace std;

//Definition of the Degrand Rossi gamma basis
//The slowest index is the gamma matrix,
//the next index is the row (top to bottom)
//the last index is the 8 integers defining
//the entries of the row.

static const int degrand_rossi[6][4][8] = {
  {
    //I
    {1,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0},
    {0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,1,0}},
  {
    //gamma 1
    {0,0,0,0,0,0,0,1},
    {0,0,0,0,0,1,0,0},
    {0,0,0,-1,0,0,0,0},
    {0,-1,0,0,0,0,0,0}},
  {
    //gamma 2
    {0,0,0,0,0,0,-1,0},
    {0,0,0,0,1,0,0,0},
    {0,0,1,0,0,0,0,0},
    {-1,0,0,0,0,0,0,0}},
  {
    //gamma 3
    {0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,-1},
    {0,-1,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0}},
  {
    //gamma 4
    {0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,1,0},
    {1,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0}},
  {
    //gamma 5
    {1,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0},
    {0,0,0,0,-1,0,0,0},
    {0,0,0,0,0,0,-1,0}},
};

template <typename Float>
void multiplySpinorByDegrandRossi(int gammaMat, Float *spinorAux, Float *spinorIn) {
  
  //Put data in complex form
  complex<Float> cSpinorIn[12];
  complex<Float> cResult[12];

  for (int site=0; site<V; site++) {
    
    for (int i=0; i<12; i++) {
      cSpinorIn[i].real(spinorIn[24*site + 2*i  ]);
      cSpinorIn[i].imag(spinorIn[24*site + 2*i+1]);
      cResult[i] = 0.0;
    }
    
    //Get gamma matrix data
    complex<Float> gamma[4][4];
    for (int row=0; row<4; row++) {
      for (int col=0; col<4; col++) {
	gamma[row][col].real(degrand_rossi[gammaMat][row][2*col]);
	gamma[row][col].imag(degrand_rossi[gammaMat][row][2*col+1]);
      }
    }
    
    //Loop over result spin
    for (int r=0; r<4; r++) {
      //Loop over spinorIn spin
      for (int s=0; s<4; s++) {
	//Loop over colour
	for (int c=0; c<3; c++) {
	  cResult[3*r + c] += gamma[r][s] * cSpinorIn[3*s + c];	
	}
      }
    }
    
    //Put data back in result
    for (int i=0; i<12; i++) {
      spinorAux[24*site + 2*i  ] = cResult[i].real();
      spinorAux[24*site + 2*i+1] = cResult[i].imag();
    }
  }
}

template <typename Float>
void insertGammaMatrix(QudaContractGamma cGamma, Float *spinorAux, Float *spinorIn) {
  
  switch(cGamma) {
  case QUDA_CONTRACT_GAMMA_I:  multiplySpinorByDegrandRossi(0, spinorAux, spinorIn);
    break;
  case QUDA_CONTRACT_GAMMA_G1: multiplySpinorByDegrandRossi(1, spinorAux, spinorIn);
    break;
  case QUDA_CONTRACT_GAMMA_G2: multiplySpinorByDegrandRossi(2, spinorAux, spinorIn);
    break;
  case QUDA_CONTRACT_GAMMA_G3: multiplySpinorByDegrandRossi(3, spinorAux, spinorIn);
    break;
  case QUDA_CONTRACT_GAMMA_G4: multiplySpinorByDegrandRossi(4, spinorAux, spinorIn);
    break;
  case QUDA_CONTRACT_GAMMA_G5: multiplySpinorByDegrandRossi(5, spinorAux, spinorIn);
    break;
  default:
    break;
  }
}



template <typename Float>
int contraction_reference(Float *spinorX, Float *spinorY, Float* spinorAux,
			  Float *result, QudaContractGamma cGamma, int X[]){
  
  int faults = 0;
  Float re=0.0, im=0.0;
  Float tol;
  if (sizeof(Float) == sizeof(double)) tol = DOUBLE_TOL;
  else tol = SINGLE_TOL;

  //Apply gamma insertion on spinorX, place in spinorAux
  insertGammaMatrix(cGamma, spinorAux, spinorX);
  
  for(int i=0; i<V; i++) {
    for(int s1=0; s1<4; s1++) {
      for(int s2=0; s2<4; s2++) {

	re = im = 0.0;
	for(int c=0; c<3; c++) {
	  re += (((Float*)spinorAux)[24*i + 6*s1 + 2*c + 0]*((Float*)spinorY)[24*i + 6*s2 + 2*c + 0] +
		 ((Float*)spinorAux)[24*i + 6*s1 + 2*c + 1]*((Float*)spinorY)[24*i + 6*s2 + 2*c + 1]);
	  
	  im += (((Float*)spinorAux)[24*i + 6*s1 + 2*c + 0]*((Float*)spinorY)[24*i + 6*s2 + 2*c + 1] -
		 ((Float*)spinorAux)[24*i + 6*s1 + 2*c + 1]*((Float*)spinorY)[24*i + 6*s2 + 2*c + 0]);
	}
	
	if ( abs( ((Float*)result)[2*(i*16 + 4*s1 + s2) + 0] - re) > tol) faults++;
	if ( abs( ((Float*)result)[2*(i*16 + 4*s1 + s2) + 1] - im) > tol) faults++;
	

#if 0
	for(int j = 0; j<V; j++) {
	  if( abs(re - result[2*(j*16 + 4*s1 + s2)]) < tol) {
	    printfQuda("%d (%d,%d,%d,%d) mapped to %d (%d,%d,%d,%d)\n",
		       i, i%X[0], (i%(X[0]*X[1]))/X[0], (i%(X[0]*X[1]*X[0]))/(X[0]*X[1]), i/(X[0]*X[1]*X[2]),
		       j, j%X[0], (j%(X[0]*X[1]))/X[0], (j%(X[0]*X[1]*X[0]))/(X[0]*X[1]), j/(X[0]*X[1]*X[2]));
	    continue;
	  }
	}
#endif
      }
    }
  }
  
  return faults;
};
