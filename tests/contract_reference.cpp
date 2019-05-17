#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DOUBLE_TOL 1e-9
#define SINGLE_TOL 2e-5

#include <test_util.h>
#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <contract_reference.h>
#include "misc.h"
#include <blas_quda.h>

#include <blas_reference.h>

using namespace quda;
using namespace std;

extern void *memset(void *s, int c, size_t n);

#include <dslash_util.h>

//Definition of the Degrand Rossi gamma basis
//The slowest index is the gamma matrix,
//the next index is the row (top to bottom)
//the last index is the 8 integers defining
//the entries of the row.

#if 0
static const int degrand_rossi[5][4][8] = {
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
void multiplySpinorByDegrandRossi(int gammaMat, Float *spinorIn) {
  
  //Put data in complex form
  complex<Float> cSpinorIn[12];
  complex<Float> cResult[12];  
  for (int i=0; i<12; i++) {
    cSpinorIn[i].real(spinorIn[2*i  ]);
    cSpinorIn[i].imag(spinorIn[2*i+1]);
    cResult[i] = 0.0;
  }
  
  //Get gamma matrix data
  complex<int> gamma[4][4];
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
    spinorIn[2*i  ] = cResult.real();
    spinorIn[2*i+1] = cResult.imag();
  }
}

template <typename Float>
void insertGammaMatrix(QudaContractGamma cGamma, Float *spinorIn) {

  switch(cGamma) {
    for(int i=0; i<V; i++) {
    case 0: //do something
      break;
    case 1: //do something else
      break;
    default:
      break;
    }
  }
}

template <typename Float>
int contraction_reference(Float *spinorX, Float *spinorY, Float *result, QudaContractGamma cGamma, int X[]) {
  
  int faults = 0;
  Float re=0.0, im=0.0;

  complex<Float> *cSpinorX = (complex<Float>*)spinorX;
  complex<Float> *cSpinorY = (complex<Float>*)spinorY;
  
  //Gamma matrix insertion
  
  //for(int i=0; i<V; i++) {
    
  
  for(int i=0; i<V; i++) {
    for(int s1=0; s1<4; s1++) {
      for(int s2=0; s2<4; s2++) {
	
	re = im = 0.0;
	for(int c=0; c<3; c++) {
	  re += (spinorX[24*i + 6*s1 + 2*c + 0] * spinorY[24*i + 6*s2 + 2*c + 0] +
		 spinorX[24*i + 6*s1 + 2*c + 1] * spinorY[24*i + 6*s2 + 2*c + 1]);
	  
	  im += (spinorX[24*i + 6*s1 + 2*c + 0] * spinorY[24*i + 6*s2 + 2*c + 1] -
		 spinorX[24*i + 6*s1 + 2*c + 1] * spinorY[24*i + 6*s2 + 2*c + 0]);
	}

	if ( abs(result[2*(i*16 + 4*s1 + s2) + 0] - re) > DOUBLE_TOL) faults++;
	if ( abs(result[2*(i*16 + 4*s1 + s2) + 1] - im) > DOUBLE_TOL) faults++;
	
	// for(int j = 0; j<V; j++) {
	//   if( abs(re - result)2*(j*16 + 4*s1 + s2)]) < DOUBLE_TOL) {
	//     printfQuda("%d (%d,%d,%d,%d) mapped to %d (%d,%d,%d,%d)\n",
	// 		 i, i%X[0], (i%(X[0]*X[1]))/X[0], (i%(X[0]*X[1]*X[0]))/(X[0]*X[1]), i/(X[0]*X[1]*X[2]),
	// 		 j, j%X[0], (j%(X[0]*X[1]))/X[0], (j%(X[0]*X[1]*X[0]))/(X[0]*X[1]), j/(X[0]*X[1]*X[2]));
	//     continue;
	//   }
	// }
      }
    }
  }
  return faults;
}
#endif
