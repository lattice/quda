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

extern void *memset(void *s, int c, size_t n);

#include <dslash_util.h>

int contraction_reference(void *spinorX, void *spinorY, void *result, QudaContractGamma cGamma, QudaPrecision cpu_prec, int X[]) {

  int faults = 0;
  
  if (cpu_prec == QUDA_DOUBLE_PRECISION) {    
    
    double re=0.0, im=0.0;    
    for(int i=0; i<V; i++) {
      for(int s1=0; s1<4; s1++) {
	for(int s2=0; s2<4; s2++) {
	  
	  re = im = 0.0;
	  for(int c=0; c<3; c++) {
	    re += (((double*)spinorX)[24*i + 6*s1 + 2*c + 0]*((double*)spinorY)[24*i + 6*s2 + 2*c + 0] +
		   ((double*)spinorX)[24*i + 6*s1 + 2*c + 1]*((double*)spinorY)[24*i + 6*s2 + 2*c + 1]);
	    
	    im += (((double*)spinorX)[24*i + 6*s1 + 2*c + 0]*((double*)spinorY)[24*i + 6*s2 + 2*c + 1] -
		   ((double*)spinorX)[24*i + 6*s1 + 2*c + 1]*((double*)spinorY)[24*i + 6*s2 + 2*c + 0]);
	  }

	  if ( abs( ((double*)result)[2*(i*16 + 4*s1 + s2) + 0] - re) > DOUBLE_TOL) faults++;
	  if ( abs( ((double*)result)[2*(i*16 + 4*s1 + s2) + 1] - im) > DOUBLE_TOL) faults++;
	  
	  // for(int j = 0; j<V; j++) {
	  //   if( abs(re - ((double*)result)[2*(j*16 + 4*s1 + s2)]) < 1e-9) {
	  //     printfQuda("%d (%d,%d,%d,%d) mapped to %d (%d,%d,%d,%d)\n",
	  // 		 i, i%X[0], (i%(X[0]*X[1]))/X[0], (i%(X[0]*X[1]*X[0]))/(X[0]*X[1]), i/(X[0]*X[1]*X[2]),
	  // 		 j, j%X[0], (j%(X[0]*X[1]))/X[0], (j%(X[0]*X[1]*X[0]))/(X[0]*X[1]), j/(X[0]*X[1]*X[2]));
	  //     continue;
	  //   }
	  // }
	}
      }
    }
  }
  else {

    float re=0.0, im=0.0;
    for(int i=0; i<V; i++) {
      for(int s1=0; s1<4; s1++) {
	for(int s2=0; s2<4; s2++) {

	  re = im = 0.0;
	  for(int c=0; c<3; c++) {
	    re += (((float*)spinorX)[24*i + 6*s1 + 2*c + 0]*((float*)spinorY)[24*i + 6*s2 + 2*c + 0] +
		   ((float*)spinorX)[24*i + 6*s1 + 2*c + 1]*((float*)spinorY)[24*i + 6*s2 + 2*c + 1]);
	    
	    im += (((float*)spinorX)[24*i + 6*s1 + 2*c + 0]*((float*)spinorY)[24*i + 6*s2 + 2*c + 1] -
		   ((float*)spinorX)[24*i + 6*s1 + 2*c + 1]*((float*)spinorY)[24*i + 6*s2 + 2*c + 0]);
	  }

	  if ( abs( ((float*)result)[2*(i*16 + 4*s1 + s2) + 0] - re) > SINGLE_TOL) faults++;
	  if ( abs( ((float*)result)[2*(i*16 + 4*s1 + s2) + 1] - im) > SINGLE_TOL) faults++;
	  
	  // printfQuda("%d %d %d (%+.16e,%+.16e) (%+.16e,%+.16e) %+.16f\n", i, s2, s2,
	  // 	     ((float*)result)[2*(i*16 + 4*s1 + s2)  ],
	  // 	     ((float*)result)[2*(i*16 + 4*s1 + s2)+1],
	  // 	     re, im,
	  // 	     abs(pow((((float*)result)[2*(i*16 + 4*s1 + s2)] - re),2) - pow((((float*)result)[2*(i*16 + 4*s1 + s2)+1] - im),2)));
	  
	}
      }
    }
  }
  return faults;
}
