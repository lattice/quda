#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>

#include <gauge_quda.h>
#include <spinor_quda.h>

#define MAX_SHORT 32767
#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

inline short floatToShort(float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline short doubleToShort(double a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}



// CPU only test of SU(3) accuracy, tests 8 and 12 component reconstruction
void SU3Test() {
  // construct input fields
  float *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = (float*)malloc(N*gaugeSiteSize*sizeof(float));

  QudaGaugeParam param;
  gauge_param = &param;
  param.anisotropy = 2.0;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
    
  printf("Randomizing fields...");
  construct_gauge_field(gauge, 1, QUDA_SINGLE_PRECISION);
  printf("done.\n");

  int fail_check = 12;
  int fail8[fail_check], fail12[fail_check];
  for (int f=0; f<fail_check; f++) {
    fail8[f] = 0;
    fail12[f] = 0;
  }

  int iter8[18], iter12[18];
  for (int i=0; i<18; i++) {
    iter8[i] = 0;
    iter12[i] = 0;
  }

  for (int eo=0; eo<2; eo++) {
    for (int i=0; i<Nh; i++) {
      int ga_idx = (eo*Nh+i);
      for (int d=0; d<4; d++) {
	float gauge8[18], gauge12[18];
	for (int j=0; j<18; j++) {
	  gauge8[j] = gauge[d][ga_idx*18+j];
	  gauge12[j] = gauge[d][ga_idx*18+j];
	}
	
	su3_construct(gauge8, QUDA_RECONSTRUCT_8, QUDA_SINGLE_PRECISION);
	su3_reconstruct(gauge8, d, i, QUDA_RECONSTRUCT_8, QUDA_SINGLE_PRECISION);
	
	su3_construct(gauge12, QUDA_RECONSTRUCT_12, QUDA_SINGLE_PRECISION);
	su3_reconstruct(gauge12, d, i, QUDA_RECONSTRUCT_12, QUDA_SINGLE_PRECISION);
	
	if (fabs(gauge8[8] - gauge[d][ga_idx*18+8]) > 1e-1) {
	  printGaugeElement(gauge[d]+ga_idx*18, 0, QUDA_SINGLE_PRECISION);printf("\n");
	  printGaugeElement(gauge8, 0, QUDA_SINGLE_PRECISION);printf("\n");
	  printGaugeElement(gauge12, 0, QUDA_SINGLE_PRECISION);
	  exit(0);
	}
	
	for (int j=0; j<18; j++) {
	  float diff8 = fabs(gauge8[j] - gauge[d][ga_idx*18+j]);
	  float diff12 = fabs(gauge12[j] - gauge[d][ga_idx*18+j]);
	  for (int f=0; f<fail_check; f++) {
	    if (diff8 > pow(10,-(f+1))) fail8[f]++;
	    if (diff12 > pow(10,-(f+1))) fail12[f]++;
	  }
	  if (diff8 > 1e-3) {
	    iter8[j]++;
	    //printf("%d %e %e\n", j, gauge[d][ga_idx*18+j], gauge8[j]);
	    //exit(0);
	  }
	  if (diff12 > 1e-3) {
	    iter12[j]++;
	  }

	}
      }
    }
  }

  for (int i=0; i<18; i++) printf("%d 12 fails = %d, 8 fails = %d\n", i, iter12[i], iter8[i]);

  for (int f=0; f<fail_check; f++) {
    printf("%e Failures: 12 component = %d / %d  = %e, 8 component = %d / %d = %e\n", 
	   pow(10,-(f+1)), fail12[f], N*4*18, fail12[f] / (float)(4*N*18),
	   fail8[f], N*4*18, fail8[f] / (float)(4*N*18));
  }

  // release memory
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);
}

int main(int argc, char **argv) {
  SU3Test();
}
