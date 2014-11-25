/*

  This is just an experiment into using polar coordinates instead of
  Cartesian coordinates for storing complex numbers.  The supposition
  is that a fixed-point form for polar coordinates will lead to a much
  more efficient use of bits leading to higher precision with 8-bit
  and 16-bit precision than otherwise possible.

*/

#include <quda_internal.h>
#include <register_traits.h>

using namespace quda;

#define MAX_USHORT 65535.0f
inline void ucopy(float &a, const ushort &b) { a = (float)b/MAX_USHORT; }
inline void ucopy(ushort &a, const float &b) { a = (short)(b*MAX_USHORT); }


void old_load_half(float spinor[24], short *in, float *norm, int idx) {

  for (int i=0; i<24; i++) {
    copy(spinor[i], in[idx*24+i]);
    spinor[i] *= norm[idx];
  }

}

void old_save_half(float spinor[24], short *out, float *norm, int idx) {

  float max = 0.0;
  for (int i=0; i<24; i++) {
    float tmp = fabs(spinor[i]);
    if (tmp > max) max = tmp;
  }
  
  norm[idx] = max;
  for (int i=0; i<24; i++) copy(out[idx*24+i], spinor[i]/max);

}

void new_load_half(float spinor[24], short *in, float *norm, int idx) {

  for (int i=0; i<12; i++) {
    float mag, phase;
    ucopy(mag,((ushort*)in)[idx*24+i*2+0]);
    mag *= norm[idx];
    copy(phase,in[idx*24+i*2+1]);
    phase *= M_PI;
    spinor[2*i+0] = mag*cos(phase);
    spinor[2*i+1] = mag*sin(phase);
  }

}

void new_save_half(float spinor[24], short *out, float *norm, int idx) {

  float max = 0.0;
  for (int i=0; i<12; i++) {
    float tmp = sqrt(spinor[2*i+0]*spinor[2*i+0] + spinor[2*i+1]*spinor[2*i+1]);
    if (tmp > max) max = tmp;
  }
  
  norm[idx] = max;
  for (int i=0; i<12; i++) {
    float phase = atan2(spinor[2*i+1], spinor[2*i+0]) / M_PI;
    float mag = sqrt(spinor[2*i+0]*spinor[2*i+0] + spinor[2*i+1]*spinor[2*i+1]) / max;

    ucopy(((ushort*)out)[idx*24+i*2+0], mag);
    copy(out[idx*24+i*2+1], phase);
  }

}

void oldCopyToHalf(short *out, float *norm, float *in, int N) {
  for (int j=0; j<N; j++) {
    float spinor[24];
    for (int i=0; i<24; i++) spinor[i] = in[j*24+i];
    old_save_half(spinor, out, norm, j);
  }

}

void oldCopyToFloat(float *out, short *in, float *norm, int N) {
  for (int j=0; j<N; j++) {
    float spinor[24];
    old_load_half(spinor, in, norm, j);
    for (int i=0; i<24; i++) out[j*24+i] = spinor[i];
  }

}

void newCopyToHalf(short *out, float *norm, float *in, int N) {
  for (int j=0; j<N; j++) {
    float spinor[24];
    for (int i=0; i<24; i++) spinor[i] = in[j*24+i];
    new_save_half(spinor, out, norm, j);
  }

}

void newCopyToFloat(float *out, short *in, float *norm, int N) {
  for (int j=0; j<N; j++) {
    float spinor[24];
    new_load_half(spinor, in, norm, j);
    for (int i=0; i<24; i++) out[j*24+i] = spinor[i];
  }

}

void insertNoise(float *field, int N, float power) {
  for (int j=0; j<N; j++) {
    for (int i=0; i<24; i++) {
      field[j*24+i] = 1000*pow(comm_drand(), power);
    }
  }
}

double l2(float *a, float *b, int N) {

  double rtn = 0.0;
  for (int j=0; j<N; j++) {
    double dif = 0;
    double nrm = 0.0;
    for (int i=0; i<24; i++) {
      dif += a[j*24+i]*a[j*24+i] - b[j*24+i]*b[j*24+i];
      nrm += a[j*24+i]*a[j*24+i];
    }
    rtn += sqrt(fabs(dif)/nrm);
  }
  return rtn/N;
}

int main() {
  const int N = 1000;

  float *ref = (float*)safe_malloc(24*N*sizeof(float));
  short *old_half = (short*)safe_malloc(24*N*sizeof(short));
  float *old_norm = (float*)safe_malloc(N*sizeof(float));
  float *old_recon = (float*)safe_malloc(24*N*sizeof(float));
  short *new_half = (short*)safe_malloc(24*N*sizeof(short));
  float *new_norm = (float*)safe_malloc(N*sizeof(float));
  float *new_recon = (float*)safe_malloc(24*N*sizeof(float));

  for (float power=0.0; power<2.0; power+=0.1) {
    insertNoise(ref, N, power);

    newCopyToHalf(new_half,new_norm,ref,N);
    newCopyToFloat(new_recon,new_half,new_norm,N);
    
    oldCopyToHalf(old_half,old_norm,ref,N);
    oldCopyToFloat(old_recon,old_half,old_norm,N);
    
    printf("pow=%e, L2 spinor deviation: old = %e, new = %e, ratio = %e\n", 
	   power, l2(ref,old_recon,N), l2(ref,new_recon,N),
	   l2(ref,old_recon,N) / l2(ref,new_recon,N));

    if (N==1) {
      for (int j=0; j<N; j++) {
	for (int i=0; i<12; i++) {
	  printf("power=%4.2f i=%d ref=(%e,%e) old=(%e,%e), new=(%e,%e)\n", 
		 power, i, ref[j*24+i*2+0], ref[j*24+i*2+1], 
		 old_recon[j*24+i*2+0], old_recon[j*24+i*2+1],
		 new_recon[j*24+i*2+0], new_recon[j*24+i*2+1]);	
	}
      }
    }
  }

  host_free(old_norm);
  host_free(old_half);
  host_free(old_recon);
  host_free(new_norm);
  host_free(new_half);
  host_free(new_recon);
  host_free(ref);
}
