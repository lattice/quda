#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <quda.h>
#include <util_quda.h>

#include <complex>

using namespace std;
typedef complex<float> Complex;
typedef complex<double> Complex16;

struct timeval startTime;

void stopwatchStart() {
  gettimeofday(&startTime, NULL);
}

double stopwatchReadSeconds() {
  struct timeval endTime;
  gettimeofday( &endTime, 0);
    
  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}

void printVector(float *v) {
  printf("{(%f %f) (%f %f) (%f %f)}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

void printSpinor(float *spinor) {
  for (int s = 0; s < 4; s++) {
    printVector(&spinor[s*(3*2)]);
  }
}

// X indexes the full lattice
void printSpinorElement(float *spinor, int X) {
  if (getOddBit(X) == 0)
    printSpinor(&spinor[(X/2)*(4*3*2)]);
  else
    printSpinor(&spinor[(X/2)*(4*3*2)+Nh*spinorSiteSize]);
}


void printGauge(float *gauge) {
  for (int m = 0; m < 3; m++) {
    printVector(&gauge[m*(3*2)]);
  }
}

// X indexes the full lattice
void printGaugeElement(float *gauge, int X) {
  if (getOddBit(X) == 0)
    printGauge(&gauge[(X/2)*gaugeSiteSize]);
  else
    printGauge(&gauge[(X/2+Nh)*gaugeSiteSize]);
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int X) {
  int x4 = X/(L3*L2*L1);
  int x3 = (X/(L2*L1)) % L3;
  int x2 = (X/L1) % L2;
  int x1 = X % L1;
  return (x4+x3+x2+x1) % 2;
}

// a+=b
void complexAddTo(float *a, float *b) {
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
void complexProduct(float *a, float *b, float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = b[0]*c[1] + b[1]*c[0];
}

// a = conj(b)*conj(c)
void complexConjugateProduct(float *a, float *b, float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
}

// a = conj(b)*c
void complexDotProduct(float *a, float *b, float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += b*c
void accumulateComplexProduct(float *a, float *b, float *c, float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a += conj(b)*c)
void accumulateComplexDotProduct(float *a, float *b, float *c) {
    a[0] += b[0]*c[0] + b[1]*c[1];
    a[1] += b[0]*c[1] - b[1]*c[0];
}

void accumulateConjugateProduct(float *a, float *b, float *c, float sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

void complexNorm(float *a, float *b) {
    a[0] = b[0]*b[0] + b[1]*b[1];
}

void su3_construct_12(float *mat) {
  Complex *w = (Complex*)&mat[2*(3*2)];
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
}

// given first two rows (u,v) of SU(3) matrix mat, reconstruct the third row
// as the cross product of the conjugate vectors: w = u* x v*
// 
// 48 flops
void su3_reconstruct_12(float *mat, int dir, int ga_idx) {
  float *u = &mat[0*(3*2)];
  float *v = &mat[1*(3*2)];
  float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  float u0 = (dir < 3 ? gauge_param->anisotropy :
	      (ga_idx >= (L4-1)*L1h*L2*L3 ? gauge_param->t_boundary : 1));
  w[0]*=u0; w[1]*=u0; w[2]*=u0; w[3]*=u0; w[4]*=u0; w[5]*=u0;
}

// SU(3) packing method taken from Bunk and Sommer
void su3_construct_8_bunk(float *mat, int dir) {
  // first calculate f(mat[0])
  // First row normality constraint
  float r0 = 1.0 / ((dir < 3) ? gauge_param->anisotropy : 1.0);
  float f0 = 1.0 / sqrt(2.0*r0*(r0+mat[0]));

  // First row and first column constraint
  float r1_2 = 0.0;
  for (int i=2; i<6; i++) r1_2 += mat[i]*mat[i];
  float r1 = sqrt(r1_2);
  float f1 = 1.0 /sqrt(2.0*r1*(r1+mat[13]));
  
  mat[0] = mat[12] * f1;
  for (int i=1; i<6; i++) mat[i] *= f0;
  for (int i=6; i<8; i++) mat[i] *= f1;
  for (int i=8; i<18; i++) mat[i] = 0.0;
}

// 110 (multiply/add flops) + 1 divide + 3 sqrt
void su3_reconstruct_8_bunk(float *mat, int dir, int ga_idx) {
  // First reconstruct first row
  float y2_sum = mat[5]*mat[5];
  y2_sum += mat[4]*mat[4];
  y2_sum += mat[3]*mat[3];
  y2_sum += mat[2]*mat[2];

  float r_column = sqrt(y2_sum); // reuse for column construct and SU(2) normalization
  y2_sum += mat[1]*mat[1]; // y2_sum now complete

  float u0 = (dir < 3 ? gauge_param->anisotropy :
	      (ga_idx >= (L4-1)*L1h*L2*L3 ? gauge_param->t_boundary : 1));
  float r0 = 1.0 / fabs(u0);
  mat[12] = mat[0]; // copy Re(U20) back to where it belongs
  mat[0] = r0*(1.0 - 2*y2_sum); // Re(U00) now restored
  float y_scale = 2.0*r0*sqrt(1.0 - y2_sum);
  mat[1] *= y_scale; 
  mat[2] *= y_scale; 
  mat[3] *= y_scale; 
  mat[4] *= y_scale; 
  mat[5] *= y_scale; 
  // First row now restored

  // Now remaining elements of first column
  r_column *= y_scale; // r = sqrt(|U10|^2 + |U20|^2) = sqrt(|U01|^2 + |U02|^2) = N
  y2_sum = mat[12]*mat[12];
  y2_sum += mat[6]*mat[6];
  y2_sum += mat[7]*mat[7];

  mat[13] = r_column * (1 - 2*y2_sum);
  y_scale = 2*r_column*sqrt(1 - y2_sum);
  mat[12] *= y_scale;
  mat[6] *= y_scale;
  mat[7] *= y_scale;
  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  //float u0 = (ga_idx >= (L4-1)*L1h*L2*L3) ? gauge_param->t_boundary : 1;
  float r_inv2 = 1.0/(u0*r_column*r_column);

  // U11
  float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}

// Stabilized Bunk and Sommer
void su3_construct_8(float *mat) {
  mat[0] = atan2(mat[1], mat[0]);
  mat[1] = atan2(mat[13], mat[12]);
  for (int i=8; i<18; i++) mat[i] = 0.0;
}

void su3_reconstruct_8(float *mat, int dir, int ga_idx) {
  // First reconstruct first row
  float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];
  float u0 = (dir < 3 ? gauge_param->anisotropy :
	      (ga_idx >= (L4-1)*L1h*L2*L3 ? gauge_param->t_boundary : 1));
  float U00_mag = sqrt(1.f/(u0*u0) - row_sum);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  float U20_mag = sqrt(1.f/(u0*u0) - column_sum);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  float r_inv2 = 1.0/(u0*row_sum);

  // U11
  float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}


int compareFloats(float *a, float *b, int len, float epsilon) {
  for (int i = 0; i < len; i++) {
    float diff = fabs(a[i] - b[i]);
    if (diff > epsilon) return 0;
  }
  return 1;
}



// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
int fullLatticeIndex(int i, int oddBit) {
  int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

void applyGaugeFieldScaling(float **gauge) {
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < gaugeSiteSize*N; i++) {
      gauge[d][i] /= gauge_param->anisotropy;
    }
  }
    
  printf("%d Boundary %d\n", gaugeSiteSize, gauge_param->t_boundary);

  // Apply boundary conditions to temporal links
  if (gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T) {
    for (int j = L1h*L2*L3*(L4-1); j < Nh; j++) {
      for (int i = 0; i < gaugeSiteSize; i++) {
	gauge[3][j*gaugeSiteSize+i] *= -1.0;
	gauge[3][(Nh+j)*gaugeSiteSize+i] *= -1.0;
      }
    }
  }
    
  if (gauge_param->gauge_fix) {
    // set all gauge links (except for the first L1h*L2*L3) to the identity,
    // to simulate fixing to the temporal gauge.
    int dir = 3; // time direction only
    float *even = gauge[dir];
    float *odd  = gauge[dir]+Nh*gaugeSiteSize;
    for (int i = L1h*L2*L3; i < Nh; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
}

void constructUnitGaugeField(float **res) {
  float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Nh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Nh; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
    
  applyGaugeFieldScaling(res);
}

// normalize the vector a
void normalize(Complex *a, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) sum += norm(a[i]);
  for (int i=0; i<len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
void orthogonalize(Complex *a, Complex *b, int len) {
  Complex16 dot = 0.0;
  for (int i=0; i<len; i++) dot += conj(a[i])*b[i];
  for (int i=0; i<len; i++) b[i] -= (Complex)dot*a[i];
}

void constructGaugeField(float **res) {
  float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Nh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Nh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (float)RAND_MAX;                    
	}
      }
      normalize((Complex*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((Complex*)(resEven[dir] + (i*3+1)*3*2), (Complex*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((Complex*)(resEven[dir] + (i*3 + 2)*3*2), 3);
      
      normalize((Complex*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((Complex*)(resOdd[dir] + (i*3+1)*3*2), (Complex*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((Complex*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	float *w = resEven[dir]+(i*3+0)*3*2;
	float *u = resEven[dir]+(i*3+1)*3*2;
	float *v = resEven[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	float *w = resOdd[dir]+(i*3+0)*3*2;
	float *u = resOdd[dir]+(i*3+1)*3*2;
	float *v = resOdd[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

    }
  }
    
  applyGaugeFieldScaling(res);
}

void constructPointSpinorField(float *res, int i0, int s0, int c0) {
  float *resEven = res;
  float *resOdd = res + Nh*spinorSiteSize;
    
  for(int i = 0; i < Nh; i++) {
    for (int s = 0; s < 4; s++) {
      for (int m = 0; m < 3; m++) {
	resEven[i*(4*3*2) + s*(3*2) + m*(2) + 0] = 0;
	resEven[i*(4*3*2) + s*(3*2) + m*(2) + 1] = 0;
	resOdd[i*(4*3*2) + s*(3*2) + m*(2) + 0] = 0;
	resOdd[i*(4*3*2) + s*(3*2) + m*(2) + 1] = 0;
	if (s == s0 && m == c0) {
	  if (fullLatticeIndex(i, 0) == i0)
	    resEven[i*(4*3*2) + s*(3*2) + m*(2) + 0] = 1;
	  if (fullLatticeIndex(i, 1) == i0)
	    resOdd[i*(4*3*2) + s*(3*2) + m*(2) + 0] = 1;
	}
      }
    }
  }
}

void constructSpinorField(float *res) {
  for(int i = 0; i < N; i++) {
    for (int s = 0; s < 4; s++) {
      for (int m = 0; m < 3; m++) {
	res[i*(4*3*2) + s*(3*2) + m*(2) + 0] = rand() / (float)RAND_MAX;
	res[i*(4*3*2) + s*(3*2) + m*(2) + 1] = rand() / (float)RAND_MAX;
      }
    }
  }
}


void applyGamma5(float *out, float *in, int sites) {
  for (int i=0; i<sites*spinorSiteSize; i+=spinorSiteSize) {
    for (int j=0; j<spinorSiteSize/2; j++) 
      out[i+j] = in[i+j];
    for (int j=0; j<spinorSiteSize/2; j++) 
      out[i+j+spinorSiteSize/2] = -in[i+j+spinorSiteSize/2];
  }
}








// OLD BROKEN METHOD
/*void su3_construct_8(float *mat) {
  Complex *u = (Complex*)&mat[0*(3*2)];
  Complex *v = (Complex*)&mat[1*(3*2)];
  Complex *w = (Complex*)&mat[2*(3*2)];

  // last elements of u are the phases for u[2] and v[1]
  v[0] = Complex(atan2(imag(u[2]),real(u[2])), atan2(imag(v[1]),real(v[1])));
  u[2] = v[2];

  v[1] = 0.0;
  v[2] = 0.0;
  for (uint i=0; i<3; i++) w[i] = 0.0; // remove last row
  }*/

/*
// given 8 numbers, reconstruct the full SU(3) matrix
void su3_reconstruct_8(float *mat) {
  mat[10] = mat[4]; // v[2] = u[2];
  mat[11] = mat[5];

  complexNorm(mat+8, mat+0); // norm(u[0])
  mat[12] = 1.f / mat[8]; // 1/norm(u[0])
  complexNorm(mat+13, mat+2); // norm(u[1])

  // Complete first row (u[2] = polar(sqrtf(1.0f - norm(u[0]) - norm(u[1])), real(u[2]))
  mat[14] = 1.f - (mat[8] + mat[13]); // (mat[8] free)
  mat[15] = sqrtf(mat[14]); // abs(u[2]) (mat[14] free)

  mat[4] = mat[15]*cos(mat[6]);  // u[2] now set
  mat[5] = mat[15]*sin(mat[6]); // (mat[6] and mat[15] now free)

  // Now the complicated second row
  mat[13] = 1.0 + mat[13]*mat[12]; // a in the quadratic

  mat[16] = cos(mat[7]);
  mat[17] = sin(mat[7]); // mat[7] now free
  
  complexDotProduct(mat+6, mat+2, mat+4); // conj(u[1]) * u[2]
  complexDotProduct(mat+8, mat+10, mat+16); // conj(v[2]) * polar(1.0, arg(v[1]))

  mat[14] = mat[6]*mat[8]; // (mat[6] now free)
  mat[14] -= mat[7]*mat[9]; // (mat[7] now free)
  mat[8] = mat[14] * mat[12]; 
  mat[8] *= 2.f; // b in the quadratic
  printf("b = %e, ", mat[8]);
  mat[9] = mat[8]*mat[8]; // b*b in the quadratic solution

  complexNorm(mat+14, mat+4); // norm(u[2])
  complexNorm(mat+15, mat+10); // norm(v[2])

  mat[6] = 1.f + mat[14] * mat[12];
  mat[14] = mat[15]*mat[6] - 1.f; // c in the quadratic (mat[6] free)

  // solve the quadratic
  mat[6] = mat[13]*mat[14];
  mat[6] *= 4.f;
  mat[7] = mat[9] - mat[6]; // b*b - 4ac
  mat[6] = sqrtf(mat[7]);
  //mat[7] = mat[6]/mat[13]; //sqrt(b*b-4ac) - b
  //mat[15] = -mat[8]/mat[13]; 
  //mat[7] *= 0.5f;
  //mat[15] *= 0.5f; // (sqrt(b*b-4ac) - b) / 2a (v[1] magnitude)
  mat[7] = mat[6] - mat[8]; //sqrt(b*b-4ac) - b
  mat[15] = mat[7] / mat[13]; 
  mat[15] *= 0.5f; // (sqrt(b*b-4ac) - b) / 2a (v[1] magnitude)

  mat[8] = (mat[15])*mat[16]; //  real(v[1]) = abs(v[1]) * cos arg(v[1]));
  mat[9] = (mat[15])*mat[17]; //  imag(v[1]) = abs(v[1]) * sin arg(v[1]));
  //printf("\nreal = %g imag = %g abs = %g cos(arg) = %g sin(arg) = %g\n", mat[8], mat[9], mat[15], mat[16], mat[17]);

  //mat[13] = mat[7]*mat[16]; // insurance terms
  //mat[14] = mat[7]*mat[17];

  // (all mat[13]-mat[17] free
  complexDotProduct(mat+16, mat+2, mat+8); // conj(u[1]) * v[1]
  complexDotProductAddTo(mat+16, mat+4, mat+10); // conj(u[1]) * v[1] + conj(u[2]) * v[2]

  complexProduct(mat+6, mat+16, mat+0); // v[0] = -(w[2] * u[0]) / norm(u[0])
  mat[6] *= -mat[12];
  mat[7] *= -mat[12];

  // Lastly do the cross product to obtain the third row
  complexConjugateProduct(mat+12, mat+2, mat+10);
  accumulateConjugateProduct(mat+12, mat+4, mat+8, -1);
  complexConjugateProduct(mat+14, mat+4, mat+6);
  accumulateConjugateProduct(mat+14, mat+0, mat+10, -1);
  complexConjugateProduct(mat+16, mat+0, mat+8);
  accumulateConjugateProduct(mat+16, mat+2, mat+6, -1);
}
*/

