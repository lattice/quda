#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <wilson_dslash_reference.h>
#include <test_util.h>

#include <face_quda.h>
#include <dslash_quda.h>
#include "misc.h"

using namespace std;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

/*	Alex	*/

extern "C"
{
	#include <lime.h>
}

#include <qcd.h>
#define MPI_READCONF

extern "C" { char* qcd_getParamComma(char token[],char* params,int len); };

/*	End Alex	*/

int Z[4];
int V;
int Vh;
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
int faceVolume[4];

//extended volume, +4
int E1, E1h, E2, E3, E4; 
int E[4];
int V_ex, Vh_ex;

int Ls;
int V5;
int V5h;

int mySpinorSiteSize;

extern float fat_link_max;


void initComms(int argc, char **argv, const int *commDims)
{
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif
  initCommsGridQuda(4, commDims, NULL, NULL);
  initRand();
}


void finalizeComms()
{
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}


void initRand()
{
  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17*rank + 137);
}


void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];
  
  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;


  E1=X[0]+4; E2=X[1]+4; E3=X[2]+4; E4=X[3]+4;
  E1h=E1/2;
  E[0] = E1;
  E[1] = E2;
  E[2] = E3;
  E[3] = E4;
  V_ex = E1*E2*E3*E4;
  Vh_ex = V_ex/2;

}


void dw_setDims(int *X, const int L5) 
{
  V = 1;
  for (int d=0; d< 4; d++) 
  {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;
  
  Ls = L5;
  V5 = V*Ls;
  V5h = Vh*Ls;

  Vs_t = Z[0]*Z[1]*Z[2]*Ls;//?
  Vsh_t = Vs_t/2;  //?
}


void setSpinorSiteSize(int n)
{
  mySpinorSiteSize = n;
}


template <typename Float>
static void printVector(Float *v) {
  printfQuda("{(%f %f) (%f %f) (%f %f)}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

// X indexes the lattice site
void printSpinorElement(void *spinor, int X, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION)
    for (int s=0; s<4; s++) printVector((double*)spinor+X*24+s*6);
  else
    for (int s=0; s<4; s++) printVector((float*)spinor+X*24+s*6);
}

// X indexes the full lattice
void printGaugeElement(void *gauge, int X, QudaPrecision precision) {
  if (getOddBit(X) == 0) {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m=0; m<3; m++) printVector((double*)gauge +(X/2)*gaugeSiteSize + m*3*2);
    else
      for (int m=0; m<3; m++) printVector((float*)gauge +(X/2)*gaugeSiteSize + m*3*2);
      
  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m = 0; m < 3; m++) printVector((double*)gauge + (X/2+Vh)*gaugeSiteSize + m*3*2);
    else
      for (int m = 0; m < 3; m++) printVector((float*)gauge + (X/2+Vh)*gaugeSiteSize + m*3*2);
  }
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int Y) {
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  return (x4+x3+x2+x1) % 2;
}

// a+=b
template <typename Float>
inline void complexAddTo(Float *a, Float *b) {
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
template <typename Float>
inline void complexProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] - b[1]*c[1];
  a[1] = b[0]*c[1] + b[1]*c[0];
}

// a = conj(b)*conj(c)
template <typename Float>
inline void complexConjugateProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] - b[1]*c[1];
  a[1] = -b[0]*c[1] - b[1]*c[0];
}

// a = conj(b)*c
template <typename Float>
inline void complexDotProduct(Float *a, Float *b, Float *c) {
  a[0] = b[0]*c[0] + b[1]*c[1];
  a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += b*c
template <typename Float>
inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a += conj(b)*c)
template <typename Float>
inline void accumulateComplexDotProduct(Float *a, Float *b, Float *c) {
  a[0] += b[0]*c[0] + b[1]*c[1];
  a[1] += b[0]*c[1] - b[1]*c[0];
}

template <typename Float>
inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

template <typename Float>
inline void su3Construct12(Float *mat) {
  Float *w = mat+12;
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;
  w[4] = 0.0;
  w[5] = 0.0;
}

// Stabilized Bunk and Sommer
template <typename Float>
inline void su3Construct8(Float *mat) {
  mat[0] = atan2(mat[1], mat[0]);
  mat[1] = atan2(mat[13], mat[12]);
  for (int i=8; i<18; i++) mat[i] = 0.0;
}

void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct12((double*)mat);
    else su3Construct12((float*)mat);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct8((double*)mat);
    else su3Construct8((float*)mat);
  }
}

// given first two rows (u,v) of SU(3) matrix mat, reconstruct the third row
// as the cross product of the conjugate vectors: w = u* x v*
// 
// 48 flops
template <typename Float>
static void su3Reconstruct12(Float *mat, int dir, int ga_idx, QudaGaugeParam *param) {
  Float *u = &mat[0*(3*2)];
  Float *v = &mat[1*(3*2)];
  Float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  Float u0 = (dir < 3 ? param->anisotropy :
	      (ga_idx >= (Z[3]-1)*Z[0]*Z[1]*Z[2]/2 ? param->t_boundary : 1));
  w[0]*=u0; w[1]*=u0; w[2]*=u0; w[3]*=u0; w[4]*=u0; w[5]*=u0;
}

template <typename Float>
static void su3Reconstruct8(Float *mat, int dir, int ga_idx, QudaGaugeParam *param) {
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];
  Float u0 = (dir < 3 ? param->anisotropy :
	      (ga_idx >= (Z[3]-1)*Z[0]*Z[1]*Z[2]/2 ? param->t_boundary : 1));
  Float U00_mag = sqrt(1.f/(u0*u0) - row_sum);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  Float U20_mag = sqrt(1.f/(u0*u0) - column_sum);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0/(u0*row_sum);

  // U11
  Float A[2];
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

void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision, QudaGaugeParam *param) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct12((double*)mat, dir, ga_idx, param);
    else su3Reconstruct12((float*)mat, dir, ga_idx, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct8((double*)mat, dir, ga_idx, param);
    else su3Reconstruct8((float*)mat, dir, ga_idx, param);
  }
}

/*
  void su3_construct_8_half(float *mat, short *mat_half) {
  su3Construct8(mat);

  mat_half[0] = floatToShort(mat[0] / M_PI);
  mat_half[1] = floatToShort(mat[1] / M_PI);
  for (int i=2; i<18; i++) {
  mat_half[i] = floatToShort(mat[i]);
  }
  }

  void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx, QudaGaugeParam *param) {

  for (int i=0; i<18; i++) {
  mat[i] = shortToFloat(mat_half[i]);
  }
  mat[0] *= M_PI;
  mat[1] *= M_PI;

  su3Reconstruct8(mat, dir, ga_idx, param);
  }*/

template <typename Float>
static int compareFloats(Float *a, Float *b, int len, double epsilon) {
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon) {
      printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return 0;
    }
  }
  return 1;
}

int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision) {
  if  (precision == QUDA_DOUBLE_PRECISION) return compareFloats((double*)a, (double*)b, len, epsilon);
  else return compareFloats((float*)a, (float*)b, len, epsilon);
}



// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
int fullLatticeIndex(int i, int oddBit) {
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];  
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2*sid + x1odd; 

  return X;
}


// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
// 
// the displacements, such as dx, refer to the full lattice coordinates. 
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1) {
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  
  // assert (oddBit == (x+y+z+t)%2);
  
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  
  return (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
}

int
neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int ret;
  
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  
  int ghost_x4 = x4+ dx4;
  
  // assert (oddBit == (x+y+z+t)%2);
  
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  
  if ( ghost_x4 >= 0 && ghost_x4 < Z[3]){
    ret = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
  }else{
    ret = (x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;    
  }

  
  return ret;
}


/*  
 * This is a computation of neighbor using the full index and the displacement in each direction
 *
 */

int
neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1) 
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }
    
  int nbr_half_idx = neighborIndex(half_idx, oddBit, dx4,dx3,dx2,dx1);
  int oddBitChanged = (dx4+dx3+dx2+dx1)%2;
  if (oddBitChanged){
    oddBit = 1 - oddBit;
  }
  int ret = nbr_half_idx;
  if (oddBit){
    ret = Vh + nbr_half_idx;
  }
    
  return ret;
}


int
neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1) 
{
  int ret;
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }
    
  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int ghost_x4 = x4+ dx4;
    
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];

  if ( ghost_x4 >= 0 && ghost_x4 < Z[3]){
    ret = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
  }else{
    ret = (x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;    
    return ret;
  }

  int oddBitChanged = (dx4+dx3+dx2+dx1)%2;
  if (oddBitChanged){
    oddBit = 1 - oddBit;
  }
    
  if (oddBit){
    ret += Vh;
  }
    
  return ret;
}


// 4d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index.
int fullLatticeIndex_4d(int i, int oddBit) {
  if (i >= Vh || i < 0) {printf("i out of range in fullLatticeIndex_4d"); exit(-1);}
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];  
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2*sid + x1odd; 

  return X;
}

// 5d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index sid.
// This function is used by neighborIndex_5d in dslash_reference.cpp.
//ok
int fullLatticeIndex_5d(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2) + i/(Z[3]*Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int 
x4_from_full_index(int i)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }
  
  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  
  return x4;
}

template <typename Float>
static void applyGaugeFieldScaling(Float **gauge, int Vh, QudaGaugeParam *param) {
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < gaugeSiteSize*Vh*2; i++) {
      gauge[d][i] /= param->anisotropy;
    }
  }
    
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  bool last_node_in_t = (commCoords(3) == commDim(3)-1) ? true : false;
#else
  bool last_node_in_t = true;
#endif

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t) {
    for (int j = (Z[0]/2)*Z[1]*Z[2]*(Z[3]-1); j < Vh; j++) {
      for (int i = 0; i < gaugeSiteSize; i++) {
	gauge[3][j*gaugeSiteSize+i] *= -1.0;
	gauge[3][(Vh+j)*gaugeSiteSize+i] *= -1.0;
      }
    }
  }
    
  if (param->gauge_fix) {
    // set all gauge links (except for the last Z[0]*Z[1]*Z[2]/2) to the identity,
    // to simulate fixing to the temporal gauge.
    int iMax = ( last_node_in_t ? (Z[0]/2)*Z[1]*Z[2]*(Z[3]-1) : Vh );
    int dir = 3; // time direction only
    Float *even = gauge[dir];
    Float *odd  = gauge[dir]+Vh*gaugeSiteSize;
    for (int i = 0; i< iMax; i++) {
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

template <typename Float>
void applyGaugeFieldScaling_long(Float **gauge, int Vh, QudaGaugeParam *param)
{

  int X1h=param->X[0]/2;
  int X1 =param->X[0];
  int X2 =param->X[1];
  int X3 =param->X[2];
  int X4 =param->X[3];

  // rescale long links by the appropriate coefficient
  for(int d=0; d<4; d++){
    for(int i=0; i < V*gaugeSiteSize; i++){
      gauge[d][i] /= (-24*param->tadpole_coeff*param->tadpole_coeff);
    }
  }

  // apply the staggered phases
  for (int d = 0; d < 3; d++) {

    //even
    for (int i = 0; i < Vh; i++) {

      int index = fullLatticeIndex(i, 0);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      int sign=1;

      if (d == 0) {
	if (i4 % 2 == 1){
	  sign= -1;
	}
      }

      if (d == 1){
	if ((i4+i1) % 2 == 1){
	  sign= -1;
	}
      }
      if (d == 2){
	if ( (i4+i1+i2) % 2 == 1){
	  sign= -1;
	}
      }

      for (int j=0;j < 6; j++){
	gauge[d][i*gaugeSiteSize + 12+ j] *= sign;
      }
    }
    //odd
    for (int i = 0; i < Vh; i++) {
      int index = fullLatticeIndex(i, 1);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      int sign=1;

      if (d == 0) {
	if (i4 % 2 == 1){
	  sign= -1;
	}
      }

      if (d == 1){
	if ((i4+i1) % 2 == 1){
	  sign= -1;
	}
      }
      if (d == 2){
	if ( (i4+i1+i2) % 2 == 1){
	  sign = -1;
	}
      }

      for (int j=0;j < 6; j++){
	gauge[d][(Vh+i)*gaugeSiteSize + 12 + j] *= sign;
      }
    }

  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T) {
    for (int j = 0; j < Vh; j++) {
      int sign =1;
      if (j >= (X4-3)*X1h*X2*X3 ){
	sign= -1;
      }

      for (int i = 0; i < 6; i++) {
	gauge[3][j*gaugeSiteSize+ 12+ i ] *= sign;
	gauge[3][(Vh+j)*gaugeSiteSize+12 +i] *= sign;
      }
    }
  }
}



template <typename Float>
static void constructUnitGaugeField(Float **res, QudaGaugeParam *param) {
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
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
    
  applyGaugeFieldScaling(res, Vh, param);
}

// normalize the vector a
template <typename Float>
static void normalize(complex<Float> *a, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) sum += norm(a[i]);
  for (int i=0; i<len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
template <typename Float>
static void orthogonalize(complex<Float> *a, complex<Float> *b, int len) {
  complex<double> dot = 0.0;
  for (int i=0; i<len; i++) dot += conj(a[i])*b[i];
  for (int i=0; i<len; i++) b[i] -= (complex<Float>)dot*a[i];
}

template <typename Float> 
static void constructGaugeField(Float **res, QudaGaugeParam *param) {
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;                    
	}
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);
      
      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;
	
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

  if (param->type == QUDA_WILSON_LINKS){  
    applyGaugeFieldScaling(res, Vh, param);
  } else if (param->type == QUDA_ASQTAD_LONG_LINKS){
    applyGaugeFieldScaling_long(res, Vh, param);      
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS){
    for (int dir = 0; dir < 4; dir++){ 
      for (int i = 0; i < Vh; i++) {
	for (int m = 0; m < 3; m++) { // last 2 rows
	  for (int n = 0; n < 3; n++) { // 3 columns
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] =1.0* rand() / (Float)RAND_MAX;
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] =2.0* rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = 3.0*rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 4.0*rand() / (Float)RAND_MAX;
	  }
	}
      }
    }
    
  }

}

template <typename Float> 
void constructUnitaryGaugeField(Float **res) 
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
  }
  
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;                    
	}
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);
      
      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }
      
      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;
	
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
}


void construct_gauge_field(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param) {
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) constructUnitGaugeField((double**)gauge, param);
    else constructUnitGaugeField((float**)gauge, param);
  } else if (type == 1) {
    if (precision == QUDA_DOUBLE_PRECISION) constructGaugeField((double**)gauge, param);
    else constructGaugeField((float**)gauge, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) applyGaugeFieldScaling((double**)gauge, Vh, param);
    else applyGaugeFieldScaling((float**)gauge, Vh, param);    
  }

}

void
construct_fat_long_gauge_field(void **fatlink, void** longlink,  
			       int type, QudaPrecision precision, QudaGaugeParam* param)
{
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) {
      constructUnitGaugeField((double**)fatlink, param);
      constructUnitGaugeField((double**)longlink, param);
    }else {
      constructUnitGaugeField((float**)fatlink, param);
      constructUnitGaugeField((float**)longlink, param);
    }
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) {
      param->type = QUDA_ASQTAD_FAT_LINKS;
      constructGaugeField((double**)fatlink, param);
      param->type = QUDA_ASQTAD_LONG_LINKS;
      constructGaugeField((double**)longlink, param);
    }else {
      param->type = QUDA_ASQTAD_FAT_LINKS;
      constructGaugeField((float**)fatlink, param);
      param->type = QUDA_ASQTAD_LONG_LINKS;
      constructGaugeField((float**)longlink, param);
    }
  }
}


template <typename Float>
static void constructCloverField(Float *res, double norm, double diag) {

  Float c = 2.0 * norm / RAND_MAX;

  for(int i = 0; i < V; i++) {
    for (int j = 0; j < 72; j++) {
      res[i*72 + j] = c*rand() - norm;
    }
    for (int j = 0; j< 6; j++) {
      res[i*72 + j] += diag;
      res[i*72 + j+36] += diag;
    }
  }
}

void construct_clover_field(void *clover, double norm, double diag, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) constructCloverField((double *)clover, norm, diag);
  else constructCloverField((float *)clover, norm, diag);
}

/*void strong_check(void *spinorRef, void *spinorGPU, int len, QudaPrecision prec) {
  printf("Reference:\n");
  printSpinorElement(spinorRef, 0, prec); printf("...\n");
  printSpinorElement(spinorRef, len-1, prec); printf("\n");    
    
  printf("\nCUDA:\n");
  printSpinorElement(spinorGPU, 0, prec); printf("...\n");
  printSpinorElement(spinorGPU, len-1, prec); printf("\n");

  compare_spinor(spinorRef, spinorGPU, len, prec);
  }*/

template <typename Float>
static void checkGauge(Float **oldG, Float **newG, double epsilon) {

  const int fail_check = 17;
  int fail[4][fail_check];
  int iter[4][18];
  for (int d=0; d<4; d++) for (int i=0; i<fail_check; i++) fail[d][i] = 0;
  for (int d=0; d<4; d++) for (int i=0; i<18; i++) iter[d][i] = 0;

  for (int d=0; d<4; d++) {
    for (int eo=0; eo<2; eo++) {
      for (int i=0; i<Vh; i++) {
	int ga_idx = (eo*Vh+i);
	for (int j=0; j<18; j++) {
	  double diff = fabs(newG[d][ga_idx*18+j] - oldG[d][ga_idx*18+j]);/// fabs(oldG[d][ga_idx*18+j]);

	  for (int f=0; f<fail_check; f++) if (diff > pow(10.0,-(f+1))) fail[d][f]++;
	  if (diff > epsilon) iter[d][j]++;
	}
      }
    }
  }

  printf("Component fails (X, Y, Z, T)\n");
  for (int i=0; i<18; i++) printf("%d fails = (%8d, %8d, %8d, %8d)\n", i, iter[0][i], iter[1][i], iter[2][i], iter[3][i]);

  printf("\nDeviation Failures = (X, Y, Z, T)\n");
  for (int f=0; f<fail_check; f++) {
    printf("%e Failures = (%9d, %9d, %9d, %9d) = (%e, %e, %e, %e)\n", pow(10.0,-(f+1)), 
	   fail[0][f], fail[1][f], fail[2][f], fail[3][f],
	   fail[0][f]/(double)(V*18), fail[1][f]/(double)(V*18), fail[2][f]/(double)(V*18), fail[3][f]/(double)(V*18));
  }

}

void check_gauge(void **oldG, void **newG, double epsilon, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) 
    checkGauge((double**)oldG, (double**)newG, epsilon);
  else 
    checkGauge((float**)oldG, (float**)newG, epsilon);
}



void 
createSiteLinkCPU(void** link,  QudaPrecision precision, int phase) 
{
    
  if (precision == QUDA_DOUBLE_PRECISION) {
    constructUnitaryGaugeField((double**)link);
  }else {
    constructUnitaryGaugeField((float**)link);
  }

  // only apply temporal boundary condition if I'm the last node in T
#ifdef MULTI_GPU
  bool last_node_in_t = (commCoords(3) == commDim(3)-1) ? true : false;
#else
  bool last_node_in_t = true;
#endif

  if(phase){
	
    for(int i=0;i < V;i++){
      for(int dir =XUP; dir <= TUP; dir++){
	int idx = i;
	int oddBit =0;
	if (i >= Vh) {
	  idx = i - Vh;
	  oddBit = 1;
	}

	int X1 = Z[0];
	int X2 = Z[1];
	int X3 = Z[2];
	int X4 = Z[3];

	int full_idx = fullLatticeIndex(idx, oddBit);
	int i4 = full_idx /(X3*X2*X1);
	int i3 = (full_idx - i4*(X3*X2*X1))/(X2*X1);
	int i2 = (full_idx - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
	int i1 = full_idx - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;	    

	double coeff= 1.0;
	switch(dir){
	case XUP:
	  if ( (i4 & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case YUP:
	  if ( ((i4+i1) & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case ZUP:
	  if ( ((i4+i1+i2) & 1) != 0){
	    coeff *= -1;
	  }
	  break;
		
	case TUP:
	  if (last_node_in_t && i4 == (X4-1)){
	    coeff *= -1;
	  }
	  break;

	default:
	  printf("ERROR: wrong dir(%d)\n", dir);
	  exit(1);
	}
	
	if (precision == QUDA_DOUBLE_PRECISION){
	  //double* mylink = (double*)link;
	  //mylink = mylink + (4*i + dir)*gaugeSiteSize;
	  double* mylink = (double*)link[dir];
	  mylink = mylink + i*gaugeSiteSize;

	  mylink[12] *= coeff;
	  mylink[13] *= coeff;
	  mylink[14] *= coeff;
	  mylink[15] *= coeff;
	  mylink[16] *= coeff;
	  mylink[17] *= coeff;
		
	}else{
	  //float* mylink = (float*)link;
	  //mylink = mylink + (4*i + dir)*gaugeSiteSize;
	  float* mylink = (float*)link[dir];
	  mylink = mylink + i*gaugeSiteSize;
		  
	  mylink[12] *= coeff;
	  mylink[13] *= coeff;
	  mylink[14] *= coeff;
	  mylink[15] *= coeff;
	  mylink[16] *= coeff;
	  mylink[17] *= coeff;
		
	}
      }
    }
  }    

    
#if 1
  for(int dir= 0;dir < 4;dir++){
    for(int i=0;i< V*gaugeSiteSize;i++){
      if (precision ==QUDA_SINGLE_PRECISION){
	float* f = (float*)link[dir];
	if (f[i] != f[i] || (fabsf(f[i]) > 1.e+3) ){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
      }else{
	double* f = (double*)link[dir];
	if (f[i] != f[i] || (fabs(f[i]) > 1.e+3)){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
	  
      }
	
    }
  }
#endif

  return;
}




template <typename Float>
int compareLink(Float **linkA, Float **linkB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[18];
  for (int i=0; i<18; i++) iter[i] = 0;
  
  for(int dir=0;dir < 4; dir++){
    for (int i=0; i<len; i++) {
      for (int j=0; j<18; j++) {
	int is = i*18+j;
	double diff = fabs(linkA[dir][is]-linkB[dir][is]);
	for (int f=0; f<fail_check; f++)
	  if (diff > pow(10.0,-(f+1))) fail[f]++;
	//if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	if (diff > 1e-3) iter[j]++;
      }
    }
  }
  
  for (int i=0; i<18; i++) printfQuda("%d fails = %d\n", i, iter[i]);
  
  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f;
    }
  }

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], 4*len*18, fail[f] / (double)(4*len*18));
  }
  
  return accuracy_level;
}

static int
compare_link(void **linkA, void **linkB, int len, QudaPrecision precision)
{
  int ret;

  if (precision == QUDA_DOUBLE_PRECISION){    
    ret = compareLink((double**)linkA, (double**)linkB, len);
  }else {
    ret = compareLink((float**)linkA, (float**)linkB, len);
  }    

  return ret;
}


// X indexes the lattice site
static void 
printLinkElement(void *link, int X, QudaPrecision precision) 
{
  if (precision == QUDA_DOUBLE_PRECISION){
    for(int i=0; i < 3;i++){
      printVector((double*)link+ X*gaugeSiteSize + i*6);
    }
	
  }
  else{
    for(int i=0;i < 3;i++){
      printVector((float*)link+X*gaugeSiteSize + i*6);
    }
  }
}

int strong_check_link(void** linkA, const char* msgA, 
		      void **linkB, const char* msgB, 
		      int len, QudaPrecision prec) 
{
  printfQuda("%s\n", msgA);
  printLinkElement(linkA[0], 0, prec); 
  printfQuda("\n");
  printLinkElement(linkA[0], 1, prec); 
  printfQuda("...\n");
  printLinkElement(linkA[3], len-1, prec); 
  printfQuda("\n");    
    
  printfQuda("\n%s\n", msgB);
  printLinkElement(linkB[0], 0, prec); 
  printfQuda("\n");
  printLinkElement(linkB[0], 1, prec); 
  printfQuda("...\n");
  printLinkElement(linkB[3], len-1, prec); 
  printfQuda("\n");
    
  int ret = compare_link(linkA, linkB, len, prec);
  return ret;
}


void 
createMomCPU(void* mom,  QudaPrecision precision) 
{
  void* temp;
    
  size_t gSize = (precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  temp = malloc(4*V*gaugeSiteSize*gSize);
  if (temp == NULL){
    fprintf(stderr, "Error: malloc failed for temp in function %s\n", __FUNCTION__);
    exit(1);
  }
    
    
    
  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
	double* thismom = (double*)mom;	    
	for(int k=0; k < momSiteSize; k++){
	  thismom[ (4*i+dir)*momSiteSize + k ]= 1.0* rand() /RAND_MAX;				
	  if (k==momSiteSize-1) thismom[ (4*i+dir)*momSiteSize + k ]= 0.0;
	}	    
      }	    
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thismom=(float*)mom;
	for(int k=0; k < momSiteSize; k++){
	  thismom[ (4*i+dir)*momSiteSize + k ]= 1.0* rand() /RAND_MAX;		
	  if (k==momSiteSize-1) thismom[ (4*i+dir)*momSiteSize + k ]= 0.0;
	}	    
      }
    }
  }
    
  free(temp);
  return;
}

void
createHwCPU(void* hw,  QudaPrecision precision)
{
  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
	double* thishw = (double*)hw;
	for(int k=0; k < hwSiteSize; k++){
	  thishw[ (4*i+dir)*hwSiteSize + k ]= 1.0* rand() /RAND_MAX;
	}
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thishw=(float*)hw;
	for(int k=0; k < hwSiteSize; k++){
	  thishw[ (4*i+dir)*hwSiteSize + k ]= 1.0* rand() /RAND_MAX;
	}
      }
    }
  }

  return;
}


template <typename Float>
int compare_mom(Float *momA, Float *momB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[momSiteSize];
  for (int i=0; i<momSiteSize; i++) iter[i] = 0;
  
  for (int i=0; i<len; i++) {
    for (int j=0; j<momSiteSize; j++) {
      int is = i*momSiteSize+j;
      double diff = fabs(momA[is]-momB[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10.0,-(f+1))) fail[f]++;
      //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
      if (diff > 1e-3) iter[j]++;
    }
  }
  
  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f+1;
    }
  }

  for (int i=0; i<momSiteSize; i++) printfQuda("%d fails = %d\n", i, iter[i]);
  
  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*momSiteSize, fail[f] / (double)(len*6));
  }
  
  return accuracy_level;
}

static void 
printMomElement(void *mom, int X, QudaPrecision precision) 
{
  if (precision == QUDA_DOUBLE_PRECISION){
    double* thismom = ((double*)mom)+ X*momSiteSize;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }else{
    float* thismom = ((float*)mom)+ X*momSiteSize;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);	
  }
}
int strong_check_mom(void * momA, void *momB, int len, QudaPrecision prec) 
{    
  printfQuda("mom:\n");
  printMomElement(momA, 0, prec); 
  printfQuda("\n");
  printMomElement(momA, 1, prec); 
  printfQuda("\n");
  printMomElement(momA, 2, prec); 
  printfQuda("\n");
  printMomElement(momA, 3, prec); 
  printfQuda("...\n");
  
  printfQuda("\nreference mom:\n");
  printMomElement(momB, 0, prec); 
  printfQuda("\n");
  printMomElement(momB, 1, prec); 
  printfQuda("\n");
  printMomElement(momB, 2, prec); 
  printfQuda("\n");
  printMomElement(momB, 3, prec); 
  printfQuda("\n");
  
  int ret;
  if (prec == QUDA_DOUBLE_PRECISION){
    ret = compare_mom((double*)momA, (double*)momB, len);
  }else{
    ret = compare_mom((float*)momA, (float*)momB, len);
  }
  
  return ret;
}


/************
 * return value
 *
 * 0: command line option matched and processed sucessfully
 * non-zero: command line option does not match
 *
 */

#ifdef MULTI_GPU
int device = -1;
#else
int device = 0;
#endif

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_INVALID;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision  prec_sloppy = QUDA_INVALID_PRECISION;
int xdim = 24;
int ydim = 24;
int zdim = 24;
int tdim = 24;
int Lsdim = 16;
QudaDagType dagger = QUDA_DAG_NO;
int gridsize_from_cmdline[4] = {1,1,1,1};
QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
char latfile[256] = "";
bool tune = true;
int niter = 10;
int test_type = 0;

/*      Alex    */

int	numberLP	 = 0;
int	numberHP	 = 1;
int	nConf		 = 0;
int	MaxP		 = 0;
int	overrideMu	 = 0;
double	newMu		 = 0.;

/*      End Alex        */

static int dim_partitioned[4] = {0,0,0,0};

int dimPartitioned(int dim)
{
  return ((gridsize_from_cmdline[dim] > 1) || dim_partitioned[dim]);
}


void __attribute__((weak)) usage_extra(char** argv){};

void usage(char** argv )
{
  printf("Usage: %s [options]\n", argv[0]);
  printf("Common options: \n");
#ifndef MULTI_GPU
  printf("    --device <n>                              # Set the CUDA device to use (default 0, single GPU only)\n");     
#endif
  printf("    --prec <double/single/half>               # Precision in GPU\n"); 
  printf("    --prec_sloppy <double/single/half>        # Sloppy precision in GPU\n"); 
  printf("    --recon <8/12/18>                         # Link reconstruction type\n"); 
  printf("    --recon_sloppy <8/12/18>                  # Sloppy link reconstruction type\n"); 
  printf("    --dagger                                  # Set the dagger to 1 (default 0)\n"); 
  printf("    --sdim <n>                                # Set space dimention(X/Y/Z) size\n"); 
  printf("    --xdim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --ydim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --zdim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --tdim <n>                                # Set T dimension size(default 24)\n");  
  printf("    --Lsdim <n>                                # Set Ls dimension size(default 16)\n");  
  printf("    --xgridsize <n>                           # Set grid size in X dimension (default 1)\n");
  printf("    --ygridsize <n>                           # Set grid size in Y dimension (default 1)\n");
  printf("    --zgridsize <n>                           # Set grid size in Z dimension (default 1)\n");
  printf("    --tgridsize <n>                           # Set grid size in T dimension (default 1)\n");
  printf("    --partition <mask>                        # Set the communication topology (X=1, Y=2, Z=4, T=8, and combinations of these)\n");
  printf("    --kernel_pack_t                           # Set T dimension kernel packing to be true (default false)\n");
  printf("    --dslash_type <type>                      # Set the dslash type, the following values are valid\n"
	 "                                                  wilson/clover/twisted_mass/asqtad/domain_wall\n");
  printf("    --load-gauge file                         # Load gauge field \"file\" for the test (requires QIO)\n");

	/*	Alex	*/

  printf("    --load <nConf>                            # Load gauge configuration number <nConf>\n");
  printf("    --mu <float>                              # Set mu at value <float>\n");
  printf("    --lp <n>                                  # Set the number of LP sources to generate (default 0)\n");
  printf("    --hp <n>                                  # Set the number of HP sources to generate (default 1)\n");
  printf("    --maxP <n>                                # Set the maximum value of p^2 (default 0)\n");

        /*      End     Alex    */

  printf("    --niter <n>                               # The number of iterations to perform (default 10)\n");
  printf("    --tune <true/false>                       # Whether to autotune or not (default true)\n");     
  printf("    --test                                    # Test method (different for each test)\n");
  printf("    --help                                    # Print out this message\n"); 
  usage_extra(argv); 
#ifdef MULTI_GPU
  char msg[]="multi";
#else
  char msg[]="single";
#endif  
  printf("Note: this program is %s GPU build\n", msg);
  exit(1);
  return ;
}

int process_command_line_option(int argc, char** argv, int* idx)
{
#ifdef MULTI_GPU
  char msg[]="multi";
#else
  char msg[]="single";
#endif

  int ret = -1;
  
  int i = *idx;

  if( strcmp(argv[i], "--help")== 0){
    usage(argv);
  }

  if( strcmp(argv[i], "--device") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    device = atoi(argv[i+1]);
    if (device < 0 || device > 16){
      printf("ERROR: Invalid CUDA device number (%d)\n", device);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    prec =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec_sloppy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    prec_sloppy =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--recon") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    link_recon =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--recon_sloppy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    link_recon_sloppy =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--xdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    xdim= atoi(argv[i+1]);
    if (xdim < 0 || xdim > 512){
      printf("ERROR: invalid X dimension (%d)\n", xdim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ydim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    ydim= atoi(argv[i+1]);
    if (ydim < 0 || ydim > 512){
      printf("ERROR: invalid T dimension (%d)\n", ydim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--zdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    zdim= atoi(argv[i+1]);
    if (zdim < 0 || zdim > 512){
      printf("ERROR: invalid T dimension (%d)\n", zdim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    tdim =  atoi(argv[i+1]);
    if (tdim < 0 || tdim > 512){
      errorQuda("Error: invalid t dimension");
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--sdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    int sdim =  atoi(argv[i+1]);
    if (sdim < 0 || sdim > 512){
      printfQuda("ERROR: invalid S dimension\n");
    }
    xdim=ydim=zdim=sdim;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--Lsdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    int Ls =  atoi(argv[i+1]);
    if (Ls < 0 || Ls > 128){
      printfQuda("ERROR: invalid Ls dimension\n");
    }
    Lsdim=Ls;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--dagger") == 0){
    dagger = QUDA_DAG_YES;
    ret = 0;
    goto out;
  }	
  
  if( strcmp(argv[i], "--partition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
#ifdef MULTI_GPU
    int value  =  atoi(argv[i+1]);
    for(int j=0; j < 4;j++){
      if (value &  (1 << j)){
	commDimPartitionedSet(j);
	dim_partitioned[j] = 1;
      }
    }
#else
    printfQuda("WARNING: Ignoring --partition option since this is a single-GPU build.\n");
#endif
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--kernel_pack_t") == 0){
    quda::setKernelPackT(true);
    ret= 0;
    goto out;
  }


  if( strcmp(argv[i], "--tune") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    

    if (strcmp(argv[i+1], "true") == 0){
      tune = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      tune = false;
    }else{
      fprintf(stderr, "ERROR: invalid tuning type\n");	
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--xgridsize") == 0){
    if (i+1 >= argc){ 
      usage(argv);
    }     
    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      errorQuda("ERROR: invalid X grid size");
    }
    gridsize_from_cmdline[0] = xsize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ygridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      errorQuda("ERROR: invalid Y grid size");
    }
    gridsize_from_cmdline[1] = ysize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--zgridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      errorQuda("ERROR: invalid Z grid size");
    }
    gridsize_from_cmdline[2] = zsize;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--tgridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      errorQuda("ERROR: invalid T grid size");
    }
    gridsize_from_cmdline[3] = tsize;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--dslash_type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    dslash_type =  get_dslash_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--load-gauge") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(latfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
	/*	Alex	*/

  if( strcmp(argv[i], "--mu") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    overrideMu = 1;
    newMu = atof(argv[i+1]);

    printfQuda		("Overriding mu with %le for strange/charm computation.\n", newMu);

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--load") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
	nConf		 = atoi(argv[i+1]);
	sprintf	(latfile,"/home/avaquero/confs/Cosa/conf.%04d", nConf);
//	sprintf		(latfile,"/lustre/avaquero/confs/B55.32/conf.%04d", nConf);
//	sprintf	(latfile,"/lustre/avaquero/confs/D15.48/conf.%04d", nConf);
//	sprintf	(latfile,"/home/avaquero/confs/conf.%04d", nConf);
	printfQuda	("Will load %s\n", latfile);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--maxP") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    MaxP =  atoi(argv[i+1]);
        printfQuda	("Will generate momenta up to p^2 = %d\n", MaxP);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--hp") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    numberHP =	atoi(argv[i+1]);

    i++;
    ret = 0;

    printfQuda	("Will generate %d HP sources\n", numberHP);

    goto out;
  }

  if( strcmp(argv[i], "--lp") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    numberLP =	atoi(argv[i+1]);
        printfQuda	("Will generate %d LP sources\n", numberLP);
    i++;
    ret = 0;
    goto out;
  }
	/*	End Alex	*/

  if( strcmp(argv[i], "--test") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    test_type = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;	    
  }
    
  if( strcmp(argv[i], "--niter") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    niter= atoi(argv[i+1]);
    if (niter < 1 || niter > 1e6){
      printf("ERROR: invalid number of iterations (%d)\n", niter);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--version") == 0){
    printf("This program is linked with QUDA library, version %s,", 
	   get_quda_ver_str());
    printf(" %s GPU build\n", msg);
    exit(0);
  }

 out:
  *idx = i;
  return ret ;

}


static struct timeval startTime;

void stopwatchStart() {
  gettimeofday(&startTime, NULL);
}

double stopwatchReadSeconds() {
  struct timeval endTime;
  gettimeofday(&endTime, NULL);

  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}


	/*	Alex	*/

void qcd_swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}


void qcd_swap_4(float *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
  {
    j=i; k=j+3;
    swap = *j; *j = *k;  *k = swap;
    j++; k--;
    swap = *j; *j = *k;  *k = swap;
  }
}

int qcd_isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}

char* qcd_getParam(char token[],char* params,int len)
{
   int i,token_len=strlen(token);

   for(i=0;i<len-token_len;i++)
   {
      if(memcmp(token,params+i,token_len)==0)
      {
         i+=token_len;
         *(strchr(params+i,'<'))='\0';
         break;
      }
   }
   return params+i;
}

int read_custom_binary_gauge_field (double **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4])
{
/*
read gauge fileld config stored in binary file
*/
	FILE			*fid;

	int			x1, x2, x3, x4, ln[4] = { 0, 0, 0, 0 };

	unsigned long long	lvol, ixh, iy, mu;

	char			tmpVar[20];
 
	double			*U = (double*)malloc(18*sizeof(double));

	double			*resEvn[4], *resOdd[4];
	int			nvh, iDummy;

	LimeReader		*limereader;
	char			*lime_type, *lime_data;
	n_uint64_t		lime_data_size;
	char			dummy;
	int			isDouble;
	int			error_occured=0;

	double			dDummy;

#ifdef	MPI_READCONF
	MPI_Offset		offset;
	MPI_Datatype		subblock;  //MPI-type, 5d subarray
	MPI_File		mpifid;
	MPI_Status		status;
	int			sizes[5], lsizes[5], starts[5];
	qcd_uint_8		i=0;
	qcd_uint_2		chunksize;
	char			*ftmp=NULL;
#else
	double			*ftmp=NULL;
	unsigned long long	iread, idx;
#endif

	fid=fopen(fname,"r");
	if(fid==NULL)
	{
		fprintf(stderr,"Error reading configuration! Could not open %s for reading\n",fname);
		error_occured=1;
	}
	
	if ((limereader = limeCreateReader(fid))==NULL)
	{
		fprintf(stderr,"Could not create limeReader\n");
		error_occured=1;
	}
	
	if(!error_occured)
	{
		while(limeReaderNextRecord(limereader) != LIME_EOF )
		{
			lime_type	= limeReaderType(limereader);

			if(strcmp(lime_type,"ildg-binary-data")==0)
				break;

			if(strcmp(lime_type,"xlf-info")==0)
			{
				lime_data_size = limeReaderBytes(limereader);
				lime_data = (char * )malloc(lime_data_size);
				limeReaderReadData((void *)lime_data,&lime_data_size, limereader);

				strcpy	(tmpVar, "kappa =");
				sscanf(qcd_getParamComma(tmpVar,lime_data, lime_data_size),"%lf",&dDummy);    
				printfQuda("Kappa:    \t%lf\n", dDummy);
				inv_param->kappa	= dDummy;

				strcpy	(tmpVar, "mu =");
				sscanf(qcd_getParamComma(tmpVar,lime_data, lime_data_size),"%lf",&dDummy);    
				printfQuda("Mu:       \t%lf\n", dDummy);

				if      (overrideMu)
				{
					printfQuda	("MU OVERRIDEN\t%lf\n", newMu);
					inv_param->mu	 = newMu;
				}
				else
					inv_param->mu		= dDummy;

				free(lime_data);
			}

			if(strcmp(lime_type,"ildg-format")==0)
			{
				lime_data_size = limeReaderBytes(limereader);
				lime_data = (char * )malloc(lime_data_size);
				limeReaderReadData((void *)lime_data,&lime_data_size, limereader);

				strcpy	(tmpVar, "<precision>");
				sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&isDouble);    
				printfQuda("Precision:\t%i bit\n",isDouble);

				strcpy	(tmpVar, "<lx>");
				sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
				param->X[0]	 = iDummy/gridSize[0];
				ln[0]		 = iDummy;

				strcpy	(tmpVar, "<ly>");
				sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
				param->X[1]	 = iDummy/gridSize[1];
				ln[1]		 = iDummy;

				strcpy	(tmpVar, "<lz>");
				sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
				param->X[2]	 = iDummy/gridSize[2];
				ln[2]		 = iDummy;

				strcpy	(tmpVar, "<lt>");
				sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
				param->X[3]	 = iDummy/gridSize[3];
				ln[3]		 = iDummy;

				printfQuda("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
				printfQuda("Subvolume:\t%ix%ix%ix%i\n", param->X[0], param->X[1], param->X[2], param->X[3]);

				free(lime_data);
			}
		}

		// Read 1 byte to set file-pointer to start of binary data

#ifdef	MPI_READCONF
		lime_data_size=1;
		limeReaderReadData(&dummy,&lime_data_size,limereader);
		offset = ftell(fid)-1;
#endif
		limeDestroyReader(limereader);
	}     

	MPI_Bcast	(&error_occured, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(error_occured)
		return	1;

	if	(isDouble == 32)
	{
		printfQuda	("Error: Unsupported precision %d bits\n", isDouble);
		return		1;
	}

	nvh = (param->X[0] * param->X[1] * param->X[2] * param->X[3]) / 2;

	for(int dir = 0; dir < 4; dir++)
	{
		resEvn[dir] = gauge[dir];
		resOdd[dir] = gauge[dir] + nvh*18;
	} 

	lvol = ln[0]*ln[1]*ln[2]*ln[3];

	if(lvol==0)
	{
		fprintf(stderr, "[] Error, zero volume\n");
		return(5);
	}
 
	int x4start = comm_coord(3)*param->X[3];
	int x4end   = x4start + param->X[3];
	int x3start = comm_coord(2)*param->X[2];
	int x3end   = x3start + param->X[2];
	int x2start = comm_coord(1)*param->X[1];
	int x2end   = x2start + param->X[1];
	int x1start = comm_coord(0)*param->X[0];
	int x1end   = x1start + param->X[0];
 
#ifdef	MPI_READCONF
	sizes[0]	 = ln[3];
	sizes[1]	 = ln[2];
	sizes[2]	 = ln[1];
	sizes[3]	 = ln[0];
	sizes[4]	 = 4*3*3*2;
	lsizes[0]	 = param->X[3];
	lsizes[1]	 = param->X[2];
	lsizes[2]	 = param->X[1];
	lsizes[3]	 = param->X[0];
	lsizes[4]	 = sizes[4];
	starts[0]	 = comm_coord(3)*param->X[3];
	starts[1]	 = comm_coord(2)*param->X[2];
	starts[2]	 = comm_coord(1)*param->X[1];
	starts[3]	 = comm_coord(0)*param->X[0];
	starts[4]	 = 0;

	strcpy	(tmpVar, "native");

	MPI_Type_create_subarray	(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
	MPI_Type_commit			(&subblock);
	
	MPI_File_open			(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
	MPI_File_set_view		(mpifid, offset, MPI_DOUBLE, subblock, tmpVar, MPI_INFO_NULL);

	//load time-slice by time-slice:

	chunksize	 = 4*3*3*sizeof(qcd_complex_16);
	ftmp		 = (char*) malloc(((unsigned int) chunksize*nvh*2));

	if	(ftmp == NULL)
	{
		fprintf(stderr,"Error in qcd_getGaugeLime! Out of memory, couldn't alloc %u bytes\n", (unsigned int) (chunksize*nvh*2));
		return	1;
	}
	else
		printfQuda	("%d bytes reserved for gauge fields (%d x %u)\n", chunksize*nvh*2, chunksize, nvh*2);

	if	(MPI_File_read_all(mpifid, ftmp, 4*3*3*2*nvh*2, MPI_DOUBLE, &status) == 1)
		printf  	("Error in MPI_File_read_all\n");
	
	if      (4*3*3*2*nvh*2*sizeof(double) > 2147483648)
	{
		printfQuda	("File too large. At least %lu processes are needed to read thils file properly.\n", (4*3*3*2*nvh*2*sizeof(double)/2147483648)+1);
		printfQuda	("If some results are wrong, try increasing the number of MPI processes.\n");
	}

	if	(!qcd_isBigEndian())
		qcd_swap_8	((double*) ftmp,2*4*3*3*nvh*2);
#else
	ftmp	 = (double*)malloc(lvol*72*sizeof(double));

	if	(ftmp == NULL)
	{
		fprintf	(stderr, "Error, could not alloc ftmp\n");
		return	6;
	}
 
	iread	 = fread	(ftmp, sizeof(double), 72*lvol, fid);

	if	(iread != 72*lvol)
	{
		fprintf(stderr, "Error, could not read proper amount of data\n");
		return	7;
	}

	fclose	(fid);

	if	(!qcd_isBigEndian())      
        	qcd_swap_8	((double*) ftmp,72*lvol);
#endif

	// reconstruct gauge field
	// - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
	//   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1

	iy	 = 0;
	ixh	 = 0;

#ifdef	MPI_READCONF
	i	 = 0;
#endif

	int lx1 = 0, lx2 = 0, lx3 = 0, lx4 = 0;

	for(x4 = x4start; x4 < x4end; x4++) 
	{  // t
		for(x3 = x3start; x3 < x3end; x3++) 
		{  // z
			for(x2 = x2start; x2 < x2end; x2++) 
			{  // y
				for(x1 = x1start; x1 < x1end; x1++) 
				{  // x
					int oddBit	 = (x1+x2+x3+x4) & 1;

//					iy		 = x1+x2*ln[0]+x3*ln[0]*ln[1]+x4*ln[0]*ln[1]*ln[2];
#ifdef	MPI_READCONF
					iy		 = ((x1-x1start)+(x2-x2start)*param->X[0]+(x3-x3start)*param->X[1]*param->X[0]+(x4-x4start)*param->X[0]*param->X[1]*param->X[2])/2;
#else
					iy		 = x1+x2*param->X[0]+x3*param->X[1]*param->X[0]+x4*param->X[0]*param->X[1]*param->X[2];
#endif
					for(mu = 0; mu < 4; mu++)
					{  
#ifdef	MPI_READCONF
						memcpy(U, &(ftmp[i]), 18*sizeof(double));

						if(oddBit)
							memcpy(&(resOdd[mu][18*iy]), U, 18*sizeof(double));
						else
							memcpy(&(resEvn[mu][18*iy]), U, 18*sizeof(double));

						i	+= 144;
#else
						ixh		 = (lx1+lx2*param->X[0]+lx3*param->X[0]*param->X[1]+lx4*param->X[0]*param->X[1]*param->X[2])/2;


						idx = mu*18 + 72*iy;

						double *links_ptr = ftmp+idx;

						memcpy(U, links_ptr, 18*sizeof(double));

						if(oddBit)
							memcpy(resOdd[mu]+18*ixh, U, 18*sizeof(double));
						else
							memcpy(resEvn[mu]+18*ixh, U, 18*sizeof(double));
#endif
					}

					++lx1;
				}

				lx1	 = 0;
				++lx2;
			}

			lx2	 = 0;
			++lx3;				
		}

		lx3	 = 0;
		++lx4;
	}

	free(ftmp);

#ifdef	MPI_READCONF
	MPI_File_close(&mpifid);
#endif  

	//	Apply BC here

	applyGaugeFieldScaling<double>((double**)gauge, nvh, param);
  
	return	0;
}

	/*	End Alex	*/
