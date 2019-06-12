#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#include <comm_quda.h>

// This contains the appropriate ifdef guards already
#include <mpi_comm_handle.h>

#include <wilson_dslash_reference.h>
#include <test_util.h>

#include <dslash_quda.h>
#include "misc.h"

using namespace std;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3


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

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */
int gridsize_from_cmdline[4] = {1,1,1,1};

void get_gridsize_from_env(int *const dims)
{
  char *grid_size_env = getenv("QUDA_TEST_GRID_SIZE");
  if (grid_size_env) {
    std::stringstream grid_list(grid_size_env);

    int dim;
    int i = 0;
    while (grid_list >> dim) {
      if (i >= 4) errorQuda("Unexpected grid size array length");
      dims[i] = dim;
      if (grid_list.peek() == ',') grid_list.ignore();
      i++;
    }
  }
}

static int lex_rank_from_coords_t(const int *coords, void *fdata)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

static int lex_rank_from_coords_x(const int *coords, void *fdata)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

static int rank_order = 0;

void initComms(int argc, char **argv, int *const commDims)
{
  if (getenv("QUDA_TEST_GRID_SIZE")) get_gridsize_from_env(commDims);

#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = { 3, 1, 2, 0 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = { 0, 1, 2, 3 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQuda(4, commDims, func, NULL);
  initRand();

  printfQuda("Rank order is %s major (%s running fastest)\n",
	     rank_order == 0 ? "column" : "row", rank_order == 0 ? "t" : "x");

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
  for (int d = 0; d < 4; d++) {
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

int fullLatticeIndex(int dim[4], int index, int oddBit){

  int za = index/(dim[0]>>1);
  int zb = za/dim[1];
  int x2 = za - zb*dim[1];
  int x4 = zb/dim[2];
  int x3 = zb - x4*dim[2];

  return  2*index + ((x2 + x3 + x4 + oddBit) & 1);
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
  int X = 2 * sid + x1odd;

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


int neighborIndex(int dim[4], int index, int oddBit, int dx[4]){

  const int fullIndex = fullLatticeIndex(dim, index, oddBit);

  int x[4];
  x[3] = fullIndex/(dim[2]*dim[1]*dim[0]);
  x[2] = (fullIndex/(dim[1]*dim[0])) % dim[2];
  x[1] = (fullIndex/dim[0]) % dim[1];
  x[0] = fullIndex % dim[0];

  for(int dir=0; dir<4; ++dir)
    x[dir] = (x[dir]+dx[dir]+dim[dir]) % dim[dir];

  return (((x[3]*dim[2] + x[2])*dim[1] + x[1])*dim[0] + x[0])/2;
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

  if ( (ghost_x4 >= 0 && ghost_x4 < Z[3]) || !comm_dim_partitioned(3)){
    ret = (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
  }else{
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  }

  return ret;
}

/*
 * This is a computation of neighbor using the full index and the displacement in each direction
 *
 */

int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1)
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
neighborIndexFullLattice(int dim[4], int index, int dx[4])
{
  const int volume = dim[0]*dim[1]*dim[2]*dim[3];
  const int halfVolume = volume/2;
  int oddBit = 0;
  int halfIndex = index;

  if(index >= halfVolume){
    oddBit = 1;
    halfIndex = index - halfVolume;
  }

  int neighborHalfIndex = neighborIndex(dim, halfIndex, oddBit, dx);

  int oddBitChanged = (dx[0]+dx[1]+dx[2]+dx[3])%2;
  if(oddBitChanged){
    oddBit = 1 - oddBit;
  }

  return neighborHalfIndex + oddBit*halfVolume;
}

int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1)
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
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
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
  int X = 2 * sid + x1odd;

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

int fullLatticeIndex_5d_4dpc(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int x4_from_full_index(int i)
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
void applyGaugeFieldScaling_long(Float **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type)
{

  int X1h=param->X[0]/2;
  int X1 =param->X[0];
  int X2 =param->X[1];
  int X3 =param->X[2];
  int X4 =param->X[3];

  // rescale long links by the appropriate coefficient
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    for(int d=0; d<4; d++){
      for(int i=0; i < V*gaugeSiteSize; i++){
	gauge[d][i] /= (-24*param->tadpole_coeff*param->tadpole_coeff);
      }
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
      int sign = 1;

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

      for (int j=0; j < 18; j++) {
	gauge[d][i*gaugeSiteSize + j] *= sign;
      }
    }
    //odd
    for (int i = 0; i < Vh; i++) {
      int index = fullLatticeIndex(i, 1);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      int sign = 1;

      if (d == 0) {
	if (i4 % 2 == 1){
	  sign = -1;
	}
      }

      if (d == 1){
	if ((i4+i1) % 2 == 1){
	  sign = -1;
	}
      }
      if (d == 2){
	if ( (i4+i1+i2) % 2 == 1){
	  sign = -1;
	}
      }

      for (int j=0; j<18; j++){
	gauge[d][(Vh+i)*gaugeSiteSize + j] *= sign;
      }
    }

  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T) {
    for (int j = 0; j < Vh; j++) {
      int sign =1;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
	if (j >= (X4-3)*X1h*X2*X3 ){
	  sign = -1;
	}
      } else {
	if (j >= (X4-1)*X1h*X2*X3 ){
	  sign = -1;
	}
      }

      for (int i=0; i<18; i++) {
	gauge[3][j*gaugeSiteSize + i] *= sign;
	gauge[3][(Vh+j)*gaugeSiteSize + i] *= sign;
      }
    }
  }

}

void applyGaugeFieldScaling_long(void **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type, QudaPrecision local_prec) {
  if (local_prec == QUDA_DOUBLE_PRECISION) {
    applyGaugeFieldScaling_long((double**)gauge, Vh, param, dslash_type);
  } else if (local_prec == QUDA_SINGLE_PRECISION) {
    applyGaugeFieldScaling_long((float**)gauge, Vh, param, dslash_type);
  } else {
    errorQuda("Invalid type %d for applyGaugeFieldScaling_long\n", local_prec);
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
static void constructGaugeField(Float **res, QudaGaugeParam *param, QudaDslashType dslash_type = QUDA_WILSON_DSLASH)
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
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
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

  if (param->type == QUDA_WILSON_LINKS) {
    applyGaugeFieldScaling(res, Vh, param);
  } else if (param->type == QUDA_ASQTAD_LONG_LINKS) {
    applyGaugeFieldScaling_long(res, Vh, param, dslash_type);
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS) {
    for (int dir = 0; dir < 4; dir++) {
      for (int i = 0; i < Vh; i++) {
	for (int m = 0; m < 3; m++) { // last 2 rows
	  for (int n = 0; n < 3; n++) { // 3 columns
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] =1.0* rand() / (Float)RAND_MAX;
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 2.0* rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = 3.0*rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 4.0*rand() / (Float)RAND_MAX;
	  }
	}
      }
    }
  }
}

template <typename Float> void constructUnitaryGaugeField(Float **res)
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
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
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

template <typename Float> static void applyStaggeredScaling(Float **res, QudaGaugeParam *param, int type)
{

  if(type == 3)  applyGaugeFieldScaling_long((Float**)res, Vh, param, QUDA_STAGGERED_DSLASH);

  return;
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
    else
      applyGaugeFieldScaling((float **)gauge, Vh, param);
  }

}

void construct_fat_long_gauge_field(void **fatlink, void **longlink, int type, QudaPrecision precision,
    QudaGaugeParam *param, QudaDslashType dslash_type)
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
      // if doing naive staggered then set to long links so that the staggered phase is applied
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if(type != 3) constructGaugeField((double**)fatlink, param, dslash_type);
      else applyStaggeredScaling((double**)fatlink, param, type);
      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH)
      {
        if(type != 3) constructGaugeField((double**)longlink, param, dslash_type);
        else applyStaggeredScaling((double**)longlink, param, type);
      }
    }else {
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if(type != 3) constructGaugeField((float**)fatlink, param, dslash_type);
      else applyStaggeredScaling((float**)fatlink, param, type);

      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if(type != 3) constructGaugeField((float**)longlink, param, dslash_type);
        else applyStaggeredScaling((float**)longlink, param, type);
      }
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      // incorporate non-trivial phase into long links
      const double phase = (M_PI * rand()) / RAND_MAX;
      const complex<double> z = polar(1.0, phase);
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gaugeSiteSize; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              complex<double> *l = (complex<double> *)(&(((double *)longlink[dir])[i * gaugeSiteSize + j]));
              *l *= z;
            } else {
              complex<float> *l = (complex<float> *)(&(((float *)longlink[dir])[i * gaugeSiteSize + j]));
              *l *= z;
            }
          }
        }
      }
    }

    if (type == 3) return;

    // set all links to zero to emulate the 1-link operator (needed for host comparison)
    if (dslash_type == QUDA_STAGGERED_DSLASH) {
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gaugeSiteSize; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              ((double *)longlink[dir])[i * gaugeSiteSize + j] = 0.0;
              ((double *)longlink[dir])[i * gaugeSiteSize + j + 1] = 0.0;
            } else {
              ((float *)longlink[dir])[i * gaugeSiteSize + j] = 0.0;
              ((float *)longlink[dir])[i * gaugeSiteSize + j + 1] = 0.0;
            }
          }
        }
      }
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

    //impose clover symmetry on each chiral block
    for (int ch=0; ch<2; ch++) {
      res[i*72 + 3 + 36*ch] = -res[i*72 + 0 + 36*ch];
      res[i*72 + 4 + 36*ch] = -res[i*72 + 1 + 36*ch];
      res[i*72 + 5 + 36*ch] = -res[i*72 + 2 + 36*ch];
      res[i*72 + 30 + 36*ch] = -res[i*72 + 6 + 36*ch];
      res[i*72 + 31 + 36*ch] = -res[i*72 + 7 + 36*ch];
      res[i*72 + 32 + 36*ch] = -res[i*72 + 8 + 36*ch];
      res[i*72 + 33 + 36*ch] = -res[i*72 + 9 + 36*ch];
      res[i*72 + 34 + 36*ch] = -res[i*72 + 16 + 36*ch];
      res[i*72 + 35 + 36*ch] = -res[i*72 + 17 + 36*ch];
    }

    for (int j = 0; j<6; j++) {
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
    printf("%e Failures = (%9d, %9d, %9d, %9d) = (%6.5f, %6.5f, %6.5f, %6.5f)\n", pow(10.0, -(f + 1)), fail[0][f],
           fail[1][f], fail[2][f], fail[3][f], fail[0][f] / (double)(V * 18), fail[1][f] / (double)(V * 18),
           fail[2][f] / (double)(V * 18), fail[3][f] / (double)(V * 18));
  }

}

void check_gauge(void **oldG, void **newG, double epsilon, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION)
    checkGauge((double**)oldG, (double**)newG, epsilon);
  else
    checkGauge((float**)oldG, (float**)newG, epsilon);
}

void createSiteLinkCPU(void **link, QudaPrecision precision, int phase)
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
        int i1 = full_idx - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;

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

void construct_spinor_source(void *v, int nSpin, int nColor, QudaPrecision precision, const int *const x, quda::RNG &rng)
{
  quda::ColorSpinorParam param;
  param.v = v;
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.setPrecision(precision);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.nDim = 4;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  for (int d = 0; d < 4; d++) param.x[d] = x[d];
  quda::cpuColorSpinorField spinor_in(param);

  quda::spinorNoise(spinor_in, rng, QUDA_NOISE_UNIFORM);
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

  if (precision == QUDA_DOUBLE_PRECISION) {
    ret = compareLink((double**)linkA, (double**)linkB, len);
  } else {
    ret = compareLink((float**)linkA, (float**)linkB, len);
  }

  return ret;
}


// X indexes the lattice site
static void printLinkElement(void *link, int X, QudaPrecision precision)
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

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec)
{
  printfQuda("%s\n", msgA);
  printLinkElement(linkA[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkA[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkA[3], len - 1, prec);
  printfQuda("\n");

  printfQuda("\n%s\n", msgB);
  printLinkElement(linkB[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkB[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkB[3], len - 1, prec);
  printfQuda("\n");

  int ret = compare_link(linkA, linkB, len, prec);
  return ret;
}

void createMomCPU(void *mom, QudaPrecision precision)
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
        double *thismom = (double *)mom;
        for(int k=0; k < momSiteSize; k++){
          thismom[(4 * i + dir) * momSiteSize + k] = 1.0 * rand() / RAND_MAX;
          if (k==momSiteSize-1) thismom[ (4*i+dir)*momSiteSize + k ]= 0.0;
        }
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thismom=(float*)mom;
	for(int k=0; k < momSiteSize; k++){
          thismom[(4 * i + dir) * momSiteSize + k] = 1.0 * rand() / RAND_MAX;
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
    for (int j=0; j<momSiteSize-1; j++) {
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
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*9, fail[f]/(double)(len*9));
  }

  return accuracy_level;
}

static void printMomElement(void *mom, int X, QudaPrecision precision)
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
int strong_check_mom(void *momA, void *momB, int len, QudaPrecision prec)
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

QudaReconstructType link_recon = QUDA_RECONSTRUCT_NO;
QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_INVALID;
QudaReconstructType link_recon_precondition = QUDA_RECONSTRUCT_INVALID;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision prec_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_refinement_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_precondition = QUDA_INVALID_PRECISION;
QudaPrecision prec_null = QUDA_INVALID_PRECISION;
QudaPrecision prec_ritz = QUDA_INVALID_PRECISION;
QudaVerbosity verbosity = QUDA_SUMMARIZE;
int xdim = 24;
int ydim = 24;
int zdim = 24;
int tdim = 24;
int Lsdim = 16;
QudaDagType dagger = QUDA_DAG_NO;
QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
int laplace3D = 4;
char latfile[256] = "";
bool unit_gauge = false;
char gauge_outfile[256] = "";
int Nsrc = 1;
int Msrc = 1;
int niter = 100;
int gcrNkrylov = 10;
QudaCABasis ca_basis = QUDA_POWER_BASIS;
double ca_lambda_min = 0.0;
double ca_lambda_max = -1.0;
int pipeline = 0;
int solution_accumulator_pipeline = 0;
int test_type = 0;
int nvec[QUDA_MAX_MG_LEVEL] = { };
char vec_infile[256] = "";
char vec_outfile[256] = "";
QudaInverterType inv_type;
QudaInverterType precon_type = QUDA_INVALID_INVERTER;
int multishift = 0;
bool verify_results = true;
bool low_mode_check = false;
bool oblique_proj_check = false;
double mass = 0.1;
double kappa = -1.0;
double mu = 0.1;
double epsilon = 0.01;
double anisotropy = 1.0;
double tadpole_factor = 1.0;
double eps_naik = 0.0;
double clover_coeff = 0.1;
bool compute_clover = false;
bool compute_fatlong = false;
double tol = 1e-7;
double tol_hq = 0.;
double reliable_delta = 0.1;
bool alternative_reliable = false;
QudaTwistFlavorType twist_flavor = QUDA_TWIST_SINGLET;
QudaMassNormalization normalization = QUDA_KAPPA_NORMALIZATION;
QudaMatPCType matpc_type = QUDA_MATPC_EVEN_EVEN;
QudaSolveType solve_type = QUDA_NORMOP_PC_SOLVE;
QudaSolutionType solution_type = QUDA_MAT_SOLUTION;

int mg_levels = 2;

QudaFieldLocation solver_location[QUDA_MAX_MG_LEVEL] = { };
QudaFieldLocation setup_location[QUDA_MAX_MG_LEVEL] = { };

int nu_pre[QUDA_MAX_MG_LEVEL] = { };
int nu_post[QUDA_MAX_MG_LEVEL] = { };
int n_block_ortho[QUDA_MAX_MG_LEVEL] = {};
double mu_factor[QUDA_MAX_MG_LEVEL] = { };
QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL] = { };
QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL] = { };
QudaSolveType coarse_solve_type[QUDA_MAX_MG_LEVEL] = { };
QudaSolveType smoother_solve_type[QUDA_MAX_MG_LEVEL] = { };
int num_setup_iter[QUDA_MAX_MG_LEVEL] = { };
double setup_tol[QUDA_MAX_MG_LEVEL] = { };
int setup_maxiter[QUDA_MAX_MG_LEVEL] = { };
int setup_maxiter_refresh[QUDA_MAX_MG_LEVEL] = { };
QudaCABasis setup_ca_basis[QUDA_MAX_MG_LEVEL] = { };
int setup_ca_basis_size[QUDA_MAX_MG_LEVEL] = { };
double setup_ca_lambda_min[QUDA_MAX_MG_LEVEL] = { };
double setup_ca_lambda_max[QUDA_MAX_MG_LEVEL] = { };
QudaSetupType setup_type = QUDA_NULL_VECTOR_SETUP;
bool pre_orthonormalize = false;
bool post_orthonormalize = true;
double omega = 0.85;
QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL] = { };
double coarse_solver_tol[QUDA_MAX_MG_LEVEL] = { };
QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL] = { };
QudaPrecision smoother_halo_prec = QUDA_INVALID_PRECISION;
double smoother_tol[QUDA_MAX_MG_LEVEL] = { };
int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL] = { };
QudaCABasis coarse_solver_ca_basis[QUDA_MAX_MG_LEVEL] = { };
int coarse_solver_ca_basis_size[QUDA_MAX_MG_LEVEL] = { };
double coarse_solver_ca_lambda_min[QUDA_MAX_MG_LEVEL] = { };
double coarse_solver_ca_lambda_max[QUDA_MAX_MG_LEVEL] = { };
bool generate_nullspace = true;
bool generate_all_levels = true;
QudaSchwarzType schwarz_type[QUDA_MAX_MG_LEVEL] = { };
int schwarz_cycle[QUDA_MAX_MG_LEVEL] = { };

int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM] = { };
int nev = 8;
int max_search_dim = 64;
int deflation_grid = 16;
double tol_restart = 5e+3*tol;

int eigcg_max_restarts = 3;
int max_restart_num = 3;
double inc_tol = 1e-2;
double eigenval_tol = 1e-1;

QudaExtLibType solver_ext_lib     = QUDA_EIGEN_EXTLIB;
QudaExtLibType deflation_ext_lib  = QUDA_EIGEN_EXTLIB;
QudaFieldLocation location_ritz   = QUDA_CUDA_FIELD_LOCATION;
QudaMemoryType    mem_type_ritz   = QUDA_MEMORY_DEVICE;

// Parameters for the stand alone eigensolver
int eig_nEv = 16;
int eig_nKr = 32;
int eig_nConv = -1; // If unchanged, will be set to nEv
bool eig_require_convergence = true;
int eig_check_interval = 10;
int eig_max_restarts = 1000;
double eig_tol = 1e-6;
bool eig_use_poly_acc = true;
int eig_poly_deg = 100;
double eig_amin = 0.1;
double eig_amax = 4.0;
bool eig_use_normop = true;
bool eig_use_dagger = false;
bool eig_compute_svd = false;
QudaEigSpectrumType eig_spectrum = QUDA_SPECTRUM_LR_EIG;
QudaEigType eig_type = QUDA_EIG_TR_LANCZOS;
bool eig_arpack_check = false;
char eig_arpack_logfile[256] = "arpack_logfile.log";
char eig_QUDA_logfile[256] = "QUDA_logfile.log";
char eig_vec_infile[256] = "";
char eig_vec_outfile[256] = "";

// Parameters for the MG eigensolver.
// The coarsest grid params are for deflation,
// all others are for PR vectors.
bool mg_eig[QUDA_MAX_MG_LEVEL] = {};
int mg_eig_nEv[QUDA_MAX_MG_LEVEL] = {};
int mg_eig_nKr[QUDA_MAX_MG_LEVEL] = {};
bool mg_eig_require_convergence[QUDA_MAX_MG_LEVEL] = {};
int mg_eig_check_interval[QUDA_MAX_MG_LEVEL] = {};
int mg_eig_max_restarts[QUDA_MAX_MG_LEVEL] = {};
double mg_eig_tol[QUDA_MAX_MG_LEVEL] = {};
bool mg_eig_use_poly_acc[QUDA_MAX_MG_LEVEL] = {};
int mg_eig_poly_deg[QUDA_MAX_MG_LEVEL] = {};
double mg_eig_amin[QUDA_MAX_MG_LEVEL] = {};
double mg_eig_amax[QUDA_MAX_MG_LEVEL] = {};
bool mg_eig_use_normop[QUDA_MAX_MG_LEVEL] = {};
bool mg_eig_use_dagger[QUDA_MAX_MG_LEVEL] = {};
QudaEigSpectrumType mg_eig_spectrum[QUDA_MAX_MG_LEVEL] = {};
QudaEigType mg_eig_type[QUDA_MAX_MG_LEVEL] = {};
bool mg_eig_coarse_guess = false;

double heatbath_beta_value = 6.2;
int heatbath_warmup_steps = 10;
int heatbath_num_steps = 10;
int heatbath_num_heatbath_per_step = 5;
int heatbath_num_overrelax_per_step = 5;
bool heatbath_coldstart = false;

QudaContractType contract_type = QUDA_CONTRACT_TYPE_OPEN;

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

  // Problem size and type parameters
  printf("    --verbosity <silent/summurize/verbose>    # The the verbosity on the top level of QUDA( default summarize)\n");
  printf("    --prec <double/single/half>               # Precision in GPU\n");
  printf("    --prec-sloppy <double/single/half>        # Sloppy precision in GPU\n");
  printf("    --prec-refine <double/single/half>        # Sloppy precision for refinement in GPU\n");
  printf("    --prec-precondition <double/single/half>  # Preconditioner precision in GPU\n");
  printf("    --prec-ritz <double/single/half>  # Eigenvector precision in GPU\n");
  printf("    --recon <8/9/12/13/18>                    # Link reconstruction type\n");
  printf("    --recon-sloppy <8/9/12/13/18>             # Sloppy link reconstruction type\n");
  printf("    --recon-precondition <8/9/12/13/18>       # Preconditioner link reconstruction type\n");
  printf("    --dagger                                  # Set the dagger to 1 (default 0)\n");
  printf("    --dim <n>                                 # Set space-time dimension (X Y Z T)\n");
  printf("    --sdim <n>                                # Set space dimension(X/Y/Z) size\n");
  printf("    --xdim <n>                                # Set X dimension size(default 24)\n");
  printf("    --ydim <n>                                # Set X dimension size(default 24)\n");
  printf("    --zdim <n>                                # Set X dimension size(default 24)\n");
  printf("    --tdim <n>                                # Set T dimension size(default 24)\n");
  printf("    --Lsdim <n>                               # Set Ls dimension size(default 16)\n");
  printf("    --gridsize <x y z t>                      # Set the grid size in all four dimension (default 1 1 1 1)\n");
  printf("    --xgridsize <n>                           # Set grid size in X dimension (default 1)\n");
  printf("    --ygridsize <n>                           # Set grid size in Y dimension (default 1)\n");
  printf("    --zgridsize <n>                           # Set grid size in Z dimension (default 1)\n");
  printf("    --tgridsize <n>                           # Set grid size in T dimension (default 1)\n");
  printf("    --partition <mask>                        # Set the communication topology (X=1, Y=2, Z=4, T=8, and combinations of these)\n");
  printf("    --rank-order <col/row>                    # Set the [t][z][y][x] rank order as either column major (t fastest, default) or row major (x fastest)\n");
  printf("    --dslash-type <type>                      # Set the dslash type, the following values are valid\n"
	 "                                                  wilson/clover/twisted-mass/twisted-clover/staggered\n"
         "                                                  /asqtad/domain-wall/domain-wall-4d/mobius/laplace\n");
  printf("    --laplace3D <n>                           # Restrict laplace operator to omit the t dimension (n=3), or include all dims (n=4) (default 4)\n");
  printf("    --flavor <type>                           # Set the twisted mass flavor type (singlet (default), deg-doublet, nondeg-doublet)\n");
  printf("    --load-gauge file                         # Load gauge field \"file\" for the test (requires QIO)\n");
  printf("    --save-gauge file                         # Save gauge field \"file\" for the test (requires QIO, "
         "heatbath test only)\n");
  printf("    --unit-gauge <true/false>                 # Generate a unit valued gauge field in the tests. If false, a "
         "random gauge is generated (default false)\n");
  printf("    --niter <n>                               # The number of iterations to perform (default 10)\n");
  printf("    --ngcrkrylov <n>                          # The number of inner iterations to use for GCR, BiCGstab-l, CA-CG (default 10)\n");
  printf("    --ca-basis-type <power/chebyshev>         # The basis to use for CA-CG (default power)\n");
  printf("    --cheby-basis-eig-min                     # Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG (default 0)\n");
  printf("    --cheby-basis-eig-max                     # Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG (default is to guess with power iterations)\n");
  printf("    --pipeline <n>                            # The pipeline length for fused operations in GCR, BiCGstab-l (default 0, no pipelining)\n");
  printf("    --solution-pipeline <n>                   # The pipeline length for fused solution accumulation (default 0, no pipelining)\n");
  printf("    --inv-type <cg/bicgstab/gcr>              # The type of solver to use (default cg)\n");
  printf("    --precon-type <mr/ (unspecified)>         # The type of solver to use (default none (=unspecified)).\n");
  printf("    --multishift <true/false>                 # Whether to do a multi-shift solver test or not (default false)\n");
  printf("    --mass                                    # Mass of Dirac operator (default 0.1)\n");
  printf("    --kappa                                   # Kappa of Dirac operator (default 0.12195122... [equiv to mass])\n");
  printf("    --mu                                      # Twisted-Mass chiral twist of Dirac operator (default 0.1)\n");
  printf(
	 "    --epsilon                                 # Twisted-Mass flavor twist of Dirac operator (default 0.01)\n");
  printf("    --tadpole-coeff                           # Tadpole coefficient for HISQ fermions (default 1.0, recommended [Plaq]^1/4)\n");
  printf("    --epsilon-naik                            # Epsilon factor on Naik term (default 0.0, suggested non-zero -0.1)\n");
  printf("    --compute-clover                          # Compute the clover field or use random numbers (default false)\n");
  printf("    --compute-fat-long                        # Compute the fat/long field or use random numbers (default false)\n");
  printf("    --clover-coeff                            # Clover coefficient (default 1.0)\n");
  printf("    --anisotropy                              # Temporal anisotropy factor (default 1.0)\n");
  printf("    --mass-normalization                      # Mass normalization (kappa (default) / mass / asym-mass)\n");
  printf("    --matpc                                   # Matrix preconditioning type (even-even, odd-odd, even-even-asym, odd-odd-asym) \n");
  printf("    --solve-type                              # The type of solve to do (direct, direct-pc, normop, "
         "normop-pc, normerr, normerr-pc) \n");
  printf("    --solution-type                           # The solution we desire (mat (default), mat-dag-mat, mat-pc, "
         "mat-pc-dag-mat-pc (default for multi-shift))\n");
  printf("    --tol  <resid_tol>                        # Set L2 residual tolerance\n");
  printf("    --tolhq  <resid_hq_tol>                   # Set heavy-quark residual tolerance\n");
  printf("    --reliable-delta <delta>                  # Set reliable update delta factor\n");
  printf("    --test                                    # Test method (different for each test)\n");
  printf("    --verify <true/false>                     # Verify the GPU results using CPU results (default true)\n");

  // Multigrid
  printf("    --mg-low-mode-check <true/false>          # Measure how well the null vector subspace overlaps with the "
         "low eigenmode subspace (default false)\n");
  printf("    --mg-oblique-proj-check <true/false>      # Measure how well the null vector subspace adjusts the low "
         "eigenmode subspace (default false)\n");
  printf("    --mg-nvec <level nvec>                    # Number of null-space vectors to define the multigrid "
         "transfer operator on a given level\n"
         "                                                If using the eigensolver of the coarsest level then this "
         "dictates the size of the deflation space.\n");
  printf("    --mg-gpu-prolongate <true/false>          # Whether to do the multigrid transfer operators on the GPU (default false)\n");
  printf("    --mg-levels <2+>                          # The number of multigrid levels to do (default 2)\n");
  printf("    --mg-nu-pre <level 1-20>                  # The number of pre-smoother applications to do at a given multigrid level (default 2)\n");
  printf("    --mg-nu-post <level 1-20>                 # The number of post-smoother applications to do at a given multigrid level (default 2)\n");
  printf("    --mg-coarse-solve-type <level solve>      # The type of solve to do on each level (direct, direct-pc) (default = solve_type)\n");
  printf("    --mg-smoother-solve-type <level solve>    # The type of solve to do in smoother (direct, direct-pc (default) )\n");
  printf("    --mg-solve-location <level cpu/cuda>      # The location where the multigrid solver will run (default cuda)\n");
  printf("    --mg-setup-location <level cpu/cuda>      # The location where the multigrid setup will be computed (default cuda)\n");
  printf("    --mg-setup-inv <level inv>                # The inverter to use for the setup of multigrid (default bicgstab)\n");
  printf("    --mg-setup-maxiter <level iter>           # The maximum number of solver iterations to use when relaxing on a null space vector (default 500)\n");
  printf("    --mg-setup-maxiter-refresh <level iter>   # The maximum number of solver iterations to use when refreshing the pre-existing null space vectors (default 100)\n");
  printf("    --mg-setup-iters <level iter>             # The number of setup iterations to use for the multigrid (default 1)\n");
  printf("    --mg-setup-tol <level tol>                # The tolerance to use for the setup of multigrid (default 5e-6)\n");
  printf("    --mg-setup-ca-basis-type <level power/chebyshev> # The basis to use for CA-CG setup of multigrid(default power)\n");
  printf("    --mg-setup-ca-basis-size <level size>     # The basis size to use for CA-CG setup of multigrid (default 4)\n");
  printf("    --mg-setup-cheby-basis-eig-min <level val> # Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default 0)\n");
  printf("    --mg-setup-cheby-basis-eig-max <level val> # Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default is to guess with power iterations)\n");
  printf("    --mg-setup-type <null/test>               # The type of setup to use for the multigrid (default null)\n");
  printf("    --mg-pre-orth <true/false>                # If orthonormalize the vector before inverting in the setup of multigrid (default false)\n");
  printf("    --mg-post-orth <true/false>               # If orthonormalize the vector after inverting in the setup of multigrid (default true)\n");
  printf("    --mg-n-block-ortho <level n>              # The number of times to run Gram-Schmidt during block "
         "orthonormalization (default 1)\n");
  printf("    --mg-omega                                # The over/under relaxation factor for the smoother of multigrid (default 0.85)\n");
  printf("    --mg-coarse-solver <level gcr/etc.>       # The solver to wrap the V cycle on each level (default gcr, only for levels 1+)\n");
  printf("    --mg-coarse-solver-tol <level gcr/etc.>   # The coarse solver tolerance for each level (default 0.25, only for levels 1+)\n");
  printf("    --mg-coarse-solver-maxiter <level n>      # The coarse solver maxiter for each level (default 100)\n");
  printf("    --mg-coarse-solver-ca-basis-type <level power/chebyshev> # The basis to use for CA-CG setup of multigrid(default power)\n");
  printf("    --mg-coarse-solver-ca-basis-size <level size>  # The basis size to use for CA-CG setup of multigrid (default 4)\n");
  printf("    --mg-coarse-solver-cheby-basis-eig-min <level val> # Conservative estimate of smallest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default 0)\n");
  printf("    --mg-coarse-solver-cheby-basis-eig-max <level val> # Conservative estimate of largest eigenvalue for Chebyshev basis CA-CG in setup of multigrid (default is to guess with power iterations)\n");
  printf("    --mg-smoother <level mr/etc.>             # The smoother to use for multigrid (default mr)\n");
  printf("    --mg-smoother-tol <level resid_tol>       # The smoother tolerance to use for each multigrid (default 0.25)\n");
  printf("    --mg-smoother-halo-prec                   # The smoother halo precision (applies to all levels - defaults to null_precision)\n");
  printf("    --mg-schwarz-type <level false/add/mul>   # Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver) (default false)\n");
  printf("    --mg-schwarz-cycle <level cycle>          # The number of Schwarz cycles to apply per smoother application (default=1)\n");
  printf("    --mg-block-size <level x y z t>           # Set the geometric block size for the each multigrid level's transfer operator (default 4 4 4 4)\n");
  printf("    --mg-mu-factor <level factor>             # Set the multiplicative factor for the twisted mass mu parameter on each level (default 1)\n");
  printf("    --mg-generate-nullspace <true/false>      # Generate the null-space vector dynamically (default true, if set false and mg-load-vec isn't set, creates free-field null vectors)\n");
  printf("    --mg-generate-all-levels <true/talse>     # true=generate null-space on all levels, false=generate on level 0 and create other levels from that (default true)\n");
  printf("    --mg-load-vec file                        # Load the vectors \"file\" for the multigrid_test (requires QIO)\n");
  printf("    --mg-save-vec file                        # Save the generated null-space vectors \"file\" from the "
         "multigrid_test (requires QIO)\n");
  printf("    --mg-verbosity <level verb>               # The verbosity to use on each level of the multigrid (default "
         "summarize)\n");

  // Deflated solvers
  printf("    --df-nev <nev>                            # Set number of eigenvectors computed within a single solve cycle (default 8)\n");
  printf("    --df-max-search-dim <dim>                 # Set the size of eigenvector search space (default 64)\n");
  printf("    --df-deflation-grid <n>                   # Set maximum number of cycles needed to compute eigenvectors(default 1)\n");
  printf("    --df-eigcg-max-restarts <n>               # Set how many iterative refinement cycles will be solved with eigCG within a single physical right hand site solve (default 4)\n");
  printf("    --df-tol-restart <tol>                    # Set tolerance for the first restart in the initCG solver(default 5e-5)\n");
  printf("    --df-tol-inc <tol>                        # Set tolerance for the subsequent restarts in the initCG solver  (default 1e-2)\n");
  printf("    --df-max-restart-num <n>                  # Set maximum number of the initCG restarts in the deflation stage (default 3)\n");
  printf("    --df-tol-eigenval <tol>                   # Set maximum eigenvalue residual norm (default 1e-1)\n");

  printf("    --solver-ext-lib-type <eigen/magma>       # Set external library for the solvers  (default Eigen library)\n");
  printf("    --df-ext-lib-type <eigen/magma>           # Set external library for the deflation methods  (default Eigen library)\n");
  printf("    --df-location-ritz <host/cuda>            # Set memory location for the ritz vectors  (default cuda memory location)\n");
  printf("    --df-mem-type-ritz <device/pinned/mapped> # Set memory type for the ritz vectors  (default device memory type)\n");

  // Eigensolver
  printf("    --eig-nEv <n>                             # The size of eigenvector search space in the eigensolver\n");
  printf("    --eig-nKr <n>                             # The size of the Krylov subspace to use in the eigensolver\n");
  printf("    --eig-nConv <n>                           # The number of converged eigenvalues requested\n");
  printf("    --eig-require-convergence  <true/false>   # If true, the solver will error out if convergence is not "
         "attained. If false, a warning will be given (default true)\n");
  printf("    --eig-check-interval <n>                  # Perform a convergence check every nth restart/iteration in "
         "the IRAM,IRLM/lanczos,arnoldi eigensolver\n");
  printf("    --eig-max-restarts <n>                    # Perform n iterations of the restart in the eigensolver\n");
  printf("    --eig-tol <tol>                           # The tolerance to use in the eigensolver\n");
  printf("    --eig-use-poly-acc <true/false>           # Use Chebyshev polynomial acceleration in the eigensolver\n");
  printf("    --eig-poly-deg <n>                        # Set the degree of the Chebyshev polynomial acceleration in "
         "the eigensolver\n");
  printf("    --eig-amin <Float>                        # The minimum in the polynomial acceleration\n");
  printf("    --eig-amax <Float>                        # The maximum in the polynomial acceleration\n");
  printf("    --eig-use-normop <true/false>             # Solve the MdagM problem instead of M (MMdag if "
         "eig-use-dagger == true) (default false)\n");
  printf("    --eig-use-dagger <true/false>             # Solve the Mdag  problem instead of M (MMdag if "
         "eig-use-normop == true) (default false)\n");
  printf("    --eig-compute-svd <true/false>            # Solve the MdagM problem, use to compute SVD of M (default "
         "false)\n");
  printf("    --eig-spectrum <SR/LR/SM/LM/SI/LI>        # The spectrum part to be calulated. S=smallest L=largest "
         "R=real M=modulus I=imaginary\n");
  printf("    --eig-type <eigensolver>                  # The type of eigensolver to use (default trlm)\n");
  printf("    --eig-QUDA-logfile <file_name>            # The filename storing the stdout from the QUDA eigensolver\n");
  printf("    --eig-arpack-check <true/false>           # Cross check the device data against ARPACK (requires ARPACK, "
         "default false)\n");
  printf("    --eig-load-vec <file>                     # Load eigenvectors to <file> (requires QIO)\n");
  printf("    --eig-save-vec <file>                     # Save eigenvectors to <file> (requires QIO)\n");

  // Multigrid Eigensolver
  printf("    --mg-eig <level> <true/false>                     # Use the eigensolver on this level (default false)\n");
  printf("    --mg-eig-nEv <level> <n>                          # The size of eigenvector search space in the "
         "eigensolver\n");
  printf("    --mg-eig-nKr <level> <n>                          # The size of the Krylov subspace to use in the "
         "eigensolver\n");
  printf("    --mg-eig-require-convergence <level> <true/false> # If true, the solver will error out if convergence is "
         "not attained. If false, a warning will be given (default true)\n");
  printf("    --mg-eig-check-interval <level> <n>               # Perform a convergence check every nth "
         "restart/iteration (only used in Implicit Restart types)\n");
  printf("    --mg-eig-max-restarts <level> <n>                 # Perform a maximun of n restarts in eigensolver "
         "(default 100)\n");
  printf("    --mg-eig-use-normop <level> <true/false>          # Solve the MdagM problem instead of M (MMdag if "
         "eig-use-dagger == true) (default false)\n");
  printf("    --mg-eig-use-dagger <level> <true/false>          # Solve the MMdag problem instead of M (MMdag if "
         "eig-use-normop == true) (default false)\n");
  printf(
    "    --mg-eig-tol <level> <tol>                        # The tolerance to use in the eigensolver (default 1e-6)\n");
  printf("    --mg-eig-use-poly-acc <level> <true/false>        # Use Chebyshev polynomial acceleration in the "
         "eigensolver (default true)\n");
  printf("    --mg-eig-poly-deg <level> <n>                     # Set the degree of the Chebyshev polynomial (default "
         "100)\n");
  printf("    --mg-eig-amin <level> <Float>                     # The minimum in the polynomial acceleration (default "
         "0.1)\n");
  printf("    --mg-eig-amax <level> <Float>                     # The maximum in the polynomial acceleration (default "
         "4.0)\n");
  printf("    --mg-eig-spectrum <level> <SR/LR/SM/LM/SI/LI>     # The spectrum part to be calulated. S=smallest "
         "L=largest R=real M=modulus I=imaginary (default SR)\n");
  printf("    --mg-eig-coarse-guess <true/false>                # If deflating on the coarse grid, optionaly use an "
         "initial guess (default = false)\n");
  printf("    --mg-eig-type <level> <eigensolver>               # The type of eigensolver to use (default trlm)\n");

  // Miscellanea
  printf("    --nsrc <n>                                # How many spinors to apply the dslash to simultaneusly (experimental for staggered only)\n");
  printf("    --msrc <n>                                # Used for testing non-square block blas routines where nsrc defines the other dimension\n");
  printf("    --heatbath-beta <beta>                    # Beta value used in heatbath test (default 6.2)\n");
  printf("    --heatbath-warmup-steps <n>               # Number of warmup steps in heatbath test (default 10)\n");
  printf("    --heatbath-num-steps <n>                  # Number of measurement steps in heatbath test (default 10)\n");
  printf("    --heatbath-num-hb-per-step <n>            # Number of heatbath hits per heatbath step (default 5)\n");
  printf("    --heatbath-num-or-per-step <n>            # Number of overrelaxation hits per heatbath step (default 5)\n");
  printf("    --heatbath-coldstart <true/false>         # Whether to use a cold or hot start in heatbath test (default false)\n");

  printf("    --contraction-type <open/dr/dp>           # Whether to leave spin elemental open, or use a gamma basis "
         "and contract on spin (default open)\n");

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

  if( strcmp(argv[i], "--verify") == 0){
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i+1], "true") == 0){
      verify_results = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      verify_results = false;
    }else{
      fprintf(stderr, "ERROR: invalid verify type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-low-mode-check") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      low_mode_check = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      low_mode_check = false;
    } else {
      fprintf(stderr, "ERROR: invalid low_mode_check type (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-oblique-proj-check") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      oblique_proj_check = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      oblique_proj_check = false;
    } else {
      fprintf(stderr, "ERROR: invalid oblique_proj_check type (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
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

  if( strcmp(argv[i], "--verbosity") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    verbosity =  get_verbosity_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec") == 0){
    if (i + 1 >= argc) { usage(argv); }
    prec =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-sloppy") == 0){
    if (i + 1 >= argc) { usage(argv); }
    prec_sloppy =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-refine") == 0){
    if (i + 1 >= argc) { usage(argv); }
    prec_refinement_sloppy =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-precondition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    prec_precondition =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-null") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    prec_null =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-ritz") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    prec_ritz =  get_prec(argv[i+1]);
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

  if( strcmp(argv[i], "--recon-sloppy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    link_recon_sloppy =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--recon-precondition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    link_recon_precondition =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--dim") == 0){
    if (i+4 >= argc){
      usage(argv);
    }
    xdim= atoi(argv[i+1]);
    if (xdim < 0 || xdim > 512){
      printf("ERROR: invalid X dimension (%d)\n", xdim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    ydim= atoi(argv[i+1]);
    if (ydim < 0 || ydim > 512){
      printf("ERROR: invalid Y dimension (%d)\n", ydim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    zdim= atoi(argv[i+1]);
    if (zdim < 0 || zdim > 512){
      printf("ERROR: invalid Z dimension (%d)\n", zdim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    tdim= atoi(argv[i+1]);
    if (tdim < 0 || tdim > 512){
      printf("ERROR: invalid T dimension (%d)\n", tdim);
      usage(argv);
    }
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
    if (i + 1 >= argc) { usage(argv); }
    tdim =  atoi(argv[i+1]);
    if (tdim < 0 || tdim > 512){
      printf("Error: invalid t dimension");
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--sdim") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int sdim =  atoi(argv[i+1]);
    if (sdim < 0 || sdim > 512){
      printf("ERROR: invalid S dimension\n");
      usage(argv);
    }
    xdim=ydim=zdim=sdim;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--Lsdim") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int Ls =  atoi(argv[i+1]);
    if (Ls < 0 || Ls > 128){
      printf("ERROR: invalid Ls dimension\n");
      usage(argv);
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

  if( strcmp(argv[i], "--multishift") == 0){
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i+1], "true") == 0){
      multishift = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      multishift = false;
    }else{
      fprintf(stderr, "ERROR: invalid multishift boolean\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--gridsize") == 0){
    if (i + 4 >= argc) { usage(argv); }
    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X grid size");
      usage(argv);
    }
    gridsize_from_cmdline[0] = xsize;
    i++;

    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y grid size");
      usage(argv);
    }
    gridsize_from_cmdline[1] = ysize;
    i++;

    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z grid size");
      usage(argv);
    }
    gridsize_from_cmdline[2] = zsize;
    i++;

    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T grid size");
      usage(argv);
    }
    gridsize_from_cmdline[3] = tsize;
    i++;

    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--xgridsize") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X grid size");
      usage(argv);
    }
    gridsize_from_cmdline[0] = xsize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ygridsize") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y grid size");
      usage(argv);
    }
    gridsize_from_cmdline[1] = ysize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--zgridsize") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z grid size");
      usage(argv);
    }
    gridsize_from_cmdline[2] = zsize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tgridsize") == 0){
    if (i + 1 >= argc) { usage(argv); }
    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T grid size");
      usage(argv);
    }
    gridsize_from_cmdline[3] = tsize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--rank-order") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    rank_order = get_rank_order(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--dslash-type") == 0){
    if (i + 1 >= argc) { usage(argv); }
    dslash_type = get_dslash_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--laplace3D") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    laplace3D = atoi(argv[i + 1]);
    if (laplace3D > 4 || laplace3D < 3) {
      printf("ERROR: invalid transverse dim %d given. Please use 3 or 4 for t,none.\n", laplace3D);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--flavor") == 0){
    if (i + 1 >= argc) { usage(argv); }
    twist_flavor = get_flavor_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--inv-type") == 0){
    if (i + 1 >= argc) { usage(argv); }
    inv_type = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--precon-type") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    precon_type = get_solver_type(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mass") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    mass = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--kappa") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    kappa = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--compute-clover") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      compute_clover = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      compute_clover = false;
    }else{
      fprintf(stderr, "ERROR: invalid compute_clover type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--clover-coeff") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    clover_coeff = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mu") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mu = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--epsilon") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    epsilon = atof(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--compute-fat-long") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      compute_fatlong = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      compute_fatlong = false;
    }else{
      fprintf(stderr, "ERROR: invalid compute_fatlong type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--epsilon-naik") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    eps_naik = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tadpole-coeff") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    tadpole_factor = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--anisotropy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    anisotropy = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tol") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    tol= atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tolhq") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    tol_hq= atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--reliable-delta") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    reliable_delta = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--alternative-reliable") == 0){
    if (i+1 >= argc) {
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0) {
      alternative_reliable = true;
    } else if (strcmp(argv[i+1], "false") == 0) {
      alternative_reliable = false;
    } else {
      fprintf(stderr, "ERROR: invalid multishift boolean\n");
      exit(1);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mass-normalization") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    normalization = get_mass_normalization_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--matpc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    matpc_type = get_matpc_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--solve-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    solve_type = get_solve_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--solution-type") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    solution_type = get_solution_type(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--load-gauge") == 0){
    if (i + 1 >= argc) { usage(argv); }
    strcpy(latfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--save-gauge") == 0){
    if (i + 1 >= argc) { usage(argv); }
    strcpy(gauge_outfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--unit-gauge") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      unit_gauge = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      unit_gauge = false;
    } else {
      fprintf(stderr, "ERROR: invalid unit-gauge type given (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--nsrc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    Nsrc = atoi(argv[i+1]);
    if (Nsrc < 0 || Nsrc > 128){ // allow 0 for testing setup in isolation
      printf("ERROR: invalid number of sources (Nsrc=%d)\n", Nsrc);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--msrc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    Msrc = atoi(argv[i+1]);
    if (Msrc < 1 || Msrc > 128){
      printf("ERROR: invalid number of sources (Msrc=%d)\n", Msrc);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--test") == 0){
    if (i + 1 >= argc) { usage(argv); }
    test_type = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-nvec") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    nvec[level] = atoi(argv[i+1]);
    if (nvec[level] < 0 || nvec[level] > 1024) {
      printf("ERROR: invalid number of vectors (%d)\n", nvec[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-levels") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mg_levels= atoi(argv[i+1]);
    if (mg_levels < 2 || mg_levels > QUDA_MAX_MG_LEVEL){
      printf("ERROR: invalid number of multigrid levels (%d)\n", mg_levels);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-nu-pre") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    nu_pre[level] = atoi(argv[i+1]);
    if (nu_pre[level] < 0 || nu_pre[level] > 20){
      printf("ERROR: invalid pre-smoother applications value (nu_pre=%d)\n", nu_pre[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-nu-post") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    nu_post[level] = atoi(argv[i+1]);
    if (nu_post[level] < 0 || nu_post[level] > 20){
      printf("ERROR: invalid pre-smoother applications value (nu_post=%d)\n", nu_post[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solve-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    coarse_solve_type[level] = get_solve_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-smoother-solve-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    smoother_solve_type[level] = get_solve_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-solver-location") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);

    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    solver_location[level] = get_location(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-location") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);

    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_location[level] = get_location(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-inv") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_inv[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-iters") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    num_setup_iter[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-tol") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_tol[level] = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--mg-setup-maxiter") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_maxiter[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-maxiter-refresh") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_maxiter_refresh[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-ca-basis-type") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i+1], "power") == 0){
      setup_ca_basis[level] = QUDA_POWER_BASIS;
    }else if (strcmp(argv[i+1], "chebyshev") == 0 || strcmp(argv[i+1], "cheby") == 0){
      setup_ca_basis[level] = QUDA_CHEBYSHEV_BASIS;
    }else{
      fprintf(stderr, "ERROR: invalid value for basis ca cg type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-ca-basis-size") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_ca_basis_size[level] = atoi(argv[i+1]);
    if (setup_ca_basis_size[level] < 1){
      printf("ERROR: invalid basis size for CA-CG (%d < 1)\n", setup_ca_basis_size[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-cheby-basis-eig-min") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_ca_lambda_min[level] = atof(argv[i+1]);
    if (setup_ca_lambda_min[level] < 0.0){
      printf("ERROR: invalid minimum eigenvalue for CA-CG (%f < 0.0)\n", setup_ca_lambda_min[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-cheby-basis-eig-max") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_ca_lambda_max[level] = atof(argv[i+1]);
    // negative value induces power iterations to guess
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if( strcmp(argv[i+1], "test") == 0)
      setup_type = QUDA_TEST_VECTOR_SETUP;
    else if( strcmp(argv[i+1], "null")==0)
      setup_type = QUDA_NULL_VECTOR_SETUP;
    else {
      fprintf(stderr, "ERROR: invalid setup type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-pre-orth") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      pre_orthonormalize = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      pre_orthonormalize = false;
    }else{
      fprintf(stderr, "ERROR: invalid pre orthogonalize type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-post-orth") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      post_orthonormalize = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      post_orthonormalize = false;
    }else{
      fprintf(stderr, "ERROR: invalid post orthogonalize type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-n-block-ortho") == 0) {
    if (i + 2 >= argc) { usage(argv); }

    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for number of block orthos", level);
      usage(argv);
    }
    i++;

    n_block_ortho[level] = atoi(argv[i + 1]);
    if (n_block_ortho[level] < 1) {
      fprintf(stderr, "ERROR: invalid number %d of block orthonormalizations", n_block_ortho[level]);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-omega") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    omega = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-verbosity") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_verbosity[level] = get_verbosity_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-smoother") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    smoother_type[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-smoother-tol") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    smoother_tol[level] = atof(argv[i+1]);

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-tol") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver_tol[level] = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-maxiter") == 0){
    if (i+2 >= argc){
      usage(argv);
    }

    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver_maxiter[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

    if( strcmp(argv[i], "--mg-coarse-solver-ca-basis-type") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i+1], "power") == 0){
      coarse_solver_ca_basis[level] = QUDA_POWER_BASIS;
    }else if (strcmp(argv[i+1], "chebyshev") == 0 || strcmp(argv[i+1], "cheby") == 0){
      coarse_solver_ca_basis[level] = QUDA_CHEBYSHEV_BASIS;
    }else{
      fprintf(stderr, "ERROR: invalid value for basis ca cg type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-ca-basis-size") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    coarse_solver_ca_basis_size[level] = atoi(argv[i+1]);
    if (coarse_solver_ca_basis_size[level] < 1){
      printf("ERROR: invalid basis size for CA-CG (%d < 1)\n", coarse_solver_ca_basis_size[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-cheby-basis-eig-min") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    coarse_solver_ca_lambda_min[level] = atof(argv[i+1]);
    if (coarse_solver_ca_lambda_min[level] < 0.0){
      printf("ERROR: invalid minimum eigenvalue for CA-CG (%f < 0.0)\n", coarse_solver_ca_lambda_min[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-cheby-basis-eig-max") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    coarse_solver_ca_lambda_max[level] = atof(argv[i+1]);
    // negative value induces power iterations to guess
    i++;
    ret = 0;
    goto out;
  }



  if( strcmp(argv[i], "--mg-smoother-halo-prec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    smoother_halo_prec =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--mg-schwarz-type") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    schwarz_type[level] = get_schwarz_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-schwarz-cycle") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    schwarz_cycle[level] = atoi(argv[i+1]);
    if (schwarz_cycle[level] < 0 || schwarz_cycle[level] >= 128) {
      printf("ERROR: invalid Schwarz cycle value requested %d for level %d",
	     level, schwarz_cycle[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-block-size") == 0){
    if (i + 5 >= argc) { usage(argv); }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X block size");
      usage(argv);
    }
    geo_block_size[level][0] = xsize;
    i++;

    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y block size");
      usage(argv);
    }
    geo_block_size[level][1] = ysize;
    i++;

    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z block size");
      usage(argv);
    }
    geo_block_size[level][2] = zsize;
    i++;

    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T block size");
      usage(argv);
    }
    geo_block_size[level][3] = tsize;
    i++;

    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-mu-factor") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    double factor =  atof(argv[i+1]);
    mu_factor[level] = factor;
    i++;
    ret=0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-generate-nullspace") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      generate_nullspace = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      generate_nullspace = false;
    }else{
      fprintf(stderr, "ERROR: invalid generate nullspace type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-generate-all-levels") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      generate_all_levels = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      generate_all_levels = false;
    }else{
      fprintf(stderr, "ERROR: invalid value for generate_all_levels type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-load-vec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(vec_infile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-save-vec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(vec_outfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-nev") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    nev = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-max-search-dim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    max_search_dim = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-deflation-grid") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    deflation_grid = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--df-eigcg-max-restarts") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    eigcg_max_restarts = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-max-restart-num") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    max_restart_num = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-tol-restart") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    tol_restart = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-tol-eigenval") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    eigenval_tol = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-tol-inc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    inc_tol = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--solver-ext-lib-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    solver_ext_lib = get_solve_ext_lib_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-ext-lib-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    deflation_ext_lib = get_solve_ext_lib_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-location-ritz") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    location_ritz = get_location(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--df-mem-type-ritz") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mem_type_ritz = get_df_mem_type_ritz(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-nEv") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_nEv = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-nKr") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_nKr = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-nConv") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_nConv = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-require-convergence") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_require_convergence = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_require_convergence = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for require-convergence (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-check-interval") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_check_interval = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-max-restarts") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_max_restarts = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-tol") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_tol = atof(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-use-poly-acc") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_use_poly_acc = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_use_poly_acc = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for use-poly-acc (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-poly-deg") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_poly_deg = atoi(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-amin") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_amin = atof(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-amax") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_amax = atof(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-use-normop") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_use_normop = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_use_normop = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for eig-use-normop (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-use-dagger") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_use_dagger = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_use_dagger = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for eig-use-dagger (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-compute-svd") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_compute_svd = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_compute_svd = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for eig-compute-svd (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-spectrum") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_spectrum = get_eig_spectrum_type(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-type") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    eig_type = get_eig_type(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-ARPACK-logfile") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    strcpy(eig_arpack_logfile, argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-QUDA-logfile") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    strcpy(eig_QUDA_logfile, argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-arpack-check") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_arpack_check = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_arpack_check = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for arpack-check (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-load-vec") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    strcpy(eig_vec_infile, argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--eig-save-vec") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    strcpy(eig_vec_outfile, argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i + 1], "true") == 0) {
      mg_eig[level] = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      mg_eig[level] = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig %d (true/false)\n", level);
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-nEv") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_nEv[level] = atoi(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-nKr") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_nKr[level] = atoi(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-require-convergence") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i + 1], "true") == 0) {
      mg_eig_require_convergence[level] = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      mg_eig_require_convergence[level] = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig-require-convergence %d (true/false)\n", level);
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-check-interval") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_check_interval[level] = atoi(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-max-restarts") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_max_restarts[level] = atoi(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-tol") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_tol[level] = atof(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-use-poly-acc") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i + 1], "true") == 0) {
      mg_eig_use_poly_acc[level] = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      mg_eig_use_poly_acc[level] = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig-use-poly-acc %d (true/false)\n", level);
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-poly-deg") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_poly_deg[level] = atoi(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-amin") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_amin[level] = atof(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-amax") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_amax[level] = atof(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-use-normop") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i + 1], "true") == 0) {
      mg_eig_use_normop[level] = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      mg_eig_use_normop[level] = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig-use-normop (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-use-dagger") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    if (strcmp(argv[i + 1], "true") == 0) {
      mg_eig_use_dagger[level] = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      mg_eig_use_dagger[level] = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig-use-dagger (true/false)\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-spectrum") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_spectrum[level] = get_eig_spectrum_type(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-type") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    int level = atoi(argv[i + 1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_eig_type[level] = get_eig_type(argv[i + 1]);

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--mg-eig-coarse-guess") == 0) {
    if (i + 1 >= argc) { usage(argv); }

    if (strcmp(argv[i + 1], "true") == 0) {
      eig_use_poly_acc = true;
    } else if (strcmp(argv[i + 1], "false") == 0) {
      eig_use_poly_acc = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for mg-eig-coarse-guess (true/false)\n");
      exit(1);
    }

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

  if( strcmp(argv[i], "--ngcrkrylov") == 0 || strcmp(argv[i], "--ca-basis-size") == 0) {
    if (i+1 >= argc){
      usage(argv);
    }
    gcrNkrylov = atoi(argv[i+1]);
    if (gcrNkrylov < 1 || gcrNkrylov > 1e6){
      printf("ERROR: invalid number of gcrkrylov iterations (%d)\n", gcrNkrylov);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ca-basis-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "power") == 0){
      ca_basis = QUDA_POWER_BASIS;
    }else if (strcmp(argv[i+1], "chebyshev") == 0 || strcmp(argv[i+1], "cheby") == 0){
      ca_basis = QUDA_CHEBYSHEV_BASIS;
    }else{
      fprintf(stderr, "ERROR: invalid value for basis ca cg type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--cheby-basis-eig-min") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    ca_lambda_min = atof(argv[i+1]);
    if (ca_lambda_min < 0.0){
      printf("ERROR: invalid minimum eigenvalue for CA-CG (%f < 0.0)\n", ca_lambda_min);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--cheby-basis-eig-max") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    ca_lambda_max = atof(argv[i+1]);
    // negative value induces power iterations to guess
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--pipeline") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    pipeline = atoi(argv[i+1]);
    if (pipeline < 0 || pipeline > 16){
      printf("ERROR: invalid pipeline length (%d)\n", pipeline);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--solution-pipeline") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    solution_accumulator_pipeline = atoi(argv[i+1]);
    if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16){
      printf("ERROR: invalid solution pipeline length (%d)\n", solution_accumulator_pipeline);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-beta") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    heatbath_beta_value = atof(argv[i+1]);
    if (heatbath_beta_value <= 0.0){
      printf("ERROR: invalid beta (%f)\n", heatbath_beta_value);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-warmup-steps") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    heatbath_warmup_steps = atoi(argv[i+1]);
    if (heatbath_warmup_steps < 0){
      printf("ERROR: invalid number of warmup steps (%d)\n", heatbath_warmup_steps);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-num-steps") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    heatbath_num_steps = atoi(argv[i+1]);
    if (heatbath_num_steps < 0){
      printf("ERROR: invalid number of heatbath steps (%d)\n", heatbath_num_steps);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-num-hb-per-step") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    heatbath_num_heatbath_per_step = atoi(argv[i+1]);
    if (heatbath_num_heatbath_per_step <= 0){
      printf("ERROR: invalid number of heatbath hits per step (%d)\n", heatbath_num_heatbath_per_step);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-num-or-per-step") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    heatbath_num_overrelax_per_step = atoi(argv[i+1]);
    if (heatbath_num_overrelax_per_step <= 0){
      printf("ERROR: invalid number of overrelaxation hits per step (%d)\n", heatbath_num_overrelax_per_step);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--heatbath-coldstart") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      heatbath_coldstart = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      heatbath_coldstart = false;
    }else{
      fprintf(stderr, "ERROR: invalid value for heatbath_coldstart type\n");
      usage(argv);
    }

    i++;
    ret = 0;
    goto out;
  }

  if (strcmp(argv[i], "--contract-type") == 0) {
    if (i + 1 >= argc) { usage(argv); }
    contract_type = get_contract_type(argv[i + 1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--version") == 0){
    printf("This program is linked with QUDA library, version %s,", get_quda_ver_str());
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
