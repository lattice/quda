#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <domain_wall_dslash_reference.h>
#include <blas_reference.h>

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <face_quda.h>


// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
// 
// the displacements, such as dx, refer to the full lattice coordinates. 
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//
//
int neighborIndex_5d(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1) {
  // fullLatticeIndex was modified for fullLatticeIndex_4d.  It is in util_quda.cpp.
  // This code bit may not properly perform 5dPC.
  int X = fullLatticeIndex_5d(i, oddBit);
  // Checked that this matches code in dslash_core_ante.h.
  int xs = X/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (X/(Z[2]*Z[1]*Z[0])) % Z[3];
  int x3 = (X/(Z[1]*Z[0])) % Z[2];
  int x2 = (X/Z[0]) % Z[1];
  int x1 = X % Z[0];
  // Displace and project back into domain 0,...,Ls-1.
  // Note that we add Ls to avoid the negative problem
  // of the C % operator.
  xs = (xs+dxs+Ls) % Ls;
  // Etc.
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  // Return linear half index.  Remember that integer division
  // rounds down.
  return (xs*(Z[3]*Z[2]*Z[1]*Z[0]) + x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
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
//
int neighborIndex_4d(int i, int oddBit, int dx4, int dx3, int dx2, int dx1) {
  // On input i should be in the range [0 , ... , Z[0]*Z[1]*Z[2]*Z[3]/2-1].
  if (i < 0 || i >= (Z[0]*Z[1]*Z[2]*Z[3]/2)) 
    { printf("i out of range in neighborIndex_4d\n"); exit(-1); }
  // Compute the linear index.  Then dissect.
  // fullLatticeIndex_4d is in util_quda.cpp.
  // The gauge fields live on a 4d sublattice.  
  int X = fullLatticeIndex_4d(i, oddBit);
  int x4 = X/(Z[2]*Z[1]*Z[0]);
  int x3 = (X/(Z[1]*Z[0])) % Z[2];
  int x2 = (X/Z[0]) % Z[1];
  int x1 = X % Z[0];
  
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  
  return (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
}

//BEGIN NEW
//#ifdef MULTI_GPU
int
neighborIndex_5d_mgpu(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  int ret;
  
  int Y = fullLatticeIndex_5d(i, oddBit);
  
  int xs = Y/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  
  int ghost_x4 = x4+ dx4;
  
  xs = (xs+dxs+Ls) % Ls;
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  
  if ( ghost_x4 >= 0 && ghost_x4 < Z[3]){
    ret = (xs*Z[3]*Z[2]*Z[1]*Z[0] + x4*Z[2]*Z[1]*Z[0] + x3*Z[1]*Z[0] + x2*Z[0] + x1) >> 1;
  }else{
    ret = (xs*Z[2]*Z[1]*Z[0] + x3*Z[1]*Z[0] + x2*Z[0] + x1) >> 1;    
  }

  
  return ret;
}

int
x4_5d_mgpu(int i, int oddBit)
{
  int Y = fullLatticeIndex_5d(i, oddBit);
  return (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
}

//#endif

//END NEW

//#ifndef MULTI_GPU
// This is just a copy of gaugeLink() from the quda code, except
// that neighborIndex() is replaced by the renamed version
// neighborIndex_4d().
//ok
template <typename Float>
Float *gaugeLink_sgpu(int i, int dir, int oddBit, Float **gaugeEven,
                Float **gaugeOdd) {
  Float **gaugeField;
  int j;
  
  // If going forward, just grab link at site, U_\mu(x).
  if (dir % 2 == 0) {
    j = i;
    // j will get used in the return statement below.
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  } else {
    // If going backward, a shift must occur, U_\mu(x-\muhat)^\dagger;
    // dagger happens elsewhere, here we're just doing index gymnastics.
    switch (dir) {
    case 1: j = neighborIndex_4d(i, oddBit, 0, 0, 0, -1); break;
    case 3: j = neighborIndex_4d(i, oddBit, 0, 0, -1, 0); break;
    case 5: j = neighborIndex_4d(i, oddBit, 0, -1, 0, 0); break;
    case 7: j = neighborIndex_4d(i, oddBit, -1, 0, 0, 0); break;
    default: j = -1; break;
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);
  }
  
  return &gaugeField[dir/2][j*(3*3*2)];
}


//#else

//Standard 4d version (nothing to change)
template <typename Float>
Float *gaugeLink_mgpu(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd, Float** ghostGaugeEven, Float** ghostGaugeOdd, int n_ghost_faces, int nbr_distance) {
  Float **gaugeField;
  int j;
  int d = nbr_distance;
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  }
  else {

    int Y = fullLatticeIndex(i, oddBit);
    int x4 = Y/(Z[2]*Z[1]*Z[0]);
    int x3 = (Y/(Z[1]*Z[0])) % Z[2];
    int x2 = (Y/Z[0]) % Z[1];
    int x1 = Y % Z[0];
    int X1= Z[0];
    int X2= Z[1];
    int X3= Z[2];
    int X4= Z[3];
    Float* ghostGaugeField;

    switch (dir) {
    case 1:
      { //-X direction
        int new_x1 = (x1 - d + X1 )% X1;
        if (x1 -d < 0){
	  ghostGaugeField = (oddBit?ghostGaugeEven[0]: ghostGaugeOdd[0]);
	  int offset = (n_ghost_faces + x1 -d)*X4*X3*X2/2 + (x4*X3*X2 + x3*X2+x2)/2;
	  return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
        break;
      }
    case 3:
      { //-Y direction
        int new_x2 = (x2 - d + X2 )% X2;
        if (x2 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[1]: ghostGaugeOdd[1]);
          int offset = (n_ghost_faces + x2 -d)*X4*X3*X1/2 + (x4*X3*X1 + x3*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
        break;

      }
    case 5:
      { //-Z direction
        int new_x3 = (x3 - d + X3 )% X3;
        if (x3 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[2]: ghostGaugeOdd[2]);
          int offset = (n_ghost_faces + x3 -d)*X4*X2*X1/2 + (x4*X2*X1 + x2*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
        break;
      }
    case 7:
      { //-T direction
        int new_x4 = (x4 - d + X4)% X4;
        if (x4 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[3]: ghostGaugeOdd[3]);
          int offset = (n_ghost_faces + x4 -d)*X1*X2*X3/2 + (x3*X2*X1 + x2*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (new_x4*(X3*X2*X1) + x3*(X2*X1) + x2*(X1) + x1) / 2;
        break;
      }//7

    default: j = -1; printf("ERROR: wrong dir \n"); exit(1);
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);

  }

  return &gaugeField[dir/2][j*(3*3*2)];
}

//A.S.: this is valid for DW dslash wiht space-time decomposition.
template <typename Float>
Float *spinorNeighbor_5d_mgpu(int i, int dir, int oddBit, Float *spinorField, Float** fwd_nbr_spinor, Float** back_nbr_spinor, int neighbor_distance, int nFace)
{
//int mpi_rank = comm_rank();
  int j;
  int nb = neighbor_distance;
  int Y = fullLatticeIndex_5d(i, oddBit);
 
  int mySpinorSiteSize = 24;
 
  int xs = Y/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  
  int X1= Z[0];
  int X2= Z[1];
  int X3= Z[2];
  int X4= Z[3];
  switch (dir) {
  case 0://+X
    {
      int new_x1 = (x1 + nb)% X1;
      if(x1+nb >=X1){
        int offset = ((x1 + nb -X1)*Ls*X4*X3*X2+xs*X4*X3*X2+x4*X3*X2 + x3*X2+x2) >> 1;
        return fwd_nbr_spinor[0] + offset*mySpinorSiteSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) >> 1;
      break;

    }
  case 1://-X
    {
      int new_x1 = (x1 - nb + X1)% X1;
      if(x1 - nb < 0){ 
        int offset = (( x1+nFace- nb)*Ls*X4*X3*X2 + xs*X4*X3*X2 + x4*X3*X2 + x3*X2 + x2) >> 1;
        return back_nbr_spinor[0] + offset*mySpinorSiteSize;
      } 
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) >> 1;
      break;
    }
  case 2://+Y
    {
      int new_x2 = (x2 + nb)% X2;
      if(x2+nb >=X2){
        int offset = (( x2 + nb -X2)*Ls*X4*X3*X1+xs*X4*X3*X1+x4*X3*X1 + x3*X1+x1) >> 1;
        return fwd_nbr_spinor[1] + offset*mySpinorSiteSize;
      } 
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) >> 1;
      break;
    }
  case 3:// -Y
    {
      int new_x2 = (x2 - nb + X2)% X2;
      if(x2 - nb < 0){ 
        int offset = (( x2 + nFace -nb)*Ls*X4*X3*X1+xs*X4*X3*X1+ x4*X3*X1 + x3*X1+x1) >> 1;
        return back_nbr_spinor[1] + offset*mySpinorSiteSize;
      } 
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) >> 1;
      break;
    }
  case 4://+Z
    {
      int new_x3 = (x3 + nb)% X3;
      if(x3+nb >=X3){
        int offset = (( x3 + nb -X3)*Ls*X4*X2*X1+xs*X4*X2*X1+x4*X2*X1 + x2*X1+x1) >> 1;
        return fwd_nbr_spinor[2] + offset*mySpinorSiteSize;
      } 
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) >> 1;
      break;
    }
  case 5://-Z
    {
      int new_x3 = (x3 - nb + X3)% X3;
      if(x3 - nb < 0){ 
        int offset = (( x3 + nFace -nb)*Ls*X4*X2*X1+xs*X4*X2*X1+x4*X2*X1+x2*X1+x1) >> 1;
        return back_nbr_spinor[2] + offset*mySpinorSiteSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) >> 1;
      break;
    }
  case 6://+T 
    {
      j = neighborIndex_5d_mgpu(i, oddBit, 0, +nb, 0, 0, 0);
      int x4 = x4_5d_mgpu(i, oddBit);
      if ( (x4 + nb) >= Z[3])
      {
        int offset = (x4+nb - Z[3])*Vsh_t;//?
        return &fwd_nbr_spinor[3][(offset+j)*mySpinorSiteSize];
      }
      break;
    }
  case 7://-T 
    {
      j = neighborIndex_5d_mgpu(i, oddBit, 0, -nb, 0, 0, 0);
      int x4 = x4_5d_mgpu(i, oddBit);
      if ( (x4 - nb) < 0){
        int offset = ( x4 - nb +nFace)*Vsh_t;//?
        return &back_nbr_spinor[3][(offset+j)*mySpinorSiteSize];
      }
      break;
    }
  default: j = -1; printf("ERROR: wrong dir\n"); exit(1);
  }

  return &spinorField[j*(mySpinorSiteSize)];
}

//#endif


template <typename Float>
Float *spinorNeighbor_5d(int i, int dir, int oddBit, Float *spinorField) {
  int j;
  switch (dir) {
  case 0: j = neighborIndex_5d(i, oddBit, 0, 0, 0, 0, +1); break;
  case 1: j = neighborIndex_5d(i, oddBit, 0, 0, 0, 0, -1); break;
  case 2: j = neighborIndex_5d(i, oddBit, 0, 0, 0, +1, 0); break;
  case 3: j = neighborIndex_5d(i, oddBit, 0, 0, 0, -1, 0); break;
  case 4: j = neighborIndex_5d(i, oddBit, 0, 0, +1, 0, 0); break;
  case 5: j = neighborIndex_5d(i, oddBit, 0, 0, -1, 0, 0); break;
  case 6: j = neighborIndex_5d(i, oddBit, 0, +1, 0, 0, 0); break;
  case 7: j = neighborIndex_5d(i, oddBit, 0, -1, 0, 0, 0); break;
  case 8: j = neighborIndex_5d(i, oddBit, +1, 0, 0, 0, 0); break;
  case 9: j = neighborIndex_5d(i, oddBit, -1, 0, 0, 0, 0); break;
  default: j = -1; break;
  }
  
  return &spinorField[j*(4*3*2)];
}


//J  Directions 0..7 were used in the 4d code.
//J  Directions 8,9 will be for P_- and P_+, chiral
//J  projectors.
const double projector[10][4][4][2] = {
  {
    {{1,0}, {0,0}, {0,0}, {0,-1}},
    {{0,0}, {1,0}, {0,-1}, {0,0}},
    {{0,0}, {0,1}, {1,0}, {0,0}},
    {{0,1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {0,1}},
    {{0,0}, {1,0}, {0,1}, {0,0}},
    {{0,0}, {0,-1}, {1,0}, {0,0}},
    {{0,-1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {1,0}},
    {{0,0}, {1,0}, {-1,0}, {0,0}},
    {{0,0}, {-1,0}, {1,0}, {0,0}},
    {{1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {-1,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{-1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,-1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,1}},
    {{0,1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,-1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,-1}},
    {{0,-1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {-1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {-1,0}},
    {{-1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {-1,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}},
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}}
  },
  // P_+ = P_R
  {
    {{2,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {2,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}}
  },
  // P_- = P_L
  {
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {2,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {2,0}}
  }
};


// todo pass projector
template <typename Float>
void multiplySpinorByDiracProjector5(Float *res, int projIdx, Float *spinorIn) {
  for (int i=0; i<4*3*2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      Float projRe = projector[projIdx][s][t][0];
      Float projIm = projector[projIdx][s][t][1];

      for (int m = 0; m < 3; m++) {
	Float spinorRe = spinorIn[t*(3*2) + m*(2) + 0];
	Float spinorIm = spinorIn[t*(3*2) + m*(2) + 1];
	res[s*(3*2) + m*(2) + 0] += projRe*spinorRe - projIm*spinorIm;
	res[s*(3*2) + m*(2) + 1] += projRe*spinorIm + projIm*spinorRe;
      }
    }
  }
}


//#ifndef MULTI_GPU
// dslashReference_4d()
//J  This is just the 4d wilson dslash of quda code, with a
//J  few small changes to take into account that the spinors
//J  are 5d and the gauge fields are 4d.
//
// if oddBit is zero: calculate odd parity spinor elements (using even parity spinor)
// if oddBit is one:  calculate even parity spinor elements
//
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
//
//An "ok" will only be granted once check2.tex is deemed complete,
//since the logic in this function is important and nontrivial.
template <typename sFloat, typename gFloat>
void dslashReference_4d_sgpu(sFloat *res, gFloat **gaugeFull, sFloat *spinorField, 
                int oddBit, int daggerBit) {
  
  // Initialize the return half-spinor to zero.  Note that it is a
  // 5d spinor, hence the use of V5h.
  for (int i=0; i<V5h*4*3*2; i++) res[i] = 0.0;
  
  // Some pointers that we use to march through arrays.
  gFloat *gaugeEven[4], *gaugeOdd[4];
  // Initialize to beginning of even and odd parts of
  // gauge array.
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    // Note the use of Vh here, since the gauge fields
    // are 4-dim'l.
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;
  }
  int sp_idx,oddBit_gge;
  for (int xs=0;xs<Ls;xs++) {
    for (int gge_idx = 0; gge_idx < Vh; gge_idx++) {
      for (int dir = 0; dir < 8; dir++) {
        sp_idx=gge_idx+Vh*xs;
        // Here is a function call to study.  It is defined near
        // Line 90 of this file.
        // Here we have to switch oddBit depending on the value of xs.  E.g., suppose
        // xs=1.  Then the odd spinor site x1=x2=x3=x4=0 wants the even gauge array
        // element 0, so that we get U_\mu(0).
        if ((xs % 2) == 0) oddBit_gge=oddBit;
        else oddBit_gge= (oddBit+1) % 2;
        gFloat *gauge = gaugeLink_sgpu(gge_idx, dir, oddBit_gge, gaugeEven, gaugeOdd);
        
        // Even though we're doing the 4d part of the dslash, we need
        // to use a 5d neighbor function, to get the offsets right.
        sFloat *spinor = spinorNeighbor_5d(sp_idx, dir, oddBit, spinorField);
        sFloat projectedSpinor[4*3*2], gaugedSpinor[4*3*2];
        int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
        multiplySpinorByDiracProjector5(projectedSpinor, projIdx, spinor);
      
        for (int s = 0; s < 4; s++) {
	        if (dir % 2 == 0) {
        	  su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
#ifdef DBUG_VERBOSE            
		  std::cout << "spinor:" << std::endl;
		  printSpinorElement(&projectedSpinor[s*(3*2)],0,QUDA_DOUBLE_PRECISION);
		  std::cout << "gauge:" << std::endl;
#endif
          } else {
        	  su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
          }
        }
      
        sum(&res[sp_idx*(4*3*2)], &res[sp_idx*(4*3*2)], gaugedSpinor, 4*3*2);
      }
    }
  }
}
//#else

template <typename sFloat, typename gFloat>
void dslashReference_4d_mgpu(sFloat *res, gFloat **gaugeFull, gFloat **ghostGauge, sFloat *spinorField, sFloat **fwdSpinor, sFloat **backSpinor, int oddBit, int daggerBit) 
{
  
  int mySpinorSiteSize = 24;		    
  for (int i=0; i<V5h*mySpinorSiteSize; i++) res[i] = 0.0;
  
  gFloat *gaugeEven[4], *gaugeOdd[4];
  gFloat *ghostGaugeEven[4], *ghostGaugeOdd[4];
  
  for (int dir = 0; dir < 4; dir++) 
  {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;

    ghostGaugeEven[dir] = ghostGauge[dir];
    ghostGaugeOdd[dir] = ghostGauge[dir] + (faceVolume[dir]/2)*gaugeSiteSize;
  }
  for (int xs=0;xs<Ls;xs++) 
  {  
    int sp_idx;
    for (int i = 0; i < Vh; i++) 
    {
      sp_idx = i + Vh*xs;
      for (int dir = 0; dir < 8; dir++) 
      {
	int oddBit_gge;

	if ((xs % 2) == 0) oddBit_gge=oddBit;
        else oddBit_gge= (oddBit+1) % 2;
	
	gFloat *gauge = gaugeLink_mgpu(i, dir, oddBit_gge, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);//this is unchanged from MPi version
	sFloat *spinor = spinorNeighbor_5d_mgpu(sp_idx, dir, oddBit, spinorField, fwdSpinor, backSpinor, 1, 1);
	
	sFloat projectedSpinor[mySpinorSiteSize], gaugedSpinor[mySpinorSiteSize];
	int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
	multiplySpinorByDiracProjector5(projectedSpinor, projIdx, spinor);
      
	for (int s = 0; s < 4; s++) 
	{
	  if (dir % 2 == 0) su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	  else su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	}
	sum(&res[sp_idx*(4*3*2)], &res[sp_idx*(4*3*2)], gaugedSpinor, 4*3*2);
      }
    }
  }
}
//#endif

//Currently we consider only spacetime decomposition (not in 5th dim), so this operator is local
template <typename sFloat>
void dslashReference_5th(sFloat *res, sFloat *spinorField, 
                int oddBit, int daggerBit, sFloat mferm) {
  for (int i = 0; i < V5h; i++) {
    for (int dir = 8; dir < 10; dir++) {
      // Calls for an extension of the original function.
      // 8 is forward hop, which wants P_+, 9 is backward hop,
      // which wants P_-.  Dagger reverses these.
      sFloat *spinor = spinorNeighbor_5d(i, dir, oddBit, spinorField);
      sFloat projectedSpinor[4*3*2];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector5(projectedSpinor, projIdx, spinor);
      //J  Need a conditional here for s=0 and s=Ls-1.
      int X = fullLatticeIndex_5d(i, oddBit);
      int xs = X/(Z[3]*Z[2]*Z[1]*Z[0]);

      if ( (xs == 0 && dir == 9) || (xs == Ls-1 && dir == 8) ) {
        ax(projectedSpinor,(sFloat)(-mferm),projectedSpinor,4*3*2);
      } 
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], projectedSpinor, 4*3*2);
    }
  }
}

// Recall that dslash is only the off-diagonal parts, so m0_dwf is not needed.
//
//#ifndef MULTI_GPU
/*
void dslash(void *res, void **gaugeFull, void *spinorField, 
	    int oddBit, int daggerBit, 
	    QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm) {
  
  if (sPrecision == QUDA_DOUBLE_PRECISION)  {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      // Do the 4d part, which hasn't changed.
      printf("doing 4d part\n"); fflush(stdout);
      dslashReference_4d_sgpu<double,double>((double*)res, (double**)gaugeFull,
                      (double*)spinorField, oddBit, daggerBit);
      // Now add in the 5th dim.
      printf("doing 5th dimen. part\n"); fflush(stdout);
      dslashReference_5th<double>((double*)res, (double*)spinorField, 
                      oddBit, daggerBit, mferm);
    } else {
      dslashReference_4d_sgpu<double,float>((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      dslashReference_5th<double>((double*)res, (double*)spinorField, oddBit, daggerBit, mferm);
    }
  } else {
    // Single-precision spinor.
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      dslashReference_4d_sgpu<float,double>((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      dslashReference_5th<float>((float*)res, (float*)spinorField, oddBit, daggerBit, mferm);
    } else {
      // Do the 4d part, which hasn't changed.
      printf("CPU reference:  doing 4d part all single precision\n"); fflush(stdout);
      dslashReference_4d_sgpu<float,float>((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      // Now add in the 5th dim.
      printf("CPU reference:  doing 5th dimen. part all single precision\n"); fflush(stdout);
      dslashReference_5th<float>((float*)res, (float*)spinorField, oddBit, daggerBit, mferm);
    }
  }
}
*/
//#endif

//BEGIN NEW
// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void dw_dslash(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) 
{
#ifndef MULTI_GPU
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_4d_sgpu((double*)out, (double**)gauge, (double*)in, oddBit, daggerBit);
    dslashReference_5th((double*)out, (double*)in, oddBit, daggerBit, mferm);
  } else {
    dslashReference_4d_sgpu((float*)out, (float**)gauge, (float*)in, oddBit, daggerBit);
    dslashReference_5th((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
  }
#else

//    void *ghostGauge[4], *sendGauge[4];
//    for (int d=0; d<4; d++) {
//      ghostGauge[d] = malloc(faceVolume[d]*gaugeSiteSize*precision);
//      sendGauge[d] = malloc(faceVolume[d]*gaugeSiteSize*precision);
//    }

//    { // Exchange gauge matrices at boundary
//      set_dim(Z);///?
//      pack_ghost(gauge, sendGauge, 1, precision);
//      int nFace = 1;
//      FaceBuffer faceBuf(Z, 4, gaugeSiteSize, nFace, precision);
//      faceBuf.exchangeCpuLink(ghostGauge, sendGauge);
//    }
    
//BEGINOFNEW    
    GaugeFieldParam gauge_field_param(gauge, gauge_param);
    cpuGaugeField cpu(gauge_field_param);
    cpu.exchangeGhost();
    void **ghostGauge = (void**)cpu.Ghost();    
//ENDOFNEW    
  
    // Get spinor ghost fields
    // First wrap the input spinor into a ColorSpinorField
    ColorSpinorParam csParam;
    csParam.v = in;
    csParam.nColor = 3;
    csParam.nSpin = 4;
    csParam.nDim = 5; //for DW dslash
    for (int d=0; d<4; d++) csParam.x[d] = Z[d];
    csParam.x[4] = Ls;//5th dimention
    csParam.precision = precision;
    csParam.pad = 0;
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  
    cpuColorSpinorField inField(csParam);

    {  // Now do the exchange
      QudaParity otherParity = QUDA_INVALID_PARITY;
      if (oddBit == QUDA_EVEN_PARITY) otherParity = QUDA_ODD_PARITY;
      else if (oddBit == QUDA_ODD_PARITY) otherParity = QUDA_EVEN_PARITY;
      else errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);

      int nFace = 1;
      FaceBuffer faceBuf(Z, 5, mySpinorSiteSize, nFace, precision, Ls);//4 <-> 5
      faceBuf.exchangeCpuSpinor(inField, otherParity, daggerBit); 
    }
    void** fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
    void** back_nbr_spinor = inField.backGhostFaceBuffer;
  //NOTE: hopping  in 5th dimension does not use MPI. 
    if (precision == QUDA_DOUBLE_PRECISION) 
    {
      dslashReference_4d_mgpu((double*)out, (double**)gauge, (double**)ghostGauge, (double*)in,(double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
      dslashReference_5th((double*)out, (double*)in, oddBit, daggerBit, mferm);    
    } else
    {
      dslashReference_4d_mgpu((float*)out, (float**)gauge, (float**)ghostGauge, (float*)in, 
		    (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
      dslashReference_5th((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);		    
    }

#endif

}

//END NEW



void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) {

  void *inEven = in;
  void *inOdd  = (char*)in + V5h*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + V5h*spinorSiteSize*precision;

  dw_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param, mferm);
  dw_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param, mferm);

  // lastly apply the kappa term
  if (precision == QUDA_DOUBLE_PRECISION) xpay((double*)in, -kappa, (double*)out, V5*spinorSiteSize);
  else xpay((float*)in, -(float)kappa, (float*)out, V5*spinorSiteSize);
}

void dw_matdagmat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm)
{

  void *tmp = malloc(V5*spinorSiteSize*precision);  
  dw_mat(tmp, gauge, in, kappa, dagger_bit, precision, gauge_param, mferm);
  dagger_bit = (dagger_bit == 1) ? 0 : 1;
  dw_mat(out, gauge, tmp, kappa, dagger_bit, precision, gauge_param, mferm);
  
  free(tmp);
}

void dw_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm)
{
  void *tmp = malloc(V5h*spinorSiteSize*precision);  
  
  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    dw_dslash(tmp, gauge, in, 1, dagger_bit, precision, gauge_param, mferm);
    dw_dslash(out, gauge, tmp, 0, dagger_bit, precision, gauge_param, mferm);
  } else {
    dw_dslash(tmp, gauge, in, 0, dagger_bit, precision, gauge_param, mferm);
    dw_dslash(out, gauge, tmp, 1, dagger_bit, precision, gauge_param, mferm);
  }

  // lastly apply the kappa term
  double kappa2 = -kappa*kappa;
  if (precision == QUDA_DOUBLE_PRECISION) xpay((double*)in, kappa2, (double*)out, V5h*spinorSiteSize);
  else xpay((float*)in, (float)kappa2, (float*)out, V5h*spinorSiteSize);

  free(tmp);
}

/*
// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void MatPC(sFloat *outEven, gFloat **gauge, sFloat *inEven, sFloat kappa,
	   QudaMatPCType matpc_type, sFloat mferm) {
  
  sFloat *tmp = (sFloat*)malloc(V5h*spinorSiteSize*sizeof(sFloat));
    
  // full dslash operator
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashReference_4d(tmp, gauge, inEven, 1, 0);
    dslashReference_5th(tmp, inEven, 1, 0, mferm);
    dslashReference_4d(outEven, gauge, tmp, 0, 0);
    dslashReference_5th(outEven, tmp, 0, 0, mferm);
  } else {
    dslashReference_4d(tmp, gauge, inEven, 0, 0);
    dslashReference_5th(tmp, inEven, 0, 0, mferm);
    dslashReference_4d(outEven, gauge, tmp, 1, 0);
    dslashReference_5th(outEven, tmp, 1, 0, mferm);
  }    
  
  // lastly apply the kappa term
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, V5h*spinorSiteSize);
  free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void MatPCDag(sFloat *outEven, gFloat **gauge, sFloat *inEven, sFloat kappa, 
	      QudaMatPCType matpc_type, sFloat mferm) {
  
  sFloat *tmp = (sFloat*)malloc(V5h*spinorSiteSize*sizeof(sFloat));    
  
  // full dslash operator
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashReference_4d(tmp, gauge, inEven, 1, 1);
    dslashReference_5th(tmp, inEven, 1, 1, mferm);
    dslashReference_4d(outEven, gauge, tmp, 0, 1);
    dslashReference_5th(outEven, tmp, 0, 1, mferm);
  } else {
    dslashReference_4d(tmp, gauge, inEven, 0, 1);
    dslashReference_5th(tmp, inEven, 0, 1, mferm);
    dslashReference_4d(outEven, gauge, tmp, 1, 1);
    dslashReference_5th(outEven, tmp, 1, 1, mferm);
  }
  
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, V5h*spinorSiteSize);
  free(tmp);
}
*/

void matpc(void *outEven, void **gauge, void *inEven, double kappa, 
	   QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision,
     double mferm) {
/*
  if (!dagger_bit) {
    if (sPrecision == QUDA_DOUBLE_PRECISION)
      if (gPrecision == QUDA_DOUBLE_PRECISION) 
	MatPC((double*)outEven, (double**)gauge, (double*)inEven, (double)kappa, matpc_type, (double)mferm);
      else
	MatPC((double*)outEven, (float**)gauge, (double*)inEven, (double)kappa, matpc_type, (double)mferm);
    else
      if (gPrecision == QUDA_DOUBLE_PRECISION) 
	MatPC((float*)outEven, (double**)gauge, (float*)inEven, (float)kappa, matpc_type, (float)mferm);
      else
	MatPC((float*)outEven, (float**)gauge, (float*)inEven, (float)kappa, matpc_type, (float)mferm);
  } else {
    if (sPrecision == QUDA_DOUBLE_PRECISION)
      if (gPrecision == QUDA_DOUBLE_PRECISION) 
	MatPCDag((double*)outEven, (double**)gauge, (double*)inEven, (double)kappa, matpc_type, (double)mferm);
      else
	MatPCDag((double*)outEven, (float**)gauge, (double*)inEven, (double)kappa, matpc_type, (double)mferm);
    else
      if (gPrecision == QUDA_DOUBLE_PRECISION) 
	MatPCDag((float*)outEven, (double**)gauge, (float*)inEven, (float)kappa, matpc_type, (float)mferm);
      else
	MatPCDag((float*)outEven, (float**)gauge, (float*)inEven, (float)kappa, matpc_type, (float)mferm);
  }
*/
}

/*
template <typename sFloat, typename gFloat> 
void MatDagMat(sFloat *out, gFloat **gauge, sFloat *in, sFloat kappa, sFloat mferm) 
{
  // Allocate a full spinor.        
  sFloat *tmp = (sFloat*)malloc(V5*spinorSiteSize*sizeof(sFloat));
  // Call templates above.
  Mat(tmp, gauge, in, kappa, mferm);
  MatDag(out, gauge, tmp, kappa, mferm);
  free(tmp);
}

template <typename sFloat, typename gFloat> 
void MatPCDagMatPC(sFloat *out, gFloat **gauge, sFloat *in, sFloat kappa, 
		   QudaMatPCType matpc_type, sFloat mferm)
{
  
  // Allocate half spinor
  sFloat *tmp = (sFloat*)malloc(V5h*spinorSiteSize*sizeof(sFloat));
  // Apply the PC templates above
  MatPC(tmp, gauge, in, kappa, matpc_type, mferm);
  MatPCDag(out, gauge, tmp, kappa, matpc_type, mferm);
  free(tmp);
}
*/
// Wrapper to templates that handles different precisions.
void matdagmat(void *out, void **gauge, void *in, double kappa,
	 QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm) 
{
/*
  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatDagMat((double*)out, (double**)gauge, (double*)in, (double)kappa,
          (double)mferm);
    else 
      MatDagMat((double*)out, (float**)gauge, (double*)in, (double)kappa, (double)mferm);
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatDagMat((float*)out, (double**)gauge, (float*)in, (float)kappa,
          (float)mferm);
    else 
      MatDagMat((float*)out, (float**)gauge, (float*)in, (float)kappa, (float)mferm);
  }
*/
}

// Wrapper to templates that handles different precisions.
void matpcdagmatpc(void *out, void **gauge, void *in, double kappa,
	 QudaPrecision sPrecision, QudaPrecision gPrecision, double mferm, QudaMatPCType matpc_type) 
{
/*
  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPCDagMatPC((double*)out, (double**)gauge, (double*)in, (double)kappa,
        matpc_type, (double)mferm);
    else 
      MatPCDagMatPC((double*)out, (float**)gauge, (double*)in, (double)kappa,
                      matpc_type, (double)mferm);
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPCDagMatPC((float*)out, (double**)gauge, (float*)in, (float)kappa,
        matpc_type, (float)mferm);
    else 
      MatPCDagMatPC((float*)out, (float**)gauge, (float*)in, (float)kappa, 
                      matpc_type, (float)mferm);
  }
*/
}


