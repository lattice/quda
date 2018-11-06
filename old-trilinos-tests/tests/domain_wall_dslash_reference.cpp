#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <domain_wall_dslash_reference.h>
#include <blas_reference.h>

#include <gauge_field.h>
#include <color_spinor_field.h>

using namespace quda;

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
        if (x1 -d < 0 && comm_dim_partitioned(0)){
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
        if (x2 -d < 0 && comm_dim_partitioned(1)){
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
        if (x3 -d < 0 && comm_dim_partitioned(2)){
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
        if (x4 -d < 0 && comm_dim_partitioned(3)){
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
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {2,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {2,0}}
  },
  // P_- = P_L
  {
    {{2,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {2,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}},
    {{0,0}, {0,0}, {0,0}, {0,0}}
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
template <QudaDWFPCType type, typename sFloat, typename gFloat>
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
  int sp_idx,gaugeOddBit;
  for (int xs=0;xs<Ls;xs++) {
    for (int gge_idx = 0; gge_idx < Vh; gge_idx++) {
      for (int dir = 0; dir < 8; dir++) {
        sp_idx=gge_idx+Vh*xs;
        // Here is a function call to study.  It is defined near
        // Line 90 of this file.
        // Here we have to switch oddBit depending on the value of xs.  E.g., suppose
        // xs=1.  Then the odd spinor site x1=x2=x3=x4=0 wants the even gauge array
        // element 0, so that we get U_\mu(0).
	gaugeOddBit = (xs%2 == 0 || type == QUDA_4D_PC) ? oddBit : (oddBit+1) % 2;
        gFloat *gauge = gaugeLink_sgpu(gge_idx, dir, gaugeOddBit, gaugeEven, gaugeOdd);
        
        // Even though we're doing the 4d part of the dslash, we need
        // to use a 5d neighbor function, to get the offsets right.
        sFloat *spinor = spinorNeighbor_5d<type>(sp_idx, dir, oddBit, spinorField);
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

#ifdef MULTI_GPU
template <QudaDWFPCType type, typename sFloat, typename gFloat>
void dslashReference_4d_mgpu(sFloat *res, gFloat **gaugeFull, gFloat **ghostGauge, sFloat *spinorField,
			     sFloat **fwdSpinor, sFloat **backSpinor, int oddBit, int daggerBit)
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
	int gaugeOddBit = (xs%2 == 0 || type == QUDA_4D_PC) ? oddBit : (oddBit + 1) % 2;
	
	gFloat *gauge = gaugeLink_mgpu(i, dir, gaugeOddBit, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);//this is unchanged from MPi version
	sFloat *spinor = spinorNeighbor_5d_mgpu<type>(sp_idx, dir, oddBit, spinorField, fwdSpinor, backSpinor, 1, 1);
	
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
#endif

//Currently we consider only spacetime decomposition (not in 5th dim), so this operator is local
template <QudaDWFPCType type, bool zero_initialize=false, typename sFloat>
void dslashReference_5th(sFloat *res, sFloat *spinorField, 
                int oddBit, int daggerBit, sFloat mferm) {
  for (int i = 0; i < V5h; i++) {
    if (zero_initialize) for(int one_site = 0 ; one_site < 24 ; one_site++)
      res[i*(4*3*2)+one_site] = 0.0;
    for (int dir = 8; dir < 10; dir++) {
      // Calls for an extension of the original function.
      // 8 is forward hop, which wants P_+, 9 is backward hop,
      // which wants P_-.  Dagger reverses these.
      sFloat *spinor = spinorNeighbor_5d<type>(i, dir, oddBit, spinorField);
      sFloat projectedSpinor[4*3*2];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector5(projectedSpinor, projIdx, spinor);
      //J  Need a conditional here for s=0 and s=Ls-1.
      int X = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
      int xs = X/(Z[3]*Z[2]*Z[1]*Z[0]);

      if ( (xs == 0 && dir == 9) || (xs == Ls-1 && dir == 8) ) {
        ax(projectedSpinor,(sFloat)(-mferm),projectedSpinor,4*3*2);
      } 
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], projectedSpinor, 4*3*2);
    }
  }
}

//Currently we consider only spacetime decomposition (not in 5th dim), so this operator is local
template <typename sFloat>
void dslashReference_5th_inv(sFloat *res, sFloat *spinorField, 
                int oddBit, int daggerBit, sFloat mferm, double *kappa) {
  double *inv_Ftr = (double*)malloc(Ls*sizeof(sFloat));
  double *Ftr = (double*)malloc(Ls*sizeof(sFloat));
  for(int xs = 0 ; xs < Ls ; xs++)
  {
    inv_Ftr[xs] = 1.0/(1.0+pow(2.0*kappa[xs], Ls)*mferm);
    Ftr[xs] = -2.0*kappa[xs]*mferm*inv_Ftr[xs]; 
    for (int i = 0; i < Vh; i++) {
      memcpy(&res[24*(i+Vh*xs)], &spinorField[24*(i+Vh*xs)], 24*sizeof(sFloat));
    }
  }
  if(daggerBit == 0)
  {
    // s = 0
    for (int i = 0; i < Vh; i++) {
      ax(&res[12+24*(i+Vh*(Ls-1))],(sFloat)(inv_Ftr[0]), &spinorField[12+24*(i+Vh*(Ls-1))], 12);
    }

    // s = 1 ... ls-2
    for(int xs = 0 ; xs <= Ls-2 ; ++xs)
    {
      for (int i = 0; i < Vh; i++) {
        axpy((sFloat)(2.0*kappa[xs]), &res[24*(i+Vh*xs)], &res[24*(i+Vh*(xs+1))], 12);
        axpy((sFloat)Ftr[xs], &res[12+24*(i+Vh*xs)], &res[12+24*(i+Vh*(Ls-1))], 12);
      }
      for (int tmp_s = 0 ; tmp_s < Ls ; tmp_s++)
        Ftr[tmp_s] *= 2.0*kappa[tmp_s];
    }
    for(int xs = 0 ; xs < Ls ; xs++)
    {
      Ftr[xs] = -pow(2.0*kappa[xs],Ls-1)*mferm*inv_Ftr[xs]; 
    }
    // s = ls-2 ... 0
    for(int xs = Ls-2 ; xs >=0 ; --xs)
    {
      for (int i = 0; i < Vh; i++) {
        axpy((sFloat)Ftr[xs], &res[24*(i+Vh*(Ls-1))], &res[24*(i+Vh*xs)], 12);
        axpy((sFloat)(2.0*kappa[xs]), &res[12+24*(i+Vh*(xs+1))], &res[12+24*(i+Vh*xs)], 12);
      }
      for (int tmp_s = 0 ; tmp_s < Ls ; tmp_s++)
        Ftr[tmp_s] /= 2.0*kappa[tmp_s];
    }
    // s = ls -1
    for (int i = 0; i < Vh; i++) {
      ax(&res[24*(i+Vh*(Ls-1))], (sFloat)(inv_Ftr[Ls-1]), &res[24*(i+Vh*(Ls-1))], 12);
    }
  }
  else
  {
    // s = 0
    for (int i = 0; i < Vh; i++) {
      ax(&res[24*(i+Vh*(Ls-1))],(sFloat)(inv_Ftr[0]), &spinorField[24*(i+Vh*(Ls-1))], 12);
    }

    // s = 1 ... ls-2
    for(int xs = 0 ; xs <= Ls-2 ; ++xs)
    {
      for (int i = 0; i < Vh; i++) {
        axpy((sFloat)Ftr[xs], &res[24*(i+Vh*xs)], &res[24*(i+Vh*(Ls-1))], 12);
        axpy((sFloat)(2.0*kappa[xs]), &res[12+24*(i+Vh*xs)], &res[12+24*(i+Vh*(xs+1))], 12);
      }
      for (int tmp_s = 0 ; tmp_s < Ls ; tmp_s++)
        Ftr[tmp_s] *= 2.0*kappa[tmp_s];
    }
    for(int xs = 0 ; xs < Ls ; xs++)
    {
      Ftr[xs] = -pow(2.0*kappa[xs],Ls-1)*mferm*inv_Ftr[xs]; 
    }
    // s = ls-2 ... 0
    for(int xs = Ls-2 ; xs >=0 ; --xs)
    {
      for (int i = 0; i < Vh; i++) {
        axpy((sFloat)(2.0*kappa[xs]), &res[24*(i+Vh*(xs+1))], &res[24*(i+Vh*xs)], 12);
        axpy((sFloat)Ftr[xs], &res[12+24*(i+Vh*(Ls-1))], &res[12+24*(i+Vh*xs)], 12);
      }
      for (int tmp_s = 0 ; tmp_s < Ls ; tmp_s++)
        Ftr[tmp_s] /= 2.0*kappa[tmp_s];
    }
    // s = ls -1
    for (int i = 0; i < Vh; i++) {
      ax(&res[12+24*(i+Vh*(Ls-1))], (sFloat)(inv_Ftr[Ls-1]), &res[12+24*(i+Vh*(Ls-1))], 12);
    }
  }
  free(inv_Ftr);
  free(Ftr);
}


// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void dw_dslash(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) 
{
#ifndef MULTI_GPU
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_4d_sgpu<QUDA_5D_PC>((double*)out, (double**)gauge, (double*)in, oddBit, daggerBit);
    dslashReference_5th<QUDA_5D_PC>((double*)out, (double*)in, oddBit, daggerBit, mferm);
  } else {
    dslashReference_4d_sgpu<QUDA_5D_PC>((float*)out, (float**)gauge, (float*)in, oddBit, daggerBit);
    dslashReference_5th<QUDA_5D_PC>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
  }
#else

    GaugeFieldParam gauge_field_param(gauge, gauge_param);
    gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuGaugeField cpu(gauge_field_param);
    void **ghostGauge = (void**)cpu.Ghost();    
  
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
    csParam.PCtype = QUDA_5D_PC;
  
    cpuColorSpinorField inField(csParam);

    {  // Now do the exchange
      QudaParity otherParity = QUDA_INVALID_PARITY;
      if (oddBit == QUDA_EVEN_PARITY) otherParity = QUDA_ODD_PARITY;
      else if (oddBit == QUDA_ODD_PARITY) otherParity = QUDA_EVEN_PARITY;
      else errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
      const int nFace = 1;

      inField.exchangeGhost(otherParity, nFace, daggerBit);
    }
    void** fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
    void** back_nbr_spinor = inField.backGhostFaceBuffer;
  //NOTE: hopping  in 5th dimension does not use MPI. 
    if (precision == QUDA_DOUBLE_PRECISION) {
      dslashReference_4d_mgpu<QUDA_5D_PC>((double*)out, (double**)gauge, (double**)ghostGauge, (double*)in,(double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
      //dslashReference_4d_sgpu<QUDA_5D_PC>((double*)out, (double**)gauge, (double*)in, oddBit, daggerBit);
      dslashReference_5th<QUDA_5D_PC>((double*)out, (double*)in, oddBit, daggerBit, mferm);
    } else {
      dslashReference_4d_mgpu<QUDA_5D_PC>((float*)out, (float**)gauge, (float**)ghostGauge, (float*)in,
					  (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
      dslashReference_5th<QUDA_5D_PC>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
    }

#endif

}

void dslash_4_4d(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) 
{
#ifndef MULTI_GPU
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_4d_sgpu<QUDA_4D_PC>((double*)out, (double**)gauge, (double*)in, oddBit, daggerBit);
  } else {
    dslashReference_4d_sgpu<QUDA_4D_PC>((float*)out, (float**)gauge, (float*)in, oddBit, daggerBit);
  }
#else

    GaugeFieldParam gauge_field_param(gauge, gauge_param);
    gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuGaugeField cpu(gauge_field_param);
    void **ghostGauge = (void**)cpu.Ghost();    
  
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
    csParam.PCtype = QUDA_4D_PC;
  
    cpuColorSpinorField inField(csParam);

    {  // Now do the exchange
      QudaParity otherParity = QUDA_INVALID_PARITY;
      if (oddBit == QUDA_EVEN_PARITY) otherParity = QUDA_ODD_PARITY;
      else if (oddBit == QUDA_ODD_PARITY) otherParity = QUDA_EVEN_PARITY;
      else errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
      const int nFace = 1;

      inField.exchangeGhost(otherParity, nFace, daggerBit);
    }
    void** fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
    void** back_nbr_spinor = inField.backGhostFaceBuffer;
    if (precision == QUDA_DOUBLE_PRECISION) {
      dslashReference_4d_mgpu<QUDA_4D_PC>((double*)out, (double**)gauge, (double**)ghostGauge, (double*)in,(double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
    } else {
      dslashReference_4d_mgpu<QUDA_4D_PC>((float*)out, (float**)gauge, (float**)ghostGauge, (float*)in,
					  (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
    }

#endif

}

void dw_dslash_5_4d(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize) dslashReference_5th<QUDA_4D_PC, true>((double*)out, (double*)in, oddBit, daggerBit, mferm);
    else dslashReference_5th<QUDA_4D_PC, false>((double*)out, (double*)in, oddBit, daggerBit, mferm);
  } else {
    if (zero_initialize) dslashReference_5th<QUDA_4D_PC, true>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
    else dslashReference_5th<QUDA_4D_PC, false>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
  }
}

void dslash_5_inv(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double *kappa) 
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference_5th_inv((double*)out, (double*)in, oddBit, daggerBit, mferm, kappa);
  } else {
    dslashReference_5th_inv((float*)out, (float*)in, oddBit, daggerBit, (float)mferm, kappa);
  }
}

void mdw_dslash_5(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double *kappa, bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize) dslashReference_5th<QUDA_4D_PC,true>((double*)out, (double*)in, oddBit, daggerBit, mferm);
    else dslashReference_5th<QUDA_4D_PC,false>((double*)out, (double*)in, oddBit, daggerBit, mferm);
  } else {
    if (zero_initialize) dslashReference_5th<QUDA_4D_PC,true>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
    else dslashReference_5th<QUDA_4D_PC,false>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
  }
  for(int xs = 0 ; xs < Ls ; xs++) {
    xpay((char*)in + precision*Vh*spinorSiteSize*xs, kappa[xs], (char*)out + precision*Vh*spinorSiteSize*xs, Vh*spinorSiteSize, precision);
  }
}

void mdw_dslash_4_pre(void *out, void **gauge, void *in, int oddBit, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double *b5, double *c5, bool zero_initialize)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    if (zero_initialize) dslashReference_5th<QUDA_4D_PC, true>((double*)out, (double*)in, oddBit, daggerBit, mferm);
    else dslashReference_5th<QUDA_4D_PC, false>((double*)out, (double*)in, oddBit, daggerBit, mferm);
    for(int xs = 0 ; xs < Ls ; xs++)
    {
      axpby(b5[xs],(double*)in  + Vh*spinorSiteSize*xs,0.5*c5[xs], (double*)out  + Vh*spinorSiteSize*xs, Vh*spinorSiteSize);
    }
  } else {
    if (zero_initialize)  dslashReference_5th<QUDA_4D_PC, true>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
    else dslashReference_5th<QUDA_4D_PC,false>((float*)out, (float*)in, oddBit, daggerBit, (float)mferm);
    for(int xs = 0 ; xs < Ls ; xs++)
    {
      axpby((float)(b5[xs]),(float*)in + Vh*spinorSiteSize*xs, (float)(0.5*c5[xs]), (float*)out + Vh*spinorSiteSize*xs, Vh*spinorSiteSize);
    }
  }
  
}

void dw_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) {

  void *inEven = in;
  void *inOdd  = (char*)in + V5h*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + V5h*spinorSiteSize*precision;

  dw_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param, mferm);
  dw_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param, mferm);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V5*spinorSiteSize, precision);
}

void dw_4d_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm) {

  void *inEven = in;
  void *inOdd  = (char*)in + V5h*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + V5h*spinorSiteSize*precision;

  dslash_4_4d(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param, mferm);
  dw_dslash_5_4d(outOdd, gauge, inOdd, 1, dagger_bit, precision, gauge_param, mferm, false);

  dslash_4_4d(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param, mferm);
  dw_dslash_5_4d(outEven, gauge, inEven, 0, dagger_bit, precision, gauge_param, mferm, false);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V5*spinorSiteSize, precision);
}

void mdw_mat(void *out, void **gauge, void *in, double *kappa_b, double *kappa_c, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double *b5, double *c5) {

  void *tmp = malloc(V5h*spinorSiteSize*precision);
  double *kappa5 = (double*)malloc(Ls*sizeof(double));

  for(int xs = 0; xs < Ls ; xs++) kappa5[xs] = 0.5*kappa_b[xs]/kappa_c[xs];

  void *inEven = in;
  void *inOdd  = (char*)in + V5h*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + V5h*spinorSiteSize*precision;

  mdw_dslash_4_pre(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, b5, c5, true);
  dslash_4_4d(outOdd, gauge, tmp, 1, dagger, precision, gauge_param, mferm);
  mdw_dslash_5(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, kappa5, true);

  for(int xs = 0 ; xs < Ls ; xs++) {
    xpay((char*)tmp + precision*Vh*spinorSiteSize*xs, -kappa_b[xs], (char*)outOdd + precision*Vh*spinorSiteSize*xs,
	 Vh*spinorSiteSize, precision);
  }

  mdw_dslash_4_pre(tmp, gauge, inOdd, 1, dagger, precision, gauge_param, mferm, b5, c5, true);
  dslash_4_4d(outEven, gauge, tmp, 0, dagger, precision, gauge_param, mferm);
  mdw_dslash_5(tmp, gauge, inEven, 0, dagger, precision, gauge_param, mferm, kappa5, true);

  for(int xs = 0 ; xs < Ls ; xs++) {
    xpay((char*)tmp + precision*Vh*spinorSiteSize*xs, -kappa_b[xs], (char*)outEven + precision*Vh*spinorSiteSize*xs,
	 Vh*spinorSiteSize, precision);
  }

  free(kappa5);
  free(tmp);
}

//
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
  xpay(in, kappa2, out, V5h*spinorSiteSize, precision);

  free(tmp);
}


void dw_4d_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm)
{
  double kappa2 = -kappa*kappa;
  double *kappa5 = (double*)malloc(Ls*sizeof(double));
  for(int xs = 0; xs < Ls ; xs++)
    kappa5[xs] = kappa;
  void *tmp = malloc(V5h*spinorSiteSize*precision);
  //------------------------------------------
  double *output = (double*)out;
  for(int k = 0 ; k< V5h*spinorSiteSize; k++)
    output[k] = 0.0;
  //------------------------------------------

  int odd_bit = (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
  bool symmetric =(matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) ? true : false;
  QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};
  
  if (symmetric && !dagger_bit) {
    dslash_4_4d(tmp, gauge, in, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[1], dagger_bit, precision, gauge_param, mferm, kappa5);
    xpay(in, kappa2, out, V5h*spinorSiteSize, precision);
  } else if (symmetric && dagger_bit) {
    dslash_5_inv(tmp, gauge, in, parity[1], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(tmp, gauge, out, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(out, gauge, tmp, parity[1], dagger_bit, precision, gauge_param, mferm);
    xpay(in, kappa2, out, V5h*spinorSiteSize, precision);
  } else {
    dslash_4_4d(tmp, gauge, in, parity[0], dagger_bit, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger_bit, precision, gauge_param, mferm, kappa5);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger_bit, precision, gauge_param, mferm);
    xpay(in, kappa2, tmp, V5h*spinorSiteSize, precision);
    dw_dslash_5_4d(out, gauge, in, parity[1], dagger_bit, precision, gauge_param, mferm, true);
    xpay(tmp, -kappa, out, V5h*spinorSiteSize, precision);
  }
  free(tmp);
  free(kappa5);
}

void mdw_matpc(void *out, void **gauge, void *in, double *kappa_b, double *kappa_c, QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param, double mferm, double *b5, double *c5)
{
  void *tmp = malloc(V5h*spinorSiteSize*precision);  
  double *kappa5 = (double*)malloc(Ls*sizeof(double));
  double *kappa2 = (double*)malloc(Ls*sizeof(double));
  double *kappa_mdwf = (double*)malloc(Ls*sizeof(double));
  for(int xs = 0; xs < Ls ; xs++)
  {
    kappa5[xs] = 0.5*kappa_b[xs]/kappa_c[xs];
    kappa2[xs] = -kappa_b[xs]*kappa_b[xs];
    kappa_mdwf[xs] = -kappa5[xs];
  }

  int odd_bit = (matpc_type == QUDA_MATPC_ODD_ODD || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
  bool symmetric =(matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) ? true : false;
  QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

  if (symmetric && !dagger) {
    mdw_dslash_4_pre(tmp, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    dslash_5_inv(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    for(int xs = 0 ; xs < Ls ; xs++) {
      xpay((char*)in + precision*Vh*spinorSiteSize*xs, kappa2[xs], (char*)out + precision*Vh*spinorSiteSize*xs,
	   Vh*spinorSiteSize, precision);
    }
  } else if (symmetric && dagger) {
    dslash_5_inv(tmp, gauge, in, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    for(int xs = 0 ; xs < Ls ; xs++) {
      xpay((char*)in + precision*Vh*spinorSiteSize*xs, kappa2[xs], (char*)out + precision*Vh*spinorSiteSize*xs,
	   Vh*spinorSiteSize, precision);
    }
  } else if (!symmetric && !dagger) {
    mdw_dslash_4_pre(out, gauge, in, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm);
    dslash_5_inv(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm, kappa_mdwf);
    mdw_dslash_4_pre(tmp, gauge, out, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_4_4d(out, gauge, tmp, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_5(tmp, gauge, in, parity[0], dagger, precision, gauge_param, mferm, kappa5, true);
    for(int xs = 0 ; xs < Ls ; xs++) {
      xpay((char*)tmp + precision*Vh*spinorSiteSize*xs, kappa2[xs], (char*)out + precision*Vh*spinorSiteSize*xs,
	   Vh*spinorSiteSize, precision);
    }
  } else if (!symmetric && dagger) {
    dslash_4_4d(out, gauge, in, parity[0], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm, b5, c5, true);
    dslash_5_inv(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, kappa_mdwf);
    dslash_4_4d(tmp, gauge, out, parity[1], dagger, precision, gauge_param, mferm);
    mdw_dslash_4_pre(out, gauge, tmp, parity[0], dagger, precision, gauge_param, mferm, b5, c5, true);
    mdw_dslash_5(tmp, gauge, in, parity[0], dagger, precision, gauge_param, mferm, kappa5, true);
    for(int xs = 0 ; xs < Ls ; xs++) {
      xpay((char*)tmp + precision*Vh*spinorSiteSize*xs, kappa2[xs], (char*)out + precision*Vh*spinorSiteSize*xs,
	   Vh*spinorSiteSize, precision);
    }
  } else {
    errorQuda("Unsupported matpc_type=%d dagger=%d", matpc_type, dagger);
  }

  free(tmp);
  free(kappa5);
  free(kappa2);
  free(kappa_mdwf);
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


