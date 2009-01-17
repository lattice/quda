#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <util_quda.h>

void sum(float *dst, float *a, float *b, int cnt) {
    for (int i = 0; i < cnt; i++)
        dst[i] = a[i] + b[i];
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
    int X = fullLatticeIndex(i, oddBit);
    int x4 = X/(L3*L2*L1);
    int x3 = (X/(L2*L1)) % L3;
    int x2 = (X/L1) % L2;
    int x1 = X % L1;

    // assert (oddBit == (x+y+z+t)%2);
    
    x4 = (x4+dx4+L4) % L4;
    x3 = (x3+dx3+L3) % L3;
    x2 = (x2+dx2+L2) % L2;
    x1 = (x1+dx1+L1) % L1;
    
    return (x4*(L3*L2*L1) + x3*(L2*L1) + x2*(L1) + x1) / 2;
}

float *gaugeLink(int i, int dir, int oddBit, float **gaugeEven, float **gaugeOdd) {
    float **gaugeField;
    int j;
    
    if (dir % 2 == 0) {
        j = i;
        gaugeField = (oddBit ? gaugeOdd : gaugeEven);
    }
    else {
        switch (dir) {
            case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
            case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
            case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
            case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
            default: j = -1; break;
        }
        gaugeField = (oddBit ? gaugeEven : gaugeOdd);
    }

    return &gaugeField[dir/2][j*(3*3*2)];
}

float *spinorNeighbor(int i, int dir, int oddBit, float *spinorField) {
    int j;
    switch (dir) {
        case 0: j = neighborIndex(i, oddBit, 0, 0, 0, +1); break;
        case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
        case 2: j = neighborIndex(i, oddBit, 0, 0, +1, 0); break;
        case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
        case 4: j = neighborIndex(i, oddBit, 0, +1, 0, 0); break;
        case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
        case 6: j = neighborIndex(i, oddBit, +1, 0, 0, 0); break;
        case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
        default: j = -1; break;
    }
    
    return &spinorField[j*(4*3*2)];
}



void dot(float* res, float* a, float* b) {
    res[0] = res[1] = 0;
    for (int m = 0; m < 3; m++) {
        float a_re = a[2*m+0];
        float a_im = a[2*m+1];
        float b_re = b[2*m+0];
        float b_im = b[2*m+1];
        res[0] += a_re * b_re - a_im * b_im;
        res[1] += a_re * b_im + a_im * b_re;
    }
}

void su3_transpose(float *res, float *mat) {
    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
            res[m*(3*2) + n*(2) + 0] = + mat[n*(3*2) + m*(2) + 0];
            res[m*(3*2) + n*(2) + 1] = - mat[n*(3*2) + m*(2) + 1];
        }
    }
}

void su3_mul(float *res, float *mat, float *vec) {
    for (int n = 0; n < 3; n++) {
        dot(&res[n*(2)], &mat[n*(3*2)], vec);
    }
}

void su3_Tmul(float *res, float *mat, float *vec) {
    float matT[3*3*2];
    su3_transpose(matT, mat);
    su3_mul(res, matT, vec);
}

const float projector[8][4][4][2] = {
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
    }
};


// todo pass projector
void multiplySpinorByDiracProjector(float *res, int projIdx, float *spinorIn) {
    zero(res, 4*3*2);

    for (int s = 0; s < 4; s++) {
        for (int t = 0; t < 4; t++) {
            float projRe = projector[projIdx][s][t][0];
            float projIm = projector[projIdx][s][t][1];
            
            for (int m = 0; m < 3; m++) {
                float spinorRe = spinorIn[t*(3*2) + m*(2) + 0];
                float spinorIm = spinorIn[t*(3*2) + m*(2) + 1];
                res[s*(3*2) + m*(2) + 0] += projRe*spinorRe - projIm*spinorIm;
                res[s*(3*2) + m*(2) + 1] += projRe*spinorIm + projIm*spinorRe;
            }
        }
    }
}


//
// dslashReference()
//
// if oddBit is zero: calculate odd parity spinor elements (using even parity spinor)
// if oddBit is one:  calculate even parity spinor elements
//
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
//
void dslashReference(float *res, float **gaugeFull, float *spinorField, int oddBit, int daggerBit) {
    zero(res, Nh*4*3*2);
    
    float *gaugeEven[4], *gaugeOdd[4];
    for (int dir = 0; dir < 4; dir++) {  
        gaugeEven[dir] = gaugeFull[dir];
        gaugeOdd[dir]  = gaugeFull[dir]+Nh*gaugeSiteSize;
    }
    
    for (int i = 0; i < Nh; i++) {
        for (int dir = 0; dir < 8; dir++) {
            float *gauge = gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd);
            float *spinor = spinorNeighbor(i, dir, oddBit, spinorField);
            
            float projectedSpinor[4*3*2], gaugedSpinor[4*3*2];
            int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
            multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
            
            for (int s = 0; s < 4; s++) {
                if (dir % 2 == 0)
                    su3_mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
                else
                    su3_Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
            }
            
            sum(&res[i*(4*3*2)], &res[i*(4*3*2)], gaugedSpinor, 4*3*2);
        }
    }
}

void Mat(float *out, float **gauge, float *in, float kappa) {
    float *inEven = in;
    float *inOdd  = in + Nh*spinorSiteSize;
    float *outEven = out;
    float *outOdd = out + Nh*spinorSiteSize;
    
    // full dslash operator
    dslashReference(outOdd, gauge, inEven, 1, 0);
    dslashReference(outEven, gauge, inOdd, 0, 0);
    
    // lastly apply the kappa term
    xpay(in, -kappa, out, N*spinorSiteSize);
}

void MatDag(float *out, float **gauge, float *in, float kappa) {
    float *inEven = in;
    float *inOdd  = in + Nh*spinorSiteSize;
    float *outEven = out;
    float *outOdd = out + Nh*spinorSiteSize;
    
    // full dslash operator
    dslashReference(outOdd, gauge, inEven, 1, 1);
    dslashReference(outEven, gauge, inOdd, 0, 1);
    
    // lastly apply the kappa term
    xpay(in, -kappa, out, N*spinorSiteSize);
}

void MatDagMat(float *out, float **gauge, float *in, float kappa) {
    float *tmp = (float*)malloc(N*spinorSiteSize*sizeof(float));
    Mat(tmp, gauge, in, kappa);
    MatDag(out, gauge, tmp, kappa);
    free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void MatPC(float *outEven, float **gauge, float *inEven, float kappa,
	   MatPCType matpc_type) {

    float *tmp = (float*)malloc(Nh*spinorSiteSize*sizeof(float));
    
    // full dslash operator
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashReference(tmp, gauge, inEven, 1, 0);
      dslashReference(outEven, gauge, tmp, 0, 0);
    } else {
      dslashReference(tmp, gauge, inEven, 0, 0);
      dslashReference(outEven, gauge, tmp, 1, 0);
    }    

    // lastly apply the kappa term
    float kappa2 = -kappa*kappa;
    xpay(inEven, kappa2, outEven, Nh*spinorSiteSize);
    free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void MatPCDag(float *outEven, float **gauge, float *inEven, float kappa, 
	      MatPCType matpc_type) {

    float *tmp = (float*)malloc(Nh*spinorSiteSize*sizeof(float));    

    // full dslash operator
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashReference(tmp, gauge, inEven, 1, 1);
      dslashReference(outEven, gauge, tmp, 0, 1);
    } else {
      dslashReference(tmp, gauge, inEven, 0, 1);
      dslashReference(outEven, gauge, tmp, 1, 1);
    }
    
    float kappa2 = -kappa*kappa;
    xpay(inEven, kappa2, outEven, Nh*spinorSiteSize);
    free(tmp);
}

void MatPCDagMatPC(float *out, float **gauge, float *in, float kappa, 
		   MatPCType matpc_type) {
    float *tmp = (float*)malloc(Nh*spinorSiteSize*sizeof(float));
    MatPC(tmp, gauge, in, kappa, matpc_type);
    MatPCDag(out, gauge, tmp, kappa, matpc_type);
    free(tmp);
}
