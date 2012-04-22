#include <stdlib.h>
#include <stdio.h>

#include "quda.h"
#include "hw_quda.h"
#include "util_quda.h"

#define hwSiteSize 12

static ParityHw 
allocateParityHw(int *X, QudaPrecision precision) 
{
    ParityHw ret;
    
    ret.precision = precision;
    ret.X[0] = X[0]/2;
    ret.volume = X[0]/2;
    for (int d=1; d<4; d++) {
	ret.X[d] = X[d];
	ret.volume *= X[d];
    }
    ret.Nc = 3;
    ret.Ns = 2;
    ret.length = ret.volume*ret.Nc*ret.Ns*2;
    
    if (precision == QUDA_DOUBLE_PRECISION) ret.bytes = ret.length*sizeof(double);
    else if (precision == QUDA_SINGLE_PRECISION) ret.bytes = ret.length*sizeof(float);
    else ret.bytes = ret.length*sizeof(short);
    
    if (cudaMalloc((void**)&ret.data, ret.bytes) == cudaErrorMemoryAllocation) {
	printf("Error allocating half wilson\n");
	exit(0);
    }
    
    cudaMemset(ret.data, 0, ret.bytes);
    
    if (precision == QUDA_HALF_PRECISION) { //FIXME not supported yet
      printf("ERROR: half precision not supporte yet in function %s\n", __FUNCTION__);
      //if (cudaMalloc((void**)&ret.dataNorm, 2*ret.bytes/spinorSiteSize) == cudaErrorMemoryAllocation) {
      //printf("Error allocating half wilson Norm\n");
      //exit(0);
      //}
    }
    
    return ret;
}


FullHw 
createHwQuda(int *X, QudaPrecision precision) 
{
    FullHw ret;
    ret.even = allocateParityHw(X, precision);
    ret.odd = allocateParityHw(X, precision);
    return ret;
}


static void
freeParityHwQuda(ParityHw parity_hw) 
{
    
    cudaFree(parity_hw.data);
    if (parity_hw.precision == QUDA_HALF_PRECISION){
	cudaFree(parity_hw.dataNorm);
    }
    
    parity_hw.data = NULL;
    parity_hw.dataNorm = NULL;
}

void 
freeHwQuda(FullHw hw) 
{
    freeParityHwQuda(hw.even);
    freeParityHwQuda(hw.odd);
}


template <typename Float>
static inline void packHwVector(float4* a, Float *b, int Vh) 
{    
    a[0*Vh].x = b[0];
    a[0*Vh].y = b[1];
    a[0*Vh].z = b[2];
    a[0*Vh].w = b[3];
    
    a[1*Vh].x = b[4];
    a[1*Vh].y = b[5];
    a[1*Vh].z = b[6];
    a[1*Vh].w = b[7];
    
    a[2*Vh].x = b[8];
    a[2*Vh].y = b[9];
    a[2*Vh].z = b[10];
    a[2*Vh].w = b[11];
    
}

template <typename Float>
static inline void packHwVector(float2* a, Float *b, int Vh) 
{    
    a[0*Vh].x = b[0];
    a[0*Vh].y = b[1];
    
    a[1*Vh].x = b[2];
    a[1*Vh].y = b[3];
    
    a[2*Vh].x = b[4];
    a[2*Vh].y = b[5];
    
    a[3*Vh].x = b[6];
    a[3*Vh].y = b[7];
    
    a[4*Vh].x = b[8];
    a[4*Vh].y = b[9];
    
    a[5*Vh].x = b[10];
    a[5*Vh].y = b[11];  
}

template <typename Float>
static inline void packHwVector(double2* a, Float *b, int Vh) 
{    
    a[0*Vh].x = b[0];
    a[0*Vh].y = b[1];
    
    a[1*Vh].x = b[2];
    a[1*Vh].y = b[3];
    
    a[2*Vh].x = b[4];
    a[2*Vh].y = b[5];
    
    a[3*Vh].x = b[6];
    a[3*Vh].y = b[7];
    
    a[4*Vh].x = b[8];
    a[4*Vh].y = b[9];
    
    a[5*Vh].x = b[10];
    a[5*Vh].y = b[11];  
}


template <typename Float>
static inline void unpackHwVector(Float *a, float4 *b, int Vh) 
{
    a[0] = a[0*Vh].x;
    a[1] = a[0*Vh].y;
    a[2] = a[0*Vh].z;
    a[3] = a[0*Vh].t;
    
    a[4] = a[1*Vh].x;
    a[5] = a[1*Vh].y;
    a[6] = a[1*Vh].z;
    a[7] = a[1*Vh].t;
    
    a[8] = a[2*Vh].x;
    a[9] = a[2*Vh].y;
    a[10] = a[2*Vh].z;
    a[11] = a[2*Vh].t;      
}


template <typename Float>
static inline void unpackHwVector(Float *a, float2 *b, int Vh) 
{    
    a[0] = b[0*Vh].x;
    a[1] = b[0*Vh].y;
    
    a[2] = b[1*Vh].x;
    a[3] = b[1*Vh].y;
    
    a[4] = b[2*Vh].x;
    a[5] = b[2*Vh].y;
    
    a[6] = b[3*Vh].x;
    a[7] = b[3*Vh].y;
    
    a[8] = b[4*Vh].x;
    a[9] = b[4*Vh].y;
    
    a[10] = b[5*Vh].x;
    a[11] = b[5*Vh].y;   

}

template <typename Float>
static inline void unpackHwVector(Float *a, double2 *b, int Vh) 
{    
    a[0] = b[0*Vh].x;
    a[1] = b[0*Vh].y;
    
    a[2] = b[1*Vh].x;
    a[3] = b[1*Vh].y;
    
    a[4] = b[2*Vh].x;
    a[5] = b[2*Vh].y;
    
    a[6] = b[3*Vh].x;
    a[7] = b[3*Vh].y;
    
    a[8] = b[4*Vh].x;
    a[9] = b[4*Vh].y;
    
    a[10] = b[5*Vh].x;
    a[11] = b[5*Vh].y;   

}

template <typename Float, typename FloatN>
void packParityHw(FloatN *res, Float *hw, int Vh) 
{
    for (int i = 0; i < Vh; i++) {
	packHwVector(res+i, hw+hwSiteSize*i, Vh);
    }
}

template <typename Float, typename FloatN>
static void unpackParityHw(Float *res, FloatN *hwPacked, int Vh) {

  for (int i = 0; i < Vh; i++) {
      unpackHwVector(res+i*hwSiteSize, hwPacked+i, Vh);
  }
}



void
static loadParityHw(ParityHw ret, void *hw, QudaPrecision cpu_prec)
{
    void *packedHw1 = 0;
    
    if (ret.precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error, cannot have CUDA double precision without double CPU precision\n");
	exit(-1);
    }
    
    if (ret.precision != QUDA_HALF_PRECISION) {	
      if (cudaMallocHost(&packedHw1, ret.bytes) == cudaErrorMemoryAllocation) {
	  errorQuda("ERROR: cudaMallocHost failed for packedHw1\n");
	}
	
	if (ret.precision == QUDA_DOUBLE_PRECISION) {
	    packParityHw((double2*)packedHw1, (double*)hw, ret.volume);
	} else {
	    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
		packParityHw((float2*)packedHw1, (double*)hw, ret.volume);
	    }
	    else {
		packParityHw((float2*)packedHw1, (float*)hw, ret.volume);
	    }
	}
	cudaMemcpy(ret.data, packedHw1, ret.bytes, cudaMemcpyHostToDevice);
	cudaFreeHost(packedHw1);
    } else {
	
	//half precision
	/*
	  ParityHw tmp = allocateParityHw(ret.X, QUDA_SINGLE_PRECISION);
	  loadParityHw(tmp, hw, cpu_prec, dirac_order);
	  copyCuda(ret, tmp);
	  freeParityHw(tmp);
	*/
    }
    
}


void
loadHwToGPU(FullHw ret, void *hw, QudaPrecision cpu_prec)
{
    void *hw_odd;
    if (cpu_prec == QUDA_SINGLE_PRECISION){
	hw_odd = ((float*)hw) + ret.even.length;
    }else{
	hw_odd = ((double*)hw) + ret.even.length;
    }
    
    loadParityHw(ret.even, hw, cpu_prec);
    loadParityHw(ret.odd, hw_odd, cpu_prec);
    
}

static void 
retrieveParityHw(void *res, ParityHw hw, QudaPrecision cpu_prec)
{
    void *packedHw1 = 0;
    if (hw.precision != QUDA_HALF_PRECISION) {
      if (cudaMallocHost((void**)&packedHw1, hw.bytes) == cudaErrorMemoryAllocation) {
	errorQuda("ERROR: cudaMallocHost failed for packedHw1\n");
      }
	cudaMemcpy(packedHw1, hw.data, hw.bytes, cudaMemcpyDeviceToHost);
	
	if (hw.precision == QUDA_DOUBLE_PRECISION) {
	    unpackParityHw((double*)res, (double2*)packedHw1, hw.volume);
	} else {
	    if (cpu_prec == QUDA_DOUBLE_PRECISION){
		unpackParityHw((double*)res, (float2*)packedHw1, hw.volume);
	    }
	    else {
		unpackParityHw((float*)res, (float2*)packedHw1, hw.volume);
	    }
	}
	cudaFreeHost(packedHw1);
	
    } else {
	//half precision
	/*
	  ParityHw tmp = allocateParityHw(hw.X, QUDA_SINGLE_PRECISION);
	  copyCuda(tmp, hw);
	  retrieveParityHw(res, tmp, cpu_prec, dirac_order);
	  freeParityHw(tmp);
	*/
    }
}


void 
retrieveHwField(void *res, FullHw hw, QudaPrecision cpu_prec)
{
    void *res_odd;
    if (cpu_prec == QUDA_SINGLE_PRECISION) res_odd = (float*)res + hw.even.length;
    else res_odd = (double*)res + hw.even.length;
        
    retrieveParityHw(res, hw.even, cpu_prec);
    retrieveParityHw(res_odd, hw.odd, cpu_prec);
    
}


/*
void hwHalfPack(float *c, short *s0, float *f0, int V) {

  float *f = f0;
  short *s = s0;
  for (int i=0; i<24*V; i+=24) {
    c[i] = sqrt(f[0]*f[0] + f[1]*f[1]);
    for (int j=0; j<24; j+=2) {
      float k = sqrt(f[j]*f[j] + f[j+1]*f[j+1]);
      if (k > c[i]) c[i] = k;
    }

    for (int j=0; j<24; j++) s[j] = (short)(MAX_SHORT*f[j]/c[i]);
    f+=24;
    s+=24;
  }

}

void hwHalfUnpack(float *f0, float *c, short *s0, int V) {
  float *f = f0;
  short *s = s0;
  for (int i=0; i<24*V; i+=24) {
    for (int j=0; j<24; j++) f[j] = s[j] * (c[i] / MAX_SHORT);
    f+=24;
    s+=24;
  }

}
*/
