#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <fat_force_quda.h>
#include <quda_internal.h>
#include <face_quda.h>
#include "misc_helpers.h"
#include <assert.h>
#include <cuda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define ALIGNMENT 4096 

#ifdef MPI_COMMS
#include "face_quda.h"
#endif

static double Anisotropy;
extern float fat_link_max;
static int X[4];
static QudaTboundary tBoundary;
static int volumeCB; // checkboarded volume
static int faceVolumeCB[4]; // checkboarded face volume

#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) / 2.f)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))

#include <pack_gauge.h>

/********************** Staple code, used by link fattening **************/

#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)

template <typename Float>
void packGhostAllLinks(Float **cpuLink, Float **cpuGhostBack,Float**cpuGhostFwd, int dir, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };
  
  
  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }
    
    //collect back ghost gauge field
    //for(int dir =0; dir < 4; dir++){
      int d;
      int a,b,c;
      
      //we need copy all 4 links in the same location
      for(int linkdir=0; linkdir < 4; linkdir ++){
	Float* even_src = cpuLink[linkdir];
	Float* odd_src = cpuLink[linkdir] + volumeCB*gaugeSiteSize;

	Float* even_dst;
	Float* odd_dst;
	
	//switching odd and even ghost cpuLink when that dimension size is odd
	//only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
	if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
	  even_dst = dst[dir] + 2*linkdir* nFace *faceVolumeCB[dir]*gaugeSiteSize;	
	  odd_dst = even_dst + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
	}else{
	  odd_dst = dst[dir] + 2*linkdir* nFace *faceVolumeCB[dir]*gaugeSiteSize;
	  even_dst = dst[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
	}

	int even_dst_index = 0;
	int odd_dst_index = 0;
	
	int startd;
	int endd; 
	if(ite == 0){ //back
	  startd = 0; 
	  endd= nFace;
	}else{//fwd
	  startd = X[dir] - nFace;
	  endd =X[dir];
	}
	for(d = startd; d < endd; d++){
	  for(a = 0; a < A[dir]; a++){
	    for(b = 0; b < B[dir]; b++){
	      for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  for(int i=0;i < 18;i++){
		    even_dst[18*even_dst_index+i] = even_src[18*index + i];
		  }
		  even_dst_index++;
		}else{ //odd
		  for(int i=0;i < 18;i++){
		    odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		  }
		  odd_dst_index++;
		}
	      }//c
	    }//b
	  }//a
	}//d
	assert( even_dst_index == nFace*faceVolumeCB[dir]);
	assert( odd_dst_index == nFace*faceVolumeCB[dir]);	
      }//linkdir
      
    //}//dir
  }//ite
}

void pack_ghost_all_links(void **cpuLink, void **cpuGhostBack, void** cpuGhostFwd, int dir, int nFace, QudaPrecision precision) {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllLinks((double**)cpuLink, (double**)cpuGhostBack, (double**) cpuGhostFwd, dir,  nFace);
  } else {
    packGhostAllLinks((float**)cpuLink, (float**)cpuGhostBack, (float**)cpuGhostFwd, dir, nFace);
  }
  
}




template <typename Float>
void packGhostAllStaples(Float *cpuStaple, Float **cpuGhostBack,Float**cpuGhostFwd, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };
  
  
  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }
    
    //collect back ghost staple
    for(int dir =0; dir < 4; dir++){
      int d;
      int a,b,c;
      
      //ther is only one staple in the same location
      for(int linkdir=0; linkdir < 1; linkdir ++){
	Float* even_src = cpuStaple;
	Float* odd_src = cpuStaple + volumeCB*gaugeSiteSize;

	Float* even_dst;
	Float* odd_dst;
	
	//switching odd and even ghost cpuLink when that dimension size is odd
	//only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
	if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
	  even_dst = dst[dir];
	  odd_dst = even_dst + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
	}else{
	  odd_dst = dst[dir];
	  even_dst = dst[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
	}

	int even_dst_index = 0;
	int odd_dst_index = 0;
	
	int startd;
	int endd; 
	if(ite == 0){ //back
	  startd = 0; 
	  endd= nFace;
	}else{//fwd
	  startd = X[dir] - nFace;
	  endd =X[dir];
	}
	for(d = startd; d < endd; d++){
	  for(a = 0; a < A[dir]; a++){
	    for(b = 0; b < B[dir]; b++){
	      for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  for(int i=0;i < 18;i++){
		    even_dst[18*even_dst_index+i] = even_src[18*index + i];
		  }
		  even_dst_index++;
		}else{ //odd
		  for(int i=0;i < 18;i++){
		    odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		  }
		  odd_dst_index++;
		}
	      }//c
	    }//b
	  }//a
	}//d
	assert( even_dst_index == nFace*faceVolumeCB[dir]);
	assert( odd_dst_index == nFace*faceVolumeCB[dir]);	
      }//linkdir
      
    }//dir
  }//ite
}


void pack_ghost_all_staples_cpu(void *staple, void **cpuGhostStapleBack, void** cpuGhostStapleFwd, int nFace, QudaPrecision precision) {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllStaples((double*)staple, (double**)cpuGhostStapleBack, (double**) cpuGhostStapleFwd, nFace);
  } else {
    packGhostAllStaples((float*)staple, (float**)cpuGhostStapleBack, (float**)cpuGhostStapleFwd, nFace);
  }
  
}

void pack_gauge_diag(void* buf, int* X, void** sitelink, int nu, int mu, int dir1, int dir2, QudaPrecision prec)
{
    /*
      nu |          |
         |__________|
            mu 
    *	    
    * nu, mu are the directions we are working on
    * Since we are packing our own data, we need to go to the north-west corner in the diagram
    * i.e. x[nu] = X[nu]-1, x[mu]=0, and looop throught x[dir1],x[dir2]
    * in the remaining two directions (dir1/dir2), dir2 is the slowest changing dim when computing
    * index
    */


  int mul_factor[4]={
    1, X[0], X[1]*X[0], X[2]*X[1]*X[0],
  };

  int even_dst_idx = 0;
  int odd_dst_idx = 0;
  char* dst_even =(char*)buf;
  char* dst_odd = dst_even + (X[dir1]*X[dir2]/2)*gaugeSiteSize*prec;
  char* src_even = (char*)sitelink[nu];
  char* src_odd = src_even + (X[0]*X[1]*X[2]*X[3]/2)*gaugeSiteSize*prec;

  if( (X[nu]+X[mu]) % 2 == 1){
    //oddness will change between me and the diagonal neighbor
    //switch it now
    char* tmp = dst_odd;
    dst_odd = dst_even;
    dst_even = tmp;
  }

  for(int i=0;i < X[dir2]; i++){
    for(int j=0; j < X[dir1]; j++){
      int src_idx = ((X[nu]-1)*mul_factor[nu]+ 0*mul_factor[mu]+i*mul_factor[dir2]+j*mul_factor[dir1])>>1;
      //int dst_idx = (i*X[dir1]+j) >> 1; 
      int oddness = ( (X[nu]-1) + 0 + i + j) %2;

      if(oddness==0){
	for(int tmpidx = 0; tmpidx < gaugeSiteSize; tmpidx++){
	  memcpy(&dst_even[(18*even_dst_idx+tmpidx)*prec], &src_even[(18*src_idx + tmpidx)*prec], prec);
	}
	even_dst_idx++;
      }else{
	for(int tmpidx = 0; tmpidx < gaugeSiteSize; tmpidx++){	
	  memcpy(&dst_odd[(18*odd_dst_idx+tmpidx)*prec], &src_odd[(18*src_idx + tmpidx)*prec], prec);
	}
	odd_dst_idx++;
      }//if

    }//for j
  }//for i
      
  if( (even_dst_idx != X[dir1]*X[dir2]/2)|| (odd_dst_idx != X[dir1]*X[dir2]/2)){
    errorQuda("even_dst_idx/odd_dst_idx(%d/%d) does not match the value of X[dir1]*X[dir2]/2 (%d)\n",
	      even_dst_idx, odd_dst_idx, X[dir1]*X[dir2]/2);
  }
  return ;
  

}


template <typename Float, typename FloatN>
void
packGaugeField(FloatN *res, Float *gauge, int oddBit, ReconstructType reconstruct, int Vh)
{
    int dir, i;
    if (reconstruct == QUDA_RECONSTRUCT_12) {
        for (dir = 0; dir < 4; dir++) {
            Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
            for (i = 0; i < Vh; i++) {
                pack12(res+i, g+4*i*gaugeSiteSize+dir*gaugeSiteSize, dir, Vh);
            }
        }
    } else if (reconstruct == QUDA_RECONSTRUCT_8){
        for (dir = 0; dir < 4; dir++) {
            Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
            for (i = 0; i < Vh; i++) {
                pack8(res+i, g+4*i*gaugeSiteSize + dir*gaugeSiteSize, dir, Vh);
            }
        }
    }else{
        for (dir = 0; dir < 4; dir++) {
            Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
            for (i = 0; i < Vh; i++) {
                pack18(res+i, g+i*gaugeSiteSize+dir, dir*gaugeSiteSize, Vh);
            }
        }
    }
}

template <typename Float, typename FloatN>
void
loadGaugeFromCPUArrayQuda(FloatN *even, FloatN *odd, Float *cpuGauge,
                          ReconstructType reconstruct, int bytes, int Vh)
{

    // Use pinned memory

    FloatN *packedEven, *packedOdd;
    cudaMallocHost((void**)&packedEven, bytes);
    cudaMallocHost((void**)&packedOdd, bytes);


    packGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh);
    packGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh);


    cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);

    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
}



void 
loadLinkToGPU_gf(FullGauge cudaGauge, void *cpuGauge, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;

    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
        loadGaugeFromCPUArrayQuda((double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double*)cpuGauge,
                                  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
    } else if (cuda_prec == QUDA_SINGLE_PRECISION) {
        if (cpu_prec == QUDA_DOUBLE_PRECISION){
            if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
                loadGaugeFromCPUArrayQuda((float4*)(cudaGauge.even), (float4*)(cudaGauge.odd), (double*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }else{
                loadGaugeFromCPUArrayQuda((float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (double*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }
        }
        else if (cpu_prec == QUDA_SINGLE_PRECISION){
            if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
                loadGaugeFromCPUArrayQuda((float4*)(cudaGauge.even), (float4*)(cudaGauge.odd), (float*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }else{
                loadGaugeFromCPUArrayQuda((float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }
        }
    } else if (cuda_prec == QUDA_HALF_PRECISION) {
        if (cpu_prec == QUDA_DOUBLE_PRECISION){
            if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
                loadGaugeFromCPUArrayQuda((short4*)(cudaGauge.even), (short4*)(cudaGauge.odd), (double*)cpuGauge,
                               cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }else{
                loadGaugeFromCPUArrayQuda((short2*)(cudaGauge.even), (short2*)(cudaGauge.odd), (double*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }
        }
        else if (cpu_prec == QUDA_SINGLE_PRECISION){
            if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
                loadGaugeFromCPUArrayQuda((short4*)(cudaGauge.even), (short4*)(cudaGauge.odd), (float*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }else{
                loadGaugeFromCPUArrayQuda((short2*)(cudaGauge.even), (short2*)(cudaGauge.odd), (float*)cpuGauge,
                                          cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB);
            }
        }
    }
}


static void 
allocateStapleQuda(FullStaple *cudaStaple, QudaPrecision precision) 
{
    cudaStaple->precision = precision;    
    cudaStaple->Nc = 3;
    
    if (precision == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: stape does not support half precision\n");
    }
    
    int elements = 18;
    
    cudaStaple->bytes = cudaStaple->stride*elements*precision;
    
    cudaMalloc((void **)&cudaStaple->even, cudaStaple->bytes);
    cudaMalloc((void **)&cudaStaple->odd, cudaStaple->bytes); 	    
}

void set_dim_ff(int *XX) {

  volumeCB = 1;
  for (int i=0; i<4; i++) {
    X[i] = XX[i];
    volumeCB *= X[i];
    faceVolumeCB[i] = 1;
    for (int j=0; j<4; j++) {
      if (i==j) continue;
      faceVolumeCB[i] *= XX[j];
    }
    faceVolumeCB[i] /= 2;
  }
  volumeCB /= 2;

}


void
createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
    }

    set_dim_ff(param->X);
    cudaStaple->volume = 1;
    for (int d=0; d<4; d++) {
      cudaStaple->X[d] = param->X[d];
      cudaStaple->volume *= param->X[d];
    }
    Anisotropy = param->anisotropy;
    tBoundary = param->t_boundary;

    //cudaStaple->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaStaple->volume /= 2;    
    cudaStaple->pad = param->staple_pad;
    cudaStaple->stride = cudaStaple->volume + cudaStaple->pad;
    allocateStapleQuda(cudaStaple,  param->cuda_prec);
    
    return;
}


void
freeStapleQuda(FullStaple *cudaStaple) 
{
    if (cudaStaple->even) {
	cudaFree(cudaStaple->even);
    }
    if (cudaStaple->odd) {
	cudaFree(cudaStaple->odd);
    }
    cudaStaple->even = NULL;
    cudaStaple->odd = NULL;
}
void
packGhostStaple(FullStaple* cudaStaple, int dir, int whichway,
		void** fwd_nbr_buf_gpu, void** back_nbr_buf_gpu,
		void** fwd_nbr_buf, void** back_nbr_buf,
		cudaStream_t* stream)
{
  int* X = cudaStaple->X;
  int Vs_x, Vs_y, Vs_z, Vs_t;
  
  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];  
  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  
  if (dir != 3){ //the code would work for dir=3 as well
    //even and odd ness switch (if necessary) is taken caren of in collectGhostStaple();
    int prec= cudaStaple->precision;
    void* gpu_buf;
    int i =dir;
    if (whichway ==  QUDA_BACKWARDS){
      gpu_buf = back_nbr_buf_gpu[i];
      collectGhostStaple(cudaStaple, gpu_buf, i, whichway, stream);
      cudaMemcpyAsync(back_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);
    }else{//whichway is  QUDA_FORWARDS;
      gpu_buf = fwd_nbr_buf_gpu[i];
      collectGhostStaple(cudaStaple, gpu_buf, i, whichway, stream);
      cudaMemcpyAsync(fwd_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);        
    }
  }else{ //special case for dir=3 since no gather kernel is required
    void* even = cudaStaple->even;
    void* odd = cudaStaple->odd;
    int Vh = cudaStaple->volume;
    int Vsh = cudaStaple->X[0]*cudaStaple->X[1]*cudaStaple->X[2]/2;
    int prec= cudaStaple->precision;
    int sizeOfFloatN = 2*prec;
    int len = Vsh*sizeOfFloatN;
    int i;
    if(cudaStaple->X[3] %2 == 0){
      //back,even
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	void* src = ((char*)even) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //back, odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)odd) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd,even
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd, odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
    }else{
      //reverse even and odd position
      //back,odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	void* src = ((char*)odd) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //back, even
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)even) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd,odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd, even
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      
    } 
  }
  
}


void 
unpackGhostStaple(FullStaple* cudaStaple, int dir, int whichway, void** fwd_nbr_buf, void** back_nbr_buf,
		  cudaStream_t* stream)
{

  int* X = cudaStaple->X;
  int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
  
  Vsh_x = X[1]*X[2]*X[3]/2;
  Vsh_y = X[0]*X[2]*X[3]/2;
  Vsh_z = X[0]*X[1]*X[3]/2;
  Vsh_t = X[0]*X[1]*X[2]/2;  
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};

  char* even = (char*)cudaStaple->even;
  char* odd = (char*)cudaStaple->odd;
  int Vh = cudaStaple->volume;
  //int Vsh = cudaStaple->X[0]*cudaStaple->X[1]*cudaStaple->X[2]/2;
  int prec= cudaStaple->precision;
  int sizeOfFloatN = 2*prec;
  int len[4] = {
    Vsh_x*sizeOfFloatN,
    Vsh_y*sizeOfFloatN,
    Vsh_z*sizeOfFloatN,
    Vsh_t*sizeOfFloatN 
  };
  
  int tmpint[4] = {
    0,
    Vsh_x, 
    Vsh_x + Vsh_y, 
    Vsh_x + Vsh_y + Vsh_z, 
  };
  
  even = ((char*)cudaStaple->even) + Vh*sizeOfFloatN + 2*tmpint[dir]*sizeOfFloatN;
  odd = ((char*)cudaStaple->odd) + Vh*sizeOfFloatN +2*tmpint[dir]*sizeOfFloatN;
  
  if(whichway == QUDA_BACKWARDS){   
    //back,even
    for(int i=0;i < 9; i++){
      void* dst = even + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)back_nbr_buf[dir]) + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
    //back, odd
    for(int i=0;i < 9; i++){
      void* dst = odd + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)back_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
  }else { //QUDA_FORWARDS
    //fwd,even
    for(int i=0;i < 9; i++){
      void* dst = even + Vsh[dir]*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)fwd_nbr_buf[dir]) + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
    //fwd, odd
    for(int i=0;i < 9; i++){
      void* dst = odd + Vsh[dir]*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)fwd_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
  }
}
#endif


/******************************** Mom code, used by Fermi force code ****************************/
#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)

static void 
allocateMomQuda(FullMom *cudaMom, QudaPrecision precision) 
{
    cudaMom->precision = precision;    
    
    if (precision == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: stape does not support half precision\n");
    }
    
    int elements = 10;
     
    cudaMom->bytes = cudaMom->volume*elements*precision*4;
    
    cudaMalloc((void **)&cudaMom->even, cudaMom->bytes);
    cudaMalloc((void **)&cudaMom->odd, cudaMom->bytes); 	    
}

void
createMomQuda(FullMom* cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
    }
    
    set_dim_ff(param->X);
    cudaMom->volume = 1;
    for (int d=0; d<4; d++) {
      cudaMom->X[d] = param->X[d];
      cudaMom->volume *= param->X[d];
    }
    Anisotropy = param->anisotropy;
    tBoundary = param->t_boundary;
 
    cudaMom->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaMom->volume /= 2;    
    
    allocateMomQuda(cudaMom,  param->cuda_prec);
    
    return;
}


void
freeMomQuda(FullMom *cudaMom) 
{
    if (cudaMom->even) {
	cudaFree(cudaMom->even);
    }
    if (cudaMom->odd) {
	cudaFree(cudaMom->odd);
    }
    cudaMom->even = NULL;
    cudaMom->odd = NULL;
}

template <typename Float, typename Float2>
inline void pack10(Float2 *res, Float *m, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	r[j*Vh].x = (float)m[j*2+0]; 
	r[j*Vh].y = (float)m[j*2+1]; 
    }
}

template <typename Float, typename Float2>
void packMomField(Float2 *res, Float *mom, int oddBit, int Vh) 
{    
    for (int dir = 0; dir < 4; dir++) {
	Float *g = mom + (oddBit*Vh*4 + dir)*momSiteSize;
	for (int i = 0; i < Vh; i++) {
	    pack10(res+i, g + 4*i*momSiteSize, dir, Vh);
	}
    }      
}

template <typename Float, typename Float2>
void loadMomField(Float2 *even, Float2 *odd, Float *mom,
		  int bytes, int Vh) 
{
    
    Float2 *packedEven, *packedOdd;
    cudaMallocHost((void**)&packedEven, bytes); 
    cudaMallocHost((void**)&packedOdd, bytes);
    
    packMomField(packedEven, (Float*)mom, 0, Vh);
    packMomField(packedOdd,  (Float*)mom, 1, Vh);
    
    cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice); 
    
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
}




void
loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param)
{
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION) {
	//loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
	//cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    } else { //single precision
	loadMomField((float2*)(cudaMom.even), (float2*)(cudaMom.odd), (float*)mom, 
		     cudaMom.bytes, cudaMom.volume);
	
    }
}


template <typename Float, typename Float2>
inline void unpack10(Float* m, Float2 *res, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	m[j*2+0] = r[j*Vh].x;
	m[j*2+1] = r[j*Vh].y;
    }
    
}

template <typename Float, typename Float2>
void 
unpackMomField(Float* mom, Float2 *res, int oddBit, int Vh) 
{
    int dir, i;
    Float *m = mom + oddBit*Vh*momSiteSize*4;
    
    for (i = 0; i < Vh; i++) {
	for (dir = 0; dir < 4; dir++) {	
	    Float* thismom = m + (4*i+dir)*momSiteSize;
	    unpack10(thismom, res+i, dir, Vh);
	}
    }
}

template <typename Float, typename Float2>
void 
storeMomToCPUArray(Float* mom, Float2 *even, Float2 *odd, 
		   int bytes, int Vh) 
{    
    Float2 *packedEven, *packedOdd;   
    cudaMallocHost((void**)&packedEven, bytes); 
    cudaMallocHost((void**)&packedOdd, bytes); 
    cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);  

    unpackMomField((Float*)mom, packedEven,0, Vh);
    unpackMomField((Float*)mom, packedOdd, 1, Vh);
        
    cudaFreeHost(packedEven); 
    cudaFreeHost(packedOdd); 
}

void 
storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec != cuda_prec){
	printf("Error:%s: cpu and gpu precison has to be the same at this moment \n", __FUNCTION__);
	exit(1);	
    }
    
    if (cpu_prec == QUDA_HALF_PRECISION){
	printf("ERROR: %s:  half precision is not supported at this moment\n", __FUNCTION__);
	exit(1);
    }
    
    if (cpu_prec == QUDA_DOUBLE_PRECISION){
	
    }else { //SINGLE PRECISIONS
	storeMomToCPUArray( (float*)mom, (float2*) cudaMom.even, (float2*)cudaMom.odd, 
			    cudaMom.bytes, cudaMom.volume);	
    }
    
}
#endif

/********** link code, used by link fattening  **********************/

#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)

static void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, QudaPrecision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  cudaGauge->Nc = 3;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  if (cudaGauge->even || cudaGauge->odd){
    errorQuda("Error: even/odd field is not null, probably already allocated(even=%p, odd=%p)\n", cudaGauge->even, cudaGauge->odd);
  }
 
  cudaGauge->bytes = 4*cudaGauge->stride*reconstruct*floatSize;
  if (!cudaGauge->even) {
    if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even gauge field");
    }
  }
   
  if (!cudaGauge->odd) {
    if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even odd gauge field");
    }
  }

}

void
createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
	printf("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
	exit(-1);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }
        
    Anisotropy = param->anisotropy;
    tBoundary = param->t_boundary;

    set_dim_ff(param->X);
    cudaGauge->anisotropy = param->anisotropy;
    cudaGauge->volumeCB = 1;
    for (int d=0; d<4; d++) {
      cudaGauge->X[d] = param->X[d];
      cudaGauge->volumeCB *= param->X[d];
    }
    //cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaGauge->volumeCB /= 2;    
    cudaGauge->pad = param->ga_pad;
    cudaGauge->stride = cudaGauge->volumeCB + cudaGauge->pad;
    cudaGauge->reconstruct = param->reconstruct;

    allocateGaugeField(cudaGauge, param->reconstruct, param->cuda_prec);
    
    return;
}


template<typename FloatN, typename Float>
static void 
do_loadLinkToGPU(int* X, FloatN *even, FloatN *odd, Float **cpuGauge, Float** ghost_cpuGauge,
		 Float** ghost_cpuGauge_diag, 
		 ReconstructType reconstruct, int bytes, int Vh, int pad, 
		 int Vsh_x, int Vsh_y, int Vsh_z, int Vsh_t,
		 QudaPrecision prec) 
{
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }

  
  int Vh_2d_max = MAX(X[0]*X[1]/2, X[0]*X[2]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[0]*X[3]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[1]*X[2]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[1]*X[3]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[2]*X[3]/2);

  int i;
  char* tmp_even;
  char* tmp_odd;
  int len = Vh*gaugeSiteSize*sizeof(Float);

#ifdef MULTI_GPU    
  int glen[4] = {
    Vsh_x*gaugeSiteSize*sizeof(Float),
    Vsh_y*gaugeSiteSize*sizeof(Float),
    Vsh_z*gaugeSiteSize*sizeof(Float),
    Vsh_t*gaugeSiteSize*sizeof(Float)
  };
  
  int ghostV = 2*(Vsh_x+Vsh_y+Vsh_z+Vsh_t)+4*Vh_2d_max;
#else
  int ghostV = 0;
#endif  

  int glen_sum = ghostV*gaugeSiteSize*sizeof(Float);
  cudaMalloc(&tmp_even, 4*(len+glen_sum)); CUERR;
  cudaMalloc(&tmp_odd, 4*(len+glen_sum)); CUERR;
  
  //even links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_even + i*(len+glen_sum), cpuGauge[i], len, cudaMemcpyHostToDevice, streams[0]); 
#else
    cudaMemcpy(tmp_even + i*(len+glen_sum), cpuGauge[i], len, cudaMemcpyHostToDevice); 
#endif
  
#ifdef MULTI_GPU  
    //dir: the source direction
    char* dest = tmp_even + i*(len+glen_sum)+len;
    for(int dir = 0; dir < 4; dir++){
#if (CUDA_VERSION >=4000)
      cudaMemcpyAsync(dest, ((char*)ghost_cpuGauge[dir])+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[0]); 
      cudaMemcpyAsync(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[0]); 	
#else
      cudaMemcpy(dest, ((char*)ghost_cpuGauge[dir])+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); 
      cudaMemcpy(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); 
#endif
      dest += 2*glen[dir];
  }
    //fill in diag 
    //@nu is @i, mu iterats from 0 to 4 and mu != nu
    int nu = i;
    for(int mu = 0; mu < 4; mu++){
      if(nu  == mu ){
	continue;
      }
      int dir1, dir2;
      for(dir1=0; dir1 < 4; dir1 ++){
	if(dir1 != nu && dir1 != mu){
	  break;
	}
      }
      for(dir2=0; dir2 < 4; dir2 ++){
	if(dir2 != nu && dir2 != mu && dir2 != dir1){
	  break;
	}
      }
#if (CUDA_VERSION >=4000)
      cudaMemcpyAsync(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu], 
		      X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice, streams[0]);	
#else	
      cudaMemcpy(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu], 
		 X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice);	
#endif
      
    }
    
#endif
  }    
  
  link_format_cpu_to_gpu((void*)even, (void*)tmp_even,  reconstruct, bytes, Vh, pad, ghostV, prec, streams[0]); CUERR;

  //odd links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_odd + i*(len+glen_sum), cpuGauge[i] + Vh*gaugeSiteSize, len, cudaMemcpyHostToDevice, streams[1]);CUERR;
#else
    cudaMemcpy(tmp_odd + i*(len+glen_sum), cpuGauge[i] + Vh*gaugeSiteSize, len, cudaMemcpyHostToDevice);CUERR;
#endif

#ifdef MULTI_GPU  
      char* dest = tmp_odd + i*(len+glen_sum)+len;
      for(int dir = 0; dir < 4; dir++){
#if (CUDA_VERSION >=4000)
	cudaMemcpyAsync(dest, ((char*)ghost_cpuGauge[dir])+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[1]); CUERR;
	cudaMemcpyAsync(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+glen[dir] +i*2*glen[dir], glen[dir], 
			cudaMemcpyHostToDevice, streams[1]); CUERR;
#else
	cudaMemcpy(dest, ((char*)ghost_cpuGauge[dir])+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); CUERR;
	cudaMemcpy(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); CUERR;

#endif

	dest += 2*glen[dir];
      }
      //fill in diag 
      //@nu is @i, mu iterats from 0 to 4 and mu != nu
      int nu = i;
      for(int mu = 0; mu < 4; mu++){
	if(nu  == mu ){
	  continue;
	}
	int dir1, dir2;
	for(dir1=0; dir1 < 4; dir1 ++){
	  if(dir1 != nu && dir1 != mu){
	    break;
	  }
	}
	for(dir2=0; dir2 < 4; dir2 ++){
	  if(dir2 != nu && dir2 != mu && dir2 != dir1){
	    break;
	  }
	}
#if (CUDA_VERSION >=4000)
	cudaMemcpyAsync(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu]+X[dir1]*X[dir2]/2*gaugeSiteSize, 
			X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice, streams[1]);	
#else
	cudaMemcpy(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu]+X[dir1]*X[dir2]/2*gaugeSiteSize, 
		   X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice );		
#endif
      }
      

#endif
  }
  link_format_cpu_to_gpu((void*)odd, (void*)tmp_odd, reconstruct, bytes, Vh, pad, ghostV, prec, streams[1]); CUERR;
  
  for(int i=0;i < 2;i++){
    cudaStreamSynchronize(streams[i]);
  }

  cudaFree(tmp_even);
  cudaFree(tmp_odd);

  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }
  CUERR;
}


void 
loadLinkToGPU(FullGauge cudaGauge, void **cpuGauge, QudaGaugeParam* param)
{

  if (param->cpu_prec  != param->cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
  QudaPrecision prec= param->cpu_prec;

#ifdef MULTI_GPU
  int* Z = param->X;
#endif
  int pad = param->ga_pad;
  int Vsh_x = param->X[1]*param->X[2]*param->X[3]/2;
  int Vsh_y = param->X[0]*param->X[2]*param->X[3]/2;
  int Vsh_z = param->X[0]*param->X[1]*param->X[3]/2;
  int Vsh_t = param->X[0]*param->X[1]*param->X[2]/2;

  int Vs[4] = {2*Vsh_x, 2*Vsh_y, 2*Vsh_z, 2*Vsh_t};

  void* ghost_cpuGauge[4];
  void* ghost_cpuGauge_diag[16];

  for(int i=0;i < 4; i++){
    
#if (CUDA_VERSION >= 4000)
    cudaMallocHost((void**)&ghost_cpuGauge[i], 8*Vs[i]*gaugeSiteSize*prec);
#else
    ghost_cpuGauge[i] = malloc(8*Vs[i]*gaugeSiteSize*prec);
#endif
    if(ghost_cpuGauge[i] == NULL){
      errorQuda("ERROR: malloc failed for ghost_sitelink[%d] \n",i);
    }
  }


#ifdef MULTI_GPU

  /*
    nu |     |
       |_____|
          mu     
  */

  int ghost_diag_len[16];
  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
        ghost_cpuGauge_diag[nu*4+mu] = NULL;
      }else{
        //the other directions
        int dir1, dir2;
        for(dir1= 0; dir1 < 4; dir1++){
          if(dir1 !=nu && dir1 != mu){
            break;
          }
        }
        for(dir2=0; dir2 < 4; dir2++){
          if(dir2 != nu && dir2 != mu && dir2 != dir1){
            break;
          }
        }
	//int rc = posix_memalign((void**)&ghost_cpuGauge_diag[nu*4+mu], ALIGNMENT, Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#if (CUDA_VERSION >= 4000)
	cudaMallocHost((void**)&ghost_cpuGauge_diag[nu*4+mu],  Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#else
	ghost_cpuGauge_diag[nu*4+mu] = malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#endif
	if(ghost_cpuGauge_diag[nu*4+mu] == NULL){
          errorQuda("malloc failed for ghost_sitelink_diag\n");
        }
	
        memset(ghost_cpuGauge_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
	ghost_diag_len[nu*4+mu] = Z[dir1]*Z[dir2]*gaugeSiteSize*prec;
      }

    }
  }

   int optflag=1;
   exchange_cpu_sitelink(param->X, cpuGauge, ghost_cpuGauge, ghost_cpuGauge_diag, prec, optflag);

#endif

   if (prec == QUDA_DOUBLE_PRECISION) {
     do_loadLinkToGPU(param->X, (double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double**)cpuGauge, 
		      (double**)ghost_cpuGauge, (double**)ghost_cpuGauge_diag, 
		      cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
		      Vsh_x, Vsh_y, Vsh_z, Vsh_t, 
		      prec);
   } else if (prec == QUDA_SINGLE_PRECISION) {
     do_loadLinkToGPU(param->X, (float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float**)cpuGauge, 
		      (float**)ghost_cpuGauge, (float**)ghost_cpuGauge_diag, 
		      cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
		      Vsh_x, Vsh_y, Vsh_z, Vsh_t, 
		      prec);    
   }else{
    printf("ERROR: half precision not supported in this funciton %s\n", __FUNCTION__);
    exit(1);
  }

   
  for(int i=0;i < 4;i++){
#if (CUDA_VERSION >= 4000)
    cudaFreeHost(ghost_cpuGauge[i]);
#else
    free(ghost_cpuGauge[i]);
#endif
  }
  for(int i=0;i <4; i++){ 
    for(int j=0;j <4; j++){
      if (i==j){
        continue;
      }
#if (CUDA_VERSION >= 4000)
      cudaFreeHost(ghost_cpuGauge_diag[i*4+j]);
#else
      free(ghost_cpuGauge_diag[i*4+j]);
#endif
    }
  }


}


template<typename FloatN, typename Float>
static void 
do_storeLinkToCPU(Float* cpuGauge, FloatN *even, FloatN *odd, 
		  int bytes, int Vh, int stride, QudaPrecision prec) 
{  
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }
  

  double* unpackedDataEven;
  double* unpackedDataOdd;
  int datalen = 4*Vh*gaugeSiteSize*sizeof(Float);
  cudaMalloc(&unpackedDataEven, datalen); CUERR;
  cudaMalloc(&unpackedDataOdd, datalen); CUERR;
  
  //unpack even data kernel
  link_format_gpu_to_cpu((void*)unpackedDataEven, (void*)even, bytes, Vh, stride, prec, streams[0]);
#if (CUDA_VERSION >= 4000)
  cudaMemcpyAsync(cpuGauge, unpackedDataEven, datalen, cudaMemcpyDeviceToHost, streams[0]);
#else
  cudaMemcpy(cpuGauge, unpackedDataEven, datalen, cudaMemcpyDeviceToHost);
#endif
  
  //unpack odd data kernel
  link_format_gpu_to_cpu((void*)unpackedDataOdd, (void*)odd,  bytes, Vh, stride, prec, streams[1]);
#if (CUDA_VERSION >=4000)
  cudaMemcpyAsync(cpuGauge + 4*Vh*gaugeSiteSize, unpackedDataOdd, datalen, cudaMemcpyDeviceToHost, streams[1]);  
  for(int i=0;i < 2; i++){
    cudaStreamSynchronize(streams[i]);
  }
#else
  cudaMemcpy(cpuGauge + 4*Vh*gaugeSiteSize, unpackedDataOdd, datalen, cudaMemcpyDeviceToHost);  
#endif
  
  cudaFree(unpackedDataEven);
  cudaFree(unpackedDataOdd);
  
  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }


  CUERR;
}



void 
storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param)
{
  
  QudaPrecision cpu_prec = param->cpu_prec;
  QudaPrecision cuda_prec= param->cuda_prec;

  if (cpu_prec  != cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
    
  if (cudaGauge->reconstruct != QUDA_RECONSTRUCT_NO){
    printf("ERROR: it makes no sense to get data back to cpu for 8/12 reconstruct, function %s\n", __FUNCTION__);
    exit(1);
  }
  
  int stride = cudaGauge->volumeCB + cudaGauge->pad;
  
  if (cuda_prec == QUDA_DOUBLE_PRECISION){
    do_storeLinkToCPU( (double*)cpuGauge, (double2*) cudaGauge->even, (double2*)cudaGauge->odd, 
		       cudaGauge->bytes, cudaGauge->volumeCB, stride, cuda_prec);
  }else if (cuda_prec == QUDA_SINGLE_PRECISION){
    do_storeLinkToCPU( (float*)cpuGauge, (float2*) cudaGauge->even, (float2*)cudaGauge->odd, 
		       cudaGauge->bytes, cudaGauge->volumeCB, stride, cuda_prec);
  }else{
    printf("ERROR: half precision not supported in this function %s\n", __FUNCTION__);
    exit(1);
  }
}


#endif

