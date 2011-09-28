#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <gauge_quda.h>
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
static QudaTboundary tBoundary;
static int X[4];
static int faceVolumeCB[4]; // checkboarded face volume
static int volumeCB; // checkboarded volume
extern float fat_link_max;

#include <pack_gauge.h>

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
// offset = 0 for body
// offset = Vh for face
// voxels = Vh for body
// voxels[i] = face volume[i]
template <typename Float, typename FloatN>
static void packQDPGaugeField(FloatN *d_gauge, Float **h_gauge, int oddBit, 
			      ReconstructType reconstruct, int Vh, int *voxels,
			      int pad, int offset, int nFace, QudaLinkType type) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack12(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack8(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      if(type == QUDA_ASQTAD_FAT_LINKS && typeid(FloatN) == typeid(short2) ){
	  //we know it is half precison with fatlink at this stage
	for (int i = 0; i < nMat; i++) 
	  fatlink_short_pack18((short2*)(d_gauge+offset+i), g+i*18, dir, Vh+pad);
      }else{
	for (int i = 0; i < nMat; i++) pack18(d_gauge+offset+i, g+i*18, dir, Vh+pad);
      }
    }
  }
}

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
template <typename Float, typename FloatN>
static void unpackQDPGaugeField(Float **h_gauge, FloatN *d_gauge, int oddBit, 
				ReconstructType reconstruct, int V, int pad) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack12(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack8(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack18(g+i*18, d_gauge+i, dir, V+pad);
    }
  }
}

// transpose and scale the matrix
template <typename Float, typename Float2>
static void transposeScale(Float *gT, Float *g, const Float2 &a) {
  for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
    gT[(ic*3+jc)*2+r] = a*g[(jc*3+ic)*2+r];
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void packCPSGaugeField(FloatN *d_gauge, Float *h_gauge, int oddBit, 
			      ReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack12(d_gauge+i, gT, dir, V+pad);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack8(d_gauge+i, gT, dir, V+pad);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack18(d_gauge+i, gT, dir, V+pad);
      }
    }
  }

}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void unpackCPSGaugeField(Float *h_gauge, FloatN *d_gauge, int oddBit, 
				ReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack12(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg, gT, Anisotropy);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack8(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack18(gT, d_gauge+i, dir, V+pad);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  }

}

template <typename Float>
void packGhost(Float **cpuLink, Float **cpuGhost, int nFace) {
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

  for(int dir =0; dir < 4; dir++)
    {
      Float* even_src = cpuLink[dir];
      Float* odd_src = cpuLink[dir] + volumeCB*gaugeSiteSize;

      Float* even_dst;
      Float* odd_dst;
     
     //switching odd and even ghost cpuLink when that dimension size is odd
     //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
        even_dst = cpuGhost[dir];
        odd_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
     }else{
	even_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
        odd_dst = cpuGhost[dir];
     }

      int even_dst_index = 0;
      int odd_dst_index = 0;

      int d;
      int a,b,c;
      for(d = X[dir]- nFace; d < X[dir]; d++){
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
    }

}


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

void set_dim(int *XX) {

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

void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhost((double**)cpuLink, (double**)cpuGhost, nFace);
  } else {
    packGhost((float**)cpuLink, (float**)cpuGhost, nFace);
  }

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
      int dst_idx = (i*X[dir1]+j) >> 1; 
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


void freeGaugeField(FullGauge *cudaGauge) {
  if (cudaGauge->even) cudaFree(cudaGauge->even);
  if (cudaGauge->odd) cudaFree(cudaGauge->odd);
  cudaGauge->even = NULL;
  cudaGauge->odd = NULL;
}

template <typename Float, typename FloatN>
static void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, 
			   GaugeFieldOrder gauge_order, ReconstructType reconstruct, 
			   int bytes, int Vh, int pad, QudaLinkType type) {
  

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);

    

  if( ! packedEven ) errorQuda( "packedEven is borked\n");
  if( ! packedOdd ) errorQuda( "packedOdd is borked\n");
  if( ! even ) errorQuda( "even is borked\n");
  if( ! odd ) errorQuda( "odd is borked\n");
  if( ! cpuGauge ) errorQuda( "cpuGauge is borked\n");

#ifdef MULTI_GPU
  if (gauge_order != QUDA_QDP_GAUGE_ORDER)
    errorQuda("Only QUDA_QDP_GAUGE_ORDER is supported for multi-gpu\n");
#endif

  //for QUDA_ASQTAD_FAT_LINKS, need to find out the max value
  //fat_link_max will be used in encoding half precision fat link
  if(type == QUDA_ASQTAD_FAT_LINKS){
    for(int dir=0; dir < 4; dir++){
      for(int i=0;i < 2*Vh*gaugeSiteSize; i++){
	Float** tmp = (Float**)cpuGauge;
	if( tmp[dir][i] > fat_link_max ){
	  fat_link_max = tmp[dir][i];
	}
      }
    }
  }
  
  double fat_link_max_double = fat_link_max;
#ifdef MULTI_GPU
  reduceMaxDouble(fat_link_max_double);
#endif
  fat_link_max = fat_link_max_double;

  int voxels[] = {Vh, Vh, Vh, Vh};
  int nFace = 1;

  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
    packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh, pad);
    packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh, pad);    
  } else {
    errorQuda("Invalid gauge_order");
  }



#ifdef MULTI_GPU
#if 1
  // three step approach
  // 1. get the links into a contiguous buffer
  // 2. communicate between nodes
  // 3. pack into padded regions

  if (type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
  else nFace = 1;

  Float *ghostLink[4]; // ghost zone links
  Float *sendLink[4]; // send buffer for communication
  for(int i=0;i < 4;i++){
    ghostLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(ghostLink[i] == NULL) errorQuda("malloc failed for ghostLink[%d]\n", i);
    sendLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(sendLink[i] == NULL) errorQuda("malloc failed for sendLink[%d]\n", i);
  }

  packGhost((Float**)cpuGauge, sendLink, nFace); // pack the ghost zones into a contiguous buffer

  FaceBuffer face(X, 4, 18, nFace, QudaPrecision(sizeof(Float))); // this is the precision of the CPU field
  face.exchangeCpuLink((void**)ghostLink, (void**)sendLink);

  packQDPGaugeField(packedEven, ghostLink, 0, reconstruct, Vh,
		    faceVolumeCB, pad, Vh, nFace, type);
  packQDPGaugeField(packedOdd,  ghostLink, 1, reconstruct, Vh, 
		    faceVolumeCB, pad, Vh, nFace, type);
  
  for(int i=0;i < 4;i++) {
    delete []ghostLink[i];
    delete []sendLink[i];
  }
#else
  // Old QMP T-split code
  QudaPrecision precision = (QudaPrecision) sizeof(even->x);
  int Nvec = sizeof(FloatN)/precision;

  // one step approach
  // transfers into the pads directly
  transferGaugeFaces((void *)packedEven, (void *)(packedEven + Vh), precision, Nvec, reconstruct, Vh, pad);
  transferGaugeFaces((void *)packedOdd, (void *)(packedOdd + Vh), precision, Nvec, reconstruct, Vh, pad);
#endif
#endif
  checkCudaError();
  cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
    
  cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
  
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}


template <typename Float, typename FloatN>
static void retrieveGaugeField(Float *cpuGauge, FloatN *even, FloatN *odd, GaugeFieldOrder gauge_order,
			       ReconstructType reconstruct, int bytes, int Vh, int pad) {

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
    
  cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);    
    
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else {
    errorQuda("Invalid gauge_order");
  }
    
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
		      GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
		      Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, QudaLinkType type)
{

  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  Anisotropy = anisotropy;
  tBoundary = t_boundary;

  cudaGauge->anisotropy = anisotropy;
  cudaGauge->tadpole_coeff = tadpole_coeff;
  cudaGauge->volumeCB = 1;
  for (int d=0; d<4; d++) {
    cudaGauge->X[d] = XX[d];
    cudaGauge->volumeCB *= XX[d];
    X[d] = XX[d];
  }
  //cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
  cudaGauge->volumeCB /= 2;
  cudaGauge->pad = pad;

  /* test disabled because of staggered pad won't pass
#ifdef MULTI_GPU
  if (pad != cudaGauge->X[0]*cudaGauge->X[1]*cudaGauge->X[2]) {
    errorQuda("Gauge padding must match spatial volume");
  }
#endif
  */

  cudaGauge->stride = cudaGauge->volumeCB + cudaGauge->pad;
  cudaGauge->gauge_fixed = gauge_fixed;
  cudaGauge->t_boundary = t_boundary;
  
  allocateGaugeField(cudaGauge, reconstruct, cuda_prec);

  set_dim(XX);

  if (cuda_prec == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (float*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    }

  } else if (cuda_prec == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    }

  } else if (cuda_prec == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION){
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    }

  }

}




void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order)
{
  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  if (cudaGauge->precision == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      retrieveGaugeField((double*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      retrieveGaugeField((float*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    }

  } else if (cudaGauge->precision == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  } else if (cudaGauge->precision == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  }

}

