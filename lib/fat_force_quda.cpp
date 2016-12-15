#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <quda_internal.h>
#include <fat_force_quda.h>
#include <face_quda.h>
#include <misc_helpers.h>
#include <assert.h>

#define gaugeSiteSize 18

#define MAX(a,b) ((a)>(b)?(a):(b))
#define ALIGNMENT 4096 

  /********************** Staple code, used by link fattening **************/

#if defined(GPU_FATLINK) || defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE) || defined(GPU_HISQ_FORCE) || defined(GPU_CLOVER_DIRAC)

namespace quda {

  template <typename Float>
  void packGhostAllStaples(Float *cpuStaple, Float **cpuGhostBack,Float**cpuGhostFwd, int nFace, int* X) {
    int XY=X[0]*X[1];
    int XYZ=X[0]*X[1]*X[2];
    int volumeCB = X[0]*X[1]*X[2]*X[3]/2;
    int faceVolumeCB[4]={
      X[1]*X[2]*X[3]/2,
      X[0]*X[2]*X[3]/2,
      X[0]*X[1]*X[3]/2,
      X[0]*X[1]*X[2]/2
    };

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


  void pack_ghost_all_staples_cpu(void *staple, void **cpuGhostStapleBack, void** cpuGhostStapleFwd, 
				  int nFace, QudaPrecision precision, int* X) {
  
    if (precision == QUDA_DOUBLE_PRECISION) {
      packGhostAllStaples((double*)staple, (double**)cpuGhostStapleBack, (double**) cpuGhostStapleFwd, nFace, X);
    } else {
      packGhostAllStaples((float*)staple, (float**)cpuGhostStapleBack, (float**)cpuGhostStapleFwd, nFace, X);
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

  void
  packGhostStaple(int* X, void* even, void* odd, int volumeCB, QudaPrecision prec,
		  int stride, 
		  int dir, int whichway,
		  void** fwd_nbr_buf_gpu, void** back_nbr_buf_gpu,
		  void** fwd_nbr_buf, void** back_nbr_buf,
		  cudaStream_t* stream)
  {
    int Vs_x, Vs_y, Vs_z, Vs_t;
  
    Vs_x = X[1]*X[2]*X[3];
    Vs_y = X[0]*X[2]*X[3];
    Vs_z = X[0]*X[1]*X[3];
    Vs_t = X[0]*X[1]*X[2];  
    int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  
    if (dir != 3){ //the code would work for dir=3 as well
      //even and odd ness switch (if necessary) is taken caren of in collectGhostStaple();
      void* gpu_buf;
      int i =dir;
      if (whichway ==  QUDA_BACKWARDS){
	gpu_buf = back_nbr_buf_gpu[i];
	collectGhostStaple(X, even, odd, volumeCB, stride, prec, gpu_buf, i, whichway, stream);
	cudaMemcpyAsync(back_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);
      }else{//whichway is  QUDA_FORWARDS;
	gpu_buf = fwd_nbr_buf_gpu[i];
	collectGhostStaple(X, even, odd, volumeCB, stride, prec, gpu_buf, i, whichway, stream);
	cudaMemcpyAsync(fwd_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);        
      }
    }else{ //special case for dir=3 since no gather kernel is required
      int Vh = volumeCB;
      int Vsh = X[0]*X[1]*X[2]/2;
      int sizeOfFloatN = 2*prec;
      int len = Vsh*sizeOfFloatN;
      int i;
      if(X[3] %2 == 0){
	//back,even
	for(i=0;i < 9; i++){
	  void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	  void* src = ((char*)even) + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//back, odd
	for(i=0;i < 9; i++){
	  void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	  void* src = ((char*)odd) + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//fwd,even
	for(i=0;i < 9; i++){
	  void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	  void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//fwd, odd
	for(i=0;i < 9; i++){
	  void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	  void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
      }else{
	//reverse even and odd position
	//back,odd
	for(i=0;i < 9; i++){
	  void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	  void* src = ((char*)odd) + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//back, even
	for(i=0;i < 9; i++){
	  void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	  void* src = ((char*)even) + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//fwd,odd
	for(i=0;i < 9; i++){
	  void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	  void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
	//fwd, even
	for(i=0;i < 9; i++){
	  void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	  void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*stride*sizeOfFloatN;
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
	}
      
      } 
    }
  
  }


  void 
  unpackGhostStaple(int* X, void* _even, void* _odd, int volume, QudaPrecision prec,
		    int stride, 
		    int dir, int whichway, void** fwd_nbr_buf, void** back_nbr_buf,
		    cudaStream_t* stream)
  {

    int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
  
    Vsh_x = X[1]*X[2]*X[3]/2;
    Vsh_y = X[0]*X[2]*X[3]/2;
    Vsh_z = X[0]*X[1]*X[3]/2;
    Vsh_t = X[0]*X[1]*X[2]/2;  
    int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};

    int Vh = volume;
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
  
    char* even = ((char*)_even) + Vh*sizeOfFloatN + 2*tmpint[dir]*sizeOfFloatN;
    char* odd = ((char*)_odd) + Vh*sizeOfFloatN +2*tmpint[dir]*sizeOfFloatN;
  
    if(whichway == QUDA_BACKWARDS){   
      //back,even
      for(int i=0;i < 9; i++){
	void* dst = even + i*stride*sizeOfFloatN;
	void* src = ((char*)back_nbr_buf[dir]) + i*len[dir] ; 
	cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); 
      }
      //back, odd
      for(int i=0;i < 9; i++){
	void* dst = odd + i*stride*sizeOfFloatN;
	void* src = ((char*)back_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
	cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); 
      }
    }else { //QUDA_FORWARDS
      //fwd,even
      for(int i=0;i < 9; i++){
	void* dst = even + Vsh[dir]*sizeOfFloatN + i*stride*sizeOfFloatN;
	void* src = ((char*)fwd_nbr_buf[dir]) + i*len[dir] ; 
	cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); 
      }
      //fwd, odd
      for(int i=0;i < 9; i++){
	void* dst = odd + Vsh[dir]*sizeOfFloatN + i*stride*sizeOfFloatN;
	void* src = ((char*)fwd_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
	cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); 
      }
    }
  }

  /*
    This is the packing kernel for the multi-dimensional ghost zone in
    the padded region.  This is called by cpuexchangesitelink in
    FaceBuffer (MPI only), which was called by loadLinkToGPU (defined at
    the bottom).  

    Not currently included since it will be replaced by Guochun's new
    routine which uses an enlarged domain instead of a ghost zone.
  */
  template <typename Float>
  void packGhostAllLinks(Float **cpuLink, Float **cpuGhostBack,Float**cpuGhostFwd, int dir, int nFace, int* X) {
    int XY=X[0]*X[1];
    int XYZ=X[0]*X[1]*X[2];

    int volumeCB = X[0]*X[1]*X[2]*X[3]/2;
    int faceVolumeCB[4]={
      X[1]*X[2]*X[3]/2,
      X[0]*X[2]*X[3]/2,
      X[0]*X[1]*X[3]/2,
      X[0]*X[1]*X[2]/2
    };

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
	  even_dst = odd_dst + nFace*faceVolumeCB[dir]*gaugeSiteSize;
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


  void pack_ghost_all_links(void **cpuLink, void **cpuGhostBack, void** cpuGhostFwd, 
			    int dir, int nFace, QudaPrecision precision, int *X) {
  
    if (precision == QUDA_DOUBLE_PRECISION) {
      packGhostAllLinks((double**)cpuLink, (double**)cpuGhostBack, (double**) cpuGhostFwd, dir,  nFace, X);
    } else {
      packGhostAllLinks((float**)cpuLink, (float**)cpuGhostBack, (float**)cpuGhostFwd, dir, nFace, X);
    }
  
  }


} // namespace quda

#undef gaugeSiteSize

#endif

