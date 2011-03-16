
#include <misc_helpers.h>
#define gaugeSiteSize 18
#define BLOCKSIZE 64

/* This function converts format in CPU form 
   into forms in GPU so as to enable coalesce access
   The function only converts half(even or odd) of the links
   Therefore the entire link conversion need to call this 
   function twice
   
   Without loss of generarity, the parity is assume to be even.
   The actual data format in cpu is following
   [a0a1 .... a17] [a18a19 .....a35] ...[b0b1 ... b17] ...
     X links                             Y links         T,Z links
   
   The GPU format of data looks like the following
   [a0a1][a18a19]  ....[pad][a2a3][a20a21]..... [b0b1][b18b19]....
    X links                                      Y links      T,Z links
   
*/

template<typename FloatN, typename Float>
__global__ void
do_link_format_cpu_to_gpu(FloatN* dst, Float* src,
			  int reconstruct,
			  int bytes, int Vh, int pad, int Vsh)
{
  int tid = blockIdx.x * blockDim.x +  threadIdx.x;
  int thread0_tid = blockIdx.x * blockDim.x;
  __shared__ FloatN buf[gaugeSiteSize/2*BLOCKSIZE];
  
  int dir;
  int j;
  
  for(dir = 0; dir < 4; dir++){
#ifdef MULTI_GPU
      FloatN* src_start = (FloatN*)( src + dir*gaugeSiteSize*(Vh+2*Vsh) + thread0_tid*gaugeSiteSize);   
#else
      FloatN* src_start = (FloatN*)( src + dir*gaugeSiteSize*(Vh) + thread0_tid*gaugeSiteSize);   
#endif
      for(j=0; j < gaugeSiteSize/2; j++){
	  buf[j*blockDim.x + threadIdx.x] =  src_start[j*blockDim.x + threadIdx.x];
      }
      __syncthreads();
      
      FloatN* dst_start = (FloatN*)(dst+dir*gaugeSiteSize/2*(Vh+pad));
      for(j=0; j < gaugeSiteSize/2; j++){
	  dst_start[tid + j*(Vh+pad)] = buf[gaugeSiteSize/2*threadIdx.x + j];
      }
      __syncthreads();
      
  }//dir
}



// we require the cpu precisision and gpu precision are the same
void 
link_format_cpu_to_gpu(void* dst, void* src, 
		       int reconstruct, int bytes, int Vh, int pad, int Vsh, 
		       QudaPrecision prec)
{
  dim3 blockDim(BLOCKSIZE);
#ifdef MULTI_GPU  
  dim3 gridDim((Vh+2*Vsh)/blockDim.x);
#else
  dim3 gridDim(Vh/blockDim.x);
#endif
  //(Vh+2*Vsh) must be multipl of BLOCKSIZE or the kernel does not work
    //because the intermediae GPU data has stride=Vh+2*Vsh and the extra two
    //Vsh is occupied by the back and forward neighbor
    if ((Vh+2*Vsh) % blockDim.x != 0){
	printf("ERROR: Vh(%d) is not multiple of blocksize(%d), exitting\n", Vh, blockDim.x);
	exit(1);
    }
    
    
    switch (prec){
    case QUDA_DOUBLE_PRECISION:
      do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>((double2*)dst, (double*)src, reconstruct, bytes, Vh, pad, Vsh);
      break;
      
    case QUDA_SINGLE_PRECISION:
      if(reconstruct == QUDA_RECONSTRUCT_NO){
	do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>((float2*)dst, (float*)src, reconstruct, bytes, Vh, pad, Vsh);   
      }else if (reconstruct == QUDA_RECONSTRUCT_12){
	//not working yet
	//do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>((float4*)dst, (float*)src, reconstruct, bytes, Vh, pad, Vsh);   
	
      }
      break;
      
    default:
      printf("ERROR: half precision not support in %s\n", __FUNCTION__);
      exit(1);
    }
    
    /*
    if (cuda_prec == QUDA_DOUBLE_PRECISION){
      do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>((double2*)dst, (double*)src, reconstruct, bytes, Vh, pad, Vsh);
    }else if( cuda_prec == QUDA_SINGLE_PRECISION){
      do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>((float2*)dst, (float*)src, reconstruct, bytes, Vh, pad, Vsh);      
    }else{
      printf("ERROR: half precision is not supported in %s\n", __FUNCTION__);
      exit(1);
    }
    */

    return;
    
}


/*
 * src format: the normal link format in GPU that has stride size @stride
 *	       the src is stored with 9 double2
 * dst format: an array of links where x,y,z,t links with the same node id is stored next to each other
 *             This format is used in destination in fatlink computation in cpu
 *    Without loss of generarity, the parity is assume to be even.
 * The actual data format in cpu is following
 *    [a0a1][a18a19]  ....[pad][a2a3][a20a21]..... [b0b1][b18b19]....
 *    X links                                      Y links      T,Z links
 * The temporary data store in GPU shared memory and the CPU format of data are the following
 *    [a0a1 .... a17] [b0b1 .....b17] [c0c1 .....c17] [d0d1 .....d17] [a18a19....a35] ....
 *    |<------------------------site 0 ---------------------------->|<----- site 2 ----->
 *
 *
 * In loading phase the indices for all threads in the first block is the following (assume block size is 64)
 * (half warp works on one direction)
 * threadIdx.x	pos_idx		mydir
 * 0		0		0
 * 1		1		0
 * 2		2		0
 * 3		3		0			
 * 4		4		0		
 * 5		5		0
 * 6		6		0
 * 7		7		0
 * 8		8		0
 * 9		9		0
 * 10		10		0
 * 11		11		0
 * 12		12		0
 * 13		13		0
 * 14		14		0
 * 15		15		0
 * 16		0		1
 * 17		1		1
 * 18	       	2		1
 * 19		3		1
 * 20		4		1
 * 21		5		1
 * 22		6		1
 * 23		7		1
 * 24		8		1
 * 25		9		1
 * 26		10		1
 * 27		11		1
 * 28		12		1
 * 29		13		1
 * 30		14		1
 * 31		15		1
 * 32		0		2
 * 33		1		2
 * 34		2		2
 * 35		3		2
 * 36		4		2
 * 37		5		2
 * 38		6		2
 * 39		7		2
 * 40		8		2
 * 41		9		2
 * 42		10		2
 * 43		11		2
 * 44		12		2
 * 45		13		2
 * 46		14		2
 * 47		15		2
 * 48		0		3
 * 49		1		3
 * 50		2		3
 * 51		3		3
 * 52		4		3
 * 53		5		3
 * 54		6		3
 * 55		7		3
 * 56		8		3
 * 57		9		3
 * 58		10		3
 * 59		11		3
 * 60		12		3
 * 61		13		3
 * 62		14		3
 * 63		15		3
 *
 */

template<typename FloatN>
__global__ void
do_link_format_gpu_to_cpu(FloatN* dst, FloatN* src,
			  int bytes, int Vh, int stride)
{
  __shared__ FloatN buf[gaugeSiteSize/2*BLOCKSIZE];
  
  int j;
  
  int block_idx = blockIdx.x*blockDim.x/4;
  int local_idx = 16*(threadIdx.x/64) + threadIdx.x%16;
  int pos_idx = blockIdx.x * blockDim.x/4 + 16*(threadIdx.x/64) + threadIdx.x%16;
  int mydir = (threadIdx.x >> 4)% 4;
  for(j=0; j < 9; j++){
    buf[local_idx*4*9+mydir*9+j] = src[pos_idx + mydir*9*stride + j*stride];
  }
  __syncthreads();
  
  for(j=0; j < 9; j++){
    dst[block_idx*9*4 + j*blockDim.x + threadIdx.x ] = buf[j*blockDim.x + threadIdx.x];    
  }  
  
}

void 
link_format_gpu_to_cpu(void* dst, void* src, 
		       int bytes, int Vh, int stride, QudaPrecision prec)
{
  
  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim(4*Vh/blockDim.x); //every 4 threads process one site's x,y,z,t links
  //4*Vh must be multipl of BLOCKSIZE or the kernel does not work
  if ((4*Vh) % blockDim.x != 0){
    printf("ERROR: Vh(%d) is not multiple of blocksize(%d), exitting\n", Vh, blockDim.x);
    exit(1);
  }
  if(prec == QUDA_DOUBLE_PRECISION){
    do_link_format_gpu_to_cpu<<<gridDim, blockDim>>>((double2*)dst, (double2*)src, bytes, Vh, stride);
  }else if(prec == QUDA_SINGLE_PRECISION){
    do_link_format_gpu_to_cpu<<<gridDim, blockDim>>>((float2*)dst, (float2*)src,  bytes, Vh, stride);
  }else{
    printf("ERROR: half precision is not supported in %s\n",__FUNCTION__);
    exit(1);
  }
  
}



#define READ_ST_SPINOR(spinor, idx, mystride)           \
  Float2 I0 = spinor[idx + 0*mystride];                 \
  Float2 I1 = spinor[idx + 1*mystride];                 \
  Float2 I2 = spinor[idx + 2*mystride];

#define WRITE_ST_SPINOR(spinor, idx, mystride)  \
  spinor[idx + 0*mystride] = I0;                        \
  spinor[idx + 1*mystride] = I1;                        \
  spinor[idx + 2*mystride] = I2;


template<int dir, int whichway, typename Float2>
__global__ void
staggeredCollectGhostSpinorKernel(Float2* in, const int oddBit,
                                  Float2* nbr_spinor_gpu)
{
#if 1
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int z1 = FAST_INT_DIVIDE(sid, X1h);
  int x1h = sid - z1*X1h;
  int z2 = FAST_INT_DIVIDE(z1, X2);
  int x2 = z1 - z2*X2;
  int x4 = FAST_INT_DIVIDE(z2, X3);
  int x3 = z2 - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  READ_ST_SPINOR(in, sid, sp_stride);
  int ghost_face_idx;

  if ( dir == 0 && whichway == QUDA_BACKWARDS){
    if (x1 < 3){
      ghost_face_idx = (x1*X4*X3*X2 + x4*(X3*X2)+x3*X2 +x2)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X3*X2/2);
    }
  }

  if ( dir == 0 && whichway == QUDA_FORWARDS){
    if (x1 >= X1 - 3){
      ghost_face_idx = ((x1-X1+3)*X4*X3*X2 + x4*(X3*X2)+x3*X2 +x2)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X3*X2/2);
    }
  }

  if ( dir == 1 && whichway == QUDA_BACKWARDS){
    if (x2 < 3){
      ghost_face_idx = (x2*X4*X3*X1 + x4*X3*X1+x3*X1+x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X3*X1/2);
    }
  }

  if ( dir == 1 && whichway == QUDA_FORWARDS){
    if (x2 >= X2 - 3){
      ghost_face_idx = ((x2-X2+3)*X4*X3*X1+ x4*X3*X1+x3*X1+x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X3*X1/2);
    }
  }

  if ( dir == 2 && whichway == QUDA_BACKWARDS){
    if (x3 < 3){
      ghost_face_idx = (x3*X4*X2*X1 + x4*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X2*X1/2);
    }
  }

  if ( dir == 2 && whichway == QUDA_FORWARDS){
    if (x3 >= X3 - 3){
      ghost_face_idx = ((x3-X3+3)*X4*X2*X1 + x4*X2*X1 + x2*X1 + x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X4*X2*X1/2);
    }
  }

  if ( dir == 3 && whichway == QUDA_BACKWARDS){
    if (x4 < 3){
      ghost_face_idx = (x4*X3*X2*X1 + x3*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X3*X2*X1/2);
    }
  }

  if ( dir == 3 && whichway == QUDA_FORWARDS){
    if (x4 >= X4 - 3){
      ghost_face_idx = ((x4-X4+3)*X3*X2*X1 + x3*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_SPINOR(nbr_spinor_gpu, ghost_face_idx, 3*X3*X2*X1/2);
    }
  }
#endif

}


//@dir can be 0, 1, 2, 3 (X,Y,Z,T directions)
//@whichway can be QUDA_FORWARDS, QUDA_BACKWORDS
void
collectGhostSpinor(void *in, const void *inNorm,
                   void* ghost_spinor_gpu,		   
		   int dir, int whichway,
                   const int parity, cudaColorSpinorField* inSpinor)
{
  
  dim3 gridDim(inSpinor->Volume()/BLOCKSIZE, 1, 1);
  dim3 blockDim(BLOCKSIZE, 1, 1);
    
  if (inSpinor->Precision() == QUDA_DOUBLE_PRECISION){
    switch(dir){
    case 0:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu);
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<0, QUDA_FORWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 1:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<1, QUDA_FORWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 2:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<2, QUDA_FORWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 3:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<3, QUDA_FORWARDS><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

      
    }
    //staggeredCollectGhostSpinorKernel<dir, whichway><<<gridDim, blockDim>>>((double2*)in, parity, (double2*)ghost_spinor_gpu);
  }else if(inSpinor->Precision() == QUDA_SINGLE_PRECISION){
    //staggeredCollectGhostSpinorKernel<dir, whichway><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu);
    switch(dir){
    case 0:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu);
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<0, QUDA_FORWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 1:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<1, QUDA_FORWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 2:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<2, QUDA_FORWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 3:
      switch(whichway){
      case QUDA_BACKWARDS:
	staggeredCollectGhostSpinorKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      case QUDA_FORWARDS:
	staggeredCollectGhostSpinorKernel<3, QUDA_FORWARDS><<<gridDim, blockDim>>>((float2*)in, parity, (float2*)ghost_spinor_gpu); CUERR;
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    }

  }else{
    printf("ERROR: half precision not implemented yet for %s\n", __FUNCTION__);
    exit(1);
  }
  cudaThreadSynchronize();
  CUERR;
}





#undef gaugeSiteSize 
#undef BLOCKSIZE 
