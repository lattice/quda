
#include <misc_helpers.h>
#define gaugeSiteSize 18
#define BLOCKSIZE 64



/*
 * MILC order, CPU->GPU
 *
 *This function converts format in CPU form 
 * into forms in GPU so as to enable coalesce access
 * The function only converts half(even or odd) of the links
 * Therefore the entire link conversion need to call this 
 * function twice
 *   
 * Without loss of generarity, the parity is assume to be even.
 * The actual data format in cpu is following
 * [a0a1 .... a17][b0b1...b17][c..][d...][a18a19 .....a35] ...[b0b1 ... b17] ...
 *  X links        Y links    T,Z links   X Links
 * where a0->a17 is the X link in the first site
 *       b0->b17 is the Y link in the first site
 *       c0->c17 is the Z link in the first site
 *       d0->d17 is the T link in the first site
 *       a18->a35 is the X link in the second site
 *       etc
 *
 * The GPU format of data looks like the following
 * [a0a1][a18a19]  ....[pad][a2a3][a20a21]..... [b0b1][b18b19]....
 *  X links                                      Y links      T,Z links
 *   
 * N: sizeof(FloatN)/sizeof(Float)
 * M: # of floats in gpu memory per link
 *    12 for 12-reconstruct
 *    18 for no-reconstruct
 */

 /*
 * N: sizeof(FloatN)/sizeof(Float)
 * M: # of floats in gpu memory per link
 *    12 for 12-reconstruct
 *    18 for no-reconstruct
 */
template<int N, int M, typename FloatN>
__global__ void
do_link_format_cpu_to_gpu_milc(FloatN* dst, FloatN* src,
				  int reconstruct, int stride)
{
  __shared__ FloatN buf[gaugeSiteSize/2*BLOCKSIZE];
  
  int j;
  
  int block_idx = blockIdx.x*blockDim.x/4;
  int local_idx = 16*(threadIdx.x/64) + threadIdx.x%16;
  int pos_idx = blockIdx.x * blockDim.x/4 + 16*(threadIdx.x/64) + threadIdx.x%16;
  int mydir = (threadIdx.x >> 4)% 4;

  for(j=0; j < 9; j++){
    buf[j*blockDim.x + threadIdx.x] = src[block_idx*9*4 + j*blockDim.x + threadIdx.x ];
  }  
  __syncthreads();

  for(j=0; j < 9; j++){
    dst[pos_idx + mydir*9*stride + j*stride] = buf[local_idx*4*9+mydir*9+j]; 
  } 
  
}

/* This function converts format in CPU form 
 * into forms in GPU so as to enable coalesce access
 * The function only converts half(even or odd) of the links
 * Therefore the entire link conversion need to call this 
 * function twice
 *   
 * Without loss of generarity, the parity is assume to be even.
 * The actual data format in cpu is following
 * [a0a1 .... a17] [a18a19 .....a35] ...[b0b1 ... b17] ...
 *  X links                             Y links         T,Z links
 *
 * The GPU format of data looks like the following
 * [a0a1][a18a19]  ....[pad][a2a3][a20a21]..... [b0b1][b18b19]....
 *  X links                                      Y links      T,Z links
 *   
 * N: sizeof(FloatN)/sizeof(Float)
 * M: # of floats in gpu memory per link
 *    12 for 12-reconstruct
 *    18 for no-reconstruct
 */

template<int N, int M, typename FloatN, typename Float>
__global__ void
do_link_format_cpu_to_gpu(FloatN* dst, Float* src,
			   int reconstruct,
			   int Vh, int pad, int ghostV, size_t threads)
{
  int tid = blockIdx.x * blockDim.x +  threadIdx.x;
  int thread0_tid = blockIdx.x * blockDim.x;
  __shared__ FloatN buf[M/N*BLOCKSIZE];
  
  int dir;
  int j;
  
  for(dir = 0; dir < 4; dir++){
#ifdef MULTI_GPU
    FloatN* src_start = (FloatN*)( src + dir*gaugeSiteSize*(Vh+ghostV) + thread0_tid*gaugeSiteSize);   
#else
    FloatN* src_start = (FloatN*)( src + dir*gaugeSiteSize*(Vh) + thread0_tid*gaugeSiteSize);   
#endif
    for(j=0; j < gaugeSiteSize/N; j++){
      if(j*blockDim.x+threadIdx.x < 9*threads){
	if( M == 18){
	  buf[j*blockDim.x + threadIdx.x] =  src_start[j*blockDim.x + threadIdx.x];
	}else{ //M==12
	  int idx = j*blockDim.x + threadIdx.x;
	  int modval = idx%(gaugeSiteSize/N);
	  int divval = idx/(gaugeSiteSize/N);
	  if(modval < (M/N)){
	    buf[divval*(M/N)+modval] = src_start[idx];
	  }
	  
	}
      }
    }
    __syncthreads();
    if(tid < threads){
      FloatN* dst_start = (FloatN*)(dst+dir*M/N*(Vh+pad));
      for(j=0; j < M/N; j++){
	dst_start[tid + j*(Vh+pad)] = buf[M/N*threadIdx.x + j];
      }
    }
    __syncthreads();
  }//dir
}



void 
link_format_cpu_to_gpu(void* dst, void* src, 
		       int reconstruct, int Vh, int pad, 
		       int ghostV,
		       QudaPrecision prec, QudaGaugeFieldOrder cpu_order, 
		       cudaStream_t stream)
{
  dim3 blockDim(BLOCKSIZE);
#ifdef MULTI_GPU  
  size_t threads=Vh+ghostV;
#else
  size_t threads=Vh;
#endif
  
  dim3 gridDim ((threads + BLOCKSIZE -1)/BLOCKSIZE);

  //(Vh+ghostV) must be multipl of BLOCKSIZE or the kernel does not work
  /*
  if ((Vh+ghostV) % blockDim.x != 0){
    errorQuda("ERROR: Vh+ghostV(%d+%d) is not multiple of blocksize(%d), exitting\n", Vh, ghostV, blockDim.x);
  }
  */
  
  int stride = Vh+pad;
  
  if(cpu_order ==  QUDA_QDP_GAUGE_ORDER){
    switch (prec){
    case QUDA_DOUBLE_PRECISION:
      switch( reconstruct){
      case QUDA_RECONSTRUCT_NO:
	do_link_format_cpu_to_gpu<2, 18><<<gridDim, blockDim, 0, stream>>>((double2*)dst, (double*)src, reconstruct, Vh, pad, ghostV, threads);
	break;
      case QUDA_RECONSTRUCT_12:
	do_link_format_cpu_to_gpu<2, 12><<<gridDim, blockDim, 0, stream>>>((double2*)dst, (double*)src, reconstruct, Vh, pad, ghostV, threads);
	break;
      default:
	errorQuda("reconstruct type not supported\n");
      }
      break;    
      
    case QUDA_SINGLE_PRECISION:
      switch( reconstruct){
      case QUDA_RECONSTRUCT_NO:
	do_link_format_cpu_to_gpu<2, 18><<<gridDim, blockDim, 0, stream>>>((float2*)dst, (float*)src, reconstruct,  Vh, pad, ghostV, threads);   
	break;
      case QUDA_RECONSTRUCT_12:
	do_link_format_cpu_to_gpu<2, 12><<<gridDim, blockDim>>>((float2*)dst, (float*)src, reconstruct, Vh, pad, ghostV, threads);   
	break;
      default:
	errorQuda("reconstruct type not supported\n");      
      }
      break;
      
    default:
      errorQuda("ERROR: half precision not support in %s\n", __FUNCTION__);
    }
  }else if (cpu_order == QUDA_MILC_GAUGE_ORDER){    

    errorQuda("QUDA_MILC_GAUGE_ORDER is disabled");

    if ((Vh+ghostV) % blockDim.x != 0){
      errorQuda("ERROR: Vh+ghostV(%d+%d) is not multiple of blocksize(%d), exitting\n", Vh, ghostV, blockDim.x);
    }
    
    switch (prec){
    case QUDA_DOUBLE_PRECISION:
      switch( reconstruct){
      case QUDA_RECONSTRUCT_NO:
	do_link_format_cpu_to_gpu_milc<2, 18><<<gridDim, blockDim, 0, stream>>>((double2*)dst, (double2*)src, reconstruct, stride);
	break;
      case QUDA_RECONSTRUCT_12:
	//do_link_format_cpu_to_gpu<2, 12><<<gridDim, blockDim, 0, stream>>>((double2*)dst, (double*)src, reconstruct, stride);
	printf("12-reconstruct not supported yet\n");
	break;
      default:
	errorQuda("reconstruct type not supported\n");
      }
      break;    
      
    case QUDA_SINGLE_PRECISION:
      switch( reconstruct){
      case QUDA_RECONSTRUCT_NO:
	do_link_format_cpu_to_gpu_milc<2, 18><<<gridDim, blockDim, 0, stream>>>((float2*)dst, (float2*)src, reconstruct,  stride);
	break;
      case QUDA_RECONSTRUCT_12:
	//do_link_format_cpu_to_gpu<2, 12><<<gridDim, blockDim>>>((float2*)dst, (float*)src, reconstruct, stride);
	printf("12-reconstruct not supported yet\n");	
	break;
      default:
	errorQuda("reconstruct type not supported\n");      
      }
      break;
      
    default:
      errorQuda("ERROR: half precision not support in %s\n", __FUNCTION__);
    }
    
  }else{
    errorQuda("ERROR: invalid cpu ordering (%d)\n", cpu_order);
  }
  
  return;
  
}
/*
 * src format: the normal link format in GPU that has stride size @stride
 *	       the src is stored with 9 double2
 * dst format: an array of links where x,y,z,t links with the same node id is stored next to each other
 *             This format is used in destination in fatlink computation in cpu
 *    Without loss of generarity, the parity is assume to be even.
 * The actual data format in GPU is the following
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
			  int Vh, int stride)
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
		       int Vh, int stride, QudaPrecision prec, cudaStream_t stream)
{
  
  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim(4*Vh/blockDim.x); //every 4 threads process one site's x,y,z,t links
  //4*Vh must be multipl of BLOCKSIZE or the kernel does not work
  if ((4*Vh) % blockDim.x != 0){
    errorQuda("ERROR: 4*Vh(%d) is not multiple of blocksize(%d), exitting\n", Vh, blockDim.x);
  }
  if(prec == QUDA_DOUBLE_PRECISION){
    do_link_format_gpu_to_cpu<<<gridDim, blockDim, 0, stream>>>((double2*)dst, (double2*)src, Vh, stride);
  }else if(prec == QUDA_SINGLE_PRECISION){
    do_link_format_gpu_to_cpu<<<gridDim, blockDim, 0, stream>>>((float2*)dst, (float2*)src, Vh, stride);
  }else{
    printf("ERROR: half precision is not supported in %s\n",__FUNCTION__);
    exit(1);
  }
  
}

#define READ_ST_STAPLE(staple, idx, mystride)           \
  Float2 P0 = staple[idx + 0*mystride];                 \
  Float2 P1 = staple[idx + 1*mystride];                 \
  Float2 P2 = staple[idx + 2*mystride];			\
  Float2 P3 = staple[idx + 3*mystride];			\
  Float2 P4 = staple[idx + 4*mystride];			\
  Float2 P5 = staple[idx + 5*mystride];			\
  Float2 P6 = staple[idx + 6*mystride];			\
  Float2 P7 = staple[idx + 7*mystride];			\
  Float2 P8 = staple[idx + 8*mystride];			

#define WRITE_ST_STAPLE(staple, idx, mystride)		\
  staple[idx + 0*mystride] = P0;                        \
  staple[idx + 1*mystride] = P1;                        \
  staple[idx + 2*mystride] = P2;			\
  staple[idx + 3*mystride] = P3;			\
  staple[idx + 4*mystride] = P4;			\
  staple[idx + 5*mystride] = P5;			\
  staple[idx + 6*mystride] = P6;			\
  staple[idx + 7*mystride] = P7;			\
  staple[idx + 8*mystride] = P8;			



template<int dir, int whichway, typename Float2>
  __global__ void
  collectGhostStapleKernel(Float2* in, const int oddBit,
			   Float2* nbr_staple_gpu)
{

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int z1 = sid / X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1 / X2;
  int x2 = z1 - z2*X2;
  int x4 = z2 / X3;
  int x3 = z2 - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;
  //int X = 2*sid + x1odd;

  READ_ST_STAPLE(in, sid, staple_stride);
  int ghost_face_idx;
  
  if ( dir == 0 && whichway == QUDA_BACKWARDS){
    if (x1 < 1){
      ghost_face_idx = (x4*(X3*X2)+x3*X2 +x2)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X3*X2/2);
    }
  }

  if ( dir == 0 && whichway == QUDA_FORWARDS){
    if (x1 >= X1 - 1){
      ghost_face_idx = (x4*(X3*X2)+x3*X2 +x2)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X3*X2/2);
    }
  }
  
  if ( dir == 1 && whichway == QUDA_BACKWARDS){
    if (x2 < 1){
      ghost_face_idx = (x4*X3*X1+x3*X1+x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X3*X1/2);
    }
  }

  if ( dir == 1 && whichway == QUDA_FORWARDS){
    if (x2 >= X2 - 1){
      ghost_face_idx = (x4*X3*X1+x3*X1+x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X3*X1/2);
    }
  }

  if ( dir == 2 && whichway == QUDA_BACKWARDS){
    if (x3 < 1){
      ghost_face_idx = (x4*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X2*X1/2);
    }
  }

  if ( dir == 2 && whichway == QUDA_FORWARDS){
    if (x3 >= X3 - 1){
      ghost_face_idx = (x4*X2*X1 + x2*X1 + x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X4*X2*X1/2);
    }
  }

  if ( dir == 3 && whichway == QUDA_BACKWARDS){
    if (x4 < 1){
      ghost_face_idx = (x3*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X3*X2*X1/2);
    }
  }
  
  if ( dir == 3 && whichway == QUDA_FORWARDS){
    if (x4 >= X4 - 1){
      ghost_face_idx = (x3*X2*X1+x2*X1+x1)>>1;
      WRITE_ST_STAPLE(nbr_staple_gpu, ghost_face_idx, X3*X2*X1/2);
    }
  }

}


//@dir can be 0, 1, 2, 3 (X,Y,Z,T directions)
//@whichway can be QUDA_FORWARDS, QUDA_BACKWORDS
void
collectGhostStaple(int* X, void* even, void* odd, int volume, QudaPrecision precision,
		   void* ghost_staple_gpu,		   
		   int dir, int whichway, cudaStream_t* stream)
{
  int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
  
  Vsh_x = X[1]*X[2]*X[3]/2;
  Vsh_y = X[0]*X[2]*X[3]/2;
  Vsh_z = X[0]*X[1]*X[3]/2;
  Vsh_t = X[0]*X[1]*X[2]/2;  
    
  dim3 gridDim(volume/BLOCKSIZE, 1, 1);
  dim3 blockDim(BLOCKSIZE, 1, 1);
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};
    
  void* gpu_buf_even = ghost_staple_gpu;
  void* gpu_buf_odd = ((char*)ghost_staple_gpu) + Vsh[dir]*gaugeSiteSize*precision ;
  if (X[dir] % 2 ==1){ //need switch even/odd
    gpu_buf_odd = ghost_staple_gpu;
    gpu_buf_even = ((char*)ghost_staple_gpu) + Vsh[dir]*gaugeSiteSize*precision ;    
  }

  int even_parity = 0;
  int odd_parity = 1;
  
  if (precision == QUDA_DOUBLE_PRECISION){
    switch(dir){
    case 0:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<0, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<0, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 1:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<1, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<1, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 2:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<2, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<2, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 3:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<3, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)even, even_parity, (double2*)gpu_buf_even);
	collectGhostStapleKernel<3, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((double2*)odd, odd_parity, (double2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;      
    }
  }else if(precision == QUDA_SINGLE_PRECISION){
   switch(dir){
    case 0:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<0, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<0, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<0, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;

    case 1:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<1, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<1, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<1, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 2:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<2, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<2, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<2, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
      
    case 3:
      switch(whichway){
      case QUDA_BACKWARDS:
	collectGhostStapleKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<3, QUDA_BACKWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      case QUDA_FORWARDS:
	collectGhostStapleKernel<3, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)even, even_parity, (float2*)gpu_buf_even);
	collectGhostStapleKernel<3, QUDA_FORWARDS><<<gridDim, blockDim, 0, *stream>>>((float2*)odd, odd_parity, (float2*)gpu_buf_odd);
	break;
      default:
	errorQuda("Invalid whichway");
	break;
      }
      break;
   }
  }else{
    printf("ERROR: invalid  precision for %s\n", __FUNCTION__);
    exit(1);
  }

  CUERR;
}


#undef gaugeSiteSize 
#undef BLOCKSIZE 
