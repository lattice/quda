

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

__global__ void
do_link_format_cpu_to_gpu(double* dst, double* src,
			  int reconstruct,
			  int bytes, int Vh, int pad, int Vsh)
{
  int tid = blockIdx.x * blockDim.x +  threadIdx.x;
  int thread0_tid = blockIdx.x * blockDim.x;
  __shared__ double2 buf[gaugeSiteSize/2*BLOCKSIZE];
  
  int dir;
  int j;

  for(dir = 0; dir < 4; dir++){
      double2* src_start = (double2*)( src + dir*gaugeSiteSize*(Vh+2*Vsh) + thread0_tid*gaugeSiteSize);   
      for(j=0; j < gaugeSiteSize/2; j++){
	  buf[j*blockDim.x + threadIdx.x] =  src_start[j*blockDim.x + threadIdx.x];
      }
      __syncthreads();
      
      double2* dst_start = (double2*)(dst+dir*gaugeSiteSize*(Vh+pad));
      for(j=0; j < gaugeSiteSize/2; j++){
	  dst_start[tid + j*(Vh+pad)] = buf[gaugeSiteSize/2*threadIdx.x + j];
      }
      __syncthreads();
      
  }//dir
}



void 
link_format_cpu_to_gpu(double* dst, double* src, 
		       int reconstruct, int bytes, int Vh, int pad, int Vsh)
{
    dim3 blockDim(BLOCKSIZE);
    dim3 gridDim((Vh+2*Vsh)/blockDim.x);
    //(Vh+2*Vsh) must be multipl of BLOCKSIZE or the kernel does not work
    //because the intermediae GPU data has stride=Vh+2*Vsh and the extra two
    //Vsh is occupied by the back and forward neighbor
    if ((Vh+2*Vsh) % blockDim.x != 0){
	printf("ERROR: Vh(%d) is not multiple of blocksize(%d), exitting\n", Vh, blockDim.x);
	exit(1);
    }
    do_link_format_cpu_to_gpu<<<gridDim, blockDim>>>(dst, src, reconstruct, bytes, Vh, pad, Vsh);
    
}


/*
 * src format: the normal link format in GPU that has stride size @stride
 *	       the src is stored with 9 double2
 * dst format: an array of links where x,y,z,t links with the same node id is stored next to each other
 *             This format is used in destination in fatlink computation in cpu
 *
 */

__global__ void
do_link_format_gpu_to_cpu(double2* dst, double2* src,
			  int reconstruct,
			  int bytes, int Vh, int stride)
{
  __shared__ double2 buf[gaugeSiteSize/2*BLOCKSIZE];
  
  int j;
  
  int block_idx = blockIdx.x*blockDim.x/4;
  int local_idx = 8*(threadIdx.x/32) + threadIdx.x%8;
  int thread_idx = blockIdx.x * blockDim.x/4 + 8*(threadIdx.x/32) + threadIdx.x%8;
  int mydir = (threadIdx.x >> 3)% 4;
  
  for(j=0; j < 9; j++){
    buf[local_idx*4*9+mydir*9+j] = src[thread_idx + mydir*9*stride + j*stride];
  }
  __syncthreads();
  
  for(j=0; j < 9; j++){
    dst[block_idx*9*4 + j*blockDim.x + threadIdx.x ] = buf[j*blockDim.x + threadIdx.x];    
  }  
  
}

void 
link_format_gpu_to_cpu(double* dst, double* src, 
		       int reconstruct, int bytes, int Vh, int stride)
{
  
  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim(4*Vh/blockDim.x); //every 4 threads process one site's x,y,z,t links
  //4*Vh must be multipl of BLOCKSIZE or the kernel does not work
  if ((4*Vh) % blockDim.x != 0){
    printf("ERROR: Vh(%d) is not multiple of blocksize(%d), exitting\n", Vh, blockDim.x);
    exit(1);
  }
  do_link_format_gpu_to_cpu<<<gridDim, blockDim>>>((double2*)dst, (double2*)src, reconstruct, bytes, Vh, stride);
  
}


#undef gaugeSiteSize 
#undef BLOCKSIZE 
