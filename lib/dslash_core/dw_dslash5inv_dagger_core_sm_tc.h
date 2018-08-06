// *** CUDA DSLASH DAGGER ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0

// This is for tensor core ONLY so assumming CUDA_VERSION >= 9000

// input spinor needs to be half
#define POW(a, b) __fast_pow(a, b)
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I0.z
#define i01_im I0.w
#define i02_re I1.x
#define i02_im I1.y
#define i10_re I1.z
#define i10_im I1.w
#define i11_re I2.x
#define i11_im I2.y
#define i12_re I2.z
#define i12_im I2.w
#define i20_re I3.x
#define i20_im I3.y
#define i21_re I3.z
#define i21_im I3.w
#define i22_re I4.x
#define i22_im I4.y
#define i30_re I4.z
#define i30_im I4.w
#define i31_re I5.x
#define i31_im I5.y
#define i32_re I5.z
#define i32_im I5.w
#define m5 param.m5_f
#define mdwf_b5 param.mdwf_b5_f
#define mdwf_c5 param.mdwf_c5_f
#define mferm param.mferm_f
#define a param.a
#define b param.b

// output spinor
float o00_re = 0.0f;
float o00_im = 0.0f;
float o01_re = 0.0f;
float o01_im = 0.0f;
float o02_re = 0.0f;
float o02_im = 0.0f;
float o10_re = 0.0f;
float o10_im = 0.0f;
float o11_re = 0.0f;
float o11_im = 0.0f;
float o12_re = 0.0f;
float o12_im = 0.0f;
float o20_re = 0.0f;
float o20_im = 0.0f;
float o21_re = 0.0f;
float o21_im = 0.0f;
float o22_re = 0.0f;
float o22_im = 0.0f;
float o30_re = 0.0f;
float o30_im = 0.0f;
float o31_re = 0.0f;
float o31_im = 0.0f;
float o32_re = 0.0f;
float o32_im = 0.0f;

MDWFSharedMemory<float4> sm_data;

const int M = param.dc.Ls*4;
const int N = 6*blockDim.x;
const int K = param.dc.Ls*4;

const int sm_pad_size = 8;

const int N_sm = N + sm_pad_size;
const int M_sm = M + sm_pad_size;

half* sm_b = (half*)((void*)sm_data);
half* sm_c = (half*)(sm_b + K*(N+sm_pad_size));
half* sm_a = (half*)(sm_c + M*(N+sm_pad_size));

#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi

#include "io_spinor.h"

bool idle = false;
int sid = (blockIdx.y*blockDim.y + threadIdx.y)*param.threads + blockIdx.x*blockDim.x+threadIdx.x;
if (blockIdx.x*blockDim.x+threadIdx.x >= param.threads){
	idle = true;
}

int X, coord[5], boundaryCrossing;

if(!idle){

  if( param.partial_length ){
    coordsFromIndexShrinked<5,QUDA_4D_PC,EVEN_X>(X, coord, sid, param);
  }else{
  
    boundaryCrossing = sid/param.dc.Xh[0] + sid/(param.dc.X[1]*param.dc.Xh[0]) + sid/(param.dc.X[2]*param.dc.X[1]*param.dc.Xh[0]);
  
    X = 2*sid + (boundaryCrossing + param.parity) % 2;
    coord[4] = X/(param.dc.X[0]*param.dc.X[1]*param.dc.X[2]*param.dc.X[3]);
  
  }

//  boundaryCrossing = sid/param.dc.Xh[0] + sid/(param.dc.X[1]*param.dc.Xh[0]) + sid/(param.dc.X[2]*param.dc.X[1]*param.dc.Xh[0]);
//  
//  X = 2*sid + (boundaryCrossing + param.parity) % 2;
//  coord[4] = X/(param.dc.X[0]*param.dc.X[1]*param.dc.X[2]*param.dc.X[3]);

READ_SPINOR( SPINORTEX, param.sp_stride, X/2, X/2 );

// data layout for tensor core B and C: spatial, color, complex, spin, s; Lsx4 by Lsx4 @ Lsx4 by 6xblockDim.x.
// lda = Lsx4, column-major
// ldb = Lsx4, column-major
// total number of halves = Ls*24*blockDim.x
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ] = i00_re;
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ] = i00_im;
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ] = i01_re;
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ] = i01_im;
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ] = i02_re;
sm_b[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ] = i02_im;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ] = i10_re;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ] = i10_im;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ] = i11_re;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ] = i11_im;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ] = i12_re;
sm_b[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ] = i12_im;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ] = i20_re;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ] = i20_im;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ] = i21_re;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ] = i21_im;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ] = i22_re;
sm_b[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ] = i22_im;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ] = i30_re;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ] = i30_im;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ] = i31_re;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ] = i31_im;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ] = i32_re;
sm_b[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ] = i32_im;

// Construct matrix A: TODO: should be careful about the idle threads.

// threadIdx.x should not be idle(?).
// With Ls=12 and blockDim.x=32 the following gives a 2-way bank conflict.  
  if(threadIdx.x < param.dc.Ls){

#ifdef MDWF_mode   // Check whether MDWF option is enabled
    half kappa = -(static_cast<half>(mdwf_c5[ threadIdx.x ])*(static_cast<half>(4.0) + static_cast<half>(m5)) - static_cast<half>(1.0))/(static_cast<half>(mdwf_b5[ threadIdx.x ])*(static_cast<half>(4.0) + static_cast<half>(m5)) + static_cast<half>(1.0));
#else
    half kappa = static_cast<half>(2.0)*static_cast<half>(a);
#endif  // select MDWF mode
  
    half inv_d_n = static_cast<half>(0.5) / ( static_cast<half>(1.0) + static_cast<half>(POW(kappa,param.dc.Ls))*static_cast<half>(mferm) );
    half factorR;
    half factorL;
  
    int exponent = threadIdx.x  > coord[4] ? param.dc.Ls-threadIdx.x+coord[4] : coord[4]-threadIdx.x;
    factorR = inv_d_n * static_cast<half>(POW(kappa,exponent))  * ( threadIdx.x > coord[4] ? static_cast<half>(-mferm) : static_cast<half>(1.0) );
    int exponent2 = threadIdx.x < coord[4] ? param.dc.Ls-coord[4]+threadIdx.x : threadIdx.x-coord[4];
    factorL = inv_d_n * static_cast<half>(POW(kappa,exponent2)) * ( threadIdx.x < coord[4] ? static_cast<half>(-mferm) : static_cast<half>(1.0) );
    // (mu, s) by (nu, t). column-major. t := threadIdx.y
  
    sm_a[ (coord[4]*4+0)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+0) ] = factorR + factorL;
    sm_a[ (coord[4]*4+0)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+1) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+0)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+2) ] = factorR - factorL;
    sm_a[ (coord[4]*4+0)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+3) ] = static_cast<half>(0.0f);
                                                                  
    sm_a[ (coord[4]*4+1)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+0) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+1)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+1) ] = factorR + factorL;
    sm_a[ (coord[4]*4+1)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+2) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+1)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+3) ] = factorR - factorL;
                                                                  
    sm_a[ (coord[4]*4+2)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+0) ] = factorR - factorL;
    sm_a[ (coord[4]*4+2)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+1) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+2)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+2) ] = factorR + factorL;
    sm_a[ (coord[4]*4+2)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+3) ] = static_cast<half>(0.0f);
                                                                  
    sm_a[ (coord[4]*4+3)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+0) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+3)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+1) ] = factorR - factorL;
    sm_a[ (coord[4]*4+3)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+2) ] = static_cast<half>(0.0f);
    sm_a[ (coord[4]*4+3)*(param.dc.Ls*4+sm_pad_size)+(threadIdx.x*4+3) ] = factorR + factorL;  
  }
}

__syncthreads();

// wmma.h
{
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int tm_dim = float(M) / float(WMMA_M);
const int tn_dim = float(N) / float(WMMA_N);
const int tk_dim = float(K) / float(WMMA_K);

// The actual/physical warp assigned to each thread in this block
int phys_warp_n_dim = float(blockDim.x)/float(warpSize); // TODO: should make sure blockDim.x is AT LEAST 32.
int phys_warp_m_dim = blockDim.y;

int phys_warp_n = float(threadIdx.x)/float(warpSize);
int phys_warp_m = threadIdx.y; 

int total_num_warp = phys_warp_n_dim*phys_warp_m_dim;
int total_num_tile = tm_dim*tn_dim;

int warp_cycle = float(total_num_tile)/float(total_num_warp);

// Set up the wmma stuff
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
// Zero the initial acc.

for(int c = 0; c < warp_cycle; c++){
  int phys_warp_index = phys_warp_n*phys_warp_m_dim+phys_warp_m + total_num_warp*c;
  // The logical warp assigned to each part of the matrix.
  int warp_m = float(phys_warp_index)/float(tn_dim);
  int warp_n = phys_warp_index-warp_m*tn_dim;
  //
  wmma::fill_fragment(c_frag, (half)0.0f);
  for( int k = 0; k < tk_dim; k++ ){
    
    int a_row = warp_m*WMMA_M;
    int a_col = k*WMMA_K;
    int b_row = k*WMMA_K;
    int b_col = warp_n*WMMA_N;
    
//    if( a_row < M && a_col < K && b_row < K && b_col < N ){    
      // Load Matrix
      wmma::load_matrix_sync(a_frag, sm_a+a_row+a_col*M_sm, M_sm);
      wmma::load_matrix_sync(b_frag, sm_b+b_col+b_row*N_sm, N_sm);
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//    }
//    __syncthreads();
  } 

//  __syncthreads();

  int c_row = warp_m*WMMA_M;
  int c_col = warp_n*WMMA_N;

  if(c_row < M && c_col < N){ 
    wmma::store_matrix_sync(sm_c+c_col+c_row*N_sm, c_frag, N_sm, wmma::mem_row_major);
  }
//  __syncthreads();
}
__syncthreads();

o00_re = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ];
o00_im = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ];
o01_re = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ];
o01_im = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ];
o02_re = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ];
o02_im = sm_c[ (coord[4]*4+0)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ];
o10_re = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ];
o10_im = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ];
o11_re = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ];
o11_im = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ];
o12_re = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ];
o12_im = sm_c[ (coord[4]*4+1)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ];
o20_re = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ];
o20_im = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ];
o21_re = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ];
o21_im = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ];
o22_re = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ];
o22_im = sm_c[ (coord[4]*4+2)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ];
o30_re = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+0 ];
o30_im = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+1 ];
o31_re = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+2 ];
o31_im = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+3 ];
o32_re = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+4 ];
o32_im = sm_c[ (coord[4]*4+3)*(blockDim.x*6+sm_pad_size)+threadIdx.x*6+5 ];

} // wmma.h

if(!idle){
// write spinor field back to device memory
WRITE_SPINOR(param.sp_stride);
}

// undefine to prevent warning when precision is changed
#undef m5
#undef mdwf_b5
#undef mdwf_c5
#undef mferm
#undef a
#undef b
#undef POW
#undef SHARED_STRIDE

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im
#undef i10_re
#undef i10_im
#undef i11_re
#undef i11_im
#undef i12_re
#undef i12_im
#undef i20_re
#undef i20_im
#undef i21_re
#undef i21_im
#undef i22_re
#undef i22_im
#undef i30_re
#undef i30_im
#undef i31_re
#undef i31_im
#undef i32_re
#undef i32_im

