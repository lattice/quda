#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sm_30_intrinsics.h>
#include <sm_35_intrinsics.h>

#include <cuComplex.h>

////////////////////////////////////////////

__device__ inline double dshfl(double x, int lane)
{
  int lo, hi;

  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));

  lo = __shfl(lo, lane);
  hi = __shfl(hi, lane);

  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi));

  return x;
}

__device__ inline cuDoubleComplex zshfl(cuDoubleComplex x, int lane)
{
  int lo, hi;// Split the double number into 2 32b registers.

  double xre = cuCreal(x);
  
  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(xre));

  lo = __shfl(lo, lane);
  hi = __shfl(hi, lane);

  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(xre) : "r"(lo), "r"(hi));

  double xim = cuCimag(x);

  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(xim));

  lo = __shfl(lo, lane);
  hi = __shfl(hi, lane);

  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(xim) : "r"(lo), "r"(hi));

  return make_cuDoubleComplex(xre, xim);
}

__device__ __noinline__ void warp_reduce_complex_32x16(double *real, double *imag, const int ltid, const int wrp_idx)
{
   if(ltid < (warpSize*8)) real[0] += real[warpSize*8], imag[0] += imag[warpSize*8];
   __syncthreads();
   if(ltid < (warpSize*4)) real[0] += real[warpSize*4], imag[0] += imag[warpSize*4];
   __syncthreads();
   if(ltid < (warpSize*2)) real[0] += real[warpSize*2], imag[0] += imag[warpSize*2];
   __syncthreads();

   if(ltid < warpSize /*the last warp*/)
   {
      real[0] += real[warpSize ];       imag[0] += imag[warpSize ];
      real[warpSize*wrp_idx] = real[0]; imag[warpSize*wrp_idx] = imag[0];
   }

   return;
}




//__global__ void SpinorMM(ReduceArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg) 
//WARNING: all matrices in column-major format
//block configuration: 32x16

/*
<----32----->                   <-16->
[|| || '' '' ......] |          [|| || .......]
[|| || '' '' ......] 16         [|| || .......]
[|| || '' '' ......] |    x     [|| || .......]
[|| || '' '' ......] 16         [|| || .......] 
<-16->               .
 
[.. .. .. .. ......]            [.. .. .......]
[.. .. .. .. ......]            [.. .. .......]
[.. .. .. .. ......]            [.. .. .......]
[..          ......]            [.. ..        ]
[..          ......]            [.. ..        ] 

*/
//__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) 
#define CHUNK_NUM 2
__global__ void  
SpinorMM_32x16(cuFloatComplex *outBuff, const int bldm, cuFloatComplex *sMat, const int srows, const int scols, const int sldm, cuDoubleComplex *cMat, const int crows, const int ccols, const int cldm) 
{
  //Note assignment due to FORTRAN data format!
  unsigned int tidY      = threadIdx.x;
  unsigned int blkY      = blockDim.x;
  unsigned int y         = blockIdx.x*(blockDim.x) + threadIdx.x;//global coordinate in spinor field length:[0 ... sldm)
  unsigned int gridSizeY = gridDim.x*blockDim.x;//total size => sldm

  unsigned int tidX      = threadIdx.y;
  unsigned int blkX      = blockDim.y;
  unsigned int x         = blockIdx.y*(blockDim.y) + threadIdx.y;//global coordinate in (restarted) search space: [0 ... max_nev = 2nev for eigCG or nev+1 for GMRESDR)
  unsigned int gridSizeX = gridDim.y*blockDim.y;//total size => max_nev

  unsigned int ltid      = threadIdx.x + blockDim.x * threadIdx.y;//linear thread index within a thread block

  if(blkY != warpSize) return;

  const int halfWarpSize = warpSize / 2; 

  cuDoubleComplex smat_accum = make_cuDoubleComplex (0.0, 0.0);//final result!

  extern __shared__ double sm_buffer[];//2*blkY*blkX*2*sizeof(double) to keep chunk of 32x16 block (currently 2 hard-coded chunks!)

  double *relem  = sm_buffer + 0*blkY*blkX;
  double *ielem  = sm_buffer + 1*blkY*blkX;

  cuDoubleComplex smat_chnk1, smat_chnk2;//in X direction two chunks

  int idy = tidY;//Y-coordinate of the "little" matrix
  int idx = tidX;//X-coordinate of a single chunk of the "big" matrix

//Start Loop in Y direction of cmat (we loop in X direction of sMat but load 2 chunks per iteration)
  while(idy < crows)
  {

    cuDoubleComplex tmp1, tmp2;

    //Load little matrix element (no textures for now):
    cuDoubleComplex cmat = cMat[idy+x*cldm];
    
    //load this to shared memory:
    relem[tidY+tidX*blkY] = cuCreal(cmat);
    ielem[tidY+tidX*blkY] = cuCimag(cmat);
   
    {
      //Load big matrix elements (mind of 2 chunks) via a texture object:
      int l = y + idx * sldm;//coordinate of the first chunk
      //cuFloatComplex smatf = tex1Dfetch<float2>(sMatTex, l); 
      //smat_chnk1 = cuComplexFloatToDouble( smatf );       
      smat_chnk1 = cuComplexFloatToDouble( sMat[l] );

      //Load the second chunk:
      //smatf   = tex1Dfetch<float2>(sMatTex, (l+blkX * sldm));        
      //smat_chnk2 = cuComplexFloatToDouble( smatf );       
      smat_chnk2 = cuComplexFloatToDouble( sMat[l] );
    }
    //Now do transposition:
    __syncthreads();

    const int locY = tidY % blkX ; //note blkY = 2*blkX = (CHUNK_NUM*blkX) in this version
    const int z    = (tidY / blkX);
    const int offs = z*(blkY / CHUNK_NUM);  
 
    tmp1 = make_cuDoubleComplex(relem[tidX+locY*blkY+offs], ielem[tidX+locY*blkY+offs]); 

    //Now exchange elements for the second chunk:
    int lane = tidY + (1-2*z)*halfWarpSize;

    tmp2 = zshfl(tmp1, lane);

    cuDoubleComplex cmatT_chnk1, cmatT_chnk2; 

    if(tidY < halfWarpSize)
    {
       cmatT_chnk1 = make_cuDoubleComplex(cuCreal(tmp1), cuCimag(tmp1));
       cmatT_chnk2 = make_cuDoubleComplex(cuCreal(tmp2), cuCimag(tmp2));
    }
    else
    {
       cmatT_chnk1 = make_cuDoubleComplex(cuCreal(tmp2), cuCimag(tmp2));
       cmatT_chnk2 = make_cuDoubleComplex(cuCreal(tmp1), cuCimag(tmp1));
    }

    //pointers for the local reductions below:
    double *reducr = sm_buffer + tidX*blkY+tidY;
    double *reduci = reducr + blkY*blkX;

#pragma unroll 
    for(int i = 0; i < 16; i++)
    {
      //Do broadcast within the warp:
      tmp1 = zshfl(cmatT_chnk1, i);//broadcast form ith lane 
      tmp2 = zshfl(cmatT_chnk2, i);//broadcast form ith lane 

      //Do complex fma:
      smat_accum = cuCfma(smat_chnk1, tmp1, smat_accum);
      smat_accum = cuCfma(smat_chnk2, tmp2, smat_accum);

      //Load smat_accum to shared memory:
      reducr[0] = cuCreal(smat_accum);
      reduci[0] = cuCimag(smat_accum); 

      warp_reduce_complex_32x16(reducr, reduci, ltid, i);//wrp_idx = i

      __syncthreads();

      if(ltid == (warpSize*i))
      {
         tmp1       = make_cuDoubleComplex(reducr[0], reduci[0]); 
         smat_accum = cuCfma(smat_accum, tmp1, smat_accum);
      }
    }
 
    idx += CHUNK_NUM*blkX; //offset for two chunks of the "big" matrix (note: blkY = CHUNK_NUM*blkX)
    idy += blkY;//continue loop in Y-direction of the little matrix (or X-direction of the big matix)
  }

  outBuff[y+x*bldm] = cuComplexDoubleToFloat( smat_accum ); //cache avoiding loads to the output buffer

  return;

}

#undef CHUNK_NUM

void sMM(void *outBuff, const int bldm,  void *sMat, const int srows, const int scols, const int sldm, void *cMat, const int crows, const int ccols, const int cldm)
{

  if(crows != scols) printf("\nError: wrong matrix dimensions\n"), exit(-1);

  const int blk_sizex = 32;
  const int blk_sizey = 16; 

  const int grid_sizex = (srows + 31) / blk_sizex;
  const int grid_sizey = ccols / blk_sizey; 

  dim3 blck_dim(blk_sizex, blk_sizey, 1);
  dim3 grid_dim(grid_sizex, grid_sizey, 1);
	
  ///For reduction operation:
  int reduce_bytes = blk_sizex * blk_sizey * sizeof(cuDoubleComplex);  

  //cudaTextureObject_t sMatTex = 0;

  //cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  //cudaResourceDesc resDesc;
  //memset(&resDesc, 0, sizeof(resDesc));
  //resDesc.resType                = cudaResourceTypeLinear;
  //resDesc.res.linear.devPtr      = sMat;
  //resDesc.res.linear.desc        = desc;
  //resDesc.res.linear.sizeInBytes = scols*sldm*sizeof(cuFloatComplex);//sldm: physical array length (including padding)
      
  //cudaTextureDesc texDesc;
  //memset(&texDesc, 0, sizeof(texDesc));
  //texDesc.readMode = cudaReadModeElementType;
      
  //cudaCreateTextureObject(&sMatTex, &resDesc, &texDesc, NULL);

  cudaFuncSetCacheConfig( SpinorMM_32x16, cudaFuncCachePreferL1 );

  SpinorMM_32x16 <<< grid_dim, blck_dim, reduce_bytes, 0 >>>((cuFloatComplex *)outBuff, bldm, (cuFloatComplex*)sMat, srows, scols, sldm, (cuDoubleComplex *)cMat, crows, ccols, cldm);

  //cudaDestroyTextureObject(sMatTex);

  return;
}

//Simplified version for the above:
#define BLOCK_SIZE 16

//Column major format: Big matrix times Little matrix.

__global__ void SMatCMatCuda_16x16(cuFloatComplex *outBuff, const int bldm, cuFloatComplex *sMat, const int sldm, cuDoubleComplex *cMat, const int cldm, const int scols)
{
    //block coords:
    int by = blockIdx.x;
    int bx = blockIdx.y;

    //local coords:
    int ty = threadIdx.x;
    int tx = threadIdx.y;

    int sBegin = BLOCK_SIZE * by;//global offset in Y-direction for the Big matrix

    int sEnd   = sBegin + sldm*scols - 1;//loop limit in X-direction for the Big matrix

    int sStep  = sldm * BLOCK_SIZE;//step in X-direction for the Big matrix

    int cBegin = cldm * BLOCK_SIZE * bx;//global offset in X-direction for the Little matrix

    int cStep  = BLOCK_SIZE;//step in Y-direction for the Little matrix

    cuDoubleComplex accum = make_cuDoubleComplex (0.0, 0.0);

    cuFloatComplex  ftmp;
    cuDoubleComplex dtmp;


    for (int s = sBegin, c = cBegin; s <= sEnd; s += sStep, c += cStep)
    {

        __shared__ float reSmat[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float imSmat[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ double reCmat[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double imCmat[BLOCK_SIZE][BLOCK_SIZE];


        ftmp = sMat[s + sldm * tx + ty];
        reSmat[ty][tx] = cuCrealf(ftmp);
        imSmat[ty][tx] = cuCimagf(ftmp);

        dtmp = cMat[c + cldm * tx + ty];
        reCmat[ty][tx] = cuCreal(dtmp);
        imCmat[ty][tx] = cuCimag(dtmp);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            ftmp = make_cuFloatComplex(reSmat[ty][k], imSmat[ty][k]);

            dtmp = make_cuDoubleComplex(reCmat[k][tx], imCmat[k][tx]);

            cuDoubleComplex dtmp2 = cuComplexFloatToDouble( ftmp );

            accum = cuCfma(dtmp2, dtmp, accum);
        }

        __syncthreads();
    }

    int idx = BLOCK_SIZE * by + bldm * BLOCK_SIZE * bx;

    outBuff[idx + bldm * tx + ty] = cuComplexDoubleToFloat( accum );

    return;
}

void sMM_v2(void *outBuff, const int bldm,  void *sMat, const int srows, const int scols, const int sldm, void *cMat, const int crows, const int ccols, const int cldm)
{
    // for test only:
    if(scols != crows) printf("\nError: wrong dimensions\n"), exit(-1);

    const int block_size = 16;

    if (ccols % block_size != 0) printf("\nError: wrong dimensions\n"), exit(-1);

    // Setup execution parameters (column-major format):
    dim3 threads(block_size, block_size);
    dim3 grid((srows+15) / threads.x, ccols / threads.y);//both ccols and srows must be multiple of block_size...

    cudaFuncSetCacheConfig( SMatCMatCuda_16x16, cudaFuncCachePreferShared );

    SMatCMatCuda_16x16<<< grid, threads >>>((cuFloatComplex*)outBuff, bldm, (cuFloatComplex*)sMat, sldm, (cuDoubleComplex*)cMat, cldm, scols);
}


