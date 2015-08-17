#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuComplex.h>

//Simplified version for the above:
#define BLOCK_SIZE 16

//Column major format: Big matrix times Little matrix.

#ifdef MAGMA_LIB
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
#endif

void sMM_v2(void *outBuff, const int bldm,  void *sMat, const int srows, const int scols, const int sldm, void *cMat, const int crows, const int ccols, const int cldm)
{
#ifdef MAGMA_LIB
    // for test only:
    if(scols != crows) printf("\nError: wrong dimensions\n"), exit(-1);

    const int block_size = 16;

    if (ccols % block_size != 0) printf("\nError: wrong dimensions\n"), exit(-1);

    // Setup execution parameters (column-major format):
    dim3 threads(block_size, block_size);
    dim3 grid((srows+15) / threads.x, ccols / threads.y);//both ccols and srows must be multiple of block_size...

    cudaFuncSetCacheConfig( SMatCMatCuda_16x16, cudaFuncCachePreferShared );

    SMatCMatCuda_16x16<<< grid, threads >>>((cuFloatComplex*)outBuff, bldm, (cuFloatComplex*)sMat, sldm, (cuDoubleComplex*)cMat, cldm, scols);
#endif
}

#undef BLOCK_SIZE
