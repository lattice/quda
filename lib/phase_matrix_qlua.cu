/* C. Kallidonis: GPU kernel to create the 
 * phase matrix, required for momentum projection
 * December 2017
 */

#include <transfer.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <mpi.h>
#include <interface_qlua_internal.h>

namespace quda {

  struct MomProjArg{
    
    const LONG_T V3;
    const int momDim;
    const int Nmoms;
    const int expSgn;
    const int csrc[QUDA_DIM];
    const int localL[QUDA_DIM];
    const int totalL[QUDA_DIM];
    const int commCoord[QUDA_DIM];
    
  MomProjArg(momProjParam param)
  :   V3(param.V3), momDim(param.momDim), Nmoms(param.Nmoms), expSgn(param.expSgn),
      csrc{param.csrc[0],param.csrc[1],param.csrc[2],param.csrc[3]},
      localL{param.localL[0],param.localL[1],param.localL[2],param.localL[3]},
      totalL{param.totalL[0],param.totalL[1],param.totalL[2],param.totalL[3]},
      commCoord{comm_coord(0),comm_coord(1),comm_coord(2),comm_coord(3)}
    { }
  }; //-- structure
  

  __global__ void phaseMatrix_kernel(complex<QUDA_REAL> *phaseMatrix, int *momMatrix, MomProjArg *arg)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if(tid < arg->V3){ // run through the spatial volume
    
      int lcoord[3]; // Need to hard code
      int gcoord[3]; // this for now
      
      int a1 = tid / arg->localL[0];
      int a2 = a1 / arg->localL[1];
      lcoord[0] = tid - a1 * arg->localL[0];
      lcoord[1] = a1  - a2 * arg->localL[1];
      lcoord[2] = a2;
      
      gcoord[0] = lcoord[0] + arg->commCoord[0] * arg->localL[0] - arg->csrc[0];
      gcoord[1] = lcoord[1] + arg->commCoord[1] * arg->localL[1] - arg->csrc[1];
      gcoord[2] = lcoord[2] + arg->commCoord[2] * arg->localL[2] - arg->csrc[2];
      
      QUDA_REAL f = (QUDA_REAL) arg->expSgn;
      for(int im=0;im<arg->Nmoms;im++){
	QUDA_REAL phase = 0.0;
	for(int id=0;id<arg->momDim;id++)
	  phase += momMatrix[id + arg->momDim*im]*gcoord[id] / (QUDA_REAL)arg->totalL[id];
	
	phaseMatrix[tid + arg->V3*im].x =   cos(2.0*PI*phase);
	phaseMatrix[tid + arg->V3*im].y = f*sin(2.0*PI*phase);
      }

    }//-- tid check

  }//--kernel
  

  void createPhaseMatrix_GPU(complex<QUDA_REAL> *phaseMatrix_dev,
			     const int *momMatrix,
			     momProjParam param){
    int *momMatrix_dev;
    cudaMalloc((void**)&momMatrix_dev, sizeof(int)*param.momDim*param.Nmoms );
    checkCudaErrorNoSync();
    cudaMemcpy(momMatrix_dev, momMatrix, sizeof(int)*param.momDim*param.Nmoms, cudaMemcpyHostToDevice);
    
    MomProjArg arg(param);
    MomProjArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(MomProjArg) );
    checkCudaErrorNoSync();
    cudaMemcpy(arg_dev, &arg, sizeof(MomProjArg), cudaMemcpyHostToDevice);
    
    //-Call the kernel
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((param.V3 + blockDim.x -1)/blockDim.x, 1, 1); // spawn threads only for the spatial volume

    phaseMatrix_kernel<<<gridDim,blockDim>>>(phaseMatrix_dev, momMatrix_dev, arg_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    
    cudaFree(momMatrix_dev);
    cudaFree(arg_dev);
  }



}//-- namespace quda
