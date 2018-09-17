#include <transfer.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <mpi.h>
#include <qlua_contract.h>
#include <qlua_contract_kernels.cuh>
#include <qlua_contract_shifts.cuh>

namespace quda {  

  const int nShiftFlag = 20;
  const int nShiftType = 3;
  
  const char *qcTMD_ShiftFlagArray = "XxYyZzTtQqRrSsUuVvWw" ;
  const char *qcTMD_ShiftTypeArray[] = {
    "Covariant",
    "Non-Covariant",
    "AdjSplitCov"};
  
  const char *qcTMD_ShiftDirArray[] = {"x", "y", "z", "t"};
  const char *qcTMD_ShiftSgnArray[] = {"-", "+"};
  
  __global__ void QluaSiteOrderCheck_kernel(QluaUtilArg *utilArg){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;    
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    if (x_cb >= utilArg->volumeCB) return;
    if (pty >= utilArg->nParity) return;

    int crd[5];
    getCoords(crd, x_cb, utilArg->lL, pty);  //-- Get local coordinates crd[] at given x_cb and pty
    crd[4] = 0;  

    int idx_cb = linkIndex(crd, utilArg->lL); //-- Checkerboard index, MUST be equal to x_cb
    
    int i_rlex = crd[0] + utilArg->lL[0]*(crd[1] + utilArg->lL[1]*(crd[2] + utilArg->lL[2]*(crd[3])));  //-- Full lattice site index
    int i_par = (crd[0] + crd[1] + crd[2] + crd[3]) & 1;

    if( (i_rlex/2 != x_cb) || (pty != i_par) || (idx_cb != x_cb) ){
      d_crdChkVal = -1;
      printf("coordCheck - ERROR: x_cb = %d, pty = %d: Site order mismatch!\n", x_cb, pty);    
    }
    else d_crdChkVal = 0;

  }//-- function

  
  int QluaSiteOrderCheck(QluaUtilArg *utilArg){
    int crdChkVal;

    QluaUtilArg *utilArg_dev;
    cudaMalloc((void**)&(utilArg_dev), sizeof(QluaUtilArg));
    checkCudaErrorNoSync();
    cudaMemcpy(utilArg_dev, utilArg,  sizeof(QluaUtilArg), cudaMemcpyHostToDevice);
    
    dim3 blockDim(THREADS_PER_BLOCK, utilArg->nParity, 1);
    dim3 gridDim((utilArg->volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
    
    QluaSiteOrderCheck_kernel<<<gridDim,blockDim>>>(utilArg_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    cudaMemcpyFromSymbol(&crdChkVal, d_crdChkVal, sizeof(crdChkVal), 0, cudaMemcpyDeviceToHost);
    checkCudaErrorNoSync();
    
    cudaFree(utilArg_dev);
    utilArg_dev = NULL;

    return crdChkVal;
  }//-- function
  //---------------------------------------------------------------------------

  
  __global__ void convertSiteOrder_QudaQDP_to_momProj_kernel(void *dst, const void *src, QluaUtilArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    int crd[5];
    getCoords(crd, x_cb, arg->lL, pty);
    int i_t = crd[arg->t_axis];
    int i_sp = 0;

    for (int i = 0 ; i < 4 ; i++)
      i_sp += arg->sp_stride[i] * crd[i];

    for (int i_f = 0 ; i_f < arg->nFldSrc ; i_f++){
      char *dst_i = (char*)dst + arg->rec_size * (
          i_sp + arg->sp_locvol * (
          i_f  + arg->nFldDst   * i_t));
      const char *src_i = (char*)src + arg->rec_size * (
          x_cb + arg->volumeCB * (
          pty + 2 * i_f));

      for (int j = 0 ; j < arg->rec_size ; j++)
        *dst_i++ = *src_i++;
    }//- i_f
    
  }//-- function

  void convertSiteOrder_QudaQDP_to_momProj(void *corrInp_dev, const void *corrQuda_dev, QluaUtilArg utilArg){

    QluaUtilArg *utilArg_dev;
    cudaMalloc((void**)&(utilArg_dev), sizeof(QluaUtilArg));
    checkCudaErrorNoSync();
    cudaMemcpy(utilArg_dev, &utilArg,  sizeof(QluaUtilArg), cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK, utilArg.nParity, 1);
    dim3 gridDim((utilArg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    convertSiteOrder_QudaQDP_to_momProj_kernel<<<gridDim,blockDim>>>(corrInp_dev, corrQuda_dev, utilArg_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    
    cudaFree(utilArg_dev);
    utilArg_dev = NULL;
  }
  //---------------------------------------------------------------------------

  qcTMD_ShiftFlag TMDparseShiftFlag(char flagStr){
    
    qcTMD_ShiftFlag shfFlag = qcShfStr_None;
    for(int iopt=0;iopt<nShiftFlag;iopt++){
      if( flagStr == qcTMD_ShiftFlagArray[iopt] ){
	shfFlag = (qcTMD_ShiftFlag)iopt;
	break;
      }
    }
    if(shfFlag==qcShfStr_None) errorQuda("TMDparseShiftFlag: Cannot parse given shift flag, flagStr = %c.\n", flagStr);
    return shfFlag;
  }

  qcTMD_ShiftType TMDparseShiftType(char *typeStr){
    
    qcTMD_ShiftType shfType = qcInvalidShift;
    for(int iopt=0;iopt<nShiftType;iopt++){
      if( strcmp(typeStr,qcTMD_ShiftTypeArray[iopt])==0 ){
	shfType = (qcTMD_ShiftType)iopt;
	break;
      }
    }
    if(shfType==qcInvalidShift) errorQuda("TMDparseShiftType: Cannot parse given shift type, typeStr = %s.\n", typeStr);
    return shfType;
  }

  qcTMD_ShiftDir TMDparseShiftDirection(qcTMD_ShiftFlag shfFlag){

    qcTMD_ShiftDir shfDir = qcShfDirNone;
    switch(shfFlag){
    case qcShfStr_x:
    case qcShfStr_X: {
      shfDir = qcShfDir_x;
    } break;
    case qcShfStr_y:
    case qcShfStr_Y: {
      shfDir = qcShfDir_y;
    } break;
    case qcShfStr_z:
    case qcShfStr_Z: {
      shfDir = qcShfDir_z;
    } break;
    case qcShfStr_t:
    case qcShfStr_T: {
      shfDir = qcShfDir_t;
    } break;
    default: errorQuda("TMDparseShiftDirection: Unsupported shift flag, shfFlag = %c.\n", (shfFlag >=0 && shfFlag<nShiftFlag) ? qcTMD_ShiftFlagArray[(int)shfFlag] : '?');
    }//-- switch    

    return shfDir;
  }
  //---------------------------------------------------------------------------

  qcTMD_ShiftSgn TMDparseShiftSign(qcTMD_ShiftFlag shfFlag, bool flipShfSgn=false){

    qcTMD_ShiftSgn shfSgn = qcShfSgnNone;
    switch(shfFlag){
    case qcShfStr_X:
    case qcShfStr_Y:
    case qcShfStr_Z:
    case qcShfStr_T: {
      if(!flipShfSgn) shfSgn = qcShfSgnPlus;
      else shfSgn = qcShfSgnMinus;
    } break;
    case qcShfStr_x:
    case qcShfStr_y:
    case qcShfStr_z:
    case qcShfStr_t: {
      if(!flipShfSgn) shfSgn = qcShfSgnMinus;
      else shfSgn = qcShfSgnPlus;
    } break;
    default: errorQuda("TMDparseShiftSign: Unsupported shift flag, shfFlag = %c.\n", (shfFlag >=0 && shfFlag<nShiftFlag) ? qcTMD_ShiftFlagArray[(int)shfFlag] : '?');
    }//-- switch    

    return shfSgn;
  }
  //---------------------------------------------------------------------------


  void qcCopyExtendedGaugeField(cudaGaugeField *dst, cudaGaugeField *src, const int *R){

    if( (dst->GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) ||
	(src->GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) )
      errorQuda("qcCopyExtendedGaugeField: Support only extended Gauge Fields!\n");

    copyExtendedGauge(*dst, *src, QUDA_CUDA_FIELD_LOCATION);
    dst->exchangeExtendedGhost(R, QCredundantComms);
  }
  //---------------------------------------------------------------------------


  __global__ void qcCopyCudaLink_kernel(Arg_CopyCudaLink *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    if(arg->extendedGauge){
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];

      Link srcU = arg->Usrc(arg->i_src, linkIndex(c2, arg->dimEx), pty);
      arg->Udst(arg->i_dst, linkIndex(c2, arg->dimEx), pty) = srcU;
    }
    else{
      Link srcU = arg->Usrc(arg->i_src, x_cb, pty);
      arg->Udst(arg->i_dst, x_cb, pty) = srcU;
    }

  }
  
  void qcCopyCudaLink(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src, const int *R){

    Arg_CopyCudaLink arg(dst, i_dst, src, i_src);
    Arg_CopyCudaLink *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    if(arg.nParity != 2) errorQuda("qcCopyCudaLink: This function supports only Full Site Subset fields!\n");

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    qcCopyCudaLink_kernel<<<gridDim,blockDim>>>(arg_dev);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeExtendedGhost(R, QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }
  //---------------------------------------------------------------------------


  __global__ void qcSetGaugeToUnity_kernel(Arg_SetUnityLink *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    Link unitU(arg->unityU); //- That's a 3x3 unity matrix

    if(arg->extendedGauge){
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];

      arg->U(arg->mu, linkIndex(c2, arg->dimEx), pty) = unitU;
    }
    else arg->U(arg->mu, x_cb, pty) = unitU;

  }

  void qcSetGaugeToUnity(cudaGaugeField *U, int mu, const int *R){

    Arg_SetUnityLink arg(U, mu);
    Arg_SetUnityLink *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
    checkCudaError();

    if(arg.nParity != 2) errorQuda("qcResetGaugeToUnity: This function supports only Full Site Subset fields!\n");

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    qcSetGaugeToUnity_kernel<<<gridDim,blockDim>>>(arg_dev);
    cudaDeviceSynchronize();
    checkCudaError();

    U->exchangeExtendedGhost(R, QCredundantComms);
    
    cudaFree(arg_dev);
    arg_dev = NULL;
  }
  //---------------------------------------------------------------------------


  void qcExchangeGhostVec(ColorSpinorField *x){
    const int nFace  = 1;
    x->exchangeGhost((QudaParity)(1), nFace, 0); //- first argument is redundant when nParity = 2. nFace MUST be 1 for now.
  }
  void qcExchangeGhostProp(ColorSpinorField **x){
    for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
      qcExchangeGhostVec(x[ivec]);
    }
  }
  //---------------------------------------------------------------------------

  
  void qcSwapCudaGauge(cudaGaugeField **x1, cudaGaugeField **x2){
    cudaGaugeField *xtmp = *x1;
    *x1 = *x2;
    *x2 = xtmp;
  }

  void qcSwapCudaVec(cudaColorSpinorField **x1, cudaColorSpinorField **x2){
    cudaColorSpinorField *xtmp = *x1;
    *x1 = *x2;
    *x2 = xtmp;
  }

  void qcCPUtoCudaVec(cudaColorSpinorField *cudaVec, cpuColorSpinorField *cpuVec){
    *cudaVec = *cpuVec;
  }  
  void qcCPUtoCudaProp(cudaColorSpinorField **cudaProp, cpuColorSpinorField **cpuProp){
    for(int i=0;i<QUDA_PROP_NVEC;i++)
      qcCPUtoCudaVec(cudaProp[i], cpuProp[i]);
  }
  //---------------------------------------------------------------------------


  void perform_ShiftCudaVec_nonCov(ColorSpinorField *dst, ColorSpinorField *src, qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftCudaVec_nonCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis Non-Covariant shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);

    qcExchangeGhostVec(src);

    Arg_ShiftCudaVec_nonCov arg(dst, src);
    Arg_ShiftCudaVec_nonCov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", funcname);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    ShiftCudaVec_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("##### %s: Kernel done in %f sec.\n", funcname, t2-t1);
    
    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftCudaVec_nonCov
  
  void perform_ShiftCudaVec_Cov(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gf,
				qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftCudaVec_Cov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis Covariant shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);

    qcExchangeGhostVec(src);

    Arg_ShiftCudaVec_Cov arg(dst, src, gf);
    Arg_ShiftCudaVec_Cov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", funcname);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    ShiftCudaVec_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("##### %s: Kernel done in %f sec.\n", funcname, t2-t1);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftCudaVec_Cov


  void perform_ShiftGauge_nonCov(cudaGaugeField *dst, cudaGaugeField *src,
				 qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftGauge_nonCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform a Gauge Link Non-Covariant shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftGauge_nonCov arg(dst, src);
    Arg_ShiftGauge_nonCov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", funcname);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    ShiftGauge_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("##### %s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftGauge_nonCov


  void perform_ShiftLink_Cov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
			     cudaGaugeField *gf, qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftLink_Cov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform a Gauge Link Covariant shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftLink_Cov arg(dst, i_dst, src, i_src, gf);
    Arg_ShiftLink_Cov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", funcname);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    ShiftLink_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("##### %s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftLink_Cov


  void perform_ShiftLink_AdjSplitCov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
				     cudaGaugeField *gf, cudaGaugeField *gf2,
				     qcTMD_ShiftFlag shfFlag, bool flipShfSgn){

    const char *funcname = "perform_ShiftLink_AdjSplitCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag, flipShfSgn);
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform a Gauge Link AdjSplitCov shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftLink_AdjSplitCov arg(dst, i_dst, src, i_src, gf, gf2);
    Arg_ShiftLink_AdjSplitCov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", funcname);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    ShiftLink_AdjSplitCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("##### %s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- ShiftLink_AdjSplitCov
  //---------------------------------------------------------------------------


  void perform_Std_Contract(complex<QUDA_REAL> *corrQuda_dev, QluaContractArg arg){

    const char *func_name = "perform_Std_Contract";
    
    QluaContractArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);    
    checkCudaError();

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    switch(arg.cntrType){
    case what_baryon_sigma_UUS: {
      baryon_sigma_twopt_asymsrc_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_qbarq_g_F_B: {
      qbarq_g_P_P_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_qbarq_g_F_aB: {
      qbarq_g_P_aP_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_qbarq_g_F_hB: {
      qbarq_g_P_hP_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_meson_F_B: {
      meson_F_B_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_meson_F_aB: {
      meson_F_aB_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    case what_meson_F_hB: {
      meson_F_hB_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
    } break;
    default: errorQuda("%s: Contraction type \'%s\' not implemented!\n", func_name, qc_contractTypeStr[arg.cntrType]);
    }//- switch
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[arg.cntrType], t2-t1);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_Std_Contract

  //-Top-level function
  void QuarkContractStd_GPU(complex<QUDA_REAL> *corrQuda_dev,
			    cudaColorSpinorField **cudaProp1,
			    cudaColorSpinorField **cudaProp2,
			    cudaColorSpinorField **cudaProp3,
			    complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
			    qudaAPI_Param paramAPI){    

    const char *func_name = "QuarkContractStd_GPU";
    
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

    if( (paramAPI.mpParam.cntrType == what_tmd_g_F_B) || (paramAPI.mpParam.cntrType == what_qpdf_g_F_B) )
      errorQuda("%s: Contraction type \'%s\' not supported!\n", func_name, qc_contractTypeStr[paramAPI.mpParam.cntrType]);

    QluaContractArg arg(cudaProp1, cudaProp2, cudaProp3, paramAPI.mpParam.cntrType, paramAPI.preserveBasis); 
    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
    if(paramAPI.mpParam.cntrType == what_baryon_sigma_UUS) copySmatricesToSymbol(S2, S1);

    perform_Std_Contract(corrQuda_dev, arg);

  }//-- QuarkContractStd_GPU
  //---------------------------------------------------------------------------


  //-- Class for TMD contractions, make it tunable
  class qcTMD : public TunableVectorY {

  protected:
    qcTMD_Arg *arg_dev;
    const cudaColorSpinorField *meta;
    qluaCntr_Type cntrType;
    complex<QUDA_REAL> *corrQuda_dev;
    int Nc;
    int Ns;
    
    long long flops() const{
      return (long long)meta->VolumeCB() * meta->SiteSubset() * (Nc*Nc*Ns*Ns*(1+Nc*Ns) * 8 + Ns*Ns*Ns * 2);
    }
    long long bytes() const{
      return (long long)meta->VolumeCB() * meta->SiteSubset() * (2*Ns*Ns*Nc*Nc + Nc*Nc) * 2*8;
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return meta->VolumeCB(); }

  public:
    qcTMD(const cudaColorSpinorField *meta_, qcTMD_Arg *arg_dev_, complex<QUDA_REAL> *corrQuda_dev_, qluaCntr_Type cntrType_)
      : TunableVectorY(meta_->SiteSubset()), meta(meta_),
	arg_dev(arg_dev_), corrQuda_dev(corrQuda_dev_), cntrType(cntrType_), Nc(QUDA_Nc), Ns(QUDA_Ns)
    {
      strcpy(aux, meta_->AuxString());
      strcat(aux, comm_dim_partitioned_string());
    }
    virtual ~qcTMD() { }

    long long getFlops(){return flops();}
    long long getBytes(){return bytes();}

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if(cntrType != what_tmd_g_F_B) errorQuda("qcTMD::apply(): Support only TMD contractions for now!\n");
      tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(corrQuda_dev, arg_dev);
      //      tmd_g_U_P_P_gvec_kernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(corrQuda_dev, arg_dev);
    }

    TuneKey tuneKey() const { return TuneKey(meta->VolString(), typeid(*this).name(), aux); }
  };


  //-Top-level function
  void QuarkContractTMD_GPU(QuarkTMD_state *qcs){

    const char *func_name = "QuarkContractTMD_GPU";

    if(qcs->cntrType != what_tmd_g_F_B)
      errorQuda("%s: This function supports only TMD contractions!\n", func_name);

    qcTMD_Arg arg(qcs->cudaPropFrw_bsh, qcs->cudaPropBkw, qcs->wlinks, qcs->i_wl_vbv, qcs->paramAPI.preserveBasis);    
    qcTMD_Arg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg) );
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", func_name);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    // tmd_g_U_P_P_gvec_kernel<<<gridDim,blockDim>>>(qcs->corrQuda_dev, arg_dev);
    tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue<<<gridDim,blockDim>>>(qcs->corrQuda_dev, arg_dev);

    // double t1 = MPI_Wtime();
    // qcTMD contractTMD(qcs->cudaPropBkw[0], arg_dev, qcs->corrQuda_dev, qcs->cntrType);
    // printfQuda("%s: contractTMD::Flops = %lld\n", func_name, contractTMD.getFlops());
    // printfQuda("%s: contractTMD::Bytes = %lld\n", func_name, contractTMD.getBytes());
    // contractTMD.apply(0);

    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[qcs->cntrType], t2-t1);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- QuarkContractTMD_GPU


} //-namespace quda
