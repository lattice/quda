/* C. Kallidonis: Wrapper functions called from interface_qlua.cpp.
 * These wrappers call GPU kernels from qlua_contract_kernels.cu,
 * qlua_contract_shifts.cu and qlua_util_kernels.cu
 * Update: September 2018
 */

#include <qlua_contract_kernels.cuh>
#include <qlua_contract_shifts.cuh>
#include <qlua_util_kernels.cuh>

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
  //---------------------------------------------------------------------------


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
  //---------------------------------------------------------------------------
  
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
  //---------------------------------------------------------------------------
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
  //---------------------------------------------------------------------------
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

  //- Deprecated function
  void qcExchangeGhostProp(ColorSpinorField **x, int Nvec=12){
    for(int ivec=0;ivec<Nvec;ivec++){
      qcExchangeGhostVec(x[ivec]);
    }
  }
  //---------------------------------------------------------------------------

  
  void qcSwapCudaGauge(cudaGaugeField **x1, cudaGaugeField **x2){
    cudaGaugeField *xtmp = *x1;
    *x1 = *x2;
    *x2 = xtmp;
  }
  //---------------------------------------------------------------------------
  void qcSwapCudaVec(cudaColorSpinorField **x1, cudaColorSpinorField **x2){
    cudaColorSpinorField *xtmp = *x1;
    *x1 = *x2;
    *x2 = xtmp;
  }
  //---------------------------------------------------------------------------
  void qcCPUtoCudaVec(cudaColorSpinorField *cudaVec, cpuColorSpinorField *cpuVec){
    *cudaVec = *cpuVec;
  }  
  void qcCPUtoCudaProp(cudaColorSpinorField **cudaProp, cpuColorSpinorField **cpuProp, int Nvec){
    for(int i=0;i<Nvec;i++)
      qcCPUtoCudaVec(cudaProp[i], cpuProp[i]);
  }
  //---------------------------------------------------------------------------

  void qcCopyGammaToConstMem(){
    qcGammaStruct gamma_h;
    for(int m=0;m<QC_LEN_G;m++){
      for(int n=0;n<QC_Ns;n++){
	gamma_h.left_ind[m][n] = gamma_left_ind_cMem(m, n);
	gamma_h.left_coeff[m][n] = {gamma_left_coeff_Re_cMem(m,n,0), gamma_left_coeff_Re_cMem(m,n,1)};
      }
    }
    qcCopyGammaToSymbol(gamma_h);
  }
  //---------------------------------------------------------------------------

  void perform_ShiftCudaVec_nonCov(ColorSpinorField *dst, ColorSpinorField *src, qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftCudaVec_nonCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  ){
      if(getVerbosity() >= QUDA_VERBOSE)
	printfQuda("%s: Will perform an On-Axis Non-Covariant shift in the %s%s direction\n",
		   funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
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
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Kernel done in %f sec.\n", funcname, t2-t1);
    
    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftCudaVec_nonCov
  //---------------------------------------------------------------------------
  
  void perform_ShiftCudaVec_Cov(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gf,
				qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftCudaVec_Cov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  ){
      if(getVerbosity() >= QUDA_VERBOSE)
	printfQuda("%s: Will perform an On-Axis Covariant shift in the %s%s direction\n",
		   funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
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
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Kernel done in %f sec.\n", funcname, t2-t1);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftCudaVec_Cov
  //---------------------------------------------------------------------------

  void perform_ShiftGauge_nonCov(cudaGaugeField *dst, cudaGaugeField *src,
				 qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftGauge_nonCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  ){
      if(getVerbosity() >= QUDA_VERBOSE)
	printfQuda("%s: Will perform a Gauge Link Non-Covariant shift in the %s%s direction\n",
		   funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
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
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftGauge_nonCov
  //---------------------------------------------------------------------------

  void perform_ShiftLink_Cov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
			     cudaGaugeField *gf, qcTMD_ShiftFlag shfFlag){

    const char *funcname = "perform_ShiftLink_Cov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag);     
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  ){
      if(getVerbosity() >= QUDA_VERBOSE)
	printfQuda("%s: Will perform a Gauge Link Covariant shift in the %s%s direction\n",
		   funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
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
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- perform_ShiftLink_Cov
  //---------------------------------------------------------------------------

  void perform_ShiftLink_AdjSplitCov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
				     cudaGaugeField *gf, cudaGaugeField *gf2,
				     qcTMD_ShiftFlag shfFlag, bool flipShfSgn){

    const char *funcname = "perform_ShiftLink_AdjSplitCov";

    qcTMD_ShiftDir  shfDir  = TMDparseShiftDirection(shfFlag);
    qcTMD_ShiftSgn  shfSgn  = TMDparseShiftSign(shfFlag, flipShfSgn);
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  ){
      if(getVerbosity() >= QUDA_VERBOSE)
	printfQuda("%s: Will perform a Gauge Link AdjSplitCov shift in the %s%s direction\n",
		   funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
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
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Kernel done in %f sec.\n", funcname, t2-t1);

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
    arg_dev = NULL;
  }//-- ShiftLink_AdjSplitCov
  //---------------------------------------------------------------------------


  //-- Class for TMD contractions with shmem, make it tunable
  class quarkContract : public TunableVectorY {

  protected:
    void *arg_dev;
    const cudaColorSpinorField *meta;
    qluaCntr_Type cntrType;
    complex<QUDA_REAL> *corrQuda_dev;
    int Nc;
    int Ns;
    int blockdimZ;
    size_t shmem_per_site;
    
    long long flops() const{
      long long flopCnt = 0;
      if(cntrType == what_tmd_g_F_B)    flopCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (Nc*Nc*Ns*Ns*(1+Nc*Ns) * 8 + Ns*Ns*Ns * 2);
      if(cntrType == what_qbarq_g_F_aB) flopCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (Nc*Nc*Ns*Ns*Ns * 8 + Ns*Ns*Ns * 2);
      if(cntrType == what_qpdf_g_F_B)   flopCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (Nc*Nc*Ns*Ns*Ns * 8 + Ns*Ns*Ns * 2);
      return flopCnt;
    }
    long long bytes() const{
      long long byteCnt = 0;
      if(cntrType == what_tmd_g_F_B)    byteCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (2*Ns*Ns*Nc*Nc + Nc*Nc) * 2*8;
      if(cntrType == what_qbarq_g_F_aB) byteCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (2*Ns*Ns*Nc*Nc) * 2*8;
      if(cntrType == what_qpdf_g_F_B)   byteCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (2*Ns*Ns*Nc*Nc) * 2*8;
      return byteCnt;
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return meta->VolumeCB(); }

    virtual unsigned int sharedBytesPerBlock(const TuneParam &param) const { 
      return param.block.x * param.block.y * shmem_per_site ; 
    }
    virtual int blockStep() const { 
      return /*FIXME*/4*((deviceProp.warpSize + blockdimZ - 1) / blockdimZ) ; 
    }
    virtual int blockMin() const { 
      return /*FIXME*/4*((deviceProp.warpSize + blockdimZ - 1) / blockdimZ) ; 
    }

  public:
    quarkContract(const cudaColorSpinorField *meta_, void *arg_dev_, 
		  complex<QUDA_REAL> *corrQuda_dev_, 
		  qluaCntr_Type cntrType_,
		  int blockdimZ_, size_t shmem_per_site_)
      : TunableVectorY(meta_->SiteSubset()), meta(meta_),
	arg_dev(arg_dev_), corrQuda_dev(corrQuda_dev_), cntrType(cntrType_), 
        Nc(QUDA_Nc), Ns(QUDA_Ns),
        blockdimZ(blockdimZ_), shmem_per_site(shmem_per_site_)
    {
      strcpy(aux, meta_->AuxString());
      strcat(aux, comm_dim_partitioned_string());
    }
    virtual ~quarkContract() { }

    long long getFlops(){return flops();}
    long long getBytes(){return bytes();}

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if((cntrType != what_tmd_g_F_B) && (cntrType != what_qbarq_g_F_aB) && (cntrType != what_qpdf_g_F_B))
        errorQuda("quarkContract::apply(): Support only what_tmd_g_F_B, what_qpdf_g_F_B and what_qbarq_g_F_aB contractions for now!\n");

      if(getVerbosity() >= QUDA_DEBUG_VERBOSE)
	printfQuda("quarkContract::apply(): grid={%ld,%ld,%ld} block={%ld,%ld,%ld} shmem=%ld\n",
		   (long)tp.grid.x, (long)tp.grid.y, (long)tp.grid.z, 
		   (long)tp.block.x, (long)tp.block.y, (long)tp.block.z,
		   (long)tp.shared_bytes);

      if(cntrType == what_tmd_g_F_B)
	tmd_g_U_D_aD_gvec_kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(corrQuda_dev, (qcTMD_Arg*)arg_dev);
      if( (cntrType == what_qbarq_g_F_aB) || (cntrType == what_qpdf_g_F_B) )
      	qbarq_g_P_aP_gvec_shMem_kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(corrQuda_dev, (QluaContractArg*)arg_dev);
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorY::initTuneParam(param);
      param.block.z = blockdimZ;
      param.grid.z  = 1;
    }
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorY::defaultTuneParam(param);
      param.block.z = blockdimZ;
      param.grid.z  = 1;
    }

    TuneKey tuneKey() const { return TuneKey(meta->VolString(), typeid(*this).name(), aux); }
  };
  //---------------------------------------------------------------------------


  //-Top-level function
  void QuarkContract_uLocal(complex<QUDA_REAL> *corrQuda_dev,
			    cudaColorSpinorField **cudaProp1,
			    cudaColorSpinorField **cudaProp2,
			    cudaColorSpinorField **cudaProp3,
			    complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
			    qudaAPI_Param paramAPI){    

    const char *func_name = "QuarkContract_uLocal";
    
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

    if( (paramAPI.mpParam.cntrType == what_tmd_g_F_B) || (paramAPI.mpParam.cntrType == what_qpdf_g_F_B) )
      errorQuda("%s: Contraction type %s not supported!\n", func_name, qc_contractTypeStr[paramAPI.mpParam.cntrType]);

    QluaContractArg arg(cudaProp1, cudaProp2, cudaProp3, paramAPI.mpParam.cntrType, paramAPI.preserveBasis, paramAPI.mpParam.nVec); 
    QluaContractArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg) );
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
    checkCudaError();

    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
    if(paramAPI.mpParam.cntrType == what_baryon_sigma_UUS) qcCopySmatricesToSymbol(S2, S1);
    if(paramAPI.mpParam.cntrType == what_qbarq_g_F_aB) qcCopyGammaToConstMem();

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
      if(!arg.preserveBasis) errorQuda("%s: qbarq_g_P_aP_gvec kernel supports only QUDA_UKQCD_GAMMA_BASIS!\n", func_name);
      quarkContract qbarq_gPaP(cudaProp1[0], (void*)arg_dev, corrQuda_dev, arg.cntrType, 
			       QC_THREADS_PER_SITE, QC_TMD_SHMEM_PER_SITE);    
      if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
	printfQuda("%s: qbarq_gPaP::Flops = %lld\n", func_name, qbarq_gPaP.getFlops());
	printfQuda("%s: qbarq_gPaP::Bytes = %lld\n", func_name, qbarq_gPaP.getBytes());
      }
      qbarq_gPaP.apply(0);
      cudaDeviceSynchronize();
      checkCudaError();
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
    default: errorQuda("%s: Contraction type %s not implemented!\n", func_name, qc_contractTypeStr[arg.cntrType]);
    }//- switch
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel %s finished in %f sec.\n", func_name, qc_contractTypeStr[arg.cntrType], t2-t1);

    cudaFree(arg_dev);
  }//-- QuarkContract_uLocal
  //---------------------------------------------------------------------------


  //-Top-level function
  void QuarkContract_TMD_QPDF(QuarkTMD_state *qcs){

    const char *func_name = "QuarkContract_TMD_QPDF";

    if( (qcs->cntrType != what_tmd_g_F_B) && (qcs->cntrType != what_qpdf_g_F_B) )
      errorQuda("%s: This function supports only TMD and PDF contractions!\n", func_name);

    if(qcs->cntrType == what_tmd_g_F_B){
      qcTMD_Arg arg(qcs->cudaPropFrw_bsh, qcs->cudaPropBkw, qcs->wlinks, qcs->i_wl_vbv, qcs->paramAPI.preserveBasis, qcs->nVec);    
      qcTMD_Arg *arg_dev;
      cudaMalloc((void**)&(arg_dev), sizeof(arg) );
      checkCudaError();
      cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
      checkCudaError();    
      if(arg.nParity != 2)   errorQuda("%s: This function supports only Full Site Subset fields!\n", func_name);
      if(!arg.preserveBasis) errorQuda("%s: TMD kernel supports only QUDA_UKQCD_GAMMA_BASIS!\n", func_name);

      quarkContract contractTMD(qcs->cudaPropBkw[0], (void*)arg_dev, qcs->corrQuda_dev, qcs->cntrType, 
				QC_THREADS_PER_SITE, QC_TMD_SHMEM_PER_SITE);

      if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
	printfQuda("%s: contractTMD::Flops = %lld\n", func_name, contractTMD.getFlops());
	printfQuda("%s: contractTMD::Bytes = %lld\n", func_name, contractTMD.getBytes());
      }

      double t1 = MPI_Wtime();
      contractTMD.apply(0);
      cudaDeviceSynchronize();
      checkCudaError();
      double t2 = MPI_Wtime();
      if(getVerbosity() >= QUDA_VERBOSE) printfQuda("TIMING - %s: Contraction kernel %s finished in %f sec.\n", func_name, qc_contractTypeStr[qcs->cntrType], t2-t1);
      cudaFree(arg_dev);
    }
    else{
      QluaContractArg arg(qcs->cudaPropFrw_bsh, qcs->cudaPropBkw, NULL, qcs->cntrType, qcs->paramAPI.preserveBasis, qcs->nVec); 
      QluaContractArg *arg_dev;
      cudaMalloc((void**)&(arg_dev), sizeof(arg) );
      checkCudaError();
      cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);    
      checkCudaError();
      if(arg.nParity != 2)   errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
      if(!arg.preserveBasis) errorQuda("%s: qPDF kernel supports only QUDA_UKQCD_GAMMA_BASIS!\n", func_name);

      quarkContract qbarq_gPaP(qcs->cudaPropBkw[0], (void*)arg_dev, qcs->corrQuda_dev, qcs->cntrType, 
			       QC_THREADS_PER_SITE, QC_TMD_SHMEM_PER_SITE);

      if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
	printfQuda("%s: qbarq_gPaP::Flops = %lld\n", func_name, qbarq_gPaP.getFlops());
	printfQuda("%s: qbarq_gPaP::Bytes = %lld\n", func_name, qbarq_gPaP.getBytes());
      }

      double t1 = MPI_Wtime();
      qbarq_gPaP.apply(0);
      cudaDeviceSynchronize();
      checkCudaError();
      double t2 = MPI_Wtime();
      if(getVerbosity() >= QUDA_VERBOSE) printfQuda("TIMING - %s: Contraction kernel %s finished in %f sec.\n", func_name, qc_contractTypeStr[qcs->cntrType], t2-t1);
      cudaFree(arg_dev);
    }

  }//-- QuarkContract_TMD_QPDF


} //-namespace quda
