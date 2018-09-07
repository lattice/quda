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

  
  int QluaSiteOrderCheck(QluaUtilArg utilArg){
    int crdChkVal;

    QluaUtilArg *utilArg_dev;
    cudaMalloc((void**)&(utilArg_dev), sizeof(QluaUtilArg));
    checkCudaErrorNoSync();
    cudaMemcpy(utilArg_dev, &utilArg,  sizeof(QluaUtilArg), cudaMemcpyHostToDevice);
    
    dim3 blockDim(THREADS_PER_BLOCK, utilArg.nParity, 1);
    dim3 gridDim((utilArg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
    
    QluaSiteOrderCheck_kernel<<<gridDim,blockDim>>>(utilArg_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    cudaMemcpyFromSymbol(&crdChkVal, d_crdChkVal, sizeof(crdChkVal), 0, cudaMemcpyDeviceToHost);
    checkCudaErrorNoSync();
    
    cudaFree(utilArg_dev);
    
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
  }
  //---------------------------------------------------------------------------

  qcTMD_ShiftFlag TMDparseShiftFlag(char flagStr){
    
    qcTMD_ShiftFlag shfFlag = qcShfStr_None;
    for(int iopt=0;iopt<nShiftFlag;iopt++){
      if( strcmp(&flagStr,qcTMD_ShiftFlagArray[iopt])==0 ){
	shfFlag = (qcTMD_ShiftFlag)iopt;
	break;
      }
    }
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

    printfQuda("TMDparseShiftDirection: Got shfFlag = %s\n", qcTMD_ShiftFlagArray[(int)shfFlag]);

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
    default: errorQuda("TMDparseShiftDirection: Unsupported shift flag, shfFlag = %s.\n", (shfFlag >=0 && shfFlag<nShiftFlag) ? qcTMD_ShiftFlagArray[(int)shfFlag] : "None");
    }//-- switch    

    return shfDir;
  }
  //---------------------------------------------------------------------------

  qcTMD_ShiftSgn TMDparseShiftSign(qcTMD_ShiftFlag shfFlag, bool flipShfSgn=false){

    printfQuda("TMDparseShiftSign: Got shfFlag = %s\n", qcTMD_ShiftFlagArray[(int)shfFlag]);

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
    default: errorQuda("TMDparseShiftSign: Unsupported shift flag, shfFlag = %s.\n", (shfFlag >=0 && shfFlag<nShiftFlag) ? qcTMD_ShiftFlagArray[(int)shfFlag] : "None");
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



  __global__ void qcSetGaugeToUnity_kernel(Arg_SetUnityLink *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    Link unitU(arg->unityU); //- That's a 3x3 unity matrix

    arg->U(arg->mu, x_cb, pty) = unitU;
  }

  void qcSetGaugeToUnity(cudaGaugeField *U, int mu){

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
    
    cudaFree(arg_dev);
  }
  //---------------------------------------------------------------------------


  void qcResetFrwVec(cudaColorSpinorField cudaVec, cpuColorSpinorField cpuVec){
    cudaVec = cpuVec;
  }  
  void qcResetFrwProp(cudaColorSpinorField **cudaProp, cpuColorSpinorField **cpuProp){
    for(int i=0;i<QUDA_PROP_NVEC;i++)
      qcResetFrwVec(&cudaProp[i], &cpuProp[i]);
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
  
  template <typename G>
  void qcSwapCudaGauge(G *x1, G *x2){
    G *xtmp = *x1;
    *x1 = *x2;
    *x2 = xtmp;
  }

  template <typename F>
  void qcSwapCudaVec(F *x1, F *x2){
    F *xtmp = x1;
    *x1 = *x2;
    *x2 = *xtmp;
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

    ShiftCudaVec_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    
    cudaFree(arg_dev);
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

    ShiftCudaVec_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    cudaFree(arg_dev);
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

    ShiftGauge_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
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

    ShiftLink_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
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

    ShiftLink_AdjSplitCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeExtendedGhost(dst->R(), QCredundantComms);

    cudaFree(arg_dev);
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
    case what_tmd_g_F_B: {
      errorQuda("%s: No support for TMD contractions! Did you intent to call \'perform_TMD_Contract\' instead?\n", func_name);
    } break;
    case what_qpdf_g_F_B:
    default: errorQuda("%s: Contraction type \'%s\' not supported!\n", func_name, qc_contractTypeStr[arg.cntrType]);
    }//- switch
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[arg.cntrType], t2-t1);

    cudaFree(arg_dev);
  }//-- perform_Std_Contract
  //---------------------------------------------------------------------------


  void perform_TMD_Contract(complex<QUDA_REAL> *corrQuda_dev, qcTMD_State argState){

    const char *func_name = "perform_TMD_Contract";

    if(argState.cntrType != what_tmd_g_F_B)
      errorQuda("%s: Supports only TMD contractions! Did you intent to call \'perform_Std_Contract\' instead?\n", func_name);
    
    qcTMD_State *argState_dev;
    cudaMalloc((void**)&(argState_dev), sizeof(qcTMD_State) );
    checkCudaError();
    cudaMemcpy(argState_dev, &argState, sizeof(qcTMD_State), cudaMemcpyHostToDevice);    
    checkCudaError();

    dim3 blockDim(THREADS_PER_BLOCK, argState.nParity, 1);
    dim3 gridDim((argState.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t1 = MPI_Wtime();
    tmd_g_U_P_P_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, argState_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[argState.cntrType], t2-t1);

    cudaFree(argState_dev);
  }//-- perform_TMD_Contract


  //-Top-level function in GPU contractions
  void QuarkContractStd_GPU(complex<QUDA_REAL> *corrQuda_dev,
			    ColorSpinorField **cudaProp1,
			    ColorSpinorField **cudaProp2,
			    ColorSpinorField **cudaProp3,
			    complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
			    qudaAPI_Param paramAPI){    

    const char *func_name = "QuarkContractStd_GPU";
    
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

    momProjParam mpParam = paramAPI.mpParam;

    QluaContractArg arg(cudaProp1, cudaProp2, cudaProp3, mpParam.cntrType, paramAPI.preserveBasis); 
    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
    if(mpParam.cntrType == what_baryon_sigma_UUS) copySmatricesToSymbol(S2, S1);

    perform_Std_Contract(corrQuda_dev, arg);

  }//-- QuarkContractStd_GPU


} //-namespace quda

  


//-- LEGACY CODE
//      ColorSpinorField *ptmp = cudaProp1[ivec]; cudaProp1[ivec] = cudaProp3[ivec]; cudaProp3[ivec] = ptmp;


  // void qcSwapCudaVec(ColorSpinorField **x1, ColorSpinorField **x2){
  //   ColorSpinorField *xtmp = *x1;
  //   *x1 = *x2;
  //   *x2 = xtmp;
  // }
  // void qcSwapCudaProp(ColorSpinorField **x1, ColorSpinorField **x2){
  //   for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++)
  //     qcSwapCudaVec(&x1[ivec], &x2[ivec]);
  // }

  //-Top-level function in GPU contractions
  // void QuarkContractTMD_GPU(complex<QUDA_REAL> *corrQuda_dev,
  // 			    ColorSpinorField **cudaProp1,
  // 			    ColorSpinorField **cudaProp2,
  // 			    ColorSpinorField **cudaProp3,
  // 			    cudaGaugeField *U, cudaGaugeField *auxU, // U is NOT an extended GaugeField!!!
  // 			    cudaGaugeField *auxU1, cudaGaugeField *auxU2, cudaGaugeField *auxU3,
  // 			    qudaAPI_Param paramAPI){

  //   const char *func_name = "QuarkContractTMD_GPU";
    
  //   if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

  //   qcTMD_ShiftFlag shfFlag = TMDparseShiftFlag(paramAPI.shfFlag);
  //   qcTMD_ShiftType propShfType = TMDparseShiftType(paramAPI.shfType);
  //   qcTMD_ShiftDir  propShfDir  = TMDparseShiftDirection(shfFlag);
  //   qcTMD_ShiftSgn  propShfSgn  = TMDparseShiftSign(shfFlag);     

  //   double t1 = MPI_Wtime();
  //   for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
  //     if(propShfType == qcCovShift)     perform_ShiftCudaVec_Cov(cudaProp3[ivec], cudaProp1[ivec], auxU, propShfDir, propShfSgn);
  //     if(propShfType == qcNonCovShift)	perform_ShiftCudaVec_nonCov(cudaProp3[ivec], cudaProp1[ivec], propShfDir, propShfSgn);
  //     qcSwapCudaVec(&(cudaProp1[ivec]), &(cudaProp3[ivec]));
  //   }
  //   double t2 = MPI_Wtime();
  //   printfQuda("TIMING - %s: Propagator ghost exchange and shift done in %f sec.\n", func_name, t2-t1);

  //   qcTMD_ShiftDir gaugeShfDir   = TMDparseShiftDirection(shfFlag);
  //   qcTMD_ShiftSgn gaugeShfSgn   = TMDparseShiftSign(shfFlag);
  //   qcTMD_ShiftType gaugeShfType = TMDparseShiftType(paramAPI.shfType);

  //   int i_src = 0;
  //   int i_dst = 0;
  //   double t3 = MPI_Wtime();
  //   if(gaugeShfType == qcNonCovShift){
  //     i_dst = 2;
  //     perform_ShiftGauge_nonCov(auxU1, auxU, gaugeShfDir, gaugeShfSgn);
  //     qcSwapCudaGauge(&auxU, &auxU1);
  //   }
  //   else if(gaugeShfType == qcCovShift){
  //     i_src = 0;
  //     i_dst = 2;
      
  //     perform_ShiftLink_Cov(auxU1, i_dst, auxU2, i_src,
  //   			    auxU, gaugeShfDir, gaugeShfSgn);
  //     qcSwapCudaGauge(&auxU, &auxU1);
  //   }
  //   else if(gaugeShfType == qcAdjSplitCovShift){
  //     i_src = 0;
  //     i_dst = 2;

  //     perform_ShiftLink_AdjSplitCov(auxU1, i_dst, auxU2, i_src,
  // 				    auxU, auxU3, gaugeShfDir, gaugeShfSgn);
  //     qcSwapCudaGauge(&auxU, &auxU1);
  //   }
  //   double t4 = MPI_Wtime();
  //   printfQuda("TIMING - %s: Gauge field shift done in %f sec.\n", func_name, t4-t3);

  //   int i_mu = i_dst;

  //   qcTMD_State argState(cudaProp1, cudaProp2, auxU, i_mu, paramAPI.mpParam.cntrType, paramAPI.preserveBasis);
 
  //   perform_TMD_Contract(corrQuda_dev, argState);

  // }//-- QuarkContractTMD_GPU


