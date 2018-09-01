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

  qcTMD_ShiftFlag TMDparseShiftFlag(char *flagStr){
    
    qcTMD_ShiftFlag shfFlag = qcShfStr_None;
    for(int iopt=0;iopt<nShiftFlag;iopt++){
      if( strcmp(flagStr,qcTMD_ShiftFlagArray[iopt])==0 ){
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

  qcTMD_ShiftSgn TMDparseShiftSign(qcTMD_ShiftFlag shfFlag){

    printfQuda("TMDparseShiftSign: Got shfFlag = %s\n", qcTMD_ShiftFlagArray[(int)shfFlag]);

    qcTMD_ShiftSgn shfSgn = qcShfSgnNone;
    switch(shfFlag){
    case qcShfStr_X:
    case qcShfStr_Y:
    case qcShfStr_Z:
    case qcShfStr_T: {
      shfSgn = qcShfSgnPlus;
    } break;
    case qcShfStr_x:
    case qcShfStr_y:
    case qcShfStr_z:
    case qcShfStr_t: {
      shfSgn = qcShfSgnMinus;
    } break;
    default: errorQuda("TMDparseShiftSign: Unsupported shift flag, shfFlag = %s.\n", (shfFlag >=0 && shfFlag<nShiftFlag) ? qcTMD_ShiftFlagArray[(int)shfFlag] : "None");
    }//-- switch    

    return shfSgn;
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
  
  void qcSwapCudaVec(ColorSpinorField *x1, ColorSpinorField *x2){
    ColorSpinorField *xtmp = x1;
    x1 = x2;
    x2 = xtmp;
  }
  void qcSwapCudaProp(ColorSpinorField **x1, ColorSpinorField **x2){
    for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++)
      qcSwapCudaVec(x1[ivec], x2[ivec]);
  }
  //---------------------------------------------------------------------------


  void perform_ShiftCudaVec_nonCov(ColorSpinorField *dst, ColorSpinorField *src, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const char *funcname = "perform_ShiftCS_noncov";
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis shift in the %s%s direction\n",
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

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    ShiftCudaVec_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();
    
    cudaFree(arg_dev);
  }//-- perform_ShiftCudaVec_nonCov


  void perform_ShiftCudaVec_Cov(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gf,
				qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const char *funcname = "perform_ShiftCudaVec_Cov";
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis shift in the %s%s direction\n",
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

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    ShiftCudaVec_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    cudaFree(arg_dev);
  }//-- perform_ShiftCudaVec_Cov


  void perform_ShiftGauge_nonCov(cudaGaugeField *dst, cudaGaugeField *src,
				 qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const char *funcname = "perform_ShiftGF_noncov";
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftGauge_nonCov arg(dst, src);
    Arg_ShiftGauge_nonCov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    ShiftGauge_nonCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeGhost();

    cudaFree(arg_dev);
  }//-- perform_ShiftGauge_nonCov


  void perform_ShiftLink_Cov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
			     cudaGaugeField *gf, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const char *funcname = "perform_ShiftLink_Cov";
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftLink_Cov arg(dst, i_dst, src, i_src, gf);
    Arg_ShiftLink_Cov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    ShiftLink_Cov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeGhost();

    cudaFree(arg_dev);
  }//-- perform_ShiftLink_Cov


  void perform_ShiftLink_AdjSplitCov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
				     cudaGaugeField *gf, cudaGaugeField *gf2,
				     qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn){

    const char *funcname = "perform_ShiftLink_AdjSplitCov";
    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4)  )
      printfQuda("%s: Will perform an On-Axis shift in the %s%s direction\n",
		 funcname, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("%s: Got invalid shfDir and/or shfSgn.\n", funcname);
    
    Arg_ShiftLink_AdjSplitCov arg(dst, i_dst, src, i_src, gf, gf2);
    Arg_ShiftLink_AdjSplitCov *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(arg));
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
    checkCudaError();

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    ShiftLink_AdjSplitCov_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    dst->exchangeGhost();

    cudaFree(arg_dev);
  }//-- ShiftLink_AdjSplitCov
  //---------------------------------------------------------------------------

  
  //-Top-level function in GPU contractions
  void QuarkContract_GPU(complex<QUDA_REAL> *corrQuda_dev,
			 ColorSpinorField **cudaProp1,
			 ColorSpinorField **cudaProp2,
			 ColorSpinorField **cudaProp3,
			 GaugeField *U, GaugeField *auxU,
			 complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
			 qudaAPI_Param paramAPI){    

    char *func_name;
    asprintf(&func_name,"QuarkContract_GPU");
    
    //-- C.K. Here we check in fact that the contractions precision (QC_REAL)
    //-- is the same as the one used throughout.
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

    momProjParam mpParam = paramAPI.mpParam;

    //-- Define the arguments structure
    QluaContractArg arg(cudaProp1, cudaProp2, cudaProp3, U, mpParam.cntrType, paramAPI.preserveBasis); 
    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
    QluaContractArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
    checkCudaError();
    cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);    
    checkCudaError();


    //-- Call kernels that perform contractions
    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    double t5 = MPI_Wtime();
    switch(mpParam.cntrType){
    case what_baryon_sigma_UUS: {
      copySmatricesToSymbol(S2, S1);
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
      qcTMD_ShiftFlag shfFlag = TMDparseShiftFlag(paramAPI.shfFlag);
      qcTMD_ShiftType propShfType = TMDparseShiftType(paramAPI.shfType);
      qcTMD_ShiftDir  propShfDir  = TMDparseShiftDirection(shfFlag);
      qcTMD_ShiftSgn  propShfSgn  = TMDparseShiftSign(shfFlag);     


      double t7 = MPI_Wtime();
      for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
        perform_ShiftCudaVec_nonCov(cudaProp3[ivec], cudaProp1[ivec], propShfDir, propShfSgn);
	qcSwapCudaVec(cudaProp1[ivec], cudaProp3[ivec]);
      }
      double t8 = MPI_Wtime();
      printfQuda("TIMING - %s: Propagator ghost exchange and shift done in %f sec.\n", func_name, t8-t7);

      //      qtmd_g_P_P_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, TMDcs_dev);
    } break;
    case what_qpdf_g_F_B:
    default: errorQuda("%s: Contraction type \'%s\' not supported!\n", func_name, qc_contractTypeStr[mpParam.cntrType]);
    }//-- switch
    cudaDeviceSynchronize();
    checkCudaError();
    double t6 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[mpParam.cntrType], t6-t5);
    
    //-- Clean-up
    free(func_name);
    cudaFree(arg_dev);

  }//-- function
  
} //-namespace quda




//-- LEGACY CODE

//      ColorSpinorField *ptmp = cudaProp1[ivec]; cudaProp1[ivec] = cudaProp3[ivec]; cudaProp3[ivec] = ptmp;
//      qcSwapVector(&TMDcs.fwdProp[ivec], &TMDcs.auxProp[ivec]);


  // void perform_ShiftGauge(TMDcontractState *TMDcs_dev,TMDcontractState *TMDcs,
  // 			  qcTMD_ShiftDir muDst, qcTMD_ShiftDir muSrc,
  // 			  qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){

  //   if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4) && ((int)shfType==0 || (int)shfType==1)  )
  //     printfQuda("perform_ShiftGauge: Will perform an On-Axis %s shift in the %s%s direction\n",
  // 		 qcTMD_ShiftTypeArray[(int)shfType], qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
  //   else
  //     errorQuda("perform_ShiftGauge: Got invalid shfDir and/or shfSgn and/or shfType.\n");

  //   cudaMemcpy(TMDcs_dev, TMDcs, sizeof(TMDcontractState), cudaMemcpyHostToDevice);    
  //   checkCudaError();

  //   dim3 blockDim(THREADS_PER_BLOCK, TMDcs->nParity, 1);
  //   dim3 gridDim((TMDcs->volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

  //   //-- C.K. For the Non-covariant case, we shift all dimensions, with the source and destination
  //   //-- dimensions the same
  //   if(shfType == qcNonCovShift){
  //     for(int mu=0;mu<4;mu++){
  // 	ShiftGauge_kernel<<<gridDim,blockDim>>>(TMDcs_dev, mu, mu, shfDir, shfSgn, shfType);
  // 	cudaDeviceSynchronize();
  // 	checkCudaError();
  //     }
  //   }


  //   cudaMemcpy(TMDcs, TMDcs_dev, sizeof(TMDcontractState), cudaMemcpyDeviceToHost);    
  //   checkCudaError();

  // }//-- perform_ShiftGauge
  // //---------------------------------------------------------------------------



  // void perform_ShiftVectorOnAxis(TMDcontractState *TMDcs_dev, TMDcontractState *TMDcs, int ivec, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType){

  //   if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4) && ((int)shfType==0 || (int)shfType==1)  )
  //     printfQuda("perform_ShiftVectorOnAxis - ivec = %2d: Will perform an On-Axis %s shift in the %s%s direction\n",
  // 		 ivec, qcTMD_ShiftTypeArray[(int)shfType], qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
  //   else
  //     errorQuda("perform_ShiftVectorOnAxis: Got invalid shfDir and/or shfSgn and/or shfType.\n");

  //   cudaMemcpy(TMDcs_dev, TMDcs, sizeof(TMDcontractState), cudaMemcpyHostToDevice);    
  //   checkCudaError();
    
  //   dim3 blockDim(THREADS_PER_BLOCK, TMDcs->nParity, 1);
  //   dim3 gridDim((TMDcs->volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

  //   ShiftVectorOnAxis_kernel<<<gridDim,blockDim>>>(TMDcs_dev, ivec, shfDir, shfSgn, shfType);
  //   cudaDeviceSynchronize();
  //   checkCudaError();

  //   cudaMemcpy(TMDcs, TMDcs_dev, sizeof(TMDcontractState), cudaMemcpyDeviceToHost);    
  //   checkCudaError();
  // }//-- perform_ShiftVectorOnAxis
