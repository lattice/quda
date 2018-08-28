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

  qcTMD_ShiftDir TMDparseShiftDirection(qcTMD_ShiftString shfFlag){

    printfQuda("TMDparseShiftDirection: Got shift flag shfFlag = %s\n", qcTMD_ShiftStringArray[(int)shfFlag]);

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
    default: errorQuda("TMDparseShiftDirection: Unsupported shift flag, shfFlag = %s.\n", (shfFlag >=0 && shfFlag<20) ? qcTMD_ShiftStringArray[(int)shfFlag] : "None");
    }//-- switch    

    return shfDir;
  }
  //---------------------------------------------------------------------------

  qcTMD_ShiftSgn TMDparseShiftSign(qcTMD_ShiftString shfFlag){

    printfQuda("TMDparseShiftSign: Got shift flag shfFlag = %s\n", qcTMD_ShiftStringArray[(int)shfFlag]);

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
    default: errorQuda("TMDparseShiftSign: Unsupported shift flag, shfFlag = %s.\n", (shfFlag >=0 && shfFlag<20) ? qcTMD_ShiftStringArray[(int)shfFlag] : "None");
    }//-- switch    

    return shfSgn;
  }
  //---------------------------------------------------------------------------


  void perform_NonCovShiftPropOnAxis(QluaContractArg *arg_dev, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, int vCB, int nPty){ // , QluaAuxCntrArg *auxArg_dev

    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4) ){
      printfQuda("perform_NonCovShiftPropOnAxis: Will perform an On-Axis non-covariant propagator shift in the %s%s direction\n", qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    }
    else{
      errorQuda("perform_NonCovShiftPropOnAxis: Got invalid shfDir and/or shfSgn.\n");
    }

    //-- Call kernel that performs non-covariant on axis propagator shift
    dim3 blockDim(THREADS_PER_BLOCK, nPty, 1);
    dim3 gridDim((vCB + blockDim.x -1)/blockDim.x, 1, 1);

    NonCovShiftPropOnAxis_kernel<<<gridDim,blockDim>>>(arg_dev, shfDir, shfSgn);  // , auxArg_dev
    cudaDeviceSynchronize();
    checkCudaError();

  }//-- perform_NonCovShiftPropOnAxis
  //---------------------------------------------------------------------------


  void perform_NonCovShiftVectorOnAxis(QluaCntrTMDArg &TMDarg, int ivec, qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, int vCB, int nPty){

    if( ((int)shfSgn>=0 && (int)shfSgn<2) && ((int)shfDir>=0 && (int)shfDir<4) )
      printfQuda("perform_NonCovShiftVectorOnAxis - ivec = %2d: Will perform an On-Axis non-covariant propagator shift in the %s%s direction\n",
		 ivec, qcTMD_ShiftSgnArray[(int)shfSgn], qcTMD_ShiftDirArray[(int)shfDir]);
    else
      errorQuda("perform_NonCovShiftVectorOnAxis: Got invalid shfDir and/or shfSgn.\n");

    //-- Call kernel that performs non-covariant on-axis vector shift
    dim3 blockDim(THREADS_PER_BLOCK, nPty, 1);
    dim3 gridDim((vCB + blockDim.x -1)/blockDim.x, 1, 1);

    QluaCntrTMDArg *TMDarg_dev;
    cudaMalloc((void**)&(TMDarg_dev), sizeof(QluaCntrTMDArg) );
    checkCudaError();
    cudaMemcpy(TMDarg_dev, &TMDarg, sizeof(QluaCntrTMDArg), cudaMemcpyHostToDevice);    
    checkCudaError();
    
    NonCovShiftVectorOnAxis_kernel<<<gridDim,blockDim>>>(TMDarg_dev, ivec, shfDir, shfSgn);
    cudaDeviceSynchronize();
    checkCudaError();

    cudaFree(TMDarg_dev);
  }//-- perform_NonCovShiftVectorOnAxis
  //---------------------------------------------------------------------------


  void qcExchangeGhostVec(ColorSpinorField **prop, int ivec){
    const int nFace  = 1;
    prop[ivec]->exchangeGhost((QudaParity)(1), nFace, 0); //- first argument is redundant when nParity = 2. nFace MUST be 1 for now.
    cudaDeviceSynchronize();
    checkCudaError();
  }

  void qcExchangeGhostProp(ColorSpinorField **prop){
    for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
      qcExchangeGhostVec(prop,ivec);
    }
  }
  //---------------------------------------------------------------------------
  

  
  //-Top-level function in GPU contractions
  void QuarkContract_GPU(complex<QUDA_REAL> *corrQuda_dev,
			 ColorSpinorField **cudaProp1,
			 ColorSpinorField **cudaProp2,
			 ColorSpinorField **cudaProp3,
			 GaugeField *U,
			 complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
			 qudaAPI_Param paramAPI){    

    char *func_name;
    asprintf(&func_name,"QuarkContract_GPU");
    
    //-- C.K. Here we check in fact that the contractions precision (QC_REAL)
    //-- is the same as the one used throughout.
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("%s: QUDA_REAL and QC_REAL type mismatch!\n", func_name);

    momProjParam mpParam = paramAPI.mpParam;

    /* C.K: Perform exchange ghost and shifts in the TMD case
     * Here we exchange ghosts vector by vector within the 
     * forward propagator (cudaProp1).
     * Then a shift takes place, again vector by vector,
     * hence TMDarg accepts (and returns) elements of the cudaColorSpinorField propagators.
     * The shifted propagator is placed into cudaProp3.
     */
    if(mpParam.cntrType == what_tmd_g_F_B){      
      qcTMD_ShiftString shfFlag = qcShfStr_Z; //-- Hard coded for now
      qcTMD_ShiftDir shfDir = TMDparseShiftDirection(shfFlag);
      qcTMD_ShiftSgn shfSgn = TMDparseShiftSign(shfFlag);     

      double t7 = MPI_Wtime();
      for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
      	qcExchangeGhostVec(cudaProp1, ivec);
	QluaCntrTMDArg TMDarg(cudaProp1[ivec], cudaProp3[ivec], mpParam.cntrType, paramAPI.preserveBasis);
      	perform_NonCovShiftVectorOnAxis(TMDarg, ivec, shfDir, shfSgn, TMDarg.volumeCB, TMDarg.nParity);
      }
      double t8 = MPI_Wtime();
      printfQuda("TIMING - %s: Propagator ghost exchange and shift done in %f sec.\n", func_name, t8-t7);
    }

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
      qtmd_g_P_P_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, arg_dev);
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
