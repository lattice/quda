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
#include <interface_qlua_internal.h>
#include <qlua_contract.h>
#include <qlua_contract_kernels.cuh>

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

  
  //-Top-level function in GPU contractions
  void QuarkContract_GPU(complex<QC_REAL> *corrQuda_dev,
			 ColorSpinorField **cudaProp1,
			 ColorSpinorField **cudaProp2,
			 ColorSpinorField **cudaProp3,
			 GaugeField *U,
			 complex<QC_REAL> *S2, complex<QC_REAL> *S1,
			 momProjParam mpParam){    

    char *func_name;
    asprintf(&func_name,"QuarkContract_GPU");
    
    if(typeid(QC_REAL) != typeid(QUDA_REAL)) errorQuda("QUDA_REAL and QC_REAL type mismatch!\n");

    printfQuda("######## %s: Running with new implementation!!!\n",func_name);
   
    //-- Define the arguments structure
    QluaContractArg arg(cudaProp1, cudaProp2, cudaProp3, mpParam.cntrType);
    if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset spinors!\n", func_name);
    QluaContractArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
    checkCudaErrorNoSync();
    cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);    
    
    LONG_T locvol = mpParam.locvol;
    copylocvolToSymbol(locvol); //- Copy the local volume to constant memory


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
    case what_qpdf_g_F_B:
    case what_tmd_g_F_B:
    default: errorQuda("%s: Contraction type \'%s\' not supported!\n", func_name, qc_contractTypeStr[mpParam.cntrType]);
    }//-- switch
    cudaDeviceSynchronize();
    checkCudaError();
    double t6 = MPI_Wtime();
    printfQuda("TIMING - %s: Contraction kernel \'%s\' finished in %f sec.\n", func_name, qc_contractTypeStr[mpParam.cntrType], t6-t5);
    
    //-- Clean-up
    cudaFree(arg_dev);
    free(func_name);
    
  }//-- function
  
} //-namespace quda
