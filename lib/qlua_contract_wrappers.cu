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
  
  // struct QluaContractArgLEGACY {

  //   typedef typename colorspinor_mapper<QC_REAL,QC_Ns,QC_Nc>::type Propagator;
    
  //   Propagator prop[QUDA_PROP_NVEC];
    
  //   const qluaCntr_Type cntrType;     // contraction type 
  //   const int nParity;                // number of parities we're working on
  //   const int nFace;                  // hard code to 1 for now
  //   const int dim[5];                 // full lattice dimensions
  //   const int commDim[4];             // whether a given dimension is partitioned or not
  //   const int lL[4];      	      // 4-d local lattice dimensions
  //   const int volumeCB;               // checkerboarded volume
  //   const int volume;                 // full-site local volume
    
  //   QluaContractArg(ColorSpinorField **propIn, qluaCntr_Type cntrType)
  // :   cntrType(cntrType), nParity(propIn[0]->SiteSubset()), nFace(1),
  //     dim{ (3-nParity) * propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3), 1 },
  //     commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
  //     lL{propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3)},
  //     volumeCB(propIn[0]->VolumeCB()),volume(propIn[0]->Volume())
  //   {
  //     for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++) prop[ivec].init(*propIn[ivec]);
  //   }

  // };//-- Structure definition
  // //---------------------------------------------------------------------------

  // struct QluaContractArgULEGACY {

  //   typedef typename colorspinor_mapper<QC_REAL,QC_Ns,QC_Nc>::type Propagator;
  //   typedef typename gauge_mapper<QC_REAL,QUDA_RECONSTRUCT_NO>::type Gauge;
    
  //   Propagator prop[QUDA_PROP_NVEC];  // Input propagator
  //   Gauge U;                          // Gauge Field

  //   const qluaCntr_Type cntrType;     // contraction type 
  //   const int nParity;                // number of parities we're working on
  //   const int nFace;                  // hard code to 1 for now
  //   const int dim[5];                 // full lattice dimensions
  //   const int commDim[4];             // whether a given dimension is partitioned or not
  //   const int lL[4];      	      // 4-d local lattice dimensions
  //   const int volumeCB;               // checkerboarded volume
  //   const int volume;                 // full-site local volume
    
  // QluaContractArgU(ColorSpinorField **propIn, GaugeField *U, qluaCntr_Type cntrType)
  // :   U(*U), cntrType(cntrType), nParity(propIn[0]->SiteSubset()), nFace(1),
  //     dim{ (3-nParity) * propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3), 1 },
  //     commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
  //     lL{propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3)},
  //     volumeCB(propIn[0]->VolumeCB()),volume(propIn[0]->Volume())
  //   {
  //     for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++) prop[ivec].init(*propIn[ivec]);
  //   }

  // };//-- Structure definition
  //---------------------------------------------------------------------------

  
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

  
  // __global__ void prepareDeviceProp_kernel(complex<QC_REAL> *devProp, QluaContractArg *arg){

  //   int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
  //   int pty  = blockIdx.y*blockDim.y + threadIdx.y;

  //   if (x_cb >= arg->volumeCB) return;
  //   if (pty >= arg->nParity) return;

  //   const int Ns = QC_Ns;
  //   const int Nc = QC_Nc;

  //   typedef ColorSpinor<QC_REAL,Nc,Ns> Vector;
  //   Vector vec[QUDA_PROP_NVEC];

  //   for(int i=0;i<QUDA_PROP_NVEC;i++){
  //     vec[i] = arg->prop[i](x_cb, pty);
  //   }
  //   rotatePropBasis(vec,QLUA_quda2qdp); //-- Rotate basis back to the QDP conventions

  //   int crd[5];
  //   getCoords(crd, x_cb, arg->lL, pty);  //-- Get local coordinates crd[] at given x_cb and pty
  //   crd[4] = 0;  

  //   int i_QudaQdp = x_cb + pty * arg->volumeCB;
  //   int lV = 2*arg->volumeCB;

  //   for(int jc = 0; jc < Nc; jc++){
  //     for(int js = 0; js < Ns; js++){
  // 	int vIdx = js + Ns*jc;     //-- vector index (which vector within propagator)
  // 	for(int ic = 0; ic < Nc; ic++){
  // 	  for(int is = 0; is < Ns; is++){
  // 	    int dIdx = ic + Nc*is; //-- spin-color index within each vector

  // 	    int pIdx = i_QudaQdp + lV*QC_QUDA_LIDX_P(ic,is,jc,js);	    
	    
  // 	    devProp[pIdx] = vec[vIdx].data[dIdx];
  // 	  }}}
  //   }
    
  // }//-- function

  
  // void prepareDeviceProp(complex<QC_REAL> *devProp, ColorSpinorField **propIn, qluaCntr_Type cntrType){
    
  //   QluaContractArg arg(propIn, cntrType);
      
  //   if(arg.nParity != 2) errorQuda("run_prepareDeviceProp: This function supports only Full Site Subset spinors!\n");
    
  //   QluaContractArg *arg_dev;
  //   cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
  //   checkCudaErrorNoSync();
  //   cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);

  //   dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  //   dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

  //   prepareDeviceProp_kernel<<<gridDim,blockDim>>>(devProp, arg_dev);
  //   cudaDeviceSynchronize();
  //   checkCudaError();

  //   cudaFree(arg_dev);
  // }

  
  // void prepareDeviceProp(complex<QC_REAL> *devProp, ColorSpinorField **propIn, GaugeField *U, qluaCntr_Type cntrType){
    
  //   QluaContractArgU arg(propIn, U, cntrType);
    
  //   if(arg.nParity != 2) errorQuda("run_prepareDeviceProp: This function supports only Full Site Subset spinors!\n");
    
  //   QluaContractArg *arg_dev;
  //   cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
  //   checkCudaErrorNoSync();
  //   cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);

  //   dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  //   dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

  //   prepareDeviceProp_kernel<<<gridDim,blockDim>>>(devProp, arg_dev);
  //   cudaDeviceSynchronize();
  //   checkCudaError();

  //   cudaFree(arg_dev);
  // }
  // //------------------------------------------------------------------------------------------

  
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
