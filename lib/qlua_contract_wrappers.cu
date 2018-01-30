#include <transfer.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <mpi.h>
#include <interface_qlua_internal.h>
#include <qlua_contract_kernels.cuh>

namespace quda {
  
  struct QluaContractArg {

    typedef typename colorspinor_mapper<QC_REAL,QC_Ns,QC_Nc>::type Propagator;

    Propagator prop[QUDA_PROP_NVEC];  // Input propagator 1

    const int parity;                 // only use this for single parity fields
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int lL[4];      	      // 4-d local lattice dimensions
    const int volumeCB;               // checkerboarded volume
    const int volume;                 // full-site local volume

  QluaContractArg(ColorSpinorField **propIn, int parity)
  :   parity(parity), nParity(propIn[0]->SiteSubset()), nFace(1),
      dim{ (3-nParity) * propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3)},
      volumeCB(propIn[0]->VolumeCB()),volume(propIn[0]->Volume())
    {
      
      for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
        prop[ivec].init(*propIn[ivec]);
      }      
      
    }//-- constructor    
  };//-- Structure definition
  //---------------------------------------------------------------------------
  
  /**
     When copying ColorSpinorFields to GPU, Quda rotates the fields to another basis using a rotation matrix.
     This function is required in order to rotate the ColorSpinorFields between the Quda and the QDP bases.
     The rotation matrix is ( with a factor sqrt(0.5) ):
              ( 0 -1  0 -1)
          M = ( 1  0  1  0)
              ( 0 -1  0  1)
              ( 1  0 -1  0)

     Before the calculation the ColorSpinorFields must be rotated as F <- M F  (quda2qdp).
     After the calculation the result must be rotated back to the Quda basis R <- M^T R (qdp2quda),
     so that when Quda copies back to the CPU the result is again rotated to the QDP convention.
   */
  __device__ __host__ inline void rotatePropBasis(ColorSpinor<QC_REAL,QC_Nc,QC_Ns> *prop, RotateType rType){

    const int Ns = QC_Ns;
    const int Nc = QC_Nc;

    typedef ColorSpinor<QC_REAL,Nc,Ns> Vector;
    Vector res[QUDA_PROP_NVEC];

    complex<QC_REAL> zro = complex<QC_REAL>{0,0};
    complex<QC_REAL> val = complex<QC_REAL>{sqrt(0.5),0};
    complex<QC_REAL> M[Ns][Ns] = { { zro, -val,  zro, -val},
				   { val,  zro,  val,  zro},
				   { zro, -val,  zro,  val},
				   { val,  zro, -val,  zro} };

    complex<QC_REAL> M_Trans[Ns][Ns];
    for(int i=0;i<Ns;i++){
      for(int j=0;j<Ns;j++){
        M_Trans[i][j] = M[j][i];
      }
    }

    complex<QC_REAL> (*A)[Ns] = NULL;
    if      (rType == QLUA_quda2qdp) A = M;
    else if (rType == QLUA_qdp2quda) A = M_Trans;

    for(int ic = 0; ic < Nc; ic++){
      for(int jc = 0; jc < Nc; jc++){
        for(int is = 0; is < Ns; is++){
          for(int js = 0; js < Ns; js++){
            int iv = js + Ns*jc;
            int id = ic + Nc*is;

            res[iv].data[id] = 0.0;
            for(int a=0;a<Ns;a++){
              int as = ic + Nc*a;

              res[iv].data[id] += A[is][a] * prop[iv].data[as];
            }
          }}}
    }

    for(int v = 0; v<QUDA_PROP_NVEC; v++)
      prop[v] = res[v];

  }
  //---------------------------------------------------------------------------

  
  __global__ void QluaCoordCheck_kernel(QluaUtilArg *utilArg){

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

  
  int QluaCoordCheck(QluaUtilArg utilArg){
    int crdChkVal;

    QluaUtilArg *utilArg_dev;
    cudaMalloc((void**)&(utilArg_dev), sizeof(QluaUtilArg));
    checkCudaErrorNoSync();
    cudaMemcpy(utilArg_dev, &utilArg,  sizeof(QluaUtilArg), cudaMemcpyHostToDevice);
    
    dim3 blockDim(THREADS_PER_BLOCK, utilArg.nParity, 1);
    dim3 gridDim((utilArg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
    
    QluaCoordCheck_kernel<<<gridDim,blockDim>>>(utilArg_dev);
    checkCudaError();
    cudaMemcpyFromSymbol(&crdChkVal, d_crdChkVal, sizeof(crdChkVal), 0, cudaMemcpyDeviceToHost);
    checkCudaErrorNoSync();
    
    cudaFree(utilArg_dev);
    
    return crdChkVal;
  }//-- function
  //---------------------------------------------------------------------------

  
  __global__ void conv_siteOrder_QudaQdp_to_momproj_kernel(void *dst, const void *src, QluaUtilArg *arg){

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
  
  void conv_siteOrder_QudaQdp_to_momproj(void *corrInp_dev, const void *corrQuda_dev, QluaUtilArg utilArg){

    QluaUtilArg *utilArg_dev;
    cudaMalloc((void**)&(utilArg_dev), sizeof(QluaUtilArg));
    checkCudaErrorNoSync();
    cudaMemcpy(utilArg_dev, &utilArg,  sizeof(QluaUtilArg), cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK, utilArg.nParity, 1);
    dim3 gridDim((utilArg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    conv_siteOrder_QudaQdp_to_momproj_kernel<<<gridDim,blockDim>>>(corrInp_dev, corrQuda_dev, utilArg_dev);
    checkCudaError();
    
    cudaFree(utilArg_dev);
  }
  //---------------------------------------------------------------------------

  
  __device__ __host__ inline void RunPropagatorTransform(complex<QC_REAL> *devProp, QluaContractArg *arg, int x_cb, int pty){

    const int Ns = QC_Ns;
    const int Nc = QC_Nc;

    typedef ColorSpinor<QC_REAL,Nc,Ns> Vector;
    Vector vec[QUDA_PROP_NVEC];

    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec[i] = arg->prop[i](x_cb, pty);
    }
    rotatePropBasis(vec,QLUA_quda2qdp); //-- Rotate basis back to the QDP conventions

    int crd[5];
    getCoords(crd, x_cb, arg->lL, pty);  //-- Get local coordinates crd[] at given x_cb and pty
    crd[4] = 0;  

    int i_QudaQdp = x_cb + pty * arg->volumeCB;
    int lV = 2*arg->volumeCB;

    for(int jc = 0; jc < Nc; jc++){
      for(int js = 0; js < Ns; js++){
	int vIdx = js + Ns*jc;     //-- vector index (which vector within propagator)
	for(int ic = 0; ic < Nc; ic++){
	  for(int is = 0; is < Ns; is++){
	    int dIdx = ic + Nc*is; //-- spin-color index within each vector

	    int pIdx = i_QudaQdp + lV*QC_QUDA_LIDX_P(ic,is,jc,js);	    
	    
	    devProp[pIdx] = vec[vIdx].data[dIdx];
	  }}}
    }
    
  }//--function
  //------------------------------------------------------------------------------------------

  
  __global__ void propagatorTransform_kernel(complex<QC_REAL> *devProp, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    RunPropagatorTransform(devProp, arg, x_cb, pty);
  }
  //------------------------------------------------------------------------------------------

  
  void propagatorTransform(complex<QC_REAL> *devProp, ColorSpinorField **propIn, int parity){//, int t_axis){
    
    QluaContractArg arg(propIn, parity);//, t_axis);
    
    if(arg.nParity != 2) errorQuda("run_propagatorTransform: This function supports only Full Site Subset spinors!\n");
    
    QluaContractArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(QluaContractArg) );
    checkCudaErrorNoSync();
    cudaMemcpy(arg_dev, &arg, sizeof(QluaContractArg), cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
    dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

    propagatorTransform_kernel<<<gridDim,blockDim>>>(devProp, arg_dev);
    checkCudaError();

    cudaFree(arg_dev);
  }
  //------------------------------------------------------------------------------------------

  
  //-Top-level function in GPU contractions
  void contractGPU_baryon_sigma_twopt_asymsrc_gvec(complex<QC_REAL> *corrQuda_dev,
						   ColorSpinorField **cudaProp1,
						   ColorSpinorField **cudaProp2,
						   ColorSpinorField **cudaProp3,
						   complex<QC_REAL> *S2, complex<QC_REAL> *S1,
						   momProjParam mpParam){

    char *func_name;
    asprintf(&func_name,"contractGPU_baryon_sigma_twopt_asymsrc_gvec");

    LONG_T locvol = mpParam.locvol;
    
    //-- Transform the propagators
    size_t propSizeCplx = sizeof(complex<QC_REAL>) * locvol * QUDA_Nc*QUDA_Nc * QUDA_Ns*QUDA_Ns;
    printfQuda("%s: propSizeCplx = %lld bytes\n", func_name, (LONG_T)propSizeCplx);
    
    complex<QC_REAL> *prop1_dev = NULL;
    complex<QC_REAL> *prop2_dev = NULL;
    complex<QC_REAL> *prop3_dev = NULL;
    
    cudaMalloc((void**)&prop1_dev, propSizeCplx );
    cudaMalloc((void**)&prop2_dev, propSizeCplx );
    cudaMalloc((void**)&prop3_dev, propSizeCplx );
    checkCudaErrorNoSync();
    cudaMemset(prop1_dev, 0, propSizeCplx);
    cudaMemset(prop2_dev, 0, propSizeCplx);
    cudaMemset(prop3_dev, 0, propSizeCplx);
    
    int parity = 0; //-- not functional for full-site fields, just set it to zero
    propagatorTransform(prop1_dev, cudaProp1, parity);
    propagatorTransform(prop2_dev, cudaProp2, parity);
    propagatorTransform(prop3_dev, cudaProp3, parity);
    printfQuda("%s: Propagators transformed\n", func_name);
    //-------------------------------------------------------------
    
    //-- allocate local volume on device
    LONG_T *locvol_dev;
    cudaMalloc((void**)&locvol_dev, sizeof(LONG_T));
    checkCudaErrorNoSync();
    cudaMemcpy(locvol_dev, &locvol, sizeof(LONG_T), cudaMemcpyHostToDevice);

    //-- allocate S-matrices on device
    size_t SmatSize = sizeof(complex<QC_REAL>)*QUDA_LEN_G;
    complex<QC_REAL> *S2_dev, *S1_dev;
    cudaMalloc((void**)&S2_dev, SmatSize);
    cudaMalloc((void**)&S1_dev, SmatSize);
    checkCudaErrorNoSync();    
    cudaMemcpy(S2_dev, S2, SmatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(S1_dev, S1, SmatSize, cudaMemcpyHostToDevice);

    
    //-- Call the kernel wrapper to perform contractions
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((locvol + blockDim.x - 1)/blockDim.x, 1, 1);

    baryon_sigma_twopt_asymsrc_gvec_kernel<<<gridDim,blockDim>>>(corrQuda_dev, locvol_dev,
    								 prop1_dev, prop2_dev, prop3_dev,
    								 S2_dev, S1_dev);
    checkCudaError();

    
    //-- Clean-up
    cudaFree(prop1_dev);
    cudaFree(prop2_dev);
    cudaFree(prop3_dev);
    cudaFree(S2_dev);
    cudaFree(S1_dev);
    cudaFree(locvol_dev);

  }//-- function
  
} //-namespace quda



    
    /* int i_rlex = crd[0] + arg->lL[0]*( */
    /* 	         crd[1] + arg->lL[1]*( */
    /* 	         crd[2] + arg->lL[2]*( */
    /* 	         crd[3])));  //-- Full lattice site index */
    /* int i_par = (crd[0] + crd[1] + crd[2] + crd[3]) & 1; */

    /* if (i_rlex / 2 != x_cb || pty != i_par) */
    /*   printf("x_cb = %d, pty = %d: site order mismatch\n", x_cb, pty); */



    /* cudaMemcpy(prop1_dev, hprop1, propSizeCplx, cudaMemcpyHostToDevice); */
    /* cudaMemcpy(prop2_dev, hprop2, propSizeCplx, cudaMemcpyHostToDevice); */
    /* cudaMemcpy(prop3_dev, hprop3, propSizeCplx, cudaMemcpyHostToDevice); */


//int pIdx = is + Ns*ic + Ns*Nc*iv + Ns*Nc*lV*js + Ns*Nc*lV*Ns*jc;  //-- QDP propagator index order (spin-inside-color-inside-volume, spin-inside-color for each vector)


//	    printf("(x_cb,pty) = %d , %d , (jc,js,ic,is,iv) = %d , %d , %d , %d , %d , devProp[%d] = %+lf  %+lf * i\n",
//		   x_cb, pty, jc, js, ic, is, iv, pIdx3, devProp[pIdx3].real(), devProp[pIdx3].imag());



    /* printfQuda("propagatorTransform:\n"); */
    /* printfQuda("  arg.parity   = %d\n", arg.parity); */
    /* printfQuda("  arg.nParity  = %d\n", arg.nParity); */
    /* printfQuda("  arg.nFace    = %d\n", arg.nFace); */
    /* printfQuda("  arg.volumeCB = %d\n", arg.volumeCB); */
    /* printfQuda("  arg.volume   = %d\n", arg.volume); */
    /* printfQuda("  arg.dim      = [%d,%d,%d,%d]\n", arg.dim[0], arg.dim[1], arg.dim[2], arg.dim[3]); */
    /* printfQuda("  arg.commDim  = [%d,%d,%d,%d]\n", arg.commDim[0], arg.commDim[1], arg.commDim[2], arg.commDim[3]); */
    /* printfQuda("  arg.lL   = [%d,%d,%d,%d]\n", arg.lL[0], arg.lL[1], arg.lL[2], arg.lL[3]); */



    /* int r1 = crd / arg->lL[0]; */
    /* int r2 = r1  / arg->lL[1]; */
    /* int locx = crd - r1 * arg->lL[0]; */
    /* int locy = r1  - r2 * arg->lL[1]; */
    /* int loct = r2 / arg->lL[2]; */
    /* int locz = r2 - loct * arg->lL[2]; */

    /* int evenSiteBit = (locx + locy + locz + loct) & 1; */
    /* int oddSiteBit  = evenSiteBit ^ 1; */
    
    /* int sBit = (pty == 0) ? evenSiteBit : oddSiteBit; //-- TODO: Need to verify/check this */
