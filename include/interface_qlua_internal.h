/* C. Kallidonis: Internal header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_INT_H__
#define INTERFACE_QLUA_INT_H__

#include <interface_qlua.h>
#include <complex_quda.h>

namespace quda {

  const bool QCredundantComms = false; //- Same as interface_quda.cpp

  //- Required for TMD contractions
  static const char ldir_list[] = "xXyYzZtTqQrRsSuUvVwW";
  static const char ldir_inv [] = "XxYyZzTtQqRrSsUuVvWw";

  struct QluaUtilArg {

    const int nParity;            // number of parities we're working on
    const int volumeCB;           // checkerboarded volume
    const int lL[4];              // 4-d local lattice dimensions
    const int t_axis;             // direction of the time-axis
    size_t rec_size;              // size of elements in correlators
    int nFldDst;                  // Number of fields in correlator to be used in momentum projection (destination)
    int nFldSrc;                  // Number of fields in correlator from contractions (source)
    int sp_stride[4];             // stride in spatial volume
    int sp_locvol;                // spatial volume
    
  QluaUtilArg(ColorSpinorField **propIn, int nFldDst, int nFldSrc, int t_axis, size_t rec_size)
  :   nParity(propIn[0]->SiteSubset()), volumeCB(propIn[0]->VolumeCB()),
      lL{propIn[0]->X(0), propIn[0]->X(1), propIn[0]->X(2), propIn[0]->X(3)},
      t_axis(t_axis), rec_size(rec_size), nFldDst(nFldDst), nFldSrc(nFldSrc),
      sp_stride{0,0,0,0}, sp_locvol(0)
    {

      if(0 <= t_axis){
        int sp_mul = 1;
        for(int i = 0; i < 4; i++){
          if(i == t_axis) sp_stride[i] = 0;
          else{
            sp_stride[i] = sp_mul;
            sp_mul *= lL[i];
          }
          sp_locvol = sp_mul;
        }//-for
      }//-if

    }//-- constructor
  };//-- Structure definition



  //- C.K. Structure holding the current state,
  //- required for TMD contractions
  struct QuarkTMD_state {

    //- contraction type
    qluaCntr_Type cntrType;
    
    //- whether to return position space correlator (true) or not
    int push_res;

    //- current link paths
    char v_lpath[1024];
    char b_lpath[1024];

    //- Phase matrix on device
    complex<QUDA_REAL> *phaseMatrix_dev  = NULL;

    //- Momentum projected data buffer on device, CK-TODO: might not need that
    complex<QUDA_REAL> *momproj_data;

    //- host forward propagator, constant throughout, will be used for resetting cudaPropFrw_bsh
    QUDA_REAL *hostPropFrw;

    //- host forward propagator, will be used to reset device forward propagator
    cpuColorSpinorField *cpuPropFrw[QUDA_PROP_NVEC];

    //- device forward (shifted) propagator, backward propagator
    //- cudaPropAux: used for shifts, will be getting swapped with cudaPropFrw_bsh
    cudaColorSpinorField *cudaPropFrw_bsh[QUDA_PROP_NVEC];
    cudaColorSpinorField *cudaPropBkw[QUDA_PROP_NVEC];
    cudaColorSpinorField *cudaPropAux;

    /* device gauge fields
     * cuda_gf: Original gauge field
     * gf_u:    Original gauge field, extended version
     * bsh_u:   Extended Gauge field shifted in the b-direction
     * aux_u:   Extended Gauge field used for non-covariant shifts of gauge fields, will be getting swapped with bsh_u
     * wlinks:  Extended Gauge field used as 'ColorMatrices' to store shifted links in the b and vbv directions.
     * The indices i_wl_b, i_wl_vbv, i_wl_tmp control which 'Lorentz' index of wlinks will be used for the shifts
     */
    cudaGaugeField *cuda_gf;
    cudaGaugeField *gf_u;
    cudaGaugeField *bsh_u;
    cudaGaugeField *aux_u;    /* for shifts of bsh_u */
    cudaGaugeField *wlinks;   /* store w_b, w_vbv */
    int i_wl_b, i_wl_vbv, i_wl_tmp;

    //- Structure holding the parameters of the contractions / momentum projection
    qudaAPI_Param paramAPI;
  };



  void qcResetFrwVec(cudaColorSpinorField **cudaVec, cpuColorSpinorField **cpuVec);
  void qcResetFrwProp(cudaColorSpinorField **cudaVec, cpuColorSpinorField **cpuVec);

  void qcSetGaugeToUnity(cudaGaugeField *U, int mu);
  
  __device__ int d_crdChkVal = 0;
  int QluaSiteOrderCheck(QluaUtilArg utilArg);

  
  void convertSiteOrder_QudaQDP_to_momProj(void *corrInp_dev, const void *corrQuda_dev, QluaUtilArg utilArg);

  
  void cudaContractQQ(
  		      ColorSpinorField **propOut,
  		      ColorSpinorField **propIn1,
  		      ColorSpinorField **propIn2,
  		      int parity, int Nc, int Ns,
  		      cntrQQParam cParam);

  
  void createPhaseMatrix_GPU(complex<QUDA_REAL> *phaseMatrix,
			     const int *momMatrix,
                             momProjParam param);

  void QuarkContractStd_GPU(complex<QUDA_REAL> *corrQuda_dev,
                            ColorSpinorField **cudaProp1,
                            ColorSpinorField **cudaProp2,
                            ColorSpinorField **cudaProp3,
                            complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
                            qudaAPI_Param paramAPI);

  void QuarkContractTMD_GPU(complex<QUDA_REAL> *corrQuda_dev,
                            ColorSpinorField **cudaProp1,
                            ColorSpinorField **cudaProp2,
                            ColorSpinorField **cudaProp3,
                            cudaGaugeField *U, cudaGaugeField *extU,
                            cudaGaugeField *extU1, cudaGaugeField *extU2, cudaGaugeField *extU3,
                            qudaAPI_Param paramAPI);
  
} //- namespace quda
  
#endif/*INTERFACE_QLUA_INT_H__*/
