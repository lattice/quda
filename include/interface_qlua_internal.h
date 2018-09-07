/* C. Kallidonis: Internal header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_INT_H__
#define INTERFACE_QLUA_INT_H__

#include <interface_qlua.h>
#include <complex_quda.h>

namespace quda {

  const bool QCredundantComms = false; //- Same as interface_quda.cpp

  //- Length of Halos in extended Gauge Field
  static int qcR[4];

  //- Required for TMD contractions
  static const char ldir_list[] = "xXyYzZtTqQrRsSuUvVwW";
  static const char ldir_inv [] = "XxYyZzTtQqRrSsUuVvWw";


  typedef enum qcTMD_ShiftFlag_s {
    qcShfStr_None = -1,
    qcShfStr_X = 0,  // +x
    qcShfStr_x = 1,  // -x
    qcShfStr_Y = 2,  // +y
    qcShfStr_y = 3,  // -y
    qcShfStr_Z = 4,  // +z
    qcShfStr_z = 5,  // -z
    qcShfStr_T = 6,  // +t
    qcShfStr_t = 7,  // -t
    qcShfStr_Q = 8,  // +x+y
    qcShfStr_q = 9,  // -x-y
    qcShfStr_R = 10, // -x+y
    qcShfStr_r = 11, // +x-y
    qcShfStr_S = 12, // +y+z
    qcShfStr_s = 13, // -y-z
    qcShfStr_U = 14, // -y+z
    qcShfStr_u = 15, // +y-z
    qcShfStr_V = 16, // +x+z
    qcShfStr_v = 17, // -x-z
    qcShfStr_W = 18, // +x-z
    qcShfStr_w = 19  // -x+z
  } qcTMD_ShiftFlag;

  typedef enum qcTMD_ShiftDir_s {
    qcShfDirNone = -1,
    qcShfDir_x = 0,
    qcShfDir_y = 1,
    qcShfDir_z = 2,
    qcShfDir_t = 3
  } qcTMD_ShiftDir;

  typedef enum qcTMD_DimU_s {
    qcShfDimU_None = -1,
    qcDimU_x = 0,
    qcDimU_y = 1,
    qcDimU_z = 2,
    qcDimU_t = 3,
    qcDimAll = 10
  } qcTMD_DimU;

  typedef enum qcTMD_ShiftSgn_s {
    qcShfSgnNone  = -1,
    qcShfSgnMinus =  0,
    qcShfSgnPlus  =  1
  } qcTMD_ShiftSgn;

  typedef enum qcTMD_ShiftType_s {
    qcInvalidShift = -1,
    qcCovShift = 0,
    qcNonCovShift = 1,
    qcAdjSplitCovShift = 2
  } qcTMD_ShiftType;


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

    int qcR[4]; //- Length of extended halos

    //- Structure holding the parameters of the contractions / momentum projection
    qudaAPI_Param paramAPI;
  };

  
  __device__ int d_crdChkVal = 0;
  int QluaSiteOrderCheck(QluaUtilArg utilArg);
  void convertSiteOrder_QudaQDP_to_momProj(void *corrInp_dev, const void *corrQuda_dev, QluaUtilArg utilArg);


  qcTMD_ShiftFlag TMDparseShiftFlag(char flagStr);


  void qcResetFrwVec(cudaColorSpinorField **cudaVec, cpuColorSpinorField **cpuVec);
  void qcResetFrwProp(cudaColorSpinorField **cudaVec, cpuColorSpinorField **cpuVec);
  void qcSetGaugeToUnity(cudaGaugeField *U, int mu);
  void qcCopyExtendedGaugeField(cudaGaugeField *dst, cudaGaugeField *src, const int *R);


  template <typename F>
  void qcSwapCudaVec(F *x1, F *x2);

  template <typename G>
  void qcSwapCudaGauge(G *x1, G *x2);


  void perform_ShiftCudaVec_nonCov(ColorSpinorField *dst, ColorSpinorField *src,
				   qcTMD_ShiftFlag shfFlag);
  void perform_ShiftLink_Cov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
                             cudaGaugeField *gf, qcTMD_ShiftFlag shfFlag);
  void perform_ShiftGauge_nonCov(cudaGaugeField *dst, cudaGaugeField *src,
                                 qcTMD_ShiftFlag shfFlag);

  
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
