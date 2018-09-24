/* C. Kallidonis: Internal header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_INT_H__
#define INTERFACE_QLUA_INT_H__

#include <interface_qlua.h>
#include <complex_quda.h>

namespace quda {

  const bool QCredundantComms = false; //- Same as interface_quda.cpp

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
    bool init;

    QluaUtilArg(cudaColorSpinorField **propIn, int nFldDst, int nFldSrc, int t_axis, size_t rec_size)
      : nParity(propIn[0]->SiteSubset()), volumeCB(propIn[0]->VolumeCB()),
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

    ~QluaUtilArg() {}

  };//-- Structure definition



  //- C.K. Structure holding the current state,
  //- required for TMD contractions
  struct QuarkTMD_state_s {

    int iStep;

    QluaUtilArg *utilArg;

    //- contraction type
    qluaCntr_Type cntrType;
    
    //- whether to return position space correlator (true) or not
    int push_res;

    //- current link paths
    char v_lpath[1024];
    char b_lpath[1024];

    //- Correlator and momentum-projection related buffers
    complex<QUDA_REAL> *phaseMatrix_dev;  //-- Device Phase Matrix buffer
    complex<QUDA_REAL> *corrQuda_dev;     //-- Device Position space correlator
    complex<QUDA_REAL> *corrInp_dev;      //-- Device Input buffer to cuBlas
    complex<QUDA_REAL> *corrOut_dev;      //-- Device output buffer of cuBlas
    complex<QUDA_REAL> *corrOut_proj;     //-- Host Final result (global summed, gathered) of momentum projection
    complex<QUDA_REAL> *corrOut_glob;     //-- Host Globally summed momentum projection buffer
    complex<QUDA_REAL> *corrOut_host;     //-- Host (local) output of cuBlas momentum projection


    //- cpuPropFrw: host forward propagator, will be used to reset device forward propagator
    //- cudaPropFrw_bsh: Device forward (shifted) propagator
    //- cudaPropBkw: Device backward propagator
    //- cudaPropAux: Device vector used for shifts, will be getting swapped with cudaPropFrw_bsh
    cpuColorSpinorField *cpuPropFrw[QUDA_PROP_NVEC];
    cudaColorSpinorField *cudaPropFrw_bsh[QUDA_PROP_NVEC];
    cudaColorSpinorField *cudaPropBkw[QUDA_PROP_NVEC];
    cudaColorSpinorField *cudaPropAux;

    /* device gauge fields
     * gf_u:    Original gauge field, extended version
     * bsh_u:   Extended Gauge field shifted in the b-direction
     * aux_u:   Extended Gauge field used for non-covariant shifts of gauge fields, will be getting swapped with bsh_u
     * wlinks:  Extended Gauge field used as 'ColorMatrices' to store shifted links in the b and vbv directions.
     * The indices i_wl_b, i_wl_vbv, i_wl_tmp control which 'Lorentz' index of wlinks will be used for the shifts
     */
    cudaGaugeField *gf_u;
    cudaGaugeField *bsh_u;
    cudaGaugeField *aux_u;    /* for shifts of bsh_u */
    cudaGaugeField *wlinks;   /* store w_b, w_vbv */
    int i_wl_b, i_wl_vbv, i_wl_tmp;

    int qcR[4]; //- Length of extended halos

    //- Structure holding the parameters of the contractions / momentum projection
    qudaAPI_Param paramAPI;
  };

  typedef struct QuarkTMD_state_s QuarkTMD_state;

  
  __device__ int d_crdChkVal = 0;
  int QluaSiteOrderCheck(QluaUtilArg *utilArg);
  void convertSiteOrder_QudaQDP_to_momProj(void *corrInp_dev, const void *corrQuda_dev, QluaUtilArg utilArg);


  qcTMD_ShiftFlag TMDparseShiftFlag(char flagStr);


  void qcCPUtoCudaVec(cudaColorSpinorField *cudaVec, cpuColorSpinorField *cpuVec);
  void qcCPUtoCudaProp(cudaColorSpinorField **cudaProp, cpuColorSpinorField **cpuProp);
  void qcSetGaugeToUnity(cudaGaugeField *U, int mu, const int *R);
  void qcCopyExtendedGaugeField(cudaGaugeField *dst, cudaGaugeField *src, const int *R);
  void qcCopyCudaLink(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src, const int *R);

  void qcSwapCudaVec(cudaColorSpinorField **x1, cudaColorSpinorField **x2);
  void qcSwapCudaGauge(cudaGaugeField **x1, cudaGaugeField **x2);


  void perform_ShiftCudaVec_nonCov(ColorSpinorField *dst, ColorSpinorField *src,
				   qcTMD_ShiftFlag shfFlag);
  void perform_ShiftCudaVec_Cov(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gf,
                                qcTMD_ShiftFlag shfFlag);
  void perform_ShiftLink_Cov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
                             cudaGaugeField *gf, qcTMD_ShiftFlag shfFlag);
  void perform_ShiftGauge_nonCov(cudaGaugeField *dst, cudaGaugeField *src,
                                 qcTMD_ShiftFlag shfFlag);
  void perform_ShiftLink_AdjSplitCov(cudaGaugeField *dst, int i_dst, cudaGaugeField *src, int i_src,
                                     cudaGaugeField *gf, cudaGaugeField *gf2,
                                     qcTMD_ShiftFlag shfFlag, bool flipShfSgn);

  void qcCopyGammaToConstMem();
  
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
                            cudaColorSpinorField **cudaProp1,
                            cudaColorSpinorField **cudaProp2,
                            cudaColorSpinorField **cudaProp3,
                            complex<QUDA_REAL> *S2, complex<QUDA_REAL> *S1,
                            qudaAPI_Param paramAPI);

  void QuarkContractTMD_GPU(QuarkTMD_state *qcs);
  
} //- namespace quda
  
#endif/*INTERFACE_QLUA_INT_H__*/
