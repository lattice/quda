/* C. Kallidonis: Header file for qlua-quda contractions
 */

#ifndef QLUA_CONTRACT_H__
#define QLUA_CONTRACT_H__

#include <transfer.h>
#include <complex_quda.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <color_spinor_field_order.h>
#include <interface_qlua_internal.h>

namespace quda {

  //-C.K. Typedef Propagator, Gauge Field and Vector Structures
  typedef typename colorspinor_mapper<QUDA_REAL,QUDA_Ns,QUDA_Nc>::type Propagator;
  typedef typename gauge_mapper<QUDA_REAL,QUDA_RECONSTRUCT_NO>::type GaugeU;
  typedef ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> Vector;
  typedef Matrix<complex<QUDA_REAL>,QUDA_Nc> Link;



  const int nShiftFlag = 20;
  const int nShiftType = 2;

  static const char *qcTMD_ShiftFlagArray[nShiftFlag] = {
    "X", "x", "Y", "y", "Z", "z", "T", "t", "Q", "q",
    "R", "r", "S", "s", "U", "u", "V", "v", "W", "w"};

  static const char *qcTMD_ShiftTypeArray[nShiftType] = {
    "Covariant",
    "Non-Covariant"};

  static const char *qcTMD_ShiftDirArray[4] = {"x", "y", "z", "t"};
  static const char *qcTMD_ShiftSgnArray[2] = {"-", "+"};


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
    qcCovShift = 0,      //-- Covariant Shift (involving a gauge link)
    qcNonCovShift = 1    //-- Non-covariant shift
  } qcTMD_ShiftType;


  /** C.K.
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
  __device__ __host__ inline void rotateVectorBasis(Vector *vecIO, RotateType rType){

    const int Ns = QUDA_Ns;
    const int Nc = QUDA_Nc;

    Vector res[QUDA_PROP_NVEC];

    complex<QUDA_REAL> zro = complex<QUDA_REAL>{0,0};
    complex<QUDA_REAL> val = complex<QUDA_REAL>{sqrt(0.5),0};
    complex<QUDA_REAL> M[Ns][Ns] = { { zro, -val,  zro, -val},
				     { val,  zro,  val,  zro},
				     { zro, -val,  zro,  val},
				     { val,  zro, -val,  zro} };

    complex<QUDA_REAL> M_Trans[Ns][Ns];
    for(int i=0;i<Ns;i++){
      for(int j=0;j<Ns;j++){
        M_Trans[i][j] = M[j][i];
      }
    }

    complex<QUDA_REAL> (*A)[Ns] = NULL;
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

              res[iv].data[id] += A[is][a] * vecIO[iv].data[as];
            }
          }}}
    }

    for(int v = 0; v<QUDA_PROP_NVEC; v++)
      vecIO[v] = res[v];

  }
  //---------------------------------------------------------------------------

  struct QluaContractArg {
    
    Propagator prop1[QUDA_PROP_NVEC]; // Input
    Propagator prop2[QUDA_PROP_NVEC]; // Propagators
    Propagator prop3[QUDA_PROP_NVEC]; //
    
    const qluaCntr_Type cntrType;     // contraction type
    const int parity;                 // hard code to 0 for now
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int lL[4];                  // 4-d local lattice dimensions
    const int volumeCB;               // checkerboarded volume
    const int volume;                 // full-site local volume
    const bool preserveBasis;         // whether to preserve the gamma basis or not
    
    QluaContractArg(ColorSpinorField **propIn1,
		    ColorSpinorField **propIn2,
		    ColorSpinorField **propIn3,
		    qluaCntr_Type cntrType, bool preserveBasis)
      : cntrType(cntrType), parity(0), nParity(propIn1[0]->SiteSubset()), nFace(1),
	dim{ (3-nParity) * propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3)},
      volumeCB(propIn1[0]->VolumeCB()),volume(propIn1[0]->Volume()), preserveBasis(preserveBasis)
    {
      for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
        prop1[ivec].init(*propIn1[ivec]);
        prop2[ivec].init(*propIn2[ivec]);
      }
      
      if(cntrType == what_baryon_sigma_UUS){
        if(propIn3 == NULL) errorQuda("QluaContractArg: Input propagator-3 is not allocated!\n");
        for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++)
          prop3[ivec].init(*propIn3[ivec]);
      }
    }

  };//-- Structure definition
  //---------------------------------------------------------------------------

  struct ArgGeom {
    
    int parity;                 // hard code to 0 for now
    int nParity;                // number of parities we're working on
    int nFace;                  // hard code to 1 for now
    int dim[5];                 // full lattice dimensions
    int commDim[4];             // whether a given dimension is partitioned or not
    int lL[4];                  // 4-d local lattice dimensions
    int volumeCB;               // checkerboarded volume
    int volume;                 // full-site local volume

    int dimEx[4]; // extended Gauge field dimensions
    int brd[4];   // Border of extended gauge field (size of extended halos)
    
    ArgGeom () {}

    ArgGeom(ColorSpinorField *x)
      : parity(0), nParity(x->SiteSubset()), nFace(1),
    	dim{ (3-nParity) * x->X(0), x->X(1), x->X(2), x->X(3), 1 },
        commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        lL{x->X(0), x->X(1), x->X(2), x->X(3)},
        volumeCB(x->VolumeCB()), volume(x->Volume())
    {}

    ArgGeom(cudaGaugeField *u) 
      : parity(0), nParity(u->SiteSubset()), nFace(1),
        commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        lL{u->X()[0], u->X()[1], u->X()[2], u->X()[3]},
        volumeCB(u->VolumeCB()), volume(u->Volume())
    {
      if(u->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED){
	for(int dir=0;dir<4;dir++){
	  dim[dir] = u->X()[dir] - 2*u->R()[dir];   //-- Actual lattice dimensions (NOT extended)
	  dimEx[dir] = dim[dir] + 2*u->R()[dir];    //-- Extended lattice dimensions
	  brd[dir] = u->R()[dir];
	}
      }
      else{
	for(int dir=0;dir<4;dir++) dim[dir] = u->X()[dir];
	dim[0] *= (3-nParity);
      }
      dim[4] = 1;
    }
  };//-- ArgGeom


  struct Arg_ShiftCudaVec_nonCov : public ArgGeom {
    
    Propagator src;
    Propagator dst;
    
    Arg_ShiftCudaVec_nonCov(ColorSpinorField *dst_, ColorSpinorField *src_)
      : ArgGeom(src_)
    {
      src.init(*src_);
      dst.init(*dst_);
    }
  };

  struct Arg_ShiftCudaVec_Cov : public ArgGeom {

    Propagator src;
    Propagator dst;
    GaugeU U;

    bool extendedGauge;

    Arg_ShiftCudaVec_Cov(ColorSpinorField *dst_, ColorSpinorField *src_, cudaGaugeField *gf_) 
      : extendedGauge((gf_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false),
	ArgGeom(gf_)
    {
      if(!extendedGauge) errorQuda("Arg_ShiftCudaVec_Cov: No support for non-extended Gauge fields for now\n");
      src.init(*src_);
      dst.init(*dst_);
      U.init(*gf_);
    }
  };
  //---------------------------------------------------------------------
  

  struct Arg_ShiftGauge_nonCov : public ArgGeom {

    GaugeU src, dst;

    Arg_ShiftGauge_nonCov(cudaGaugeField *dst_, cudaGaugeField *src_)
      : ArgGeom(src_)
    {
      src.init(*src_);
      dst.init(*dst_);
    }
  };
  
  struct Arg_ShiftLink_Cov : public ArgGeom {

    int i_src, i_dst;
    GaugeU src, dst;
    GaugeU gf_u;

    Arg_ShiftLink_Cov(cudaGaugeField *dst_, int i_dst_,
		      cudaGaugeField *src_, int i_src_, 
		      cudaGaugeField *gf_u_)
      : ArgGeom(gf_u_), i_src(i_src_), i_dst(i_dst_)
    {
      src.init(*src_);
      dst.init(*dst_);
      gf_u.init(*gf_u_);
    }
  };

  struct Arg_ShiftLink_AdjSplitCov : public ArgGeom {

    GaugeU src, dst;
    GaugeU gf_u, bsh_u;
    int i_src, i_dst;

    Arg_ShiftLink_AdjSplitCov(cudaGaugeField *dst_, int i_dst_,
			      cudaGaugeField *src_, int i_src_,
			      cudaGaugeField *gf_u_, cudaGaugeField *bsh_u_)
      : ArgGeom(gf_u_), i_src(i_src_), i_dst(i_dst_)
    {
      src.init(*src_);
      dst.init(*dst_);
      gf_u.init(*gf_u_);
      bsh_u.init(*bsh_u_);
    }
  };
  //---------------------------------------------------------------------


  struct qcTMD_State : public ArgGeom {

    Propagator fwdProp[QUDA_PROP_NVEC];
    Propagator bwdProp[QUDA_PROP_NVEC];
    GaugeU U;

    bool preserveBasis;
    qluaCntr_Type cntrType;
    int i_mu;
    
    qcTMD_State () {}
    
    qcTMD_State(ColorSpinorField **fwdProp_, ColorSpinorField **bwdProp_,
		cudaGaugeField *U_, int i_mu_, qluaCntr_Type cntrType_, bool preserveBasis_)
      : ArgGeom(fwdProp_[0]), i_mu(i_mu_), cntrType(cntrType_), preserveBasis(preserveBasis_)
    {
      for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++){
	fwdProp[ivec].init(*fwdProp_[ivec]);
	bwdProp[ivec].init(*bwdProp_[ivec]);
      }
      U.init(*U_);
    }

  };
  //---------------------------------------------------------------------

  
}//-- namespace quda


#endif //-- QLUA_CONTRACT_H__


  // //- Additional structure required for the TMD contractions
  // struct TMDcontractState {

  //   Propagator fwdProp[QUDA_PROP_NVEC];  // This will be a pointer to the (shifted) forward propagator
  //   Propagator bwdProp[QUDA_PROP_NVEC];  // This will be a pointer to the backward propagator
  //   Propagator auxProp[QUDA_PROP_NVEC];  // This will be an auxilliary propagator

  //   GaugeU U;                   // Original Gauge Field
  //   GaugeU auxU;                // Auxilliary Gauge Field

  //   bool csInit = false;
    
  //   qluaCntr_Type cntrType;     // contraction type
  //   int parity;                 // hard code to 0 for now
  //   int nParity;                // number of parities we're working on
  //   int nFace;                  // hard code to 1 for now
  //   int dim[5];                 // full lattice dimensions
  //   int commDim[4];             // whether a given dimension is partitioned or not
  //   int lL[4];                  // 4-d local lattice dimensions
  //   int volumeCB;               // checkerboarded volume
  //   int volume;                 // full-site local volume
  //   bool preserveBasis;         // whether to preserve the gamma basis or not

  //   TMDcontractState () {}

  //   TMDcontractState(ColorSpinorField **prop, GaugeField *U_, qluaCntr_Type cntrType, bool preserveBasis)
  //     : cntrType(cntrType), preserveBasis(preserveBasis),
  // 	parity(0), nParity(prop[0]->SiteSubset()), nFace(1),
  //   	dim{ (3-nParity) * prop[0]->X(0), prop[0]->X(1), prop[0]->X(2), prop[0]->X(3), 1 },
  //       commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
  //       lL{prop[0]->X(0), prop[0]->X(1), prop[0]->X(2), prop[0]->X(3)},
  //       volumeCB(prop[0]->VolumeCB()), volume(prop[0]->Volume())
  //   {
  //     if(U_==NULL) errorQuda("TMDcontractState: Gauge Field U is not allocated!\n");
  //     U.init(*U_);
  //     csInit = true;
  //   }
    
  //   TMDcontractState(ColorSpinorField *fwdVec, ColorSpinorField *auxVec, int ivec,
  //   		     GaugeField *U_, GaugeField *auxU_,
  //   		     qluaCntr_Type cntrType, bool preserveBasis)
  //     : cntrType(cntrType), parity(0), nParity(fwdVec->SiteSubset()), nFace(1),
  //   	dim{ (3-nParity) * fwdVec->X(0), fwdVec->X(1), fwdVec->X(2), fwdVec->X(3), 1 },
  //       commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
  //       lL{fwdVec->X(0), fwdVec->X(1), fwdVec->X(2), fwdVec->X(3)},
  //       volumeCB(fwdVec->VolumeCB()), volume(fwdVec->Volume()), preserveBasis(preserveBasis)
  //   {
  //     fwdProp[ivec].init(*fwdVec);
  //     auxProp[ivec].init(*auxVec);
  //     if(U_    == NULL) errorQuda("TMDcontractState: Gauge Field U    is not allocated!\n");
  //     if(auxU_ == NULL) errorQuda("TMDcontractState: Gauge Field auxU is not allocated!\n");
  //     U.init(*U_);
  //     auxU.init(*auxU_);
  //     csInit = true;
  //   }

  //   void initPropShf(ColorSpinorField *fwdVec, ColorSpinorField *auxVec, int ivec){
  //     if(!csInit) errorQuda("TMDcontractState: Need to initialize first!\n");
  //     fwdProp[ivec].init(*fwdVec);
  //     auxProp[ivec].init(*auxVec);
  //   }//-- initPropShf
    
  //   void initGaugeShf(GaugeField *U_, GaugeField *auxU_){
  //     if(!csInit) errorQuda("TMDcontractState: Need to initialize first!\n");
  //     if(U_    == NULL) errorQuda("TMDcontractState: Gauge Field U    is not allocated!\n");
  //     if(auxU_ == NULL) errorQuda("TMDcontractState: Gauge Field auxU is not allocated!\n");
  //     U.init(*U_);
  //     auxU.init(*auxU_);
  //   }//-- initGaugeShf
    
  //   void initGaugeShf(GaugeField *auxU_){
  //     if(!csInit) errorQuda("TMDcontractState: Need to initialize first!\n");
  //     if(auxU_ == NULL) errorQuda("TMDcontractState: Gauge Field auxU is not allocated!\n");
  //     auxU.init(*auxU_);
  //   }//-- initGaugeShf

  //   void initBwdProp(ColorSpinorField **bwdProp_){
  //     if(!csInit) errorQuda("TMDcontractState: Need to initialize first!\n");
  //     for(int ivec=0;ivec<QUDA_PROP_NVEC;ivec++)
  //       bwdProp[ivec].init(*bwdProp_[ivec]);
  //   }

  // };//-- Structure definition
  // //---------------------------------------------------------------------------
