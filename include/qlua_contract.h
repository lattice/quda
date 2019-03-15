/* C. Kallidonis: Header file for qlua-quda contractions
 */

#ifndef QLUA_CONTRACT_H__
#define QLUA_CONTRACT_H__

#include <mpi.h>
#include <transfer.h>
#include <complex_quda.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <interface_qlua_internal.h>

#define QUDA_LEN_G (QUDA_Ns*QUDA_Ns)

namespace quda {

  //-C.K. Typedef Propagator, Gauge Field and Vector Structures
  typedef typename colorspinor_mapper<QUDA_REAL,QUDA_Ns,QUDA_Nc>::type Propagator;
  typedef typename gauge_mapper<QUDA_REAL,QUDA_RECONSTRUCT_NO>::type GaugeU;
  typedef ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> Vector;
  typedef Matrix<complex<QUDA_REAL>,QUDA_Nc> Link;

  extern const int nShiftFlag;
  extern const int nShiftType;
  extern const char *qcTMD_ShiftFlagArray;

  extern const char *qcTMD_ShiftTypeArray[];
  extern const char *qcTMD_ShiftDirArray[];
  extern const char *qcTMD_ShiftSgnArray[];


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
    const int NV = QUDA_NVEC_PROP;

    Vector res[NV];

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

    for(int v = 0; v<NV; v++)
      vecIO[v] = res[v];

  }
  //---------------------------------------------------------------------------

  struct ContractQQArg {

    Propagator pIn1[QUDA_MAX_NVEC];  // Input propagator 1
    Propagator pIn2[QUDA_MAX_NVEC];  // Input propagator 2
    Propagator pOut[QUDA_MAX_NVEC];  // Output propagator

    const int parity;                 // only use this for single parity fields
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int volumeCB;               // checkerboarded volume
    const int nVec;                   // number of vectors in the propagator (usually 12)

    qluaCntrQQ_Id cntrID;             // contract index

    ContractQQArg(ColorSpinorField **propOut, ColorSpinorField **propIn1, ColorSpinorField **propIn2, int parity, cntrQQParam cQQParam)
      :   parity(parity), nParity(propIn1[0]->SiteSubset()), nFace(1),
	  dim{ (3-nParity) * propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(propIn1[0]->VolumeCB()), nVec(cQQParam.nVec), cntrID(cQQParam.cntrID)
    {

      if(nVec != QUDA_NVEC_PROP)
	errorQuda("ContractQQArg: QuarkQQ Contractions support only nVec = %d. Got nVec = %d\n", QUDA_NVEC_PROP, nVec);

      for(int ivec=0;ivec<nVec;ivec++){
        pIn1[ivec].init(*propIn1[ivec]);
        pIn2[ivec].init(*propIn2[ivec]);
        pOut[ivec].init(*propOut[ivec]);
      }

    }
  }; //-- Structure definition
  //---------------------------------------------------------------------------


  struct QluaContractArg {
    
    Propagator prop1[QUDA_MAX_NVEC]; // Input
    Propagator prop2[QUDA_MAX_NVEC]; // Propagators
    Propagator prop3[QUDA_MAX_NVEC]; //
    
    const int nVec;                   // Number of vectors within Propagator Structure (for "regular" prop, nVec = 12)
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
    
    QluaContractArg(cudaColorSpinorField **propIn1,
		    cudaColorSpinorField **propIn2,
		    cudaColorSpinorField **propIn3,
		    qluaCntr_Type cntrType, bool preserveBasis, int nVec_)
      : cntrType(cntrType), parity(0), nParity(propIn1[0]->SiteSubset()), nFace(1),
	dim{ (3-nParity) * propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3)},
      volumeCB(propIn1[0]->VolumeCB()),volume(propIn1[0]->Volume()), preserveBasis(preserveBasis), nVec(nVec_)
    {

      if((cntrType != what_qpdf_g_F_B) && (cntrType != what_qbarq_g_F_aB) && (nVec != QUDA_NVEC_PROP))
	errorQuda("QluaContractArg: Ultra-local contractions apart from qbarq_g_F_aB support only nVec = %d. Got nVec = %d\n", QUDA_NVEC_PROP, nVec);

      for(int ivec=0;ivec<nVec;ivec++){
        prop1[ivec].init(*propIn1[ivec]);
        prop2[ivec].init(*propIn2[ivec]);
      }
      
      if(cntrType == what_baryon_sigma_UUS){
        if(propIn3 == NULL) errorQuda("QluaContractArg: Input propagator-3 is not allocated!\n");
        for(int ivec=0;ivec<nVec;ivec++)
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
    { }

    ArgGeom(cudaGaugeField *u) 
      : parity(0), nParity(u->SiteSubset()), nFace(1),
        commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        lL{u->X()[0], u->X()[1], u->X()[2], u->X()[3]}
    {
      if(u->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED){
	volume = 1;
	for(int dir=0;dir<4;dir++){
	  dim[dir] = u->X()[dir] - 2*u->R()[dir];   //-- Actual lattice dimensions (NOT extended)
	  dimEx[dir] = dim[dir] + 2*u->R()[dir];    //-- Extended lattice dimensions
	  brd[dir] = u->R()[dir];
	  volume *= dim[dir];
	}
	volumeCB = volume/2;
      }
      else{
	volume = 1;
	for(int dir=0;dir<4;dir++){
	  dim[dir] = u->X()[dir];
	  volume *= dim[dir];
	}
	volumeCB = volume/2;
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
      :	ArgGeom(gf_), extendedGauge((gf_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false)
    {
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


  struct Arg_SetUnityLink : public ArgGeom {

    GaugeU U;
    int mu;

    bool extendedGauge;

    complex<QUDA_REAL> unityU[QUDA_Nc*QUDA_Nc];

    Arg_SetUnityLink(cudaGaugeField *U_, int mu_)
      : ArgGeom(U_), mu(mu_), extendedGauge((U_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false)
    {
      U.init(*U_);
      
      for(int ic=0;ic<QUDA_Nc;ic++){
        for(int jc=0;jc<QUDA_Nc;jc++){
	  if(ic==jc) unityU[jc + QUDA_Nc*ic] = complex<QUDA_REAL> {1.0,0.0};
	  else unityU[jc + QUDA_Nc*ic] = complex<QUDA_REAL> {0.0,0.0};
        }
      }
    }
  };
  //---------------------------------------------------------------------


  struct Arg_CopyCudaLink : public ArgGeom {

    GaugeU Udst;
    GaugeU Usrc;
    int i_dst, i_src;

    bool extendedGauge_dst;
    bool extendedGauge_src;
    bool extendedGauge;

    Arg_CopyCudaLink(cudaGaugeField *Udst_, int i_dst_, cudaGaugeField *Usrc_, int i_src_)
      : ArgGeom(Usrc_), i_dst(i_dst_), i_src(i_src_),
	extendedGauge_src((Usrc_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false),
	extendedGauge_dst((Udst_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false)
    {
      if(extendedGauge_src != extendedGauge_dst)
	errorQuda("Arg_CopyCudaLink: Source and Destination GaugeFields ghost properties differ!\n");
      extendedGauge = extendedGauge_src;
      
      Udst.init(*Udst_);
      Usrc.init(*Usrc_);
    }
  };
  //---------------------------------------------------------------------


  struct qcTMD_Arg : public ArgGeom {

    Propagator fwdProp[QUDA_MAX_NVEC];
    Propagator bwdProp[QUDA_MAX_NVEC];
    GaugeU U;

    int nVec;
    int i_mu;
    bool preserveBasis;
    bool extendedGauge;

    qcTMD_Arg () {}
    
    qcTMD_Arg(cudaColorSpinorField **fwdProp_, cudaColorSpinorField **bwdProp_,
	      cudaGaugeField *U_, int i_mu_, bool preserveBasis_, int nVec_)
      : ArgGeom(U_), i_mu(i_mu_), preserveBasis(preserveBasis_),
      extendedGauge((U_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? true : false), nVec(nVec_)
    {
      for(int ivec=0;ivec<nVec;ivec++){
	fwdProp[ivec].init(*fwdProp_[ivec]);
	bwdProp[ivec].init(*bwdProp_[ivec]);
      }
      U.init(*U_);
    }

  };
  //---------------------------------------------------------------------


  struct qcGammaStruct {
    int left_ind[QUDA_LEN_G][QUDA_Ns];
    complex<QUDA_REAL> left_coeff[QUDA_LEN_G][QUDA_Ns];
  };

  
}//-- namespace quda


#endif //-- QLUA_CONTRACT_H__
