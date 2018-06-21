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
#include <interface_qlua.h>


namespace quda {

  //-C.K. Typedef Propagator, Gauge Field and Vector Structures
  typedef typename colorspinor_mapper<QUDA_REAL,QUDA_Ns,QUDA_Nc>::type Propagator;
  typedef typename gauge_mapper<QUDA_REAL,QUDA_RECONSTRUCT_NO>::type GaugeU;
  typedef ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> Vector;

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
  __device__ __host__ inline void rotatePropBasis(Vector *prop, RotateType rType){

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

              res[iv].data[id] += A[is][a] * prop[iv].data[as];
            }
          }}}
    }

    for(int v = 0; v<QUDA_PROP_NVEC; v++)
      prop[v] = res[v];

  }
  //---------------------------------------------------------------------------

  struct QluaContractArg {

    Propagator prop1[QUDA_PROP_NVEC]; // Input
    Propagator prop2[QUDA_PROP_NVEC]; // Propagators
    Propagator prop3[QUDA_PROP_NVEC]; //
    
    GaugeU U;                         // Gauge Field

    const qluaCntr_Type cntrType;     // contraction type
    const int nParity;                // number of parities we're working on
    const int nFace;                  // hard code to 1 for now
    const int dim[5];                 // full lattice dimensions
    const int commDim[4];             // whether a given dimension is partitioned or not
    const int lL[4];                  // 4-d local lattice dimensions
    const int volumeCB;               // checkerboarded volume
    const int volume;                 // full-site local volume
    
    QluaContractArg(ColorSpinorField **propIn1,
                    ColorSpinorField **propIn2,
                    ColorSpinorField **propIn3,
                    GaugeField *Uin,
                    qluaCntr_Type cntrType)
      : cntrType(cntrType), nParity(propIn1[0]->SiteSubset()), nFace(1),
	dim{ (3-nParity) * propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3), 1 },
        commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        lL{propIn1[0]->X(0), propIn1[0]->X(1), propIn1[0]->X(2), propIn1[0]->X(3)},
        volumeCB(propIn1[0]->VolumeCB()),volume(propIn1[0]->Volume())
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

      if((cntrType == what_qpdf_g_F_B) || (cntrType == what_tmd_g_F_B)){
	if(Uin == NULL) errorQuda("QluaContractArg: Gauge Field is not allocated!\n");
	U.init(*Uin);
      }

    }

  };//-- Structure definition
  //---------------------------------------------------------------------------
  
}//-- namespace quda


#endif //-- QLUA_CONTRACT_H__
