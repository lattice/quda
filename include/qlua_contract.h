/* C. Kallidonis: Header file for qlua-quda contractions
 */

#ifndef QLUA_CONTRACT_H__
#define QLUA_CONTRACT_H__

#include <interface_qlua.h>
#include <complex_quda.h>
#include <color_spinor.h>
#include <color_spinor_field_order.h>


namespace quda {

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
  __device__ __host__ inline void rotatePropBasis(ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> *prop, RotateType rType){

    const int Ns = QUDA_Ns;
    const int Nc = QUDA_Nc;

    typedef ColorSpinor<QUDA_REAL,Nc,Ns> Vector;
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
  
}//-- namespace quda


#endif //-- QLUA_CONTRACT_H__
