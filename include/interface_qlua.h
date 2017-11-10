/* C. Kallidonis: Header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_H__
#define INTERFACE_QLUA_H__

#ifdef __cplusplus

#define EXTRN_C extern "C"

#define delete_not_null(p) do { if (NULL != (p)) delete (p) ; } while(0)

#else /*!defined(__cplusplus)*/
#define EXTRN_C
#endif/*__cplusplus*/

#include <color_spinor_field.h>

namespace quda {

#define QUDA_Nc 3
#define QUDA_Ns 4
#define QUDA_DIM 4
#define QUDA_MAX_RANK 6

  typedef long long LONG_T;
  typedef double QUDA_REAL;
  

  enum qudaAPI_ContractId{
    cntr12 = 12,
    cntr13 = 13,
    cntr14 = 14,
    cntr23 = 23,
    cntr24 = 24,
    cntr34 = 34,
    cntr_INVALID = 0
  };
 
  typedef struct {
    int node;
    int rank;
    int net[QUDA_MAX_RANK];
    int net_coord[QUDA_MAX_RANK];
    /* local volume : lo[mu] <= x[mu] < hi[mu] */
    int site_coord_lo[QUDA_MAX_RANK];
    int site_coord_hi[QUDA_MAX_RANK];
    LONG_T locvol;
    LONG_T *ind_qdp2quda;
  } qudaLattice;

  typedef struct {
    QUDA_REAL alpha[QUDA_MAX_RANK];
    QUDA_REAL beta;
    int Nstep;
    QudaVerbosity verbosity;
  } wuppertalParam;

  typedef struct {
    int nVec;
    qudaAPI_ContractId cntrID = cntr_INVALID;
  } contractParam;

  typedef struct {
    QudaVerbosity verbosity;
    wuppertalParam wParam;
    contractParam cParam;
  } qudaAPI_Param;

  typedef struct{
    QUDA_REAL re;
    QUDA_REAL im;
  } QLUA_COMPLEX;

#define CMUL(x,y)  ( (QLUA_COMPLEX) {(x.re) * (y.re) - (x.im) * (y.im), (x.re) * (y.im) + (x.im) * (y.re) } )
#define CADD(x,y)  ( (QLUA_COMPLEX) {(x.re)+(y.re), (x.im)+(y.im)})
#define CSUB(x,y)  ( (QLUA_COMPLEX) {(x.re)-(y.re), (x.im)-(y.im)})
#define PROP_ELEM(ptr,crd,lV,c1,s1,c2,s2) ((QLUA_COMPLEX){ (ptr)[(c2)+QUDA_Nc*(s2)][0+2*(crd)+2*(lV)*(c1)+2*(lV)*QUDA_Nc*(s1)], (ptr)[(c2)+QUDA_Nc*(s2)][1+2*(crd)+2*(lV)*(c1)+2*(lV)*QUDA_Nc*(s1)] } )


  EXTRN_C
    QudaVerbosity parseVerbosity(const char *v);
  
  EXTRN_C int
    doQQ_contract_Quda(
		       QUDA_REAL *hprop_out,
		       QUDA_REAL *hprop_in1,
		       QUDA_REAL *hprop_in2,
		       const qudaLattice *qS,
		       int nColor, int nSpin,
		       const qudaAPI_Param qAparam);

  EXTRN_C int
    laplacianQuda(
		  QUDA_REAL *hv_out,
		  QUDA_REAL *hv_in,
		  QUDA_REAL *h_gauge[],
		  const qudaLattice *qS,
		  int nColor, int nSpin,
		  const qudaAPI_Param qAparam);


  EXTRN_C int
    Qlua_invertQuda(
		    QUDA_REAL *hv_out,
		    QUDA_REAL *hv_in,
		    QUDA_REAL *h_gauge[],
		    const qudaLattice *qS,
		    int nColor, int nSpin,
		    qudaAPI_Param qAparam);


  void cudaContractQQ(
		      ColorSpinorField &propOut,
		      ColorSpinorField &propIn1,
		      ColorSpinorField &propIn2,
		      int Nc, int Ns,
		      contractParam cParam);

} //- namespace quda
  
#endif/*INTERFACE_QLUA_H__*/



