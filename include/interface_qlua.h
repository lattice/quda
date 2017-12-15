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

#define QUDA_Nc 3
#define QUDA_Ns 4
#define QUDA_DIM 4
#define QUDA_MAX_RANK 6
#define QUDA_PROP_NVEC 12

#define PI 2*asin(1.0)
#define THREADS_PER_BLOCK 64

#ifndef LONG_T
#define LONG_T long long
#endif
typedef double QUDA_REAL;

typedef enum qudaAPI_ContractId_s{
  cntr12 = 12,
  cntr13 = 13,
  cntr14 = 14,
  cntr23 = 23,
  cntr24 = 24,
  cntr34 = 34,
  cntr_INVALID = 0
} qudaAPI_ContractId;

struct qudaLattice_s { 
  int node;
  int rank;
  int net[QUDA_MAX_RANK];
  int net_coord[QUDA_MAX_RANK];
  /* local volume : lo[mu] <= x[mu] < hi[mu] */
  int site_coord_lo[QUDA_MAX_RANK];
  int site_coord_hi[QUDA_MAX_RANK];
  LONG_T locvol;
  LONG_T *ind_qdp2quda;
};
typedef struct qudaLattice_s qudaLattice;

typedef struct {
  QUDA_REAL alpha[QUDA_MAX_RANK];
  QUDA_REAL beta;
  int Nstep;
  QudaVerbosity verbosity;
} wuppertalParam;

typedef struct {
  int nVec;
  qudaAPI_ContractId cntrID;
} contractParam;

typedef struct {
  // first four parameters are set within qudaAPI as user input
  int Ndata;
  int QsqMax;
  int expSgn;
  int GPU_phaseMatrix;
  
  // these are set in the qlua-interface function
  int V3;
  int momDim;
  int Nmoms;
} momProjParam;

typedef struct {
  QudaVerbosity verbosity;
  wuppertalParam wParam;
  contractParam cParam;
  momProjParam mpParam;
} qudaAPI_Param;

typedef struct{
  QUDA_REAL re;
  QUDA_REAL im;
} QLUA_COMPLEX;

typedef enum RotateType_s {
  QLUA_qdp2quda = 1,
  QLUA_quda2qdp = 2
} RotateType;


EXTRN_C
QudaVerbosity parseVerbosity(const char *v);

EXTRN_C
qudaAPI_ContractId parseContractIdx(const char *v);

EXTRN_C int
doQQ_contract_Quda(
		   QUDA_REAL *hprop_out,
		   QUDA_REAL *hprop_in1,
		   QUDA_REAL *hprop_in2,
		   const qudaLattice *qS,
		   int nColor, int nSpin,
		   const qudaAPI_Param qAparam);

EXTRN_C int
momentumProjectionPropagator_Quda(
				  QUDA_REAL *corrOut,
				  QUDA_REAL *corrIn,
				  const qudaLattice *qS,
				  qudaAPI_Param paramAPI);

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


#endif/*INTERFACE_QLUA_H__*/
