#ifndef _QUDA_H
#define _QUDA_H

#include <cuda_runtime.h>

#define L1 32 // "x" dimension
#define L2 32 // "y" dimension
#define L3 32 // "z" dimension
#define L4 32 // "time" dimension
#define L1h (L1/2) // half of the full "x" dimension, useful for even/odd lattice indexing

#define N (L1*L2*L3*L4) // total number of lattice points
#define Nh (N/2) // total number of even/odd lattice points

// The Quda is added to avoid collisions with other libs
#define GaugeFieldOrder QudaGaugeFieldOrder
#define DiracFieldOrder QudaDiracFieldOrder
#define InverterType QudaInverterType  
#define Precision QudaPrecision
#define MatPCType QudaMatPCType
#define SolutionType QudaSolutionType
#define MassNormalization QudaMassNormalization
#define PreserveSource QudaPreserveSource
#define ReconstructType QudaReconstructType
#define GaugeFixed QudaGaugeFixed
#define DagType QudaDagType
#define Tboundary QudaTboundary

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef struct ParityGauge_s *ParityGauge;
  typedef struct ParitySpinor_s *ParitySpinor;

  typedef struct {
    ParitySpinor odd;
    ParitySpinor even;
  } FullSpinor;
  
  typedef struct {
    ParityGauge odd;
    ParityGauge even;
  } FullGauge;
  
#ifdef __cplusplus
}
#endif

#include <invert_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>

#endif // _QUDA_H

