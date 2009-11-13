#ifndef _QUDA_H
#define _QUDA_H

#include <cuda_runtime.h>

//#define L1 4 // "x" dimension
//#define L2 4 // "y" dimension
//#define L3 4 // "z" dimension
//#define L4 4 // "time" dimension
//#define L1h (L1/2) // half of the full "x" dimension, useful for even/odd lattice indexing

//#define N (L1*L2*L3*L4) // total number of lattice points
//#define Nh (N/2) // total number of even/odd lattice points

#define MAX_SHORT 32767

// The Quda is added to avoid collisions with other libs
#define GaugeFieldOrder QudaGaugeFieldOrder
#define DiracFieldOrder QudaDiracFieldOrder
#define CloverFieldOrder QudaCloverFieldOrder
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

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef void *ParityGauge;

  typedef struct {
    size_t bytes;
    Precision precision;
    int length; // total length
    int real_length; // physical length (excluding padding)
    int volume; // geometric volume (single parity)
    int pad; // padding from end of array to start of next
    int stride; // geometric stride between volume lengthed arrays
    int X[4]; // the geometric lengths (single parity)
    int Nc; // number of colors
    ReconstructType reconstruct;
    ParityGauge odd;
    ParityGauge even;
    double anisotropy;
  } FullGauge;
  
  typedef struct {
    size_t bytes;
    Precision precision;
    int length;
    int real_length; // physical length (excluding padding)
    int volume;
    int pad; // padding from end of array to start of next
    int stride; // geometric stride between volume lengthed arrays
    int X[4];
    int Nc;
    int Ns;
    void *clover;
    float *cloverNorm;
  } ParityClover;

  typedef struct {
    ParityClover odd;
    ParityClover even;
  } FullClover;

  typedef struct {
    size_t bytes;
    Precision precision;
    int length; // total length
    int real_length; // physical length (excluding padding)
    int volume; // geometric volume (single parity)
    int pad; // padding from end of array to start of next
    int stride; // geometric stride between volume lengthed arrays
    int X[4]; // the geometric lengths (single parity)
    int Nc; // length of color dimension
    int Ns; // length of spin dimension
    void *spinor; // either (double2*), (float4 *) or (short4 *), depending on precision
    float *spinorNorm; // used only when precision is QUDA_HALF_PRECISION
  } ParitySpinor;

  typedef struct {
    ParitySpinor odd;
    ParitySpinor even;
  } FullSpinor;
  
#ifdef __cplusplus
}
#endif

#include <invert_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>

#endif // _QUDA_H
