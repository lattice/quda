#ifndef _HISQ_FORCE_UTILS_H
#define _HISQ_FORCE_UTILS_H

#include <quda_internal.h>
#include <quda.h>


// The following routines are used to test the force calculation from hisq smearing
namespace quda{
  namespace fermion_force{

    typedef struct  {
      size_t bytes;
      QudaPrecision precision;
      int length; // total length
      int volume; // geometric volume (single parity)
      int X[4];   // the geometric lengths (single parity)
      int Nc; // length of color dimension
      void *data; // either (double2*) or (float2*) or (float4*)
    } ParityMatrix;

    typedef struct {
      ParityMatrix odd;
      ParityMatrix even;
    } FullMatrix;

    typedef struct {
      ParityMatrix odd;
      ParityMatrix even;
    } FullCompMatrix; // compressed matrix 


    FullMatrix createMatQuda(const int X[4], QudaPrecision precision);
    FullCompMatrix createCompMatQuda(const int X[4], QudaPrecision precision);
    void freeMatQuda(FullMatrix mat);
    void freeCompMatQuda(FullCompMatrix mat);


    typedef struct {
      size_t bytes;
      QudaPrecision precision;
      int length; // total length
      int volume; // geometric volume
      int X[4];
      int Nc; // length of color dimension
      void *data[8]; // array of 8 pointers (Not a pointer to an array!) 
    } ParityOprod;


    typedef struct {
      ParityOprod odd;
      ParityOprod even;
    } FullOprod;


    void loadOprodToGPU(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, int vol);
    void allocateOprodFields(void **cudaOprodEven, void **cudaOprodOdd, int vol);
    void fetchOprodFromGPU(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, int vol);

    FullOprod createOprodQuda(int *X, QudaPrecision precision);
    void copyOprodToGPU(FullOprod cudaOprod, void *oprod, int half_volume);


  } // namespace fermion_force
} // namespace quda


#endif // _HISQ_FORCE_UTILS_H
