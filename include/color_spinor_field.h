#include <quda_internal.h>
#include <cuComplex.h>
#include <quda.h>

#ifndef __COLORSPINORFIELD_QUDA_H
#define __COLORSPINORFIELD_QUDA_H

// Probably want some checking for this limit
#define QUDA_MAX_DIM 6

class ColorSpinorParam {
 public:
  FieldType fieldType; // cpu, cuda etc. 
  int nColor; // Number of colors of the field
  int nSpin; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  int nDim; // number of spacetime dimensions
  int x[QUDA_MAX_DIM]; // size of each dimension
  QudaPrecision precision; // Precision of the field
  int pad; // volumetric padding

  FieldSubset fieldSubset; // Full, even or odd
  SubsetOrder subsetOrder; // defined for full fields
  
  QudaColorSpinorOrder fieldOrder; // Float, Float2, Float4 etc.
  GammaBasis basis;
  FieldCreate create; // 

  void *v; // pointer to field
  void *norm;

 ColorSpinorParam()
   : fieldType(QUDA_INVALID_FIELD), nColor(0), nSpin(0), nDim(0), precision(QUDA_INVALID_PRECISION), 
    pad(0), fieldSubset(QUDA_INVALID_SUBSET), subsetOrder(QUDA_INVALID_SUBSET_ORDER), 
    fieldOrder(QUDA_INVALID_ORDER), basis(QUDA_INVALID_BASIS), create(QUDA_INVALID_CREATE)
  { for(int d=0; d<QUDA_MAX_DIM; d++) x[d] = 0;}

  // used to create cpu params
 ColorSpinorParam(void *V, QudaInvertParam &inv_param, int *X)
   : fieldType(QUDA_CPU_FIELD), nColor(3), nSpin(4), nDim(4), precision(inv_param.cpu_prec), 
    pad(0), fieldSubset(QUDA_INVALID_SUBSET), subsetOrder(QUDA_INVALID_SUBSET_ORDER), 
    fieldOrder(QUDA_INVALID_ORDER), basis(QUDA_DEGRAND_ROSSI_BASIS), create(QUDA_REFERENCE_CREATE), v(V)
  { 

    if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
    for (int d=0; d<nDim; d++) x[d] = X[d];

    if (inv_param.dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_SPIN_COLOR_ORDER;
      subsetOrder = QUDA_ODD_EVEN_SUBSET_ORDER;
    } else if (inv_param.dirac_order == QUDA_QDP_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_COLOR_SPIN_ORDER;
      subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;
    } else if (inv_param.dirac_order == QUDA_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_SPIN_COLOR_ORDER;
      subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;
    } else {
      errorQuda("Dirac order %d not supported", inv_param.dirac_order);
    }
  }

  // used to create cuda param from a cpu param
  ColorSpinorParam(ColorSpinorParam &cpuParam, QudaInvertParam &inv_param) 
    : fieldType(QUDA_CUDA_FIELD), nColor(cpuParam.nColor), nSpin(cpuParam.nSpin), nDim(cpuParam.nDim), 
    precision(inv_param.cuda_prec), pad(inv_param.sp_pad), fieldSubset(cpuParam.fieldSubset), 
    subsetOrder(QUDA_EVEN_ODD_SUBSET_ORDER), fieldOrder(QUDA_INVALID_ORDER), basis(QUDA_UKQCD_BASIS),
    create(QUDA_COPY_CREATE), v(0)
  {
    if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
    for (int d=0; d<nDim; d++) x[d] = cpuParam.x[d];

    if (precision == QUDA_DOUBLE_PRECISION) {
      fieldOrder = QUDA_FLOAT2_ORDER;
    } else {
      fieldOrder = QUDA_FLOAT4_ORDER;
    }

  }

  void print() {
    printfQuda("fieldType = %d\n", fieldType);
    printfQuda("nColor = %d\n", nColor);
    printfQuda("nSpin = %d\n", nSpin);
    printfQuda("nDim = %d\n", nDim);
    for (int d=0; d<nDim; d++) printfQuda("x[%d] = %d\n", d, x[d]);
    printfQuda("precision = %d\n", precision);
    printfQuda("pad = %d\n", pad);
    printfQuda("fieldSubset = %d\n", fieldSubset);
    printfQuda("subsetOrder = %d\n", subsetOrder);
    printfQuda("fieldOrder = %d\n", fieldOrder);
    printfQuda("basis = %d\n", basis);
    printfQuda("create = %d\n", create);
    printfQuda("v = %lx\n", (ulong)v);
    printfQuda("norm = %lx\n", (ulong)norm);
  }

  virtual ~ColorSpinorParam() {
  }
};

class ColorSpinorField {

 private:
  void create(int nDim, const int *x, int Nc, int Ns, QudaPrecision precision, 
	      int pad, FieldType type, FieldSubset subset, 
	      SubsetOrder subsetOrder, QudaColorSpinorOrder order, GammaBasis basis);
  void destroy();  

 protected:
  bool init;
  QudaPrecision precision;

  int nColor;
  int nSpin;
  
  int nDim;
  int x[QUDA_MAX_DIM];

  int volume;
  int pad;
  int stride;
  
  int real_length;
  int length;
  size_t bytes;
  
  FieldType type;
  FieldSubset subset;
  SubsetOrder subset_order;
  QudaColorSpinorOrder order;
  GammaBasis basis;
  
  // in the case of full fields, these are references to the even / odd sublattices
  ColorSpinorField *even;
  ColorSpinorField *odd;

  // resets the above attributes based on contents of param
  void reset(const ColorSpinorParam &);
  void fill(ColorSpinorParam &);
  static void checkField(const ColorSpinorField &, const ColorSpinorField &);

 public:
  ColorSpinorField();
  ColorSpinorField(const ColorSpinorField &);
  ColorSpinorField(const ColorSpinorParam &);

  virtual ~ColorSpinorField();

  ColorSpinorField& operator=(const ColorSpinorField &);

  QudaPrecision Precision() const { return precision; }
  int Ncolor() const { return nColor; } 
  int Nspin() const { return nSpin; } 
  int Length() const { return length; }
  int Stride() const { return stride; }
  int Volume() const { return volume; }

  FieldType fieldType() const { return type; }
  FieldSubset fieldSubset() const { return subset; }
  SubsetOrder subsetOrder() const { return subset_order; }
  QudaColorSpinorOrder fieldOrder() const { return order; }
  GammaBasis gammaBasis() const { return basis; }
};

class cpuColorSpinorField;

// CUDA implementation
class cudaColorSpinorField : public ColorSpinorField {

  friend class cpuColorSpinorField;

  friend double normEven(const cudaColorSpinorField &b);

  friend class DiracWilson;
  friend class DiracClover;
  friend class DiracCloverPC;
  friend void zeroCuda(cudaColorSpinorField &a);
  friend void copyCuda(cudaColorSpinorField &, const cudaColorSpinorField &);
  friend double axpyNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  friend double sumCuda(cudaColorSpinorField &b);
  friend double normCuda(const cudaColorSpinorField &b);
  friend double reDotProductCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
  friend double xmyNormCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
  friend void axpbyCuda(const double &a, cudaColorSpinorField &x, const double &b, cudaColorSpinorField &y);
  friend void axpyCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  friend void axCuda(const double &a, cudaColorSpinorField &x);
  friend void xpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y);
  friend void xpayCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y);
  friend void mxpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y);
  friend void axpyZpbxCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y, 
			   cudaColorSpinorField &z, const double &b);
  friend void caxpbyCuda(const double2 &a, cudaColorSpinorField &x, const double2 &b, cudaColorSpinorField &y);
  friend void caxpyCuda(const double2 &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  friend void cxpaypbzCuda(cudaColorSpinorField &, const double2 &b, cudaColorSpinorField &y, 
			   const double2 &c, cudaColorSpinorField &z);
  friend void caxpbypzYmbwCuda(const double2 &, cudaColorSpinorField &, const double2 &, cudaColorSpinorField &, 
			       cudaColorSpinorField &, cudaColorSpinorField &); 
  friend cuDoubleComplex cDotProductCuda(cudaColorSpinorField &, cudaColorSpinorField &);
  friend cuDoubleComplex xpaycDotzyCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y, 
					cudaColorSpinorField &z);
  friend double3 cDotProductNormACuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
  friend double3 cDotProductNormBCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
  friend double3 caxpbypzYmbwcDotProductWYNormYCuda(const double2 &a, cudaColorSpinorField &x, const double2 &b, 
						    cudaColorSpinorField &y, cudaColorSpinorField &z, 
						    cudaColorSpinorField &w, cudaColorSpinorField &u);

  friend void dslashCuda(cudaColorSpinorField&, FullGauge, cudaColorSpinorField&, int, int);
  friend void dslashXpayCuda(cudaColorSpinorField&, FullGauge, cudaColorSpinorField&, int, int, 
			     cudaColorSpinorField&, double);
  
  friend void MatCuda(cudaColorSpinorField&, FullGauge, cudaColorSpinorField &, double, int);
  friend void MatPCCuda(cudaColorSpinorField&, FullGauge, cudaColorSpinorField &, 
		 double, cudaColorSpinorField&, MatPCType, int);
  friend void MatPCDagMatPCCuda(cudaColorSpinorField&, FullGauge, cudaColorSpinorField &, double,
				cudaColorSpinorField&, MatPCType);
  friend void cloverDslashCuda(cudaColorSpinorField&, FullGauge, FullClover, cudaColorSpinorField &, int, int);
  friend void cloverDslashXpayCuda(cudaColorSpinorField&, FullGauge, FullClover, cudaColorSpinorField &, int,
			    int, cudaColorSpinorField &, double);
  friend void cloverMatPCCuda(cudaColorSpinorField &, FullGauge, FullClover, FullClover, 
			      cudaColorSpinorField &, double, cudaColorSpinorField&, MatPCType, int);
  friend void cloverMatPCDagMatPCCuda(cudaColorSpinorField &, FullGauge, FullClover, FullClover,
			       cudaColorSpinorField &, double, cudaColorSpinorField &, MatPCType);
  friend void cloverMatCuda(cudaColorSpinorField &, FullGauge, FullClover,
			    cudaColorSpinorField &, double, cudaColorSpinorField &, int);
  friend void cloverCuda(cudaColorSpinorField &, FullGauge, FullClover, cudaColorSpinorField &, int);
  
 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool init;

  static void *buffer;// pinned memory
  static bool bufferInit;
  static size_t bufferBytes;

  void create(const FieldCreate);
  void destroy();
  void zero();
  void copy(const cudaColorSpinorField &);

 public:
  cudaColorSpinorField();
  cudaColorSpinorField(const cudaColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorField&, const ColorSpinorParam&);
  cudaColorSpinorField(const ColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorParam&);
  virtual ~cudaColorSpinorField();

  cudaColorSpinorField& operator=(const cudaColorSpinorField&);
  cudaColorSpinorField& operator=(const cpuColorSpinorField&);

  void loadCPUSpinorField(const cpuColorSpinorField &src);
  void saveCPUSpinorField (cpuColorSpinorField &src) const;

  cudaColorSpinorField& Even() const;
  cudaColorSpinorField& Odd() const;

  static void freeBuffer();

};


// CPU implementation
class cpuColorSpinorField : public ColorSpinorField {

  friend class cudaColorSpinorField;

  friend double normCpu(const cpuColorSpinorField &);
  friend double dslashCUDA();
  friend void dslashRef();

 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool init;

  void create(const FieldCreate);
  void destroy();
  void copy(const cpuColorSpinorField&);
  void zero();

 public:
  cpuColorSpinorField();
  cpuColorSpinorField(const cpuColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorParam&);
  virtual ~cpuColorSpinorField();

  cpuColorSpinorField& operator=(const cpuColorSpinorField&);
  cpuColorSpinorField& operator=(const cudaColorSpinorField&);

  //cpuColorSpinorField& Even() const;
  //cpuColorSpinorField& Odd() const;

  void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0);
  static void Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, const int resolution=1);
  void PrintVector(int vol);
};

#endif

/*

// experimenting with functors for arbitrary ordering
class spinorFunctor {

 protected:
  const void *v;
  const int nColor;
  const int nSpin;
  const int volume;
  const Precision precision;

 public:
 spinorFunctor(void *V, int Volume, int Nc, int Ns, Precision prec)
   : v(V), nColor(Nc), nSpin(Ns), volume(Volume), precision(prec) { ; }
  virtual ~spinorFunctor();
  // return element at parity p, linear index x, spin s, color c and complexity z
  
  virtual void* operator()(int p, int x, int s, int c, int z) const = 0;
};

// accessor for SPACE_SPIN_COLOR_ORDER
class SSCfunctor : public spinorFunctor {
  
 public:
 SSCfunctor(void *V, int volume, int Nc, int Ns, Precision prec)
   : spinorFunctor(V, volume, Nc, Ns, prec) { ; }
  virtual ~SSCfunctor();

  void* operator()(int p, int x, int s, int c, int z) const {
    switch (precision) {
      case QUDA_DOUBLE_PRECISION:
	return ((double*)v)+(((p*volume+x)*nSpin+s)*nColor+c)*2+z;
      case QUDA_SINGLE_PRECISION:
	return ((float*)v)+(((p*volume+x)*nSpin+s)*nColor+c)*2+z;
      default:
	errorQuda("Precision not defined");
    }
  }

};

// accessor for SPACE_COLOR_SPIN_ORDER
class SCSfunctor : public spinorFunctor {
  
 public:
  SCSfunctor(void *V, int volume, int Nc, int Ns, Precision prec)
    : spinorFunctor(V, volume, Nc, Ns, prec) { ; }
  virtual ~SCSfunctor();

  void* operator()(int p, int x, int s, int c, int z) const {
    switch (precision) {
      case QUDA_DOUBLE_PRECISION:
	return ((double*)v)+(((p*volume+x)*nColor+c)*nSpin+s)*2+z;
      case QUDA_SINGLE_PRECISION:
	return ((float*)v)+(((p*volume+x)*nColor+c)*nSpin+s)*2+z;
      default:
	errorQuda("Precision not defined");
    }
  }

};

*/


