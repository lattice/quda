#include <quda_internal.h>

#ifndef __COLORSPINORFIELD_H
#define __COLORSPINORFIELD_H

typedef enum FieldType_s {
  CPU_FIELD,
  CUDA_FIELD
} FieldType;

typedef enum FieldSubset_s {
  FULL_FIELD_SUBSET,
  EVEN_FIELD_SUBSET,
  ODD_FIELD_SUBSET
} FieldSubset;

typedef enum SubsetOrder_s {
  LEXICOGRAPHIC_SUBSET_ORDER, // lexicographic ordering
  EVEN_ODD_SUBSET_ORDER, // QUDA and qdp use this
  ODD_EVEN_SUBST_PARITY // cps uses this
} SubsetOrder;

// Unless otherwise stated, color runs faster than spin
typedef enum FieldOrder_s {
  FLOAT_ORDER, // spin-color-complex-space
  FLOAT2_ORDER, // (spin-color-complex)/2-space-(spin-color-complex)%2
  FLOAT4_ORDER, // (spin-color-complex)/4-space-(spin-color-complex)%4
  SPACE_SPIN_COLOR_ORDER,
  SPACE_COLOR_SPIN_ORDER // QLA ordering (spin inside color)
} FieldOrder;

typedef enum FieldCreate_s {
  CREATE_ZERO, // create new field
  CREATE_COPY, // create copy to field
  CREATE_REFERENCE // create reference to field
} FieldCreate;

typedef enum GammaBasis_s {
  DEGRAND_ROSSI_BASIS,
  UKQCD_BASIS
} GammaBasis;

typedef struct ColorSpinorParam_s {
  Precision prec; // Precision of the field
  int nColor; // Number of colors of the field
  int nSpin; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  int nDim; // number of spacetime dimensions
  int *x; // size of each dimension
  int pad; // volumetric padding
  FieldType fieldType; // cpu, cuda etc. 

  FieldSubset fieldSubset; // Full, even or odd
  SubsetOrder subsetOrder; // defined for full fields

  FieldOrder fieldOrder; // Float, Float2, Float4 etc.
  GammaBasis basis;
  FieldCreate create; // 

} ColorSpinorParam;

class ColorSpinorField {

 private:
  bool init;
  void create(int nDim, int *x, int Nc, int Ns, Precision precision, 
	      int pad, FieldType type, FieldSubset subset, 
	      SubsetOrder subsetOrder, FieldOrder order, GammaBasis basis);
  void destroy();  

 protected:
  Precision prec;

  int nColor;
  int nSpin;
  
  int nDim;
  int *x;

  int volume;
  int pad;
  int stride;
  
  int real_length;
  int length;
  size_t bytes;
  
  FieldType type;
  FieldSubset subset;
  SubsetOrder subset_order;
  FieldOrder order;
  GammaBasis basis;
  
 public:
  ColorSpinorField();
  ColorSpinorField(const ColorSpinorField &);
  ColorSpinorField(const ColorSpinorParam &);

  virtual ~ColorSpinorField();

  ColorSpinorField& operator=(const ColorSpinorField &);

  Precision precision() const { return prec; }
  int Ncolor() const { return nColor; } 
  int Nspin() const { return nSpin; } 

  FieldType fieldType() const { return type; }
  FieldSubset fieldSubset() const { return subset; }
  SubsetOrder subsetOrder() const { return subset_order; }
  FieldOrder fieldOrder() const { return order; }
  GammaBasis gammaBasis() const { return basis; }

  virtual ColorSpinorField& operator+=(const ColorSpinorField&) = 0;
};

class cpuColorSpinorField;

// CUDA implementation
class cudaColorSpinorField : public ColorSpinorField {

 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool init;

  static void *buffer;// pinned memory
  static bool bufferInit;
  static size_t bufferBytes;

 public:
  cudaColorSpinorField();
  cudaColorSpinorField(const cudaColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorParam&);
  virtual ~cudaColorSpinorField();

  void create();
  void destroy();

  ColorSpinorField& operator+=(const ColorSpinorField&);

  void loadCPUSpinorField(const cpuColorSpinorField &src);
  void saveCPUSpinorField (cpuColorSpinorField &src) const;
};


// CPU implementation
class cpuColorSpinorField : public ColorSpinorField {

  friend void cudaColorSpinorField::loadCPUSpinorField(const cpuColorSpinorField &);
  friend void cudaColorSpinorField::saveCPUSpinorField(cpuColorSpinorField &) const;

 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool init;

 public:
  cpuColorSpinorField();
  cpuColorSpinorField(const cpuColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorParam&);
  virtual ~cpuColorSpinorField();

  void create();
  void destroy();

  ColorSpinorField& operator+=(const ColorSpinorField&);

};

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

#endif
