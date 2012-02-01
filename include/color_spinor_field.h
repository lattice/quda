#include <quda_internal.h>
#include <quda.h>

#include <iostream>
#include <complex>

namespace quda {
  typedef std::complex<double> Complex;
}

#ifndef _COLOR_SPINOR_FIELD_H
#define _COLOR_SPINOR_FIELD_H

// Probably want some checking for this limit
#define QUDA_MAX_DIM 6

#include <lattice_field.h>

struct FullClover;

class ColorSpinorParam {
 public:
  QudaFieldLocation fieldLocation; // cpu, cuda etc. 
  int nColor; // Number of colors of the field
  int nSpin; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  int nDim; // number of spacetime dimensions
  int x[QUDA_MAX_DIM]; // size of each dimension
  QudaPrecision precision; // Precision of the field
  int pad; // volumetric padding

  QudaTwistFlavorType twistFlavor; // used by twisted mass

  QudaSiteSubset siteSubset; // Full, even or odd
  QudaSiteOrder siteOrder; // defined for full fields
  
  QudaFieldOrder fieldOrder; // Float, Float2, Float4 etc.
  QudaGammaBasis gammaBasis;
  QudaFieldCreate create; // 

  void *v; // pointer to field
  void *norm;

  ColorSpinorParam(const ColorSpinorField &a);

  QudaVerbosity verbose;

 ColorSpinorParam()
   : fieldLocation(QUDA_INVALID_FIELD_LOCATION), nColor(0), nSpin(0), nDim(0), 
    precision(QUDA_INVALID_PRECISION), pad(0), twistFlavor(QUDA_TWIST_INVALID),
    siteSubset(QUDA_INVALID_SITE_SUBSET), siteOrder(QUDA_INVALID_SITE_ORDER), 
    fieldOrder(QUDA_INVALID_FIELD_ORDER), gammaBasis(QUDA_INVALID_GAMMA_BASIS), 
    create(QUDA_INVALID_FIELD_CREATE), verbose(QUDA_SILENT)
    { 
      for(int d=0; d<QUDA_MAX_DIM; d++) {
	x[d] = 0; 
      }
    }
  
  // used to create cpu params
 ColorSpinorParam(void *V, QudaInvertParam &inv_param, const int *X, const bool pc_solution)
   : fieldLocation(QUDA_CPU_FIELD_LOCATION), nColor(3), 
    nSpin(inv_param.dslash_type == QUDA_ASQTAD_DSLASH ? 1 : 4), nDim(4), 
    precision(inv_param.cpu_prec), pad(0), twistFlavor(inv_param.twist_flavor), 
    siteSubset(QUDA_INVALID_SITE_SUBSET), siteOrder(QUDA_INVALID_SITE_ORDER), 
    fieldOrder(QUDA_INVALID_FIELD_ORDER), gammaBasis(inv_param.gamma_basis), 
    create(QUDA_REFERENCE_FIELD_CREATE), v(V), verbose(inv_param.verbosity)
  { 

    if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
    for (int d=0; d<nDim; d++) x[d] = X[d];

    if (!pc_solution) {
      siteSubset = QUDA_FULL_SITE_SUBSET;;
    } else {
      x[0] /= 2; // X defined the full lattice dimensions
      siteSubset = QUDA_PARITY_SITE_SUBSET;
    }

    if (inv_param.dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      nDim++;
      x[4] = inv_param.Ls;
    }

    if (inv_param.dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      siteOrder = QUDA_ODD_EVEN_SITE_ORDER;
    } else if (inv_param.dirac_order == QUDA_QDP_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
      siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    } else if (inv_param.dirac_order == QUDA_DIRAC_ORDER) {
      fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    } else {
      errorQuda("Dirac order %d not supported", inv_param.dirac_order);
    }
  }

  // used to create cuda param from a cpu param
 ColorSpinorParam(ColorSpinorParam &cpuParam, QudaInvertParam &inv_param) 
    : fieldLocation(QUDA_CUDA_FIELD_LOCATION), nColor(cpuParam.nColor), nSpin(cpuParam.nSpin), 
    nDim(cpuParam.nDim), precision(inv_param.cuda_prec), pad(inv_param.sp_pad),  
    twistFlavor(cpuParam.twistFlavor), siteSubset(cpuParam.siteSubset), 
    siteOrder(QUDA_EVEN_ODD_SITE_ORDER), fieldOrder(QUDA_INVALID_FIELD_ORDER), 
    gammaBasis(nSpin == 4? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS), 
    create(QUDA_COPY_FIELD_CREATE), v(0), verbose(cpuParam.verbose)
  {
    if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
    for (int d=0; d<nDim; d++) x[d] = cpuParam.x[d];

    if (precision == QUDA_DOUBLE_PRECISION || nSpin == 1) {
      fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }

  }

  void print() {
    printfQuda("fieldLocation = %d\n", fieldLocation);
    printfQuda("nColor = %d\n", nColor);
    printfQuda("nSpin = %d\n", nSpin);
    printfQuda("twistFlavor = %d\n", twistFlavor);
    printfQuda("nDim = %d\n", nDim);
    for (int d=0; d<nDim; d++) printfQuda("x[%d] = %d\n", d, x[d]);
    printfQuda("precision = %d\n", precision);
    printfQuda("pad = %d\n", pad);
    printfQuda("siteSubset = %d\n", siteSubset);
    printfQuda("siteOrder = %d\n", siteOrder);
    printfQuda("fieldOrder = %d\n", fieldOrder);
    printfQuda("gammaBasis = %d\n", gammaBasis);
    printfQuda("create = %d\n", create);
    printfQuda("v = %lx\n", (unsigned long)v);
    printfQuda("norm = %lx\n", (unsigned long)norm);
  }

  virtual ~ColorSpinorParam() {
  }
};

class cpuColorSpinorField;
class cudaColorSpinorField;

class ColorSpinorField {

 private:
  void create(int nDim, const int *x, int Nc, int Ns, QudaTwistFlavorType Twistflavor, 
	      QudaPrecision precision, int pad, QudaFieldLocation location, QudaSiteSubset subset, 
	      QudaSiteOrder siteOrder, QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis);
  void destroy();  

  QudaVerbosity verbose;

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

  QudaTwistFlavorType twistFlavor;
  
  int real_length; // physical length only
  int length; // length including pads, but not ghost zone - used for BLAS

  // multi-GPU parameters
  int ghostFace[QUDA_MAX_DIM];// the size of each face
  int ghostOffset[QUDA_MAX_DIM]; // offsets to each ghost zone
  int ghostNormOffset[QUDA_MAX_DIM]; // offsets to each ghost zone for norm field

  int ghost_length; // length of ghost zone
  int ghost_norm_length; // length of ghost zone for norm
  int total_length; // total length of spinor (physical + pad + ghost)
  int total_norm_length; // total length of norm

  size_t bytes; // size in bytes of spinor field
  size_t norm_bytes; // size in bytes of norm field

  QudaFieldLocation fieldLocation;
  QudaSiteSubset siteSubset;
  QudaSiteOrder siteOrder;
  QudaFieldOrder fieldOrder;
  QudaGammaBasis gammaBasis;
  
  // in the case of full fields, these are references to the even / odd sublattices
  ColorSpinorField *even;
  ColorSpinorField *odd;

  void createGhostZone();

  // resets the above attributes based on contents of param
  void reset(const ColorSpinorParam &);
  void fill(ColorSpinorParam &) const;
  static void checkField(const ColorSpinorField &, const ColorSpinorField &);

 public:
  //ColorSpinorField();
  ColorSpinorField(const ColorSpinorField &);
  ColorSpinorField(const ColorSpinorParam &);

  virtual ~ColorSpinorField();

  ColorSpinorField& operator=(const ColorSpinorField &);

  QudaPrecision Precision() const { return precision; }
  int Ncolor() const { return nColor; } 
  int Nspin() const { return nSpin; } 
  int TwistFlavor() const { return twistFlavor; } 
  int Ndim() const { return nDim; }
  const int* X() const { return x; }
  int X(int d) const { return x[d]; }
  int RealLength() const { return real_length; }
  int Length() const { return length; }
  int Stride() const { return stride; }
  int Volume() const { return volume; }
  size_t Bytes() const { return bytes; }
  size_t NormBytes() const { return norm_bytes; }
  void PrintDims() const { printf("dimensions=%d %d %d %d\n",
				  x[0], x[1], x[2], x[3]);}
  
  QudaFieldLocation FieldLocation() const { return fieldLocation; }
  QudaSiteSubset SiteSubset() const { return siteSubset; }
  QudaSiteOrder SiteOrder() const { return siteOrder; }
  QudaFieldOrder FieldOrder() const { return fieldOrder; }
  QudaGammaBasis GammaBasis() const { return gammaBasis; }

  int GhostLength() const { return ghost_length; }
  const int *GhostFace() const { return ghostFace; }  
  int GhostOffset(const int i) const { return ghostOffset[i]; }  
  int GhostNormOffset(const int i ) const { return ghostNormOffset[i]; }  

  friend std::ostream& operator<<(std::ostream &out, const ColorSpinorField &);
  friend class ColorSpinorParam;
};

// CUDA implementation
class cudaColorSpinorField : public ColorSpinorField {

  friend class cpuColorSpinorField;

 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool alloc; // whether we allocated memory
  bool init;

  static void *buffer;// pinned memory
  static bool bufferInit;
  static size_t bufferBytes;

  static void* fwdGhostFaceBuffer[QUDA_MAX_DIM]; //gpu memory
  static void* backGhostFaceBuffer[QUDA_MAX_DIM]; //gpu memory
  static int initGhostFaceBuffer;
  static QudaPrecision facePrecision;

  void create(const QudaFieldCreate);
  void destroy();
  void copy(const cudaColorSpinorField &);

 public:
  //cudaColorSpinorField();
  cudaColorSpinorField(const cudaColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorField&, const ColorSpinorParam&);
  cudaColorSpinorField(const ColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorParam&);
  virtual ~cudaColorSpinorField();

  cudaColorSpinorField& operator=(const cudaColorSpinorField&);
  cudaColorSpinorField& operator=(const cpuColorSpinorField&);

  void loadCPUSpinorField(const cpuColorSpinorField &src);
  void saveCPUSpinorField (cpuColorSpinorField &src) const;

  void allocateGhostBuffer(void);
  static void freeGhostBuffer(void);

  void packGhost(const int dim, const QudaDirection dir, const QudaParity parity, 
		 const int dagger, cudaStream_t* stream);
  void sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir,
		 const int dagger, cudaStream_t *stream);
  void unpackGhost(void* ghost_spinor, const int dim, const QudaDirection dir, 
		   const int dagger, cudaStream_t* stream);

  void* V() {return v;}
  const void* V() const {return v;}
  void* Norm(){return norm;}
  const void* Norm() const {return norm;}

  cudaColorSpinorField& Even() const;
  cudaColorSpinorField& Odd() const;

  static void freeBuffer();

  void zero();
};

// Forward declaration of accessor functors
template <typename Float> class ColorSpinorFieldOrder;
template <typename Float> class SpaceColorSpinOrder;
template <typename Float> class SpaceSpinColorOrder;
template <typename Float> class QOPDomainWallOrder;

// CPU implementation
class cpuColorSpinorField : public ColorSpinorField {

  friend class cudaColorSpinorField;

  template <typename Float> friend class SpaceColorSpinOrder;
  template <typename Float> friend class SpaceSpinColorOrder;
  template <typename Float> friend class QOPDomainWallOrder;

 public:
  static void* fwdGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
  static void* backGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
  static void* fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
  static void* backGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
  static int initGhostFaceBuffer;

 private:
  void *v; // the field elements
  void *norm; // the normalization field
  bool init;
  
  void create(const QudaFieldCreate);
  void destroy();

  void createOrder(); // create the accessor for a given field ordering
  ColorSpinorFieldOrder<double> *order_double; // accessor functor used to access fp64 elements
  ColorSpinorFieldOrder<float> *order_single; // accessor functor used to access fp32 elements

 public:
  //cpuColorSpinorField();
  cpuColorSpinorField(const cpuColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorParam&);
  virtual ~cpuColorSpinorField();

  cpuColorSpinorField& operator=(const cpuColorSpinorField&);
  cpuColorSpinorField& operator=(const cudaColorSpinorField&);

  //cpuColorSpinorField& Even() const;
  //cpuColorSpinorField& Odd() const;

  void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0);
  static int Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, const int resolution=1);
  void PrintVector(unsigned int x);

  void allocateGhostBuffer(void);
  static void freeGhostBuffer(void);
	
  void packGhost(void* ghost_spinor, const int dim, 
		 const QudaDirection dir, const QudaParity parity, const int dagger);
  void unpackGhost(void* ghost_spinor, const int dim, 
		   const QudaDirection dir, const int dagger);
  
  void* V() { return v; }
  const void * V() const { return v; }

  void copy(const cpuColorSpinorField&);
  void zero();
};



#endif // _COLOR_SPINOR_FIELD_H
