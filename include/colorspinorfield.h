typedef enum FieldLocation_s {
  CPU_FIELD,
  CUDA_FIELD
} FieldLocation;

typedef enum FieldType_s {
  FULL_FIELD,
  EVEN_FIELD,
  ODD_FIELD
} FieldType;

// Unless otherwise stated, color runs faster than spin
typedef enum FieldOrder_s {
  FLOAT_ORDER,
  FLOAT2_ORDER,
  FLOAT4_ORDER,
  FLOATN_ORDER, // One array of Nc * Ns complex elements
  FLOATN_QLA_ORDER // As above but spin runs faster than color
} FieldOrder;

typedef enum FieldCreate_s {
  CREATE_ZERO, // create new field
  CREATE_COPY, // create copy to field
  CREATE_REFERENCE // create reference to field
} FieldCreate;

typedef struct ColorSpinorParam_s {
  Precision prec; // Precision of the field
  int nColor; // Number of colors of the field
  int nSpin; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor
  int nDim; // number of spacetime dimensions
  int *x; // size of each dimension
  int pad; // volumetric padding
  FieldLocation fieldLocation; // cpu, cuda etc. 
  FieldType fieldType; // Full, even or odd
  FieldOrder fieldOrder; // Float, Float2, Float4 etc.

  FieldCreate create; // 
  void *v; // Pointer from which to copy data or to reference
} ColorSpinorParam;

class ColorSpinorField {

 protected:
  Precision prec;
  void *v; // the field elements
  void *norm; // the normalization field

  int nColor;
  int nSpin;
  
  int nDim;
  int *x;

  int volume;
  int pad;
  int stride;
  
  int length;
  
  FieldLocation fieldLocation;
  FieldType fieldType;
  FieldOrder fieldOrder;
  
 public:
  ColorSpinorField();
  ColorSpinorField(const ColorSpinorField &);

  ColorSpinorField(const ColorSpinorParam &);

  virtual ~ColorSpinorField();
  
  ColorSpinorField& operator=(const ColorSpinorField &);

  Precision precision() const { return prec; }
  int Ncolor() const { return nColor; } 
  int Nspin() const { return nSpin; } 

  virtual ColorSpinorField& operator+=(const ColorSpinorField&) = 0;
};

// CUDA implementation
class cudaColorSpinorField : public ColorSpinorField {

 private:

 public:
  cudaColorSpinorField();
  cudaColorSpinorField(const cudaColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorParam&);
  virtual ~cudaColorSpinorField();

  ColorSpinorField& operator+=(const ColorSpinorField&);
};


// CPU implementation
class cpuColorSpinorField : public ColorSpinorField {

 private:

 public:
  cpuColorSpinorField();
  cpuColorSpinorField(const cpuColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorParam&);
  virtual ~cpuColorSpinorField();

  ColorSpinorField& operator+=(const ColorSpinorField&);
};
