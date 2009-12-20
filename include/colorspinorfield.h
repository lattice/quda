typedef enum FieldLocation_s {
  CPU_FIELD,
  CUDA_FIELD
} FieldLocation;

typedef enum FieldType_s {
  FULL_FIELD,
  EVEN_FIELD,
  ODD_FIELD
} FieldType;

typedef enum FieldOrder_s {
  FLOAT_ORDER,
  FLOAT2_ORDER,
  FLOAT4_ORDER,
  COLOR_SPIN_ORDER, // 
  SPIN_COLOR_ORDER  //
} FieldOrder;

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
};

class cudaColorSpinorField : public ColorSpinorField {

 private:

 public:
  cudaColorSpinorField();
  cudaColorSpinorField(const cudaColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorField&);
  cudaColorSpinorField(const ColorSpinorParam&);
  virtual ~cudaColorSpinorField();

};

class cpuColorSpinorField : public ColorSpinorField {

 private:

 public:
  cpuColorSpinorField();
  cpuColorSpinorField(const cpuColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorField&);
  cpuColorSpinorField(const ColorSpinorParam&);
  virtual ~cpuColorSpinorField();

};
