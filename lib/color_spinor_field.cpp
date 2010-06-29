#include <color_spinor_field.h>
#include <string.h>
#include <iostream>

/*ColorSpinorField::ColorSpinorField() : init(false) {

}*/

ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) : init(false), even(0), odd(0) {
  create(param.nDim, param.x, param.nColor, param.nSpin, param.precision, param.pad, 
	 param.fieldLocation, param.siteSubset, param.siteOrder, param.fieldOrder, 
	 param.gammaBasis);
}

ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) : init(false), even(0), odd(0) {
  create(field.nDim, field.x, field.nColor, field.nSpin, field.precision, field.pad,
	 field.fieldLocation, field.siteSubset, field.siteOrder, field.fieldOrder, 
	 field.gammaBasis);
}

ColorSpinorField::~ColorSpinorField() {
  destroy();
}

void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, QudaPrecision Prec, 
			      int Pad, QudaFieldLocation fieldLocation, 
			      QudaSiteSubset siteSubset, QudaSiteOrder siteOrder, 
			      QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis) {
  if (Ndim > QUDA_MAX_DIM){
    errorQuda("Number of dimensions nDim = %d too great", Ndim);
  }
  nDim = Ndim;
  nColor = Nc;
  nSpin = Ns;

  precision = Prec;
  volume = 1;
  for (int d=0; d<nDim; d++) {
    x[d] = X[d];
    volume *= x[d];
  }
  pad = Pad;
  
  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    stride = volume/2 + pad; // padding is based on half volume
    length = 2*stride*nColor*nSpin*2;
    
  } else {
    stride = volume + pad;
    length = stride*nColor*nSpin*2;
  }

  real_length = volume*nColor*nSpin*2;

  bytes = length * precision;

  this->fieldLocation = fieldLocation;
  this->siteSubset = siteSubset;
  this->siteOrder = siteOrder;
  this->fieldOrder = fieldOrder;
  this->gammaBasis = gammaBasis;

  init = true;
}

void ColorSpinorField::destroy() {
  init = false;
}

ColorSpinorField& ColorSpinorField::operator=(const ColorSpinorField &src) {
  if (&src != this) {
    create(src.nDim, src.x, src.nColor, src.nSpin, src.precision, src.pad,
	   src.fieldLocation, src.siteSubset, src.siteOrder, src.fieldOrder, 
	   src.gammaBasis);    
  }
  return *this;
}

// Resets the attributes of this field if param disagrees (and is defined)
void ColorSpinorField::reset(const ColorSpinorParam &param) {

  if (param.nColor != 0) nColor = param.nColor;
  if (param.nSpin != 0) nSpin = param.nSpin;

  if (param.precision != QUDA_INVALID_PRECISION)  precision = param.precision;
  if (param.nDim != 0) nDim = param.nDim;

  volume = 1;
  for (int d=0; d<nDim; d++) {
    if (param.x[0] != 0) x[d] = param.x[d];
    volume *= x[d];
  }
  
  if (param.pad != 0) pad = param.pad;

  if (param.siteSubset == QUDA_FULL_SITE_SUBSET){
    stride = volume/2 + pad;
    length = 2*stride*nColor*nSpin*2;
  }else if (param.siteSubset == QUDA_PARITY_SITE_SUBSET){
    stride = volume + pad;
    length = stride*nColor*nSpin*2;  
  }else{
    //do nothing, not an error
  }

  real_length = volume*nColor*nSpin*2;

  bytes = length * precision;

  if (param.fieldLocation != QUDA_INVALID_FIELD_LOCATION) fieldLocation = param.fieldLocation;
  if (param.siteSubset != QUDA_INVALID_SITE_SUBSET) siteSubset = param.siteSubset;
  if (param.siteOrder != QUDA_INVALID_SITE_ORDER) siteOrder = param.siteOrder;
  if (param.fieldOrder != QUDA_INVALID_FIELD_ORDER) fieldOrder = param.fieldOrder;
  if (param.gammaBasis != QUDA_INVALID_GAMMA_BASIS) gammaBasis = param.gammaBasis;

  if (!init) errorQuda("Shouldn't be resetting a non-inited field\n");
}

// Fills the param with the contents of this field
void ColorSpinorField::fill(ColorSpinorParam &param) {
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.precision = precision;
  param.nDim = nDim;
  memcpy(param.x, x, QUDA_MAX_DIM*sizeof(int));
  param.pad = pad;
  param.fieldLocation = fieldLocation;
  param.siteSubset = siteSubset;
  param.siteOrder = siteOrder;
  param.fieldOrder = fieldOrder;
  param.gammaBasis = gammaBasis;
  param.create = QUDA_INVALID_FIELD_CREATE;
}

// For kernels with precision conversion built in
void ColorSpinorField::checkField(const ColorSpinorField &a, const ColorSpinorField &b) {
  if (a.Length() != b.Length()) {
    errorQuda("checkSpinor: lengths do not match: %d %d", a.Length(), b.Length());
  }

  if (a.Ncolor() != b.Ncolor()) {
    errorQuda("checkSpinor: colors do not match: %d %d", a.Ncolor(), b.Ncolor());
  }

  if (a.Nspin() != b.Nspin()) {
    errorQuda("checkSpinor: spins do not match: %d %d", a.Nspin(), b.Nspin());
  }
}

double norm2(const ColorSpinorField &a) {

  if (a.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
    return normCuda(dynamic_cast<const cudaColorSpinorField&>(a));
  } else if (a.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
    return normCpu(dynamic_cast<const cpuColorSpinorField&>(a));
  } else {
    errorQuda("Field type %d not supported", a.FieldLocation());
  }

}
