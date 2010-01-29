#include <color_spinor_field.h>

ColorSpinorField::ColorSpinorField() : init(false) {

}

ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) : init(false) {
  create(param.nDim, param.x, param.nColor, param.nSpin, param.precision, param.pad, 
	 param.fieldType, param.fieldSubset, param.subsetOrder, param.fieldOrder, param.basis);
}

ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) : init(false) {
  create(field.nDim, field.x, field.nColor, field.nSpin, field.precision, field.pad,
	 field.type, field.subset, field.subset_order, field.order, field.basis);
}

ColorSpinorField::~ColorSpinorField() {
  destroy();
}

void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, QudaPrecision Prec, 
			      int Pad, FieldType Type, FieldSubset Subset, 
			      SubsetOrder Subset_order, QudaColorSpinorOrder Order,
			      GammaBasis Basis) {
  if (Ndim > QUDA_MAX_DIM) errorQuda("Number of dimensions nDim = %d too great", nDim);
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
  stride = volume + pad;

  length = stride*nColor*nSpin*2;
  real_length = volume*nColor*nSpin*2;

  bytes = length * precision;

  type = Type;
  subset = Subset;
  subset_order = Subset_order;
  order = Order;

  basis = Basis;

  init = true;
}

void ColorSpinorField::destroy() {
}

ColorSpinorField& ColorSpinorField::operator=(const ColorSpinorField &src) {
  if (&src != this) {
    create(src.nDim, src.x, src.nColor, src.nSpin, src.precision, src.pad,
	   src.type, src.subset, src.subset_order, src.order, src.basis);    
  }
  return *this;
}

// Resets the attributes of this field if param disagrees (and is defined)
void ColorSpinorField::reset(const ColorSpinorParam &param) {

  if (param.nColor != 0) nColor = param.nColor;
  if (param.nSpin != 0) nSpin = param.nSpin;

  if (param.precision != QUDA_INVALID_PRECISION)  precision = param.precision;
  if (param.nDim != 0 && nDim != param.nDim) {
    nDim = param.nDim;
  }

  // only check the that the first dimension is non-zero
  if (param.x[0] != 0) {
    volume = 1;
    for (int d=0; d<nDim; d++) {
      x[d] = param.x[d];
      volume *= x[d];
    }
  }
  
  if (param.pad != 0) pad = param.pad;
  stride = volume + pad;

  length = stride*nColor*nSpin*2;
  real_length = volume*nColor*nSpin*2;

  bytes = length * precision;

  if (param.fieldType != QUDA_INVALID_FIELD) type = param.fieldType;
  if (param.fieldSubset != QUDA_INVALID_SUBSET) subset = param.fieldSubset;
  if (param.subsetOrder != QUDA_INVALID_SUBSET_ORDER) subset_order = param.subsetOrder;
  if (param.fieldOrder != QUDA_INVALID_ORDER) order = param.fieldOrder;

  if (param.basis != QUDA_INVALID_BASIS) basis = param.basis;
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

