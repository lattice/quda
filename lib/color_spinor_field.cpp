#include <colorspinorfield.h>

ColorSpinorField::ColorSpinorField() : init(false) {

}

ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) : init(false) {
  create(param.nDim, param.x, param.nColor, param.nSpin, param.prec, param.pad, 
	 param.fieldType, param.fieldSubset, param.subsetOrder, param.fieldOrder, param.basis);
}

ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) : init(false) {
  create(field.nDim, field.x, field.nColor, field.nSpin, field.prec, field.pad,
	 field.type, field.subset, field.subset_order, field.order, field.basis);
}

ColorSpinorField::~ColorSpinorField() {
  destroy();
}

void ColorSpinorField::create(int Ndim, int *X, int Nc, int Ns, Precision Prec, 
			      int Pad, FieldType Type, FieldSubset Subset, 
			      SubsetOrder Subset_order, FieldOrder Order,
			      GammaBasis Basis) {
  nDim = Ndim;
  nColor = Nc;
  nSpin = Ns;

  prec = Prec;
  volume = 1;
  x = new int[nDim];
  for (int d=0; d<nDim; d++) {
    x[d] = X[d];
    volume *= x[d];
  }
  pad = Pad;
  stride = volume + pad;

  length = stride*nColor*nSpin*2;
  real_length = volume*nColor*nSpin*2;

  bytes = length * prec;

  type = Type;
  subset = Subset;
  subset_order = Subset_order;
  order = Order;

  basis = Basis;

  init = true;
}

void ColorSpinorField::destroy() {
  if (init) delete []x;
}
