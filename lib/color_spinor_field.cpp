#include <color_spinor_field.h>
#include <string.h>
#include <iostream>
#include <face_quda.h>

// forward declarations
double normCpu(const cpuColorSpinorField &b);
double normCuda(const cudaColorSpinorField &b);


/*ColorSpinorField::ColorSpinorField() : init(false) {

}*/

ColorSpinorParam::ColorSpinorParam(const ColorSpinorField &field) {
  field.fill(*this);
}

ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) : verbose(param.verbose), init(false), 
								    even(0), odd(0) 
{
  create(param.nDim, param.x, param.nColor, param.nSpin, param.twistFlavor, param.precision, param.pad, 
	 param.fieldLocation, param.siteSubset, param.siteOrder, param.fieldOrder, 
	 param.gammaBasis);

}

ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) : verbose(field.verbose), init(false),
								    even(0), odd(0)
{
  create(field.nDim, field.x, field.nColor, field.nSpin, field.twistFlavor, field.precision, field.pad,
	 field.fieldLocation, field.siteSubset, field.siteOrder, field.fieldOrder, 
	 field.gammaBasis);

}

ColorSpinorField::~ColorSpinorField() {
  destroy();
}

void ColorSpinorField::createGhostZone() {

  if (verbose == QUDA_DEBUG_VERBOSE && fieldLocation == QUDA_CUDA_FIELD_LOCATION) 
    printfQuda("Location = %d, Precision = %d, Subset = %d\n", fieldLocation, precision, siteSubset);

  int num_faces = 1;
  int num_norm_faces=2;
  if (nSpin == 1) { //staggered
    num_faces=6;
    num_norm_faces=6;
  }

  // calculate size of ghost zone required
  int ghostVolume = 0;
  for (int i=0; i<nDim; i++) {
    ghostFace[i] = 0;
    if (commDimPartitioned(i)) {
      ghostFace[i] = 1;
      for (int j=0; j<nDim; j++) {
	if (i==j) continue;
	ghostFace[i] *= x[j];
      }
      if (i==0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
      if (siteSubset == QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
      ghostVolume += ghostFace[i];
    }
    if(i==0){
      ghostOffset[i] = 0;
      ghostNormOffset[i] = 0;
    }else{
      ghostOffset[i] = ghostOffset[i-1] + num_faces*ghostFace[i-1];
      ghostNormOffset[i] = ghostNormOffset[i-1] + num_norm_faces*ghostFace[i-1];
    }

    if (verbose == QUDA_DEBUG_VERBOSE && fieldLocation == QUDA_CUDA_FIELD_LOCATION) 
      printfQuda("face %d = %6d commDimPartitioned = %6d ghostOffset = %6d ghostNormOffset = %6d\n", 
		 i, ghostFace[i], commDimPartitioned(i), ghostOffset[i], ghostNormOffset[i]);
  }
  int ghostNormVolume = num_norm_faces * ghostVolume;
  ghostVolume *= num_faces;

  if (verbose == QUDA_DEBUG_VERBOSE && fieldLocation == QUDA_CUDA_FIELD_LOCATION) 
    printfQuda("Allocated ghost volume = %d, ghost norm volume %d\n", ghostVolume, ghostNormVolume);

// ghost zones are calculated on c/b volumes
#ifdef MULTI_GPU
  ghost_length = ghostVolume*nColor*nSpin*2; 
  ghost_norm_length = (precision == QUDA_HALF_PRECISION) ? ghostNormVolume : 0;
#else
  ghost_length = 0;
  ghost_norm_length = 0;
#endif

  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    total_length = length + 2*ghost_length; // 2 ghost zones in a full field
    total_norm_length = 2*(stride + ghost_norm_length); // norm length = 2*stride
  } else {
    total_length = length + ghost_length;
    total_norm_length = (precision == QUDA_HALF_PRECISION) ? stride + ghost_norm_length : 0; // norm length = stride
  }

  // no ghost zones for cpu fields (yet?)
  if (fieldLocation == QUDA_CPU_FIELD_LOCATION) {
    ghost_length = 0;
    ghost_norm_length = 0;
    total_length = length;
    total_norm_length = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*stride : stride;
  }

  if (precision != QUDA_HALF_PRECISION) total_norm_length = 0;

  if (verbose == QUDA_DEBUG_VERBOSE && fieldLocation == QUDA_CUDA_FIELD_LOCATION) {
    printfQuda("ghost length = %d, ghost norm length = %d\n", ghost_length, ghost_norm_length);
    printfQuda("total length = %d, total norm length = %d\n", total_length, total_norm_length);
  }
}

void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, QudaTwistFlavorType Twistflavor, 
			      QudaPrecision Prec, int Pad, QudaFieldLocation fieldLocation, 
			      QudaSiteSubset siteSubset, QudaSiteOrder siteOrder, 
			      QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis) {
  this->fieldLocation = fieldLocation;
  this->siteSubset = siteSubset;
  this->siteOrder = siteOrder;
  this->fieldOrder = fieldOrder;
  this->gammaBasis = gammaBasis;

  if (Ndim > QUDA_MAX_DIM){
    errorQuda("Number of dimensions nDim = %d too great", Ndim);
  }
  nDim = Ndim;
  nColor = Nc;
  nSpin = Ns;
  twistFlavor = Twistflavor;

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

  real_length = volume*nColor*nSpin*2; // physical length

  createGhostZone();

  bytes = total_length * precision; // includes pads and ghost zones
  bytes = ALIGNMENT_ADJUST(bytes);
  norm_bytes = total_norm_length * sizeof(float);
  norm_bytes = ALIGNMENT_ADJUST(norm_bytes);
  init = true;
}

void ColorSpinorField::destroy() {
  init = false;
}

ColorSpinorField& ColorSpinorField::operator=(const ColorSpinorField &src) {
  if (&src != this) {
    create(src.nDim, src.x, src.nColor, src.nSpin, src.twistFlavor, 
	   src.precision, src.pad, src.fieldLocation, src.siteSubset, 
	   src.siteOrder, src.fieldOrder, src.gammaBasis);    
  }
  return *this;
}

// Resets the attributes of this field if param disagrees (and is defined)
void ColorSpinorField::reset(const ColorSpinorParam &param) {

  if (param.fieldLocation != QUDA_INVALID_FIELD_LOCATION) fieldLocation = param.fieldLocation;

  if (param.nColor != 0) nColor = param.nColor;
  if (param.nSpin != 0) nSpin = param.nSpin;
  if (param.twistFlavor != QUDA_TWIST_INVALID) twistFlavor = param.twistFlavor;

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
  } else if (param.siteSubset == QUDA_PARITY_SITE_SUBSET){
    stride = volume + pad;
    length = stride*nColor*nSpin*2;  
  } else {
    //errorQuda("SiteSubset not defined %d", param.siteSubset);
    //do nothing, not an error (can't remember why - need to document this sometime! )
  }

  if (param.siteSubset != QUDA_INVALID_SITE_SUBSET) siteSubset = param.siteSubset;
  if (param.siteOrder != QUDA_INVALID_SITE_ORDER) siteOrder = param.siteOrder;
  if (param.fieldOrder != QUDA_INVALID_FIELD_ORDER) fieldOrder = param.fieldOrder;
  if (param.gammaBasis != QUDA_INVALID_GAMMA_BASIS) gammaBasis = param.gammaBasis;

  createGhostZone();

  real_length = volume*nColor*nSpin*2;

  bytes = total_length * precision;
  bytes = ALIGNMENT_ADJUST(bytes);
  norm_bytes = total_norm_length * sizeof(float);
  norm_bytes = ALIGNMENT_ADJUST(norm_bytes);
  if (!init) errorQuda("Shouldn't be resetting a non-inited field\n");
}

// Fills the param with the contents of this field
void ColorSpinorField::fill(ColorSpinorParam &param) const {
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.twistFlavor = twistFlavor;
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
  param.verbose = verbose;
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

  if (a.TwistFlavor() != b.TwistFlavor()) {
    errorQuda("checkSpinor: twist flavors do not match: %d %d", a.TwistFlavor(), b.TwistFlavor());
  }
}

double norm2(const ColorSpinorField &a) {

  double rtn = 0.0;
  if (a.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
    rtn = normCuda(dynamic_cast<const cudaColorSpinorField&>(a));
  } else if (a.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
    rtn = normCpu(dynamic_cast<const cpuColorSpinorField&>(a));
  } else {
    errorQuda("Field type %d not supported", a.FieldLocation());
  }

  return rtn;
}

std::ostream& operator<<(std::ostream &out, const ColorSpinorField &a) {
  out << "fieldLocation = " << a.fieldLocation << std::endl;
  out << "nColor = " << a.nColor << std::endl;
  out << "nSpin = " << a.nSpin << std::endl;
  out << "twistFlavor = " << a.twistFlavor << std::endl;
  out << "nDim = " << a.nDim << std::endl;
  for (int d=0; d<a.nDim; d++) out << "x[" << d << "] = " << a.x[d] << std::endl;
  out << "volume = " << a.volume << std::endl;
  out << "precision = " << a.precision << std::endl;
  out << "pad = " << a.pad << std::endl;
  out << "stride = " << a.stride << std::endl;
  out << "real_length = " << a.real_length << std::endl;
  out << "length = " << a.length << std::endl;
  out << "ghost_length = " << a.ghost_length << std::endl;
  out << "total_length = " << a.total_length << std::endl;
  out << "ghost_norm_length = " << a.ghost_norm_length << std::endl;
  out << "total_norm_length = " << a.total_norm_length << std::endl;
  out << "siteSubset = " << a.siteSubset << std::endl;
  out << "siteOrder = " << a.siteOrder << std::endl;
  out << "fieldOrder = " << a.fieldOrder << std::endl;
  out << "gammaBasis = " << a.gammaBasis << std::endl;
  return out;
}
