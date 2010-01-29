#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <color_spinor_field.h>

cpuColorSpinorField::cpuColorSpinorField() : 
  ColorSpinorField(), init(false) {

}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
  ColorSpinorField(param), init(false) {
  create(param.create);
  if (param.create == QUDA_NULL_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_CREATE) {
    v = param.v;
  } else {
    errorQuda("Creation type not supported");
  }
}

cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create(QUDA_COPY_CREATE);
  memcpy(v,src.v,bytes);
}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create(QUDA_COPY_CREATE);
  if (src.fieldType() == QUDA_CPU_FIELD) {
    memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
  } else if (src.fieldType() == QUDA_CUDA_FIELD) {
    dynamic_cast<const cudaColorSpinorField&>(src).saveCPUSpinorField(*this);
  } else {
    errorQuda("FieldType not supported");
  }
}

cpuColorSpinorField::~cpuColorSpinorField() {
  destroy();
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cpuColorSpinorField &src) {
  if (&src != this) {
    destroy();
    // keep current attributes unless unset
    if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
    type = QUDA_CPU_FIELD;
    create(QUDA_COPY_CREATE);
    copy(src);
  }
  return *this;
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cudaColorSpinorField &src) {
  destroy();
  // keep current attributes unless unset
  if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
  type = QUDA_CPU_FIELD;
  create(QUDA_COPY_CREATE);
  src.saveCPUSpinorField(*this);
  return *this;
}

void cpuColorSpinorField::create(const FieldCreate create) {
  if (pad != 0) {
    errorQuda("Non-zero pad not supported");
  }
  
  if (precision == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported");
  }

  if (basis != QUDA_DEGRAND_ROSSI_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (order != QUDA_SPACE_COLOR_SPIN_ORDER && order != QUDA_SPACE_SPIN_COLOR_ORDER) {
    errorQuda("Field order %d not supported", order);
  }

  if (create != QUDA_REFERENCE_CREATE) {
    v = (void*)malloc(bytes);
    init = true;
  }
  
}

void cpuColorSpinorField::destroy() {

  if (init) free(v);

}

void cpuColorSpinorField::copy(const cpuColorSpinorField &src) {
  checkField(*this, src);
  memcpy(v, src.v, bytes);
}

void cpuColorSpinorField::zero() {
  memset(v, '0', bytes);
}



//sets the elements of the field to random [0, 1]
template <typename Float>
void random(Float *v, const int length) {
  for(int i = 0; i < length; i++) {
    v[i] = rand() / (double)RAND_MAX;
  }
}

// create a point source at spacetime point st, spin s and colour c
template <typename Float>
void point(Float *v, const int st, const int s, const int c, const int nSpin, 
	   const int nColor, const QudaColorSpinorOrder field_order) {
  switch(field_order) {
  case QUDA_SPACE_SPIN_COLOR_ORDER: 
    v[(st*nSpin+s)*nColor+c] = 1.0;
    break;
  case QUDA_SPACE_COLOR_SPIN_ORDER:
    v[(st*nColor+c)*nSpin+s] = 1.0;
    break;
  default:
    errorQuda("Field ordering %d not supported", field_order);
  }
}

void cpuColorSpinorField::Source(const QudaSourceType sourceType, const int st, const int s, const int c) {

  switch(sourceType) {

  case QUDA_RANDOM_SOURCE:
    if (precision == QUDA_DOUBLE_PRECISION) random((double*)v, length);
    else if (precision == QUDA_SINGLE_PRECISION) random((float*)v, length);
    else errorQuda("Precision not supported");
    break;

  case QUDA_POINT_SOURCE:
    if (precision == QUDA_DOUBLE_PRECISION) point((double*)v, st, s, c, nSpin, nColor, order);
    else if (precision == QUDA_SINGLE_PRECISION) point((float*)v, st, s, c, nSpin, nColor, order);
    else errorQuda("Precision not supported");
    break;

  default:
    errorQuda("Source type %d not implemented", sourceType);

  }

}

template <typename FloatA, typename FloatB>
static void compareSpinor(const FloatA *u, const FloatB *v, const int volume, 
			  const int N, const int resolution) {
  int fail_check = 16*resolution;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int *iter = new int[N];

  for (int i=0; i<N; i++) iter[i] = 0;

  for (int i=0; i<volume; i++) {
    for (int j=0; j<N; j++) {
      int is = i*N+j;
      double diff = fabs(u[is]-v[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10.0,-(f+1)/(double)resolution)) fail[f]++;
      if (diff > 1e-3) iter[j]++;
    }
  }
    
  for (int i=0; i<N; i++) printf("%d fails = %d\n", i, iter[i]);
    
  for (int f=0; f<fail_check; f++) {
    printf("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)/(double)resolution), 
	   fail[f], volume*N, fail[f] / (double)(volume*N));
  }

  delete []iter;
}

void cpuColorSpinorField::Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, 
				  const int resolution) {
  checkField(a, b);
  if (a.precision == QUDA_HALF_PRECISION || b.precision == QUDA_HALF_PRECISION) 
    errorQuda("Half precision not implemented");
  if (a.order != b.order || 
      (a.order != QUDA_SPACE_COLOR_SPIN_ORDER && a.order != QUDA_SPACE_SPIN_COLOR_ORDER))
    errorQuda("Field ordering not supported");

  if (a.precision == QUDA_DOUBLE_PRECISION) 
    if (b.precision == QUDA_DOUBLE_PRECISION)
      compareSpinor((double*)a.v, (double*)b.v, a.volume, 2*a.nSpin*a.nColor, resolution);
    else
      compareSpinor((double*)a.v, (float*)b.v, a.volume, 2*a.nSpin*a.nColor, resolution);
  else 
    if (b.precision == QUDA_DOUBLE_PRECISION)
      compareSpinor((float*)a.v, (double*)b.v, a.volume, 2*a.nSpin*a.nColor, resolution);
    else
      compareSpinor((float*)a.v, (float*)b.v, a.volume, 2*a.nSpin*a.nColor, resolution);
}

template <typename Float>
void print_vector(const Float *v, const int vol, const int Ns, const int Nc, 
		  const QudaColorSpinorOrder order) {

  switch(order) {

  case QUDA_SPACE_SPIN_COLOR_ORDER:

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	std::cout << "( " << v[((vol*Ns+s)*Nc+c)*2] << " , " << v[((vol*Ns+s)*Nc+c)*2+1] << " ) ";
      }
      std::cout << std::endl;
    }
    break;

  case QUDA_SPACE_COLOR_SPIN_ORDER:

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	std::cout << "( " << v[((vol*Ns+s)*Nc+c)*2] << " , " << v[((vol*Ns+s)*Nc+c)*2+1] << " ) ";
      }
      std::cout << std::endl;
    }
    break;

  default:
    errorQuda("Field oredring not supported");
      
  }
}

// print out the vector at volume point vol
void cpuColorSpinorField::PrintVector(int vol) {
  
  switch(precision) {
  case QUDA_DOUBLE_PRECISION:
    print_vector((double*)v, vol, nSpin, nColor, order);
    break;
  case QUDA_SINGLE_PRECISION:
    print_vector((float*)v, vol, nSpin, nColor, order);
    break;
  default:
    errorQuda("Precision %d not implemented", precision); 
  }

}

double normCpu(const cpuColorSpinorField &a) {
  
  double norm2 = 0.0;
  if (a.precision == QUDA_DOUBLE_PRECISION)
    for (int i=0; i<a.length; i++) norm2 += ((double*)a.v)[i]*((double*)a.v)[i];
  else if (a.precision == QUDA_SINGLE_PRECISION)
    for (int i=0; i<a.length; i++) norm2 += ((float*)a.v)[i]*((float*)a.v)[i];
  else
    errorQuda("Precision type %d not implemented", a.precision);

  return norm2;
}
