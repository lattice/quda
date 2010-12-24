#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

/*
Maybe this will be useful at some point

#define myalloc(type, n, m0) (type *) aligned_malloc(n*sizeof(type), m0)

#define ALIGN 16
void *
aligned_malloc(size_t n, void **m0)
{
  size_t m = (size_t) malloc(n+ALIGN);
  *m0 = (void*)m;
  size_t r = m % ALIGN;
  if(r) m += (ALIGN - r);
  return (void *)m;
}
*/

 /*cpuColorSpinorField::cpuColorSpinorField() : 
  ColorSpinorField(), init(false) {

  }*/

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
  ColorSpinorField(param), init(false), order_double(NULL), order_single(NULL) {
  create(param.create);
  if (param.create == QUDA_NULL_FIELD_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
    v = param.v;
  } else {
    errorQuda("Creation type not supported");
  }
}

cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
  ColorSpinorField(src), init(false), order_double(NULL), order_single(NULL) {
  create(QUDA_COPY_FIELD_CREATE);
  memcpy(v,src.v,bytes);
}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
  ColorSpinorField(src), init(false), order_double(NULL), order_single(NULL) {
  create(QUDA_COPY_FIELD_CREATE);
  if (src.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
    memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
  } else if (src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
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
    fieldLocation = QUDA_CPU_FIELD_LOCATION;
    create(QUDA_COPY_FIELD_CREATE);
    copy(src);
  }
  return *this;
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cudaColorSpinorField &src) {
  destroy();
  // keep current attributes unless unset
  if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
  fieldLocation = QUDA_CPU_FIELD_LOCATION;
  create(QUDA_COPY_FIELD_CREATE);
  src.saveCPUSpinorField(*this);
  return *this;
}

void cpuColorSpinorField::create(const QudaFieldCreate create) {
  if (pad != 0) {
    errorQuda("Non-zero pad not supported");
  }
  
  if (precision == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported");
  }

  if (gammaBasis != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && 
      fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    errorQuda("Field order %d not supported", fieldOrder);
  }

  if (create != QUDA_REFERENCE_FIELD_CREATE) {
    v = (void*)malloc(bytes);
    init = true;
  }
 
  createOrder(); // need to do this for references?
}

void cpuColorSpinorField::createOrder() {

  if (precision == QUDA_DOUBLE_PRECISION) {
    if (fieldOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      order_double = new SpaceSpinColorOrder<double>(*this);
    else if (fieldOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      order_double = new SpaceColorSpinOrder<double>(*this);
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", fieldOrder);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if (fieldOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      order_single = new SpaceSpinColorOrder<float>(*this);
    else if (fieldOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      order_single = new SpaceColorSpinOrder<float>(*this);
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", fieldOrder);
  } else {
    errorQuda("Precision %d not supported", precision);
  }
  
}

void cpuColorSpinorField::destroy() {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    delete order_double;
  } else if (precision == QUDA_SINGLE_PRECISION) {
    delete order_single;
  } else {
    errorQuda("Precision %d not supported", precision);
  }
  
  if (init) {
    free(v);
    init = false;
  }

}

template <class D, class S>
void genericCopy(D &dst, const S &src) {

  for (int x=0; x<dst.Volume(); x++) {
    for (int s=0; s<dst.Nspin(); s++) {
      for (int c=0; c<dst.Ncolor(); c++) {
	for (int z=0; z<2; z++) {
	  dst(x, s, c, z) = src(x, s, c, z);
	}
      }
    }
  }

}

void cpuColorSpinorField::copy(const cpuColorSpinorField &src) {
  checkField(*this, src);
  if (fieldOrder == src.fieldOrder) {
    memcpy(v, src.v, bytes);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (src.precision == QUDA_DOUBLE_PRECISION) {
	genericCopy(*order_double, *(src.order_double));
      } else {
	genericCopy(*order_double, *(src.order_single));
      }
    } else {
      if (src.precision == QUDA_DOUBLE_PRECISION) {
	genericCopy(*order_single, *(src.order_double));
      } else {
	genericCopy(*order_single, *(src.order_single));
      }
    }
  }
}

void cpuColorSpinorField::zero() {
  memset(v, '0', bytes);
}

/*
cpuColorSpinorField& cpuColorSpinorField::Even() const { 
  if (subset == QUDA_FULL_FIELD_SUBSET) {
    return *(dynamic_cast<cpuColorSpinorField*>(even)); 
  } else {
    errorQuda("Cannot return even subset of %d subset", subset);
  }
}

cpuColorSpinorField& cpuColorSpinorField::Odd() const {
  if (subset == QUDA_FULL_FIELD_SUBSET) {
    return *(dynamic_cast<cpuColorSpinorField*>(odd)); 
  } else {
    errorQuda("Cannot return odd subset of %d subset", subset);
  }
}
*/

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
	   const int nColor, const QudaFieldOrder fieldOrder) {
  switch(fieldOrder) {
  case QUDA_SPACE_SPIN_COLOR_FIELD_ORDER: 
    v[(st*nSpin+s)*nColor+c] = 1.0;
    break;
  case QUDA_SPACE_COLOR_SPIN_FIELD_ORDER:
    v[(st*nColor+c)*nSpin+s] = 1.0;
    break;
  default:
    errorQuda("Field ordering %d not supported", fieldOrder);
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
    if (precision == QUDA_DOUBLE_PRECISION) point((double*)v, st, s, c, nSpin, nColor, fieldOrder);
    else if (precision == QUDA_SINGLE_PRECISION) point((float*)v, st, s, c, nSpin, nColor, fieldOrder);
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
  int *fail = new int[fail_check];
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
  delete []fail;
}

void cpuColorSpinorField::Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, 
				  const int resolution) {
  checkField(a, b);
  if (a.precision == QUDA_HALF_PRECISION || b.precision == QUDA_HALF_PRECISION) 
    errorQuda("Half precision not implemented");
  if (a.fieldOrder != b.fieldOrder || 
      (a.fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && a.fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER))
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
		  const QudaFieldOrder fieldOrder) {

  switch(fieldOrder) {

  case QUDA_SPACE_SPIN_COLOR_FIELD_ORDER:

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	std::cout << "( " << v[((vol*Ns+s)*Nc+c)*2] << " , " << v[((vol*Ns+s)*Nc+c)*2+1] << " ) ";
      }
      std::cout << std::endl;
    }
    break;

  case QUDA_SPACE_COLOR_SPIN_FIELD_ORDER:

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	std::cout << "( " << v[((vol*Nc+c)*Ns+s)*2] << " , " << v[((vol*Nc+c)*Ns+s)*2+1] << " ) ";
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
    print_vector((double*)v, vol, nSpin, nColor, fieldOrder);
    break;
  case QUDA_SINGLE_PRECISION:
    print_vector((float*)v, vol, nSpin, nColor, fieldOrder);
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
