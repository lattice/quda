#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <typeinfo>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <eig_variables.h>
#include <comm_quda.h> // for comm_drand()

/*
Maybe this will be useful at some point

#define myalloc(type, n, m0) (type *) aligned_malloc(n*sizeof(type), m0)

#define ALIGN 16
void *
aligned_malloc(size_t n, void **m0)
{
  size_t m = (size_t) safe_malloc(n+ALIGN);
  *m0 = (void*)m;
  size_t r = m % ALIGN;
  if(r) m += (ALIGN - r);
  return (void *)m;
}
*/

namespace quda {
  //cudaEigValueSet class definition
  cudaEigValueSet::cudaEigValueSet(const EigValueSet &src, const EigParam &param) :
    EigValueSet(param), init(false)
  {
    create(param.create);
    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      copy(src, cudaMemcpyHostToDevice);
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
    } else {
      errorQuda("Creation type %d not supported", param.create);
    }
    checkCudaError();
  }

  cudaEigValueSet::~cudaEigValueSet() {
    destroy();
  }
  
  cudaEigValueSet& cudaEigValueSet::operator=(const cudaEigValueSet &src) {
    if (&src != this) {
      if (!reference) {
        destroy();
        // keep current attributes unless unset
        if (!EigValueSet::init) EigValueSet::operator=(src);
        create(QUDA_COPY_FIELD_CREATE);
      }
      copy(src, cudaMemcpyDevicetoDevice);
    }
    return *this;
  }
  
  cudaEigValueSet& cudaEigValueSet::operator=(const cpuEigValueSet &src) {
    if (!EigValueSet::init) {
      destroy();
      // keep current attributes unless unset
      EigValueSet::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    copy(src, cudaMemcpyHostToDevice);
    return *this;
  }

  void cudaEigValueSet::create(const QudaFieldCreate create) {

    //eigen values are complex number
    bytes = 2*(self_param.nk+self_param.np)*self_param.precision;

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = device_malloc(bytes);
      init = true;
    }
  }

  void cudaEigValueSet::destroy() {
    if (init) {
      device_free(v);
      init = false;
    }
  }

  void cudaEigValueSet::save(EigValueSet &dest, enum cudaMemcpyKind) {
    cudaMemcpy(dest.v, v, bytes, cudaMemcpyKind);
  }

  void cudaEigValueSet::copy(const EigValueSet &src, enum cudaMemcpyKind) {
    cudaMemcpy(v, src.v, bytes, cudaMemcpyKind);
  }

  void cudaEigValueSet::zero() {
    cudaMemset(v, 0, bytes);
  }


  // Return the location of the field
  QudaFieldLocation cudaEigValueSet::Location() const { 
    return QUDA_CPU_FIELD_LOCATION;
  }

  //cudaEigVecSet class definition
  cudaEigVecSet::cudaEigVecSet(const EigVecSet &src, const EigParam &param) :
    EigVecSet(param), init(false)
  {
    create(param);
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
    }
  }

  cudaEigVecSet::~cudaEigVecSet() {
    destroy();
  }
  
  cudaEigVecSet& cudaEigVecSet::operator=(const cudaEigVecSet &src) {
    if (&src != this) {
      if (!reference) {
        destroy();
        EigParam Tmp_param(src);
        Tmp_param.create = QUDA_COPY_FIELD_CREATE;
        // keep current attributes unless unset
        if (!EigVecSet::init) EigVecSet::operator=(src);
        create(Tmp_param);
      }
    }
    return *this;
  }
  
  cudaEigVecSet& cudaEigVecSet::operator=(const cpuEigVecSet &src) {
    if (!reference) {
      destroy();
      EigParam Tmp_param(src);
      Tmp_param.create = QUDA_COPY_FIELD_CREATE;
      // keep current attributes unless unset
      if (!EigVecSet::init) EigVecSet::operator=(src);
      create(Tmp_param);
    }
    return *this;
  }

  void cudaEigVecSet::create(const EigParam &param) {

    //eigen Vectors are set of complex number
    bytes = (param.nk+param.np)*param.fsize*param.precision;

    if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
      
//      v = (void**)device_malloc((param.nk+param.np)*sizeof(void*));
//      cudaMemcpyToSymbol(cuda_eigvec, &v, sizeof(void*));
      v = (void**)safe_malloc((param.nk+param.np)*sizeof(void*));

      for(int k = 0; k < param.nk+param.np; k++)
      {
        param.CSPparam.v = ((void**)param.v)[k]; //current code supports double precision only
        ((void**)v)[k] = new cudaColorSpinorField(param.CSPparam);
      }
//      cudaMemcpy(v,tmp_addr,(param.nk+param.np)*sizeof(void*), cudaMemcpyHostToDevice);
      init = true;
    }
    checkCudaError();
  }

  void cudaEigVecSet::destroy() {
    if (init) {
      for(int k = 0; k < param.nk+param.np ; k++)
        device_free(((void**)v)[k]);
      device_free(v);
      init = false;
    }
  }

  std::ostream& operator<<(std::ostream &out, const cudaEigVecSet &a) {
    out << (const EigVecSet&)a;
    out << "v = " << a.v << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }
  // Return the location of the field
  QudaFieldLocation cudaEigVecSet::Location() const { 
    return QUDA_CPU_FIELD_LOCATION;
  }

} // namespace quda
