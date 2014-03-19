#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <color_spinor_field.h>
#include <eig_variables.h>
#include <color_spinor_field_order.h>
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
  //cpuEigValueSet class definition
  cpuEigValueSet::cpuEigValueSet(const EigParam &param) :
    EigValueSet(param), init(false)
  {
    create(param.create);
    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
    } else {
      errorQuda("Creation type %d not supported", param.create);
    }
  }

  cpuEigValueSet::~cpuEigValueSet() {
    destroy();
  }
  
  cpuEigValueSet& cpuEigValueSet::operator=(const cpuEigValueSet &src) {
    if (&src != this) {
      if (!reference) {
        destroy();
        // keep current attributes unless unset
        if (!EigValueSet::init) EigValueSet::operator=(src);
        create(QUDA_COPY_FIELD_CREATE);
      }
      copy(src);
    }
    return *this;
  }

  void cpuEigValueSet::create(const QudaFieldCreate create) {

    //eigen values are complex number
    bytes = 2*(self_param.nk+self_param.np)*self_param.precision;

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = safe_malloc(bytes);
      init = true;
    }
  }

  void cpuEigValueSet::destroy() {
    if (init) {
      host_free(v);
      init = false;
    }
  }

  void cpuEigValueSet::copy(const cpuEigValueSet &src) {
    memcpy(v, src.v, bytes);
  }

  void cpuEigValueSet::zero() {
    memset(v, '\0', bytes);
  }

  // print out the n-th eigen value  
  void cpuEigValueSet::PrintEigVal(unsigned int n) {
    std::cout << n <<"-th Eig value =" << v[2*n] << "+i " << v[2*n+1] << std::endl;
  }

  // Return the location of the field
  QudaFieldLocation cpuEigValueSet::Location() const { 
    return QUDA_CPU_FIELD_LOCATION;
  }

  //cpuEigVecSet class definition
  cpuEigVecSet::cpuEigVecSet(const EigParam &param) :
    EigVecSet(param), init(false)
  {
    create(param);
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
    } else {
      errorQuda("Creation type %d not supported", param.create);
    }
  }

  cpuEigVecSet::~cpuEigVecSet() {
    destroy();
  }
  
  cpuEigVecSet& cpuEigVecSet::operator=(const cudaEigVecSet &src) {
    if (!reference) {
      destroy();
      EigParam Tmp_param(src);
      Tmp_param.create = QUDA_COPY_FIELD_CREATE
        // keep current attributes unless unset
        if (!EigVecSet::init) EigVecSet::operator=(src);
      create(Tmp_param);
    }
    src.save(*this, cudaMemcpyDeviceToHost);
    return *this;
  }
  
  cpuEigVecSet& cpuEigVecSet::operator=(const cpuEigVecSet &src) {
    if (&src != this) {
      if (!reference) {
        destroy();
        EigParam Tmp_param(src);
        Tmp_param.create = QUDA_COPY_FIELD_CREATE
        // keep current attributes unless unset
        if (!EigVecSet::init) EigVecSet::operator=(src);
        create(Tmp_param);
      }
    }
    return *this;
  }

  void cpuEigVecSet::create(const EigParam &param) {

    //eigen Vectors are set of complex number
    bytes = (param.nk+param.np)*param.fsize*param.precision;

    if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
      
      v = (void**)safe_malloc((param.nk+param.np)*sizeof(void*));
      for(int k = 0; k < param.nk+param.np; k++)
      {
        param.CSPparam.v = ((void**)param.v)[k]; //current code supports double precision only
        ((void**)v)[k] = new cpuColorSpinorField(param.CSPparam);
      }
      init = true;
    }
  }

  void cpuEigVecSet::destroy() {
    if (init) {
      for(int k = 0; k < param.nk+param.np ; k++)
        host_free(((void**)v)[k]);
      host_free(v);
      init = false;
    }
  }

  // print out the vector at volume point x
  void cpuEigVecSet::PrintEigVal(unsigned int n) {
    for(int k = 0; k < param.fsize/2 ; k++)
      std::cout << n <<"-th Eig, vector["<< k <<"] =" << ((void**)v)[n][2*k] << "+i " << ((void**)v)[2*k+1] << std::endl;
  }

  // Return the location of the field
  QudaFieldLocation cpuEigVecSet::Location() const { 
    return QUDA_CPU_FIELD_LOCATION;
  }

} // namespace quda
