#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <string.h>
#include <iostream>
#include <limits>

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <device.h>

namespace quda {

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : ColorSpinorField(param)
  {
    create2(param.create);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break; // do nothing;
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: errorQuda("Copy field create not implemented for this constructor"); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : ColorSpinorField(src)
  {
    create2(QUDA_COPY_FIELD_CREATE);
    copy(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) : ColorSpinorField(src)
  {
    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) || 
         (param.create == QUDA_REFERENCE_FIELD_CREATE && (param.is_composite || param.is_component))) {
      reset(param);
    } else {
      errorQuda("Undefined behaviour"); // else silent bug possible?
    }

    // This must be set before create is called
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      if (typeid(src) == typeid(cudaColorSpinorField)) {
	v = (void*)src.V();
	norm = (void*)src.Norm();
      } else {
	errorQuda("Cannot reference a non-cuda field");
      }

      if (composite_descr.is_component && !(src.SiteSubset() == QUDA_FULL_SITE_SUBSET && this->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) 
      {//setup eigenvector form the set
        v    = (void*)((char*)v    + composite_descr.id*bytes);         
        norm = (void*)((char*)norm + composite_descr.id*norm_bytes);         
      }
    }

    create2(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      copy(src);
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else {
      errorQuda("CreateType %d not implemented", param.create);
    }

  }

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) : ColorSpinorField(src)
  {
    create2(QUDA_COPY_FIELD_CREATE);
    copy(src);
  }

  cudaColorSpinorField::~cudaColorSpinorField() {
    destroyComms();
    destroy2();
  }

  //! for composite fields:
  cudaColorSpinorField& cudaColorSpinorField::Component(const int idx) const {
    
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {//setup eigenvector form the set
        return *(dynamic_cast<cudaColorSpinorField*>(components[idx])); 
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
  }

  // copyCuda currently cannot not work with set of spinor fields..
  void cudaColorSpinorField::CopySubset(cudaColorSpinorField &, const int, const int) const
  {
#if 0
    if (first_element < 0) errorQuda("\nError: trying to set negative first element.\n");
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (first_element == 0 && range == this->EigvDim())
      {
        if (range != dst.EigvDim())errorQuda("\nError: eigenvector range to big.\n");
        checkField(dst, *this);
        copyCuda(dst, *this);
      }
      else if ((first_element+range) < this->EigvDim()) 
      {//setup eigenvector subset

        cudaColorSpinorField *eigv_subset;

        ColorSpinorParam param;

        param.nColor = nColor;
        param.nSpin = nSpin;
        param.twistFlavor = twistFlavor;
        param.precision = precision;
        param.nDim = nDim;
        param.pad = pad;
        param.siteSubset = siteSubset;
        param.siteOrder = siteOrder;
        param.fieldOrder = fieldOrder;
        param.gammaBasis = gammaBasis;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
 
        param.eigv_dim  = range;
        param.eigv_id   = -1;
        param.v = (void*)((char*)v + first_element*eigv_bytes);
        param.norm = (void*)((char*)norm + first_element*eigv_norm_bytes);

        eigv_subset = new cudaColorSpinorField(param);

        //Not really needed:
        eigv_subset->eigenvectors.reserve(param.eigv_dim);
        for (int id = first_element; id < (first_element+range); id++)
        {
            param.eigv_id = id;
            eigv_subset->eigenvectors.push_back(new cudaColorSpinorField(*this, param));
        }
        checkField(dst, *eigv_subset);
        copyCuda(dst, *eigv_subset);

        delete eigv_subset;
      } else {
        errorQuda("Incorrect eigenvector dimension...");
      }
    } else{
      errorQuda("Eigenvector must be a parity spinor");
      exit(-1);
    }
#endif
  }

  const void *cudaColorSpinorField::Ghost2() const
  {
    if (bufferIndex < 2) {
      return ghost_recv_buffer_d[bufferIndex];
    } else {
      return ghost_pinned_recv_buffer_hd[bufferIndex % 2];
    }
  }

} // namespace quda
