#include <propagator.h>

namespace quda {
  
  Propagator::Propagator(const ColorSpinorParam &param_)
  {
    ColorSpinorParam param(param_);
    prop_dim = param.nColor * param.nSpin;
    prop_vectors.reserve(prop_dim);
    
    // Allocate memory on host or device in a contiguous chunk.    
    size_t volume = 1;
    for (int d = 0; d < param.nDim; d++) volume *= param.x[d];
    prop_location = param.location;
    prop_precision = param.Precision();

    if(prop_location == QUDA_CPU_FIELD_LOCATION) printfQuda("prop_location == QUDA_CPU_FIELD_LOCATION)\n");
    
    size_t prop_size_bytes = prop_dim * prop_dim * volume * 2 * prop_precision;
    printfQuda("prop_size_bytes = %lu\n", prop_size_bytes); 
    prop_data = (prop_location == QUDA_CPU_FIELD_LOCATION ?
		 (void *)malloc(prop_size_bytes) : (void *)device_malloc(prop_size_bytes));
    
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    for (size_t i = 0; i < prop_dim; i++) {
      switch(prop_precision) {
      case QUDA_DOUBLE_PRECISION : 
	param.v = (double *)prop_data + i * volume * prop_dim * 2; break;
      case QUDA_SINGLE_PRECISION : 
	param.v = (float *)prop_data + i * volume * prop_dim * 2; break;
      case QUDA_HALF_PRECISION : 
	param.v = (short *)prop_data + i * volume * prop_dim * 2; break;
      case QUDA_QUARTER_PRECISION : 
	param.v = (char *)prop_data + i * volume * prop_dim * 2; break;
      default :
	errorQuda("Unknown precision type %d given", prop_precision);
      }	
      prop_vectors[i] = ColorSpinorField::Create(param);
    }
    
    prop_init = true;
  }
  
  Propagator *Propagator::Create(const ColorSpinorParam &param)
  {
    Propagator *prop = new Propagator(param);
    return prop;
  }
  
  Propagator &Propagator::operator=(const Propagator &src)
  {
    if (&src != this) { *this = (dynamic_cast<const Propagator &>(src)); }
    return *this;
  }

  void Propagator::copyVectors(const std::vector<ColorSpinorField *> &vecs)
  {
    // Sanity checks
    if (!prop_init) errorQuda("Propgator not initialised");
    if (vecs.size() == 0) errorQuda("Zero sized vector set in Propagator");
    size_t n_vecs = vecs[0]->Ncolor() * vecs[0]->Nspin();
    if (vecs.size() != n_vecs) errorQuda("Propgator expected %lu vectors, %lu passed", n_vecs, vecs.size());
    
    printfQuda("Here!\n");
    
    // Copy vectors from input
    for (size_t i = 0; i < n_vecs; i++) {
      printfQuda("Here %lu!\n", i);
      prop_vectors[i]->PrintVector(0);	    
      vecs[i]->PrintVector(0);
      *prop_vectors[i] = *vecs[i];
      printfQuda("Here %lu!\n", i);
    }
  }
  
  ColorSpinorField *Propagator::selectVector(const int vec)
  {
    // Sanity checks
    if (!prop_init) errorQuda("Propgator not initialised");
    size_t n_vecs = prop_vectors.size();
    if ((size_t)vec >= n_vecs) errorQuda("Propgator has %lu vectors, vector[%d] requested", n_vecs, vec);

    return prop_vectors[vec];
  }

  void *Propagator::V()
  {
    if (!prop_init) errorQuda("Propgator not initialised");
    if (prop_vectors.size() == 0) errorQuda("Zero sized vector set in Propagator");
    return (void*)prop_data;
  }
  
  Propagator::~Propagator()
  {
    if (prop_init) {
      for (auto &vec : prop_vectors)
        if (vec) delete vec;
      prop_vectors.resize(0);
      prop_init = false;
    }
  }
} // namespace quda
