#include <propagator.h>

namespace quda {

  Propagator::Propagator() :
    prop_data(nullptr),
    prop_init(false),
    prop_dim(0),
    prop_location(QUDA_INVALID_FIELD_LOCATION),
    prop_precision(QUDA_INVALID_PRECISION)
  {    
  }

  Propagator::Propagator(const Propagator &src) :
    prop_data(nullptr),
    prop_init(false),
    prop_dim(0),
    prop_location(QUDA_INVALID_FIELD_LOCATION),
    prop_precision(QUDA_INVALID_PRECISION)
  {
    *this = src;
  }

  
  Propagator::Propagator(const ColorSpinorParam &param_) :
    prop_data(nullptr),
    prop_init(false),
    prop_dim(0),
    prop_location(QUDA_INVALID_FIELD_LOCATION),
    prop_precision(QUDA_INVALID_PRECISION)
  {
    ColorSpinorParam param(param_);
    size_t volume = 1;
    for (int d = 0; d < param.nDim; d++) volume *= param.x[d];
    
    prop_dim = param.nColor * param.nSpin;
    prop_location = param.location;
    prop_precision = param.Precision();
    
    // Allocate memory on host or device.
    size_t prop_size_bytes = prop_dim * prop_dim * volume * 2 * prop_precision;
    prop_data = (prop_location == QUDA_CPU_FIELD_LOCATION ?
		 (void *)malloc(prop_size_bytes) : (void *)device_malloc(prop_size_bytes));

    param.create = QUDA_REFERENCE_FIELD_CREATE;
    prop_vectors.reserve(prop_dim);
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
      prop_vectors.push_back(ColorSpinorField::Create(param));
    }    
    if(prop_dim == prop_vectors.size()) prop_init = true;
    else errorQuda("prop_dim %lu not equal to vector set size %lu", prop_dim, prop_vectors.size());
  }

  Propagator::Propagator(const ColorSpinorParam &param_, void **host_data) :
    prop_data(nullptr),
    prop_init(false),
    prop_dim(0),
    prop_location(QUDA_INVALID_FIELD_LOCATION),
    prop_precision(QUDA_INVALID_PRECISION)
  {
    ColorSpinorParam param(param_);
    if(param.create != QUDA_REFERENCE_FIELD_CREATE) errorQuda("This constructor is for referenece fields only");
    
    size_t volume = 1;
    for (int d = 0; d < param.nDim; d++) volume *= param.x[d];

    prop_dim = param.nColor * param.nSpin;
    prop_location = param.location;
    prop_precision = param.Precision();

    // Use host memory provided
    prop_vectors.reserve(prop_dim);
    for (size_t i = 0; i < prop_dim; i++) {
      param.v = host_data[i];
      prop_vectors.push_back(ColorSpinorField::Create(param));
    }
    if(prop_dim == prop_vectors.size()) prop_init = true;
    else errorQuda("prop_dim %lu not equal to vector set size %lu", prop_dim, prop_vectors.size());
  }
    
  Propagator &Propagator::operator=(const Propagator &src)
  {
    // Sanity checks
    if (!prop_init) errorQuda("Propagator not initialised");
    if (src.Dim() != prop_dim) errorQuda("Propgator expected %lu vectors, %lu passed", prop_dim, src.Dim());
    
    // Copy vectors from input
    for (size_t i = 0; i < prop_dim; i++) {
      *prop_vectors[i] = *src.Vectors(i);
    }
    
    // Update Propagator attributes
    prop_precision = src.Precision();    
    return *this;
  }

  Propagator::~Propagator()
  {
    if (prop_init) {
      for (auto &vec : prop_vectors)
        if (vec) delete vec;
      prop_vectors.resize(0);
      prop_init = false;
      if(prop_create != QUDA_REFERENCE_FIELD_CREATE) {
	if(prop_location == QUDA_CPU_FIELD_LOCATION) free(prop_data);
	else device_free(prop_data);
      }
    }
  }
} // namespace quda
