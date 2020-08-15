#include <propagator.h>

namespace quda {

  Propagator::Propagator(const ColorSpinorParam &param)
  {
    prop_vectors.reserve(param.nColor * param.nSpin);
    for(int i=0; i<(param.nColor*param.nSpin); i++) prop_vectors.push_back(ColorSpinorField::Create(param));
    prop_init = true;
  }    
      
  Propagator* Propagator::Create(const ColorSpinorParam &param)
  {    
    Propagator *prop = new Propagator(param);    
    return prop;
  }

  Propagator& Propagator::operator=(const Propagator &src)
  {
    if (&src != this) {
      *this = (dynamic_cast<const Propagator&>(src));
    }
    return *this;
  }  
  
  void Propagator::copyVectors(const std::vector<ColorSpinorField*> &vecs)
  {
    // Sanity checks
    if(!prop_init) errorQuda("Propgator not initialised");
    if(vecs.size() == 0) errorQuda("zero sized vector set");
    size_t n_vecs = vecs[0]->Ncolor() * vecs[0]->Nspin();
    if(vecs.size() != n_vecs) errorQuda("Propgator expected %lu vectors, %lu passed", n_vecs, vecs.size());

    // Copy vectors from input
    for(size_t i=0; i<n_vecs; i++) prop_vectors[i] = vecs[i];
  }


  ColorSpinorField* Propagator::selectVector(const int vec)
  {
    // Sanity checks
    if(!prop_init) errorQuda("Propgator not initialised");
    size_t n_vecs = prop_vectors.size();
    if((size_t)vec >= n_vecs) errorQuda("Propgator has %lu vectors, vector[%d] requested", n_vecs, vec);
    
    return prop_vectors[vec];
  }

  Propagator::~Propagator()
  {
    if (prop_init) {
      for (auto &vec : prop_vectors)
	if(vec) delete vec;
      prop_vectors.resize(0);
      prop_init = false;
    }
  }
}
