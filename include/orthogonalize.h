#ifndef __ORTHOGONALIZE_H__
#define __ORTHOGONALIZE_H__
#include "color_spinor_field.h"

namespace quda {

  class Orthogonalizer {
   
    public:
      virtual void operator()(cudaColorSpinorField out[], cudaColorSpinorField in[], int Nvec) const = 0;
      virtual void operator()(cudaColorSpinorField v[], int Nvec) const = 0;
      virtual ~Orthogonalizer() {}
      // Checks to find the maximum deviation from orthogonality in a set of vectors.
      // Returns both the maximum deviation in 'max_error' and the indices of the vectors whose dot product gives the maximum deviation
      void getMaxError(double& max_error, int &index1, int &index2, cudaColorSpinorField v[], int Nvec) const; 

    protected:
      // remove from v the component along the direction of u
      void project(cudaColorSpinorField& v, cudaColorSpinorField& u) const;
      // project out the u component of v and write the result to w
      void project(cudaColorSpinorField& w, cudaColorSpinorField& v, cudaColorSpinorField& u) const;

  };

  class ModifiedGramSchmidt : public Orthogonalizer 
  { 
    void operator()(cudaColorSpinorField out[], cudaColorSpinorField in[], int Nvec) const;
    void operator()(cudaColorSpinorField v[], int Nvec) const;
  };



} // namespace quda

#endif
