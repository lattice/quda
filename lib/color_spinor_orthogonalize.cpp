#include <orthogonalize.h>
#include <blas_quda.h>

namespace quda{

  void Orthogonalizer::getMaxError(double& max_error, int& index1, int& index2, cudaColorSpinorField v[], int nitem) const 
  {
    if(nitem <= 1){
      max_error = 0.0;
      index1 = index2 = 0;
      return;
    }

    max_error = 0.0;
   
    double error;
    for(int i=1; i<nitem; ++i){
      for(int j=0; j<i; ++j){
        error = std::abs(cDotProductCuda(v[i],v[j])); 
        if(max_error < error){
          max_error = error;
          index1 = i; index2 = j;        
        }
      }
    }
  }

  // removes from v the component of v along the vector u 
  void Orthogonalizer::project(cudaColorSpinorField& v, cudaColorSpinorField& u) const 
  {
   double3 temp = cDotProductNormACuda(u,v);
   double u2 = temp.z;
   if (u2 == 0) return;
   Complex uv(temp.x,temp.y);
   caxpyCuda(-uv/u2, u, v); // v -= (uv/u2)*u
   return;
  }

  // returns v with the component of v along the vector u removed in w
  void Orthogonalizer::project(cudaColorSpinorField& w, cudaColorSpinorField& v, cudaColorSpinorField& u) const
  {
    w = v;
    project(w,u);
    return;
  }

  // in-place orthogonalization
  void ModifiedGramSchmidt::operator()(cudaColorSpinorField v[], int nitem) const 
  {
    for(int i=0; i<nitem; ++i){
      for(int j=0; j<i; ++j){
        project(v[i], v[j]);
      }    
    }
    return;
  }
 
   
  void ModifiedGramSchmidt::operator()(cudaColorSpinorField out[], cudaColorSpinorField in[], int nitem) const 
  {
    for(int i=0; i<nitem; ++i) out[i] = in[i];
    this->operator()(out, nitem);
    return;
  }

} // namespace quda
