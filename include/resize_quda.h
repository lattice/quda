#ifndef _RESIZE_QUDA_H
#define _RESIZE_QUDA_H

//struct DecompParam;
//class cudaColorSpinorField;

#include <domain_decomposition.h>
#include <color_spinor_field.h>
#include <face_quda.h>

namespace quda {
  
  void cropCuda(cudaColorSpinorField& dst, const cudaColorSpinorField &src, const DecompParam& params);


  class FaceBufferHandle 
  {

  };



  class Extender 
  {
    private:
      FaceBuffer* face;

    public:
      Extender(const cudaColorSpinorField& field); 
      virtual ~Extender(){ if(face) delete face; }
      // should probably change this as well
      void operator()(cudaColorSpinorField& dst, cudaColorSpinorField& src, const DecompParam& params, const int* const domain_overlap);
  };
} // namespace quda




#endif // _RESIZE_QUDA_H
