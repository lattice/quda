#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>
#include <iostream>


namespace quda {

#ifdef MULTI_GPU
  XSD::XSD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param,profile), mat(mat)
  {
    sd = new SD(mat,param,profile);
    for(int i=0; i<4; ++i) R[i] = param.overlap_precondition*commDimPartitioned(i);
  }

  XSD::~XSD(){
    if(param.inv_type_precondition != QUDA_GCR_INVERTER) profile.TPSTART(QUDA_PROFILE_FREE);
    if(init){
      delete xx;
      delete bx;
    }
    delete sd;
    if(param.inv_type_precondition != QUDA_GCR_INVERTER) profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void XSD::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
  {
    if(!init){
     ColorSpinorParam csParam(b);
     for(int i=0; i<4; ++i) csParam.x[i] += 2*R[i];
     xx = new cudaColorSpinorField(csParam);
     bx = new cudaColorSpinorField(csParam);	 	
     init = true;
    }

    int parity = mat.getMatPCType();
    copyExtendedColorSpinor(*bx, b, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
    exchangeExtendedGhost(bx, R, parity, streams);

    sd->operator()(*xx,*bx); // actuall run SD

    // copy the interior region of the solution back
    copyExtendedColorSpinor(x, *xx, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
    return;
  }

#endif // MULTI_GPU
} // namespace quda
