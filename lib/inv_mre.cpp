#include <invert_quda.h>

namespace quda {

  MinResExt::MinResExt(DiracMatrix &dirac, QudaInvertParam &param, TimeProfile &profile) 
    : Solver(param, profile) {

  }

  virtual MinResExt::~MinResExt() {

  }

  MinResExt::operator(cudaColorSpinorField &x, const cudaColorSpinorField &b, cudaColorSpinorField **p) {

  }

};
