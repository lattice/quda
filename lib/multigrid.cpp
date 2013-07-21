#include <multigrid.h>

namespace quda {

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), smoother(0), coarse(0), fine(param.fine) {

    if (param.level > QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level);

    // create the smoother for this level
    smoother = Solver::create(param, param.matResidual, param.matSmooth, param.matSmooth, profile);

    // if not on the coarsest level, construct it
    if (param.level < param.Nlevel) {
      // create transfer operator
      transfer = new Transfer(param.B, param.Nvec, param.geoBlockSize, param.spinBlockSize);

      // create coarse grid operator
      ColorSpinorField *tmp1_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);
      ColorSpinorField *tmp2_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);
      DiracCoarse matCoarse(param.matResidual.Expose(), transfer, *tmp1_coarse, *tmp2_coarse);

      // create the next multigrid level
      MGParam coarse_param = param;
      coarse_param.level++;

      coarse_param.matResidual = matCoarse;
      coarse_param.matSmooth = matCoarse;

      coarse_param.fine = this;

      coarse = new MG(coarse_param, profile);
    }
  }

  MG::~MG() {
    if (param.level < param.Nlevel) {
      if (tmp1_coarse) delete tmp1_coarse;
      if (tmp2_coarse) delete tmp2_coarse;
      if (coarse) delete coarse;
      if (transfer) delete transfer;
    }
    if (smoother) delete smoother;
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      param.maxiter = param.nu_pre;
      (*smoother)(x, b);

      // restrict to the coarse grid
      transfer->R(*r, *r_coarse);

      // recurse to the next lower level
      (*coarse)(*x_coarse, *r_coarse); 

      // prolongate back to this grid
      transfer->P(x, *x_coarse); // FIXME: need to ensure the prolongator sums to x here

      // do the post smoothing
      param.maxiter = param.nu_post;
      (*smoother)(x, b);

    } else { // do the coarse grid solve

      (*smoother)(x, b);

    }

  }

}
