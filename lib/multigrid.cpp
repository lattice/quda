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
      // first two need to be cpu fields
      ColorSpinorParam csParam(*param.B[0]);
      ColorSpinorField *tmp1;
      ColorSpinorField *tmp2;

      // first two need to be gpu fields with native ordering basis
      ColorSpinorField *tmp3;
      ColorSpinorField *tmp4;
      DiracCoarse matCoarse(param.matResidual.Expose(), transfer, *tmp1, *tmp2, *tmp3, *tmp4);
      delete tmp1;
      delete tmp2;
      delete tmp3;
      delete tmp4;

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
