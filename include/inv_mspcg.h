#ifndef _MSPCG_H
#define _MSPCG_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <vector>

#include <invert_quda.h>

namespace quda {

  class MSPCG : public Solver { // Multisplitting Preconditioned CG

    private:

      DiracMobiusPC* mat;
      DiracMobiusPC* mat_sloppy;
      DiracMobiusPC* mat_precondition;
      //    DiracDubiusPC* mat_extended;
      DiracMdagM* nrm_op;
      DiracMdagM* nrm_op_sloppy;
      DiracMdagM* nrm_op_precondition;

      DiracParam dirac_param;
      DiracParam dirac_param_sloppy;
      DiracParam dirac_param_precondition;

      cudaGaugeField* padded_gauge_field;
      cudaGaugeField* padded_gauge_field_precondition;

      Solver *solver_prec;
      SolverParam solver_prec_param;

      std::array<int, 4> R;
      
      cudaColorSpinorField* vct_dr;
      cudaColorSpinorField* vct_dp;
      cudaColorSpinorField* vct_dmmp;
      cudaColorSpinorField* vct_dtmp;

      cudaColorSpinorField* r;
      cudaColorSpinorField* x;
      cudaColorSpinorField* p;
      cudaColorSpinorField* z;
      cudaColorSpinorField* mmp;
      cudaColorSpinorField* tmp;

      cudaColorSpinorField* fr;
      cudaColorSpinorField* fz;

      cudaColorSpinorField* immp;
      cudaColorSpinorField* ip;
      cudaColorSpinorField* itmp;

      Timer copier_timer;
      Timer preconditioner_timer;
      Timer sloppy_timer;
      Timer precise_timer;
  
    public:

      MSPCG(QudaInvertParam* inv_param, SolverParam& _param, TimeProfile& profile, int ps=1);

      double Gflops;
      double fGflops;

      void test_dslash( const ColorSpinorField& tb );

      virtual ~MSPCG();

      void operator()(ColorSpinorField& out, ColorSpinorField& in);
      void inner_cg( ColorSpinorField& ix, ColorSpinorField& ib );
      int  outer_cg( ColorSpinorField& dx, ColorSpinorField& db, double quit );
  };

}

#endif
