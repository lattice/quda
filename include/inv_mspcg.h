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
      
      Solver *solver_prec;
      SolverParam solver_prec_param;

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

      std::array<int, 4> R;
      
      cudaColorSpinorField* vct_dr;
      cudaColorSpinorField* vct_dp;
      cudaColorSpinorField* vct_dmmp;
      cudaColorSpinorField* vct_dtmp;
      cudaColorSpinorField* vct_dtmp2;

      cudaColorSpinorField* r;
      cudaColorSpinorField* x;
      cudaColorSpinorField* p;
      cudaColorSpinorField* z;
      cudaColorSpinorField* mmp;
      cudaColorSpinorField* tmp;
      cudaColorSpinorField* tmp2;
      
      cudaColorSpinorField* r_old;

      cudaColorSpinorField* fr;

      cudaColorSpinorField* immp;
      cudaColorSpinorField* ip;
      
      cudaColorSpinorField* ifmmp;
      cudaColorSpinorField* ifp;
      cudaColorSpinorField* iftmp;
      cudaColorSpinorField* ifset;

      Timer copier_timer;
      Timer preconditioner_timer;
      Timer sloppy_timer;
      Timer precise_timer;
 
      int sp_len2, sp_len1, sp_len0;
      int RR2[4], RR1[4], RR0[4];
      int_fastdiv Xs2[4], Xs1[4], Xs0[4];

    public:

      MSPCG(QudaInvertParam* inv_param, SolverParam& _param, TimeProfile& profile, int ic=6);

      int inner_iterations;

      double Gflops;
      double fGflops;

      double reliable_update_delta;

      void allocate(ColorSpinorField& db);
      void deallocate();

      void test_dslash( const ColorSpinorField& tb );

      virtual ~MSPCG();

      void operator()(ColorSpinorField& out, ColorSpinorField& in);
      void reliable_update(ColorSpinorField& dx, ColorSpinorField& db);
      
      void inner_dslash( ColorSpinorField& out, const ColorSpinorField& in );
      void inner_cg( ColorSpinorField& ix, ColorSpinorField& ib );
      int  outer_cg( ColorSpinorField& dx, ColorSpinorField& db, double quit );
  };

}

#endif
