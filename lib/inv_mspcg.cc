#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <limits>

#include <inv_mspcg.h>

extern quda::cudaGaugeField* gaugePrecondition;
extern quda::cudaGaugeField* gaugePrecise;
extern quda::cudaGaugeField* gaugeSloppy;

namespace quda {

  using namespace blas;

  static cudaGaugeField* createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile,
      bool redundant_comms=false, QudaReconstructType recon=QUDA_RECONSTRUCT_INVALID)
  {
    int y[4];
    for (int dir=0; dir<4; ++dir) y[dir] = in.X()[dir] + 2*R[dir];
    int pad = 0;

    GaugeFieldParam gParamEx(y, in.Precision(), recon != QUDA_RECONSTRUCT_INVALID ? recon : in.Reconstruct(), pad,
        in.Geometry(), QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = in.Order();
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = in.TBoundary();
    gParamEx.nFace = 1;
    gParamEx.tadpole = in.Tadpole();
    for (int d=0; d<4; d++) gParamEx.r[d] = R[d];

    cudaGaugeField *out = new cudaGaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);

    // now fill up the halos
    profile.TPSTART(QUDA_PROFILE_COMMS);
    out->exchangeExtendedGhost(R,redundant_comms);
    profile.TPSTOP(QUDA_PROFILE_COMMS);

    return out;
  } 

  static void set_mobius_dirac_param(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    double kappa = inv_param->kappa;
    if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      kappa *= gaugePrecise->Anisotropy();
    }

    if(inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH){
      errorQuda("ONLY Mobius.\n"); 
    }
    
    diracParam.type = pc ? QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC : QUDA_MOBIUS_DOMAIN_WALL_DIRAC;
    diracParam.Ls = inv_param->Ls;
    memcpy(diracParam.b_5, inv_param->b_5, sizeof(double)*inv_param->Ls);
    memcpy(diracParam.c_5, inv_param->c_5, sizeof(double)*inv_param->Ls);
    
    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = gaugePrecise;
//    diracParam.fatGauge = gaugeFatPrecise;
//    diracParam.longGauge = gaugeLongPrecise;
//    diracParam.clover = cloverPrecise;
//    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;

    for (int i=0; i<4; i++) diracParam.commDim[i] = 1;   // comms are always on
  }


/*
  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam& inner, const SolverParam& outer)
  {
    //    inner.tol = outer.tol_precondition;
    inner.tol = 5e-2;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver

    inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  }
*/
  MSPCG::MSPCG(QudaInvertParam* inv_param, SolverParam& _param, TimeProfile& profile, int ic) :
    Solver(_param, profile), solver_prec(0), solver_prec_param(_param), 
    mat(NULL),  mat_sloppy(NULL), mat_precondition(NULL),
    nrm_op(NULL), nrm_op_sloppy(NULL), nrm_op_precondition(NULL), 
    vct_dr(NULL), vct_dp(NULL), vct_dmmp(NULL), vct_dtmp(NULL),
    r(NULL), p(NULL), z(NULL), mmp(NULL), tmp(NULL), fr(NULL), 
    immp(NULL), ip(NULL), 
    ifmmp(NULL), ifp(NULL), iftmp(NULL), 
    inner_iterations(ic), reliable_update_delta(inv_param->reliable_delta)
  { 

    printfQuda("MSPCG constructor starts.\n");

    R[0]=1;
    R[1]=2;
    R[2]=2;
    R[3]=2;
    // TODO: R is the checkerboarded size.


    if(inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH){
      errorQuda("ONLY works for QUDA_MOBIUS_DWF_DSLASH.");
    }

    // create extended gauge field
    // TODO: dynamical allocation need fix
    if(not gaugePrecondition){
      errorQuda("gaugePrecondition not valid.");
    }
    
    if(not gaugeSloppy){
      errorQuda("gaugeSloppy not valid.");
    }

    int gR[4] = {2*R[0], R[1], R[2], R[3]}; 
    padded_gauge_field = createExtendedGauge(*gaugePrecise, gR, profile, true);
    padded_gauge_field_precondition = createExtendedGauge(*gaugePrecondition, gR, profile, true);

    set_mobius_dirac_param(dirac_param, inv_param, true); // pc = true

    set_mobius_dirac_param(dirac_param_sloppy, inv_param, true); // pc = true
    dirac_param_sloppy.gauge = gaugeSloppy;

    set_mobius_dirac_param(dirac_param_precondition, inv_param, true); // pc = true
    dirac_param_precondition.gauge = padded_gauge_field_precondition;

    for(int i = 0; i < 4; i++){
      dirac_param.commDim[i] = 1; 
      dirac_param_sloppy.commDim[i] = 1; 
      dirac_param_precondition.commDim[i] = 0;
    }

    dirac_param.print();
    dirac_param_sloppy.print();
    dirac_param_precondition.print();
    
    mat = new DiracMobiusPC(dirac_param);
    nrm_op = new DiracMdagM(mat);
     
    mat_sloppy = new DiracMobiusPC(dirac_param_sloppy);
    nrm_op_sloppy = new DiracMdagM(mat_sloppy);
    
    mat_precondition = new DiracMobiusPC(dirac_param_precondition);
    nrm_op_precondition = new DiracMdagM(mat_precondition);

//    fillInnerSolverParam(solver_prec_param, param);

    printfQuda("MSPCG constructor ends.\n");
    
    copier_timer.Reset("woo", "hoo", 0);
    precise_timer.Reset("woo", "hoo", 0);
    sloppy_timer.Reset("woo", "hoo", 0);
    preconditioner_timer.Reset("woo", "hoo", 0);
  }

  MSPCG::~MSPCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);
    /*
       if(solver_prec) 
       delete solver_prec;

       if( MdagM ) 
       delete MdagM;
       if( MdagM_precondition ) 
       delete MdagM_precondition;
       if( mat )
       delete mat;
       if( mat_precondition )
       delete mat_precondition;
       */
    delete nrm_op_precondition;
    delete mat_precondition;
    
    delete nrm_op_sloppy;
    delete mat_sloppy;
    
    delete nrm_op;
    delete mat;

    delete padded_gauge_field;
    delete padded_gauge_field_precondition;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void MSPCG::test_dslash( const ColorSpinorField& b ){
    
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.print();
    // TODO: def
    cudaColorSpinorField* tx = NULL;
    cudaColorSpinorField* tt = NULL;
    cudaColorSpinorField* tb = NULL;

    tx  = new cudaColorSpinorField(csParam);
    tt  = new cudaColorSpinorField(csParam);
    tb  = new cudaColorSpinorField(csParam);

    blas::copy( *tb, b );

    double b2 = blas::norm2(*tb);
    printfQuda("Test b2 before = %16.12e.\n", b2);
    if( comm_rank() ){ blas::zero(*tb); }
    b2 = blas::norm2(*tb);
    printfQuda("Test b2 after  = %16.12e.\n", b2);
    (*nrm_op)(*tx, *tb, *tt);
//    mat->Dslash4(*tx, *tb, QUDA_EVEN_PARITY);
    double x2 = blas::norm2(*tx);
    printfQuda("Test     x2/b2 = %16.12e/%16.12e.\n", x2, b2);
    if( comm_rank() ){
      blas::zero(*tx);
    }
    x2 = blas::norm2(*tx);
    printfQuda("Chopping x2/b2 = %16.12e/%16.12e.\n", x2, b2);
    
    cudaColorSpinorField* fx = NULL;
    cudaColorSpinorField* fy = NULL;
    cudaColorSpinorField* fb = NULL;
    cudaColorSpinorField* ft = NULL;

    for(int i=0; i<4; ++i){
      csParam.x[i] += 2*R[i];
    }

    csParam.setPrecision(dirac_param_precondition.gauge->Precision());
    csParam.print();

    // TODO: def
    fx  = new cudaColorSpinorField(csParam);
    fy  = new cudaColorSpinorField(csParam);
    fb  = new cudaColorSpinorField(csParam);
    ft  = new cudaColorSpinorField(csParam);
    blas::zero(*fb);
    blas::zero(*fx);

    copyExtendedColorSpinor(*fb, *tb, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL); // parity = 0

    //    quda::pack::initConstants(*dirac_param_precondition.gauge, profile);
    double fb2 = norm2(*fb);

//    (*nrm_op_precondition)(*fx, *fb, *ft);
/*    
    int odd_bit = 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    mat_precondition->Dagger(QUDA_DAG_NO);
    
    mat_precondition->Dslash4pre(*ft, *fb, parity[1]);                   // +0
//    mat_precondition->Dslash4prePartial(*ft, *fb, parity[1], sp_len0, RR0, Xs0);                   // +0
    mat_precondition->Dslash4Partial(*fy, *ft, parity[0], sp_len1, RR1, Xs1); // +1
//    mat_precondition->Dslash5inv(*ft, *fy, parity[0]);                   // +1
    mat_precondition->Dslash5invPartial(*ft, *fy, parity[0], sp_len1, RR1, Xs1);                   // +1
    mat_precondition->Dslash4pre(*fy, *ft, parity[0]);                   // +1
//    mat_precondition->Dslash4prePartial(*fy, *ft, parity[0], sp_len1, RR1, Xs1);                   // +1
    mat_precondition->Dslash4(*ft, *fy, parity[1]);                      // +2
    mat_precondition->Dslash5invXpay(*fy, *ft, parity[1], *fb, -1.0);      // +2

    mat_precondition->Dagger(QUDA_DAG_YES);
    
    mat_precondition->Dslash5inv(*ft, *fy, parity[1]);                  // +2
//    mat_precondition->Dslash4Partial(*fx, *ft, parity[0], sp_len1, RR1, Xs1); // +1
    mat_precondition->Dslash4(*fx, *ft, parity[0]);
    mat_precondition->Dslash4pre(*ft, *fx, parity[0]);                 // +1
//    mat_precondition->Dslash4prePartial(*ft, *fx, parity[0], sp_len1, RR1, Xs1);                 // +1
//    mat_precondition->Dslash5inv(*fx, *ft, parity[0]);                 // +1
    mat_precondition->Dslash5invPartial(*fx, *ft, parity[0], sp_len1, RR1, Xs1);                 // +1
//    mat_precondition->Dslash4(*ft, *fx, parity[1]);
    mat_precondition->Dslash4Partial(*ft, *fx, parity[1], sp_len0, RR0, Xs0); // +0
    mat_precondition->Dslash4preXpay(*fx, *ft, parity[1], *fy, -1.0);   // +0
//    mat_precondition->Dslash4preXpayPartial(*fx, *ft, parity[1], *fy, -1.0, sp_len0, RR0, Xs0);   // +0
    
    mat_precondition->Dagger(QUDA_DAG_NO);
*/

    inner_dslash(*fx, *fb);

    double fx2 = norm2(*fx);
    printfQuda("Test   fx2/fb2 = %16.12e/%16.12e.\n", fx2, fb2);
    zero_extended_color_spinor_interface( *fx, R, QUDA_CUDA_FIELD_LOCATION, 0);
    fx2 = norm2(*fx);
    printfQuda("Chopping   fx2 = %16.12e.\n", fx2);

    copyExtendedColorSpinor(*tx, *fx, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL); // parity = 0
    double x2_ = blas::norm2(*tx);
    printfQuda("Rebuild     x2 = %16.12e.\n", x2_);
    printfQuda("%% diff      x2 = %16.12e (This number is SUPPOSED to be tiny).\n", (x2-x2_)/x2);

    delete tx;
    delete tt;
    delete tb;
    delete fx;
    delete fy;
    delete fb;
    delete ft;

    printfQuda("dslash test completed.\n");
  }

  void MSPCG::inner_dslash( ColorSpinorField& out, const ColorSpinorField& in ){
    
    
    int odd_bit = 0; // Even-Even
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    mat_precondition->Dagger(QUDA_DAG_NO);
    
    blas::zero(*iftmp);
//    mat_precondition->Dslash4pre(*iftmp, in, parity[1]);                                // +0
    mat_precondition->Dslash4prePartial(*iftmp, in, parity[1], sp_len0, RR0, Xs0);        // +0
    
    // TODO: Test
//    mat_precondition->Dslash4(*ifset, *iftmp, parity[0]);                               // +1
//    mat_precondition->Dslash4Partial(*ifset, *iftmp, parity[0], sp_len1, RR1, Xs1, true, {2,2,2,2});       // +1
    mat_precondition->dslash4_dslash5inv_dslash4pre_partial(*ifset, *iftmp, parity[0], sp_len1, RR1, Xs1, true, {2,2,2,2});
//    mat_precondition->dslash4_dslash5inv_dslash4pre_partial(*ifset, *iftmp, parity[0], sp_len2, RR2, Xs2, false, {2,2,2,2});
//    mat_precondition->Dslash5inv(*iftmp, *ifset, parity[0]);                            // +1
//    mat_precondition->Dslash5invPartial(*iftmp, *ifset, parity[0], sp_len1, RR1, Xs1);    // +1
    blas::copy(*iftmp, *ifset);
    // TODO: Test

//    mat_precondition->Dslash4pre(*ifset, *iftmp, parity[0]);                            // +1
    mat_precondition->Dslash4prePartial(*ifset, *iftmp, parity[0], sp_len1, RR1, Xs1);    // +1
//    mat_precondition->Dslash4(*iftmp, *ifset, parity[1]);                      // +2
//    mat_precondition->Dslash4(*iftmp, *ifset, parity[1]);                      // +2
    mat_precondition->Dslash4Partial(*iftmp, *ifset, parity[1], sp_len2, RR2, Xs2, true, {1,1,1,1});                      // +2
    mat_precondition->Dslash5invXpay(*ifset, *iftmp, parity[1], in, -1.0);      // +2

    mat_precondition->Dagger(QUDA_DAG_YES);
    
//    mat_precondition->Dslash5inv(*iftmp, *ifset, parity[1]);                  // +2
    mat_precondition->Dslash5inv(*iftmp, *ifset, parity[1]);                  // +2
//    mat_precondition->Dslash4(out, *iftmp, parity[0]);
    mat_precondition->Dslash4Partial(out, *iftmp, parity[0], sp_len1, RR1, Xs1); // +1
//    mat_precondition->Dslash4pre(*iftmp, out, parity[0]);                 // +1
    mat_precondition->Dslash4prePartial(*iftmp, out, parity[0], sp_len1, RR1, Xs1);                 // +1
//    mat_precondition->Dslash5inv(out, *iftmp, parity[0]);                 // +1
    mat_precondition->Dslash5invPartial(out, *iftmp, parity[0], sp_len1, RR1, Xs1);                 // +1
//    mat_precondition->Dslash4(*iftmp, out, parity[1]);
    mat_precondition->Dslash4Partial(*iftmp, out, parity[1], sp_len0, RR0, Xs0); // +0
//    mat_precondition->Dslash4preXpay(out, *iftmp, parity[1], *ifset, -1.0);   // +0
    mat_precondition->Dslash4preXpayPartial(out, *iftmp, parity[1], *ifset, -1.0, sp_len0, RR0, Xs0);   // +0
    
    mat_precondition->Dagger(QUDA_DAG_NO);
 
  }

  void MSPCG::inner_cg(ColorSpinorField& ix, ColorSpinorField& ib )
  {
    commGlobalReductionSet(false);

    blas::zero(ix);

    double rk2 = blas::norm2(ib);
    double Mpk2, alpha, beta, rkp12;

//    printfQuda("inner_cg: before starting: r2 = %8.4e \n", rk2);
    blas::copy(*ip, ib);

    for(int local_loop_count = 0; local_loop_count < inner_iterations; local_loop_count++){
      
//      copier_timer.Start("woo", "hoo", 0);
      copyExtendedColorSpinor(*ifp, *ip, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL);
//      copier_timer.Stop("woo", "hoo", 0);
//      zero_extended_color_spinor_interface( *ifp, R, QUDA_CUDA_FIELD_LOCATION, 0);
      inner_dslash(*ifmmp, *ifp);
//      (*nrm_op_precondition)(*ifmmp, *ifp, *iftmp);
//      copier_timer.Start("woo", "hoo", 0);
      copyExtendedColorSpinor(*immp, *ifmmp, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL);
//      copier_timer.Stop("woo", "hoo", 0);
      
      Mpk2 = reDotProduct(*ip, *immp);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *ip, ix);
      rkp12 = axpyNorm(-alpha, *immp, ib);
      
      beta = rkp12 / rk2;
      rk2 = rkp12;

      xpay(ib, beta, *ip);
      
//      printfQuda("inner_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n",
//          local_loop_count, rk2, alpha, beta, Mpk2);
    }

    commGlobalReductionSet(true);
    
    return;
  }

  int MSPCG::outer_cg( ColorSpinorField& dx, ColorSpinorField& db, double quit )
  {
    double Mpk2, alpha, beta, rkp12;
    (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = nrm_op * x
    double rk2 = xmyNorm(db, *vct_dr); // r = b - nrm_op * x
    
    printfQuda("outer_cg: before starting: r2 = %8.4e \n", rk2);
    if(rk2 < quit){
      printfQuda("outer_cg: CONVERGED with ZERO effort.\n");
      return 0;
    }
    
    blas::copy(*vct_dp, *vct_dr);
    
    int loop_count;
    for(loop_count = 0; loop_count < param.maxiter; loop_count++){
      
      (*nrm_op)(*vct_dmmp, *vct_dp, *vct_dtmp);
      Mpk2 = reDotProduct(*vct_dp, *vct_dmmp);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *vct_dp, dx);
      rkp12 = axpyNorm(-alpha, *vct_dmmp, *vct_dr);

//      rkp12 = blas::norm2(*vct_dr);
      
      beta = rkp12 / rk2;
      rk2 = rkp12;
      if(rkp12 < quit) break;

      xpay(*vct_dr, beta, *vct_dp);

      printfQuda("outer_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n", loop_count, rk2, alpha, beta, Mpk2);
    }
    
    printfQuda("outer_cg: CONVERGED after %04d iterations: r2/target_r2 = %8.4e/%8.4e.\n", loop_count+1, rk2, quit);

    return loop_count;
  }

  void MSPCG::allocate(ColorSpinorField& db){
 
// initializing the fermion vectors.
    ColorSpinorParam csParam(db);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

// d* means precise
    vct_dr  =  new cudaColorSpinorField(csParam);
    vct_dp  =  new cudaColorSpinorField(csParam);
    vct_dmmp = new cudaColorSpinorField(csParam);
    vct_dtmp = new cudaColorSpinorField(csParam);
    x  =   new cudaColorSpinorField(csParam);

// sloppy
    csParam.setPrecision(dirac_param_sloppy.gauge->Precision());
    
    r  =   new cudaColorSpinorField(csParam);
    z  =   new cudaColorSpinorField(csParam);
    p  =   new cudaColorSpinorField(csParam);
    mmp  = new cudaColorSpinorField(csParam);
    tmp  = new cudaColorSpinorField(csParam);

// TODO: test
    r_old = new cudaColorSpinorField(csParam);

    csParam.setPrecision(dirac_param_precondition.gauge->Precision());

// f* means fine/preconditioning
    fr  =  new cudaColorSpinorField(csParam);
    blas::zero(*fr);

// i* means inner preconditioning
    immp=  new cudaColorSpinorField(csParam);
    ip  =  new cudaColorSpinorField(csParam);
    
    for(int i=0; i<4; ++i){
      csParam.x[i] += 2*R[i];
    }
    csParam.setPrecision(dirac_param_precondition.gauge->Precision());
    ifmmp=  new cudaColorSpinorField(csParam);
    ifp  =  new cudaColorSpinorField(csParam);
    iftmp=  new cudaColorSpinorField(csParam);
    ifset=  new cudaColorSpinorField(csParam);


    // numbers to enable faster preconditioner dslash
    for(int d = 0; d < 4; d++){
      RR2[d] = 0;
      RR1[d] = 1;
      RR0[d] = 2;
    }
    Xs2[0] = csParam.x[0]*2 - RR2[0]*2;
    Xs2[1] = csParam.x[1]   - RR2[1]*2;
    Xs2[2] = csParam.x[2]   - RR2[2]*2;
    Xs2[3] = csParam.x[3]   - RR2[3]*2;
 
    Xs1[0] = csParam.x[0]*2 - RR1[0]*2;
    Xs1[1] = csParam.x[1]   - RR1[1]*2;
    Xs1[2] = csParam.x[2]   - RR1[2]*2;
    Xs1[3] = csParam.x[3]   - RR1[3]*2;
    
    Xs0[0] = csParam.x[0]*2 - RR0[0]*2;
    Xs0[1] = csParam.x[1]   - RR0[1]*2;
    Xs0[2] = csParam.x[2]   - RR0[2]*2;
    Xs0[3] = csParam.x[3]   - RR0[3]*2;   
   
    sp_len2 = Xs2[0]*Xs2[1]*Xs2[2]*Xs2[3]/2;
    sp_len1 = Xs1[0]*Xs1[1]*Xs1[2]*Xs1[3]/2;
    sp_len0 = Xs0[0]*Xs0[1]*Xs0[2]*Xs0[3]/2;
   
  }

  void MSPCG::deallocate(){
    
    delete r;
    delete x;
    delete z;
    delete p;
    delete mmp;
    delete tmp;

    delete r_old;

    delete fr;
    
    delete immp;
    delete ip;
 
    delete ifmmp;
    delete ifp;
    delete iftmp;
    delete ifset;

    delete vct_dr;
    delete vct_dp;
    delete vct_dmmp;
    delete vct_dtmp;
   
  }

  void MSPCG::operator()(ColorSpinorField& dx, ColorSpinorField& db)
  {

    Gflops = 0.;
    fGflops = 0.;
    
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(db);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      printfQuda("Warning: inverting on zero-field source\n");
      dx = db;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }
    
    this->allocate( db );

    int k;
    //    int parity = nrm_op->getMatPCType();
    double alpha, beta, rkzk, pkApk, zkP1rkp1;

    double stop = stopping(param.tol, b2, param.residual_type);

//    test_dslash(db);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // end of initializing.

    precise_timer.Start("woo", "hoo", 0);
    for(int cycle=0; cycle<5; cycle++){
      
      (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = MdagM * x
      double r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x
      printfQuda("Cycle #%02d.\n", cycle);
      printfQuda("True precise residual is %8.4e\n", r2);
      if(r2 < stop) break;
    
      double sloppy_solver_stop = r2*param.tol*param.tol*1e4>stop ? r2*param.tol*param.tol*1e4 : stop;

      blas::copy(*r, *vct_dr); // throw true residual into the sloppy solver.
      blas::zero(*x);
//      copier_timer.Start("woo", "hoo", 0);
//      copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//      copier_timer.Stop("woo", "hoo", 0);

      blas::copy(*fr, *r);

      sloppy_timer.Start("woo", "hoo", 0);
      
      preconditioner_timer.Start("woo", "hoo", 0);
      if(inner_iterations <= 0){
        blas::copy(*z, *fr);
      }else{
        inner_cg(*z, *fr);
      }
      preconditioner_timer.Stop("woo", "hoo", 0);
 
//      copier_timer.Start("woo", "hoo", 0);
//      copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//      copier_timer.Stop("woo", "hoo", 0);
      blas::copy(*p, *z);

      k = 0;
      while( k < param.maxiter ){
        rkzk = reDotProduct(*r, *z);

        (*nrm_op_sloppy)(*mmp, *p, *tmp);
        pkApk = reDotProduct(*p, *mmp);
        alpha = rkzk / pkApk;


        axpy(alpha, *p, *x); // x_k+1 = x_k + alpha * p_k
        double rr2 = axpyNorm(-alpha, *mmp, *r); // r_k+1 = r_k - alpha * Ap_k
//        double rr2 = blas::norm2(*r);
        if(rr2 < sloppy_solver_stop) break;

        // z = M^-1 r
//        copier_timer.Start("woo", "hoo", 0);
//        copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//        copier_timer.Stop("woo", "hoo", 0);

        blas::copy(*fr, *r);

        preconditioner_timer.Start("woo", "hoo", 0);
        if(inner_iterations <= 0){
          blas::copy(*z, *fr);
        }else{
          inner_cg(*z, *fr);
        }
        preconditioner_timer.Stop("woo", "hoo", 0);
        
//        copier_timer.Start("woo", "hoo", 0);
//        copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//        copier_timer.Stop("woo", "hoo", 0);
// TODO: test
//        double diff_real =  cDotProduct(*z, *r_old).real();
//        double diff_imag =  cDotProduct(*z, *r_old).imag();
//        printfQuda("MSPCG/iter.count/diff: %05d %8.4e +i %8.4e\n", k, diff_real, diff_imag);
// TODO:
        
        zkP1rkp1 = reDotProduct(*z, *r);
        beta = zkP1rkp1 / rkzk;
//        beta = (zkP1rkp1-diff_real) / rkzk;
        xpay(*z, beta, *p);

        double zz2 = blas::norm2(*z);
        printfQuda("z2/r2: %8.4e/%8.4e.\n", zz2, rr2);

        ++k;
        printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop, std::sqrt(rr2/b2), param.tol);

      }

      sloppy_timer.Stop("woo", "hoo", 0);
      
      blas::copy(*vct_dtmp, *x);
      xpy(*vct_dtmp, dx);

    }
    
    
    k = outer_cg(dx, db, stop);

    precise_timer.Stop("woo", "hoo", 0);
    
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    param.iter += k;

    double precise_tflops = nrm_op->flops()*1e-12;
    double sloppy_tflops = nrm_op_sloppy->flops()*1e-12;
    double preconditioner_tflops = (nrm_op_precondition->flops()+blas::flops)*1e-12;
    reduceDouble(precise_tflops);
    reduceDouble(sloppy_tflops);
    reduceDouble(preconditioner_tflops);
    param.gflops = (preconditioner_tflops+sloppy_tflops+precise_tflops)*1e3;

    double prec_time = preconditioner_timer.time; 
//    reduceMaxDouble(prec_time);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual 

    (*nrm_op)(*vct_dr, dx, *vct_dtmp);
    double true_res = xmyNorm(db, *vct_dr);
    param.true_res = sqrt(true_res/b2);

    printfQuda("True residual/target_r2: %8.4e/%8.4e.\n", true_res, stop);
    printfQuda("Performance precise:        %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      precise_tflops/(precise_timer.time-sloppy_timer.time), precise_timer.time-sloppy_timer.time, precise_timer.count);
    printfQuda("Performance sloppy:         %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      sloppy_tflops/(sloppy_timer.time-prec_time), sloppy_timer.time-prec_time, sloppy_timer.count);
    printfQuda("Performance preconditioner: %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      preconditioner_tflops/prec_time, prec_time, preconditioner_timer.count);
    printfQuda("Performance copier:                         in %8.4f secs with %05d calls.\n",
      copier_timer.time, copier_timer.count);

    // reset the flops counters
    blas::flops = 0;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    deallocate();
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

  void MSPCG::reliable_update(ColorSpinorField& dx, ColorSpinorField& db)
  {

    Gflops = 0.;
    fGflops = 0.;
    
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(db);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      printfQuda("Warning: inverting on zero-field source\n");
      dx = db;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }
    
    this->allocate( db );

    int k;
    //    int parity = nrm_op->getMatPCType();
    double alpha, beta, rkzk, pkApk, zkP1rkp1;
    double stop = stopping(param.tol, b2, param.residual_type);

    test_dslash(db);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // end of initializing.


// START of the main loop

    (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = MdagM * x
    double r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x
    printfQuda("True precise residual is %8.4e\n", r2);

    double r2_max = r2;
    int num_reliable_updates = 0;
// reliable update
		
		const double u = param.precision_sloppy==8?std::numeric_limits<double>::epsilon()/2.:((param.precision_sloppy==4)?std::numeric_limits<float>::epsilon()/2.:pow(2.,-13));
		const double uhigh = param.precision==8?std::numeric_limits<double>::epsilon()/2.:((param.precision==4)?std::numeric_limits<float>::epsilon()/2.:pow(2.,-13));
		const double deps = sqrt(u);
		const double dfac = 1.1;
		double d_new = 0.;
		double d = 0.;
		double dinit = 0.;

		double xnorm = 0.;
		double pnorm = 0.;
		double ppnorm = 0.;

		double rNorm = sqrt(r2);
		double r2_old;
		double r0Norm = rNorm;

		(*nrm_op)(*vct_dp, db, *vct_dtmp);
		double Anorm = sqrt( blas::norm2(*vct_dp)/b2 ); // the matrix norm

		dinit = uhigh * rNorm;
		d = dinit;

// reliable update


    blas::copy(*r, *vct_dr); // throw true residual into the sloppy solver.
    blas::zero(*x);
    //      copier_timer.Start("woo", "hoo", 0);
    //      copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
    //      copier_timer.Stop("woo", "hoo", 0);

    blas::copy(*fr, *r);

    sloppy_timer.Start("woo", "hoo", 0);

		double rr2 = r2;

    preconditioner_timer.Start("woo", "hoo", 0);
    if(inner_iterations <= 0){
      blas::copy(*z, *fr);
    }else{
      inner_cg(*z, *fr);
    }
    preconditioner_timer.Stop("woo", "hoo", 0);

    //      copier_timer.Start("woo", "hoo", 0);
    //      copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
    //      copier_timer.Stop("woo", "hoo", 0);
    blas::copy(*p, *z);

    k = 0;
    while( k < param.maxiter ){
      rkzk = reDotProduct(*r, *z);

      (*nrm_op_sloppy)(*mmp, *p, *tmp);
      
			r2_old = rr2;
			double3 pAppp = blas::cDotProductNormA(*p, *mmp);
//			pkApk = reDotProduct(*p, *mmp);
			pkApk = pAppp.x;
			ppnorm = pAppp.z;

			alpha = rkzk / pkApk;

      blas::copy(*r_old, *r);

      axpy(alpha, *p, *x); // x_k+1 = x_k + alpha * p_k
      rr2 = axpyNorm(-alpha, *mmp, *r); // r_k+1 = r_k - alpha * Ap_k
			rNorm = sqrt(rr2);

      // reliable update
   
      
      if( rr2 > r2_max ) r2_max = rr2;
//      if( rr2 < reliable_update_delta*reliable_update_delta*r2_max || rr2 < stop ){
      if( rr2 < stop or ( ( (d <= deps*sqrt(r2_old)) or (dfac*dinit > deps*r0Norm) ) and (d_new > deps*rNorm) and (d_new > dfac*dinit) ) ){
				
				printfQuda("Reliable update conditions: \n    d_n-1 < eps*r2_old: %8.4e < %8.4e,\n    dn    > eps*r_n: %8.4e    > %8.4e,\n    dnew  > 1.1*dinit: %8.4e  > (1.1*)%8.4e.\n",
	   			d, deps*sqrt(r2_old), d_new,deps*rNorm, d_new, dinit);

        precise_timer.Start("woo", "hoo", 0);
        
        blas::copy(*vct_dtmp, *x);
        xpy(*vct_dtmp, dx);
        
        (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = MdagM * x
        r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x
        
        blas::copy(*r, *vct_dr);
        blas::zero(*x);
        
				dinit = uhigh*( sqrt(r2) + Anorm*sqrt(blas::norm2(dx)) );
				d = d_new;
				xnorm = 0.;
				pnorm = 0.;

        r2_max = r2;

        num_reliable_updates++;
        printfQuda( "reliable update: sloppy r2 = %8.4e; precise r2 = %8.4e.\n", rr2, r2 );
        
        rr2 = r2;

				d_new = dinit;

        precise_timer.Stop("woo", "hoo", 0);
      }else{
				d = d_new;
				pnorm = pnorm + alpha*alpha*ppnorm;
				xnorm = sqrt(pnorm);
				d_new = d + u*rNorm + uhigh*Anorm*xnorm;
			}
      
      
      if(rr2 < stop) break;

      // z = M^-1 r
      //        copier_timer.Start("woo", "hoo", 0);
      //        copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      //        copier_timer.Stop("woo", "hoo", 0);

      blas::copy(*fr, *r);

      preconditioner_timer.Start("woo", "hoo", 0);
      if(inner_iterations <= 0){
        blas::copy(*z, *fr);
      }else{
        inner_cg(*z, *fr);
      }
      preconditioner_timer.Stop("woo", "hoo", 0);

      //        copier_timer.Start("woo", "hoo", 0);
      //        copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      //        copier_timer.Stop("woo", "hoo", 0);
      // TODO: test
      //        double diff_real =  cDotProduct(*z, *r_old).real();
      //        double diff_imag =  cDotProduct(*z, *r_old).imag();
      //        printfQuda("MSPCG/iter.count/diff: %05d %8.4e +i %8.4e\n", k, diff_real, diff_imag);
      // TODO:

      xpay(*r, -1., *r_old);
      
      zkP1rkp1 = reDotProduct(*z, *r_old);
//      zkP1rkp1 = reDotProduct(*z, *r);
      beta = zkP1rkp1 / rkzk;
      //        beta = (zkP1rkp1-diff_real) / rkzk;
      xpay(*z, beta, *p);

//      double zz2 = blas::norm2(*z);
//      printfQuda("z2/r2: %8.4e/%8.4e.\n", zz2, rr2);

      ++k;
      printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop, std::sqrt(rr2/b2), param.tol);

    }

    sloppy_timer.Stop("woo", "hoo", 0);

// END of main loop 
    
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    param.iter += k;

    double precise_tflops = nrm_op->flops()*1e-12;
    double sloppy_tflops = nrm_op_sloppy->flops()*1e-12;
    double preconditioner_tflops = (nrm_op_precondition->flops()+blas::flops)*1e-12;
    
    param.gflops = (preconditioner_tflops+sloppy_tflops+precise_tflops)*1e3;
    
    reduceDouble(precise_tflops);
    reduceDouble(sloppy_tflops);
    reduceDouble(preconditioner_tflops);

    double prec_time = preconditioner_timer.time; 
//    reduceMaxDouble(prec_time);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual 

    (*nrm_op)(*vct_dr, dx, *vct_dtmp);
    double true_res = xmyNorm(db, *vct_dr);
    param.true_res = sqrt(true_res/b2);

    printfQuda("-------- END --------\n");
    printfQuda("MSPCG CONVERGED in %05d iterations with %03d reliable updates.\n", k, num_reliable_updates);
    printfQuda("True residual/target_r2: %8.4e/%8.4e.\n", true_res, stop);
    printfQuda("Performance precise:        %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      precise_tflops/precise_timer.time, precise_timer.time, precise_timer.count);
    printfQuda("Performance sloppy:         %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      sloppy_tflops/(sloppy_timer.time-prec_time-precise_timer.time), sloppy_timer.time-prec_time-precise_timer.time, sloppy_timer.count);
    printfQuda("Performance preconditioner: %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      preconditioner_tflops/prec_time, prec_time, preconditioner_timer.count);
    printfQuda("Performance copier:                         in %8.4f secs with %05d calls.\n",
      copier_timer.time, copier_timer.count);
    printfQuda("Flops ratio sloppy/preconditioner: %.2f.\n", preconditioner_tflops/sloppy_tflops/(double)inner_iterations);

    // reset the flops counters
    blas::flops = 0;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    deallocate();
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
