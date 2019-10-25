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

#include <mpi.h>

#include <gauge_tools.h>

#include <copy_color_spinor_field_5d.h>

#include <random>

extern quda::cudaGaugeField* gaugePrecondition;
extern quda::cudaGaugeField* gaugePrecise;
extern quda::cudaGaugeField* gaugeSloppy;

namespace quda {

  using namespace blas;

  static cudaGaugeField* createExtendedGauge(cudaGaugeField& in, const int* R, TimeProfile& profile,
      bool redundant_comms = false, QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID) {
    int y[4];
    for (int dir = 0; dir < 4; ++dir) y[dir] = in.X()[dir] + 2 * R[dir];
    int pad = 0;

    GaugeFieldParam gParamEx(y, in.Precision(), recon != QUDA_RECONSTRUCT_INVALID ? recon : in.Reconstruct(), pad,
        in.Geometry(), QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = in.Order();
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = in.TBoundary();
    gParamEx.nFace = 1;
    gParamEx.tadpole = in.Tadpole();
    for (int d = 0; d < 4; d++) gParamEx.r[d] = R[d];

    gParamEx.setPrecision(in.Precision(), true);

    cudaGaugeField* out = new cudaGaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);

    // now fill up the halos
    profile.TPSTART(QUDA_PROFILE_COMMS);
    out->exchangeExtendedGhost(R, redundant_comms);
    profile.TPSTOP(QUDA_PROFILE_COMMS);

    return out;
  }

  static void set_mobius_dirac_param(DiracParam& diracParam, QudaInvertParam* inv_param, const bool pc) {
    double kappa = inv_param->kappa;
    if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) { kappa *= gaugePrecise->Anisotropy(); }

    if (inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH) { errorQuda("ONLY Mobius.\n"); }

    diracParam.type = pc ? QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC : QUDA_MOBIUS_DOMAIN_WALL_DIRAC;
    diracParam.Ls = inv_param->Ls;
    memcpy(diracParam.b_5, inv_param->b_5, sizeof(double _Complex) * inv_param->Ls);
    memcpy(diracParam.c_5, inv_param->c_5, sizeof(double _Complex) * inv_param->Ls);

    if (inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC
        || inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      errorQuda("Currently MSPCG does NOT support asymmetric preconditioning.\n");
    }
    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = gaugePrecise;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;

    for (int i = 0; i < 4; i++) diracParam.commDim[i] = 1; // comms are always on
  }
  
  MSPCG::MSPCG(QudaInvertParam* inv_param, SolverParam& param_, TimeProfile& profile)
      : Solver(param_, profile)
      , solver_prec(0)
      , solver_prec_param(param_)
      , mat(nullptr)
      , mat_sloppy(nullptr)
      , mat_precondition(nullptr)
      , nrm_op(nullptr)
      , nrm_op_sloppy(nullptr)
      , nrm_op_precondition(nullptr)
      , vct_dr(nullptr)
      , vct_dp(nullptr)
      , vct_dmmp(nullptr)
      , vct_dtmp(nullptr)
      , vct_dtmp2(nullptr)
      , r(nullptr)
      , p(nullptr)
      , z(nullptr)
      , mmp(nullptr)
      , tmp(nullptr)
      , tmp2(nullptr)
      , fr(nullptr)
      , fz(nullptr)
      , immp(nullptr)
      , ip(nullptr)
      , ifmmp(nullptr)
      , ifp(nullptr)
      , iftmp(nullptr)
      , ifset(nullptr)
      , inner_iterations(param.maxiter_precondition) {

    printfQuda("MSPCG constructor starts.\n");

    R[0] = 2;
    R[1] = 2;
    R[2] = 2;
    R[3] = 2;

    if (inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH) { errorQuda("ONLY works for QUDA_MOBIUS_DWF_DSLASH."); }

    // create extended gauge field
    if (not gaugePrecondition) { errorQuda("gaugePrecondition not valid."); }

    if (not gaugeSloppy) { errorQuda("gaugeSloppy not valid."); }

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    if (device_prop.major < 7) {
      tc = false;
      errorQuda("Sorry we need Volta(device_prop.major >= 7 and you have %d) for this to work.\n", device_prop.major);
    } else {
      tc = true;
      printfQuda("We will be using the tensor core implementation.\n");
    }

    int gR[4] = {R[0], R[1], R[2], R[3]};
    padded_gauge_field = createExtendedGauge(*gaugePrecise, gR, profile, true);
    padded_gauge_field_precondition = createExtendedGauge(*gaugePrecondition, gR, profile, true);

    set_mobius_dirac_param(dirac_param, inv_param, true); // pc = true

    set_mobius_dirac_param(dirac_param_sloppy, inv_param, true); // pc = true
    dirac_param_sloppy.gauge = gaugeSloppy;

    set_mobius_dirac_param(dirac_param_precondition, inv_param, true); // pc = true
    dirac_param_precondition.gauge = padded_gauge_field_precondition;

    for (int i = 0; i < 4; i++) {
      dirac_param_precondition.commDim[i] = 0;
    }

    mat = new DiracMobiusPC(dirac_param);
    nrm_op = new DiracMdagM(mat);

    mat_sloppy = new DiracMobiusPC(dirac_param_sloppy);
    nrm_op_sloppy = new DiracMdagM(mat_sloppy);

    mat_precondition = new DiracMobiusPC(dirac_param_precondition);
    nrm_op_precondition = new DiracMdagMLocal(mat_precondition);
    
    const char fname[] = "MSPCG::MSPCG(QudaInvertParam*, SolverParam&, TimeProfile&)";
    const char cname[] = __FILE__;

    copier_timer.Reset(fname, cname, __LINE__);
    precise_timer.Reset(fname, cname, __LINE__);
    sloppy_timer.Reset(fname, cname, __LINE__);
    preconditioner_timer.Reset("woo", cname, __LINE__);
    for (int i = 0; i < 2; i++){
      linalg_timer[i].Reset(fname, cname, __LINE__);
    }
    
    printfQuda("MSPCG constructor ends.\n");
  }

  MSPCG::~MSPCG() {
    profile.TPSTART(QUDA_PROFILE_FREE);
    
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

  void MSPCG::inner_dslash(ColorSpinorField& out, const ColorSpinorField& in) {
    (*nrm_op_precondition)(out, in);
  }

  static int transfer_index_from_st(int s, int t, int Ls_in, int Ls_out){
    return (s * Ls_in + t);
  }

  double MSPCG::calculate_chi(ColorSpinorField& out, const ColorSpinorField& in, std::vector<float>& v, int Ls_hat){
    
    ColorSpinorParam csParam(in);
    
    cudaColorSpinorField tmp(csParam);
    
    csParam.x[4] = Ls_hat;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    
    dirac_param_precondition.Ls = Ls_hat;
    
    DiracMobiusPC mat_precondition_truncated(dirac_param_precondition);

    cudaColorSpinorField truncated_cs_field_in(csParam);
    cudaColorSpinorField truncated_cs_field_out(csParam);
   
    // T * phi
    transfer_5d_hh(truncated_cs_field_in, in, reinterpret_cast<complex<float>*>(v.data()), false);
    // A * T * phi
    mat_precondition_truncated.MdagMLocal(truncated_cs_field_out, truncated_cs_field_in);
    // T^ * A * T * phi
    transfer_5d_hh(tmp, truncated_cs_field_out, reinterpret_cast<complex<float>*>(v.data()), true);

    // B * phi
    inner_dslash(out, in);
   
    // T^ * A * T * phi - B * phi
    return xmyNorm(tmp, out);
  }
  
  void MSPCG::ATx(ColorSpinorField& out,  const ColorSpinorField& in, std::vector<float>& v){
  
    int Ls_out = out.X(4);

    ColorSpinorParam csParam(in);
    
    csParam.x[4] = Ls_out;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    
    dirac_param_precondition.Ls = Ls_out;
    
    DiracMobiusPC mat_precondition_truncated(dirac_param_precondition);
    cudaColorSpinorField truncated_cs_field_in(csParam);
   
    // T * phi
    transfer_5d_hh(truncated_cs_field_in, in, reinterpret_cast<complex<float>*>(v.data()), false);
    // A * T * phi
    mat_precondition_truncated.MdagMLocal(out, truncated_cs_field_in);
  }

  void fill_random(std::vector<float>& v){
    static std::random_device rd;
		// the good rng
  	static std::mt19937 rng(23ul*comm_rank());
		// The gaussian distribution
  	static std::normal_distribution<double> n(0., 1.);
   
    for(auto& x : v){
      x = 1e-1*n(rng);
    }
  
  }

  void MSPCG::inner_cg(ColorSpinorField& ix, ColorSpinorField& ib) {
  
    commGlobalReductionSet(false);
    
    constexpr int color_spin_dim = 12;

    int Ls_in = ib.X(4);
    int Ls_out = 8;
  
    std::vector<float> v(Ls_in*Ls_out*color_spin_dim*color_spin_dim*2, 0.0f);
    fill_random(v);
    
    ColorSpinorParam csParam(ib);
    cudaColorSpinorField chi(csParam);
    cudaColorSpinorField phi(csParam);
   
    phi = ib;

    csParam.x[4] = Ls_out;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
 
    cudaColorSpinorField ATchi(csParam);
    cudaColorSpinorField ATphi(csParam);
    
    std::vector<float> d1(Ls_in*Ls_out*color_spin_dim*color_spin_dim*2, 0.0f);
    std::vector<float> d2(Ls_in*Ls_out*color_spin_dim*color_spin_dim*2, 0.0f);

    float alpha = 1e-8f;

    for(int iteration = 0; iteration < 10000; iteration++){
      
      double chi2 = calculate_chi(chi, phi, v, Ls_out);
     
      printfQuda("gradient min iteration %04d chi2 = %8.4e\n", iteration, chi2);

      ATx(ATphi, phi, v);
      ATx(ATchi, chi, v);

      tensor_5d_hh(ATphi, chi, reinterpret_cast<complex<float>*>(d1.data()));
      tensor_5d_hh(ATchi, phi, reinterpret_cast<complex<float>*>(d2.data()));
    
      for(int i = 0; i < v.size(); i++){
        v[i] -= alpha * 2.0f * (d1[i] + d2[i]);
      }

    }
      
    blas::copy(ix, ib);

#if 0
    
    double d = 0.4;
    for(int i = 0; i < 197; i++){
      v[i] += d/2;
      double chi2_p = calculate_chi(chi, phi, v, Ls_out);
      v[i] -= d;
      double chi2_m = calculate_chi(chi, phi, v, Ls_out);
      printfQuda("d_chi2 norm [%04d]  = %f vs. %f \n", i, (chi2_p - chi2_m)/d, (2*(d1[i]+d2[i])));
      v[i] += d/2;
    }

#endif

#if 0

    blas::zero(ix);
    double rk2 = blas::norm2(ib);
    if(rk2 == 0.0){
      commGlobalReductionSet(true);
      return;
    }
    double Mpk2, alpha, beta, rkp12;
    blas::copy(*ip, ib);
    for (int local_loop_count = 0; local_loop_count < inner_iterations; local_loop_count++) {
      inner_dslash(*immp, *ip);
      Mpk2 = reDotProduct(*ip, *immp);
      alpha = rk2 / Mpk2;
      rkp12 = axpyNorm(-alpha, *immp, ib);
      beta = rkp12 / rk2;
      rk2 = rkp12;
      axpyZpbx(alpha, *ip, ix, ib, beta);
    }

#endif
    
    commGlobalReductionSet(true);
    return;
  }

  int MSPCG::outer_cg(ColorSpinorField& dx, ColorSpinorField& db, double quit) {
    double Mpk2, alpha, beta, rkp12;
    (*nrm_op)(*vct_dr, dx, *vct_dtmp, *vct_dtmp2); // r = nrm_op * x
    double rk2 = xmyNorm(db, *vct_dr); // r = b - nrm_op * x

    printfQuda("outer_cg: before starting: r2 = %8.4e \n", rk2);
    if (rk2 < quit) {
      printfQuda("outer_cg: CONVERGED with ZERO effort.\n");
      return 0;
    }

    blas::copy(*vct_dp, *vct_dr);

    int loop_count;
    for (loop_count = 0; loop_count < param.maxiter; loop_count++) {

      (*nrm_op)(*vct_dmmp, *vct_dp, *vct_dtmp, *vct_dtmp2);
      Mpk2 = reDotProduct(*vct_dp, *vct_dmmp);

      alpha = rk2 / Mpk2;

      axpy(alpha, *vct_dp, dx);
      rkp12 = axpyNorm(-alpha, *vct_dmmp, *vct_dr);

      beta = rkp12 / rk2;
      rk2 = rkp12;
      if (rkp12 < quit) break;

      xpay(*vct_dr, beta, *vct_dp);

      printfQuda(
          "outer_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n", loop_count, rk2, alpha, beta, Mpk2);
    }

    printfQuda("outer_cg: CONVERGED after %04d iterations: r2/target_r2 = %8.4e/%8.4e.\n", loop_count + 1, rk2, quit);

    return loop_count;
  }

  void MSPCG::Minv(ColorSpinorField& out, const ColorSpinorField& in) {
    preconditioner_timer.Start("woo", "hoo", 0);
    if (inner_iterations <= 0) {
      blas::copy(out, in);
    } else {
      blas::copy(*fr, in);
      if (dirac_param_sloppy.gauge->Precision() == dirac_param_precondition.gauge->Precision()) {
        inner_cg(out, *fr);
      } else {
        inner_cg(*fz, *fr);
        blas::copy(out, *fz);
      }
    }
    preconditioner_timer.Stop("woo", "hoo", 0);
  }

  void MSPCG::allocate(ColorSpinorField& db) {

    // initializing the fermion vectors.
    ColorSpinorParam csParam(db);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    // d* means precise
    vct_dr = new cudaColorSpinorField(csParam);
    vct_dp = new cudaColorSpinorField(csParam);
    vct_dmmp = new cudaColorSpinorField(csParam);
    vct_dtmp = new cudaColorSpinorField(csParam);
    vct_dtmp2 = new cudaColorSpinorField(csParam);
    x = new cudaColorSpinorField(csParam);

    // sloppy
    csParam.setPrecision(dirac_param_sloppy.gauge->Precision());

    r = new cudaColorSpinorField(csParam);
    z = new cudaColorSpinorField(csParam);
    p = new cudaColorSpinorField(csParam);
    mmp = new cudaColorSpinorField(csParam);
    tmp = new cudaColorSpinorField(csParam);
    tmp2 = new cudaColorSpinorField(csParam);

    r_old = new cudaColorSpinorField(csParam);

    csParam.setPrecision(dirac_param_precondition.gauge->Precision());

    // f* means fine/preconditioning
    fr = new cudaColorSpinorField(csParam);
    fz = new cudaColorSpinorField(csParam);

    // i* means inner preconditioning
    immp = new cudaColorSpinorField(csParam);
    ip = new cudaColorSpinorField(csParam);

    csParam.x[0] += 2 * R[0] / 2; // x direction is checkerboarded
    for (int i = 1; i < 4; ++i) { csParam.x[i] += 2 * R[i]; }
    csParam.setPrecision(dirac_param_precondition.gauge->Precision());
    ifmmp = new cudaColorSpinorField(csParam);
    ifp = new cudaColorSpinorField(csParam);
    iftmp = new cudaColorSpinorField(csParam);
    ifset = new cudaColorSpinorField(csParam);

    // numbers to enable faster preconditioner dslash
    for (int d = 0; d < 4; d++) {
      RR2[d] = 0;
      RR1[d] = 1;
      RR0[d] = 2;
    }
    Xs2[0] = csParam.x[0] * 2 - RR2[0] * 2;
    Xs2[1] = csParam.x[1] - RR2[1] * 2;
    Xs2[2] = csParam.x[2] - RR2[2] * 2;
    Xs2[3] = csParam.x[3] - RR2[3] * 2;

    Xs1[0] = csParam.x[0] * 2 - RR1[0] * 2;
    Xs1[1] = csParam.x[1] - RR1[1] * 2;
    Xs1[2] = csParam.x[2] - RR1[2] * 2;
    Xs1[3] = csParam.x[3] - RR1[3] * 2;

    Xs0[0] = csParam.x[0] * 2 - RR0[0] * 2;
    Xs0[1] = csParam.x[1] - RR0[1] * 2;
    Xs0[2] = csParam.x[2] - RR0[2] * 2;
    Xs0[3] = csParam.x[3] - RR0[3] * 2;

    sp_len2 = Xs2[0] * Xs2[1] * Xs2[2] * Xs2[3] / 2;
    sp_len1 = Xs1[0] * Xs1[1] * Xs1[2] * Xs1[3] / 2;
    sp_len0 = Xs0[0] * Xs0[1] * Xs0[2] * Xs0[3] / 2;
  }

  void MSPCG::deallocate() {

    delete r;
    delete x;
    delete z;
    delete p;
    delete mmp;
    delete tmp;
    delete tmp2;

    delete r_old;

    delete fr;
    delete fz;

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
    delete vct_dtmp2;
  }

  void MSPCG::operator()(ColorSpinorField& dx, ColorSpinorField& db) {

    const char fname[] = "MSPCG::operator()(ColorSpinorField&, ColorSpinorField&)";
    const char cname[] = __FILE__;

    Gflops = 0.;
    fGflops = 0.;

    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(db);
    if (b2 == 0.) {
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      printfQuda("Warning: inverting on zero-field source\n");
      dx = db;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }

    this->allocate(db);

    int k;
    //    int parity = nrm_op->getMatPCType();
    double alpha, beta, rkzk, pkApk, zkP1rkp1;
    double stop = stopping(param.tol, b2, param.residual_type);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // end of initializing.

    // START of the main loop

    precise_timer.Start(fname, cname, __LINE__);
    (*nrm_op)(*vct_dr, dx, *vct_dtmp, *vct_dtmp2); // r = MdagM * x
    precise_timer.Stop(fname, cname, __LINE__);

    double r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x
    printfQuda("True precise residual is %8.4e\n", r2);
    printfQuda("Using a sophisticated reliable update scheme.\n");

    double r2_max = r2;
    int num_reliable_updates = 0;
    // reliable update

    constexpr double fp16_eps = std::pow(2., -13);
    const double u = param.precision_sloppy == 8
        ? std::numeric_limits<double>::epsilon() / 2.
        : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() / 2. : fp16_eps);
    const double uhigh = param.precision == 8
        ? std::numeric_limits<double>::epsilon() / 2.
        : ((param.precision == 4) ? std::numeric_limits<float>::epsilon() / 2. : fp16_eps);
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

    (*nrm_op)(*vct_dp, db, *vct_dtmp, *vct_dtmp2);
    double Anorm = sqrt(blas::norm2(*vct_dp) / b2); // the matrix norm

    dinit = uhigh * rNorm;
    d = dinit;

    // reliable update

    blas::copy(*r, *vct_dr); // throw true residual into the sloppy solver.
    blas::zero(*x);

    double rr2 = r2;

    Minv(*z, *r);

    blas::copy(*p, *z);

    k = 0;
    while (k < param.maxiter) {

      sloppy_timer.Start(fname, cname, __LINE__);
      (*nrm_op_sloppy)(*mmp, *p, *tmp, *tmp2);
      sloppy_timer.Stop(fname, cname, __LINE__);

      linalg_timer[0].Start(fname, cname, __LINE__);
      r2_old = rr2;
#if 0
      rkzk = reDotProduct(*r, *z);
      double3 pAppp = blas::cDotProductNormA(*p, *mmp);
#else
      // single multi-reduction that computes all the required inner products
      // ||p||^2, (r, z), ||M p||^2 (note that tmp2 contains M*p vector)
      std::vector<ColorSpinorField*> lhs {p, tmp2, r};
      std::vector<ColorSpinorField*> rhs {p, tmp2, z};
      Complex dot[9];
      cDotProduct(dot, lhs, rhs);
      double3 pAppp = make_double3(dot[4].real(), 0.0, dot[0].real());
      rkzk = dot[8].real();
#endif
      // printfQuda("rkzk = %e\n", rkzk);
      // for (int i=0; i<6; i++) printfQuda("%d %e %e\n", i, dot[i].real(), dot[i].imag());

      //      pkApk = reDotProduct(*p, *mmp);
      pkApk = pAppp.x;
      ppnorm = pAppp.z;

      alpha = rkzk / pkApk;

      blas::copy(*r_old, *r);

      axpy(alpha, *p, *x); // x_k+1 = x_k + alpha * p_k
      rr2 = axpyNorm(-alpha, *mmp, *r); // r_k+1 = r_k - alpha * Ap_k
      rNorm = sqrt(rr2);

      // the more sophisticated reliable update
      if (rr2 > r2_max) r2_max = rr2;
      if (rr2 < stop
          or (((d <= deps * sqrt(r2_old)) or (dfac * dinit > deps * r0Norm)) and (d_new > deps * rNorm)
              and (d_new > dfac * dinit))) {

        printfQuda("Reliable update conditions: \n    d_n-1 < eps*r2_old: %8.4e < %8.4e,\n    dn    > eps*r_n: %8.4e   "
                   " > %8.4e,\n    dnew  > 1.1*dinit: %8.4e  > (1.1*)%8.4e.\n",
            d, deps * sqrt(r2_old), d_new, deps * rNorm, d_new, dinit);

        precise_timer.Start("woo", "hoo", 0);

        blas::copy(*vct_dtmp, *x);
        xpy(*vct_dtmp, dx);

        (*nrm_op)(*vct_dr, dx, *vct_dtmp, *vct_dtmp2); // r = MdagM * x
        r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x

        blas::copy(*r, *vct_dr);
        blas::zero(*x);

        dinit = uhigh * (sqrt(r2) + Anorm * sqrt(blas::norm2(dx)));
        d = d_new;
        xnorm = 0.;
        pnorm = 0.;

        r2_max = r2;

        num_reliable_updates++;
        printfQuda("reliable update: sloppy r2 = %8.4e; precise r2 = %8.4e.\n", rr2, r2);

        rr2 = r2;

        d_new = dinit;

        precise_timer.Stop("woo", "hoo", 0);
      } else {
        d = d_new;
        pnorm = pnorm + alpha * alpha * ppnorm;
        xnorm = sqrt(pnorm);
        d_new = d + u * rNorm + uhigh * Anorm * xnorm;
      }

      if (rr2 < stop) break;

      linalg_timer[0].Stop(fname, cname, __LINE__);

      Minv(*z, *r);

      linalg_timer[1].Start(fname, cname, __LINE__);
      // replace with fused kernel?
      xpay(*r, -1., *r_old);
      zkP1rkp1 = reDotProduct(*z, *r_old);

      // zkP1rkp1 = reDotProduct(*z, *r);
      beta = zkP1rkp1 / rkzk;
      xpay(*z, beta, *p);

      linalg_timer[1].Stop(fname, cname, __LINE__);

      k++;
      printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop,
          std::sqrt(rr2 / b2), param.tol);
    }

    // END of main loop

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    param.iter += k;

    double precise_tflops = nrm_op->flops() * 1e-12;
    double sloppy_tflops = nrm_op_sloppy->flops() * 1e-12;
    double preconditioner_tflops = nrm_op_precondition->flops() * 1e-12;
    double linalg_tflops = blas::flops * 1e-12;

    param.gflops = (preconditioner_tflops + sloppy_tflops + precise_tflops + linalg_tflops) * 1e3;

    reduceDouble(precise_tflops);
    reduceDouble(sloppy_tflops);
    reduceDouble(preconditioner_tflops);
    reduceDouble(linalg_tflops);

    double prec_time = preconditioner_timer.time;
    //    reduceMaxDouble(prec_time);

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d.\n", param.maxiter);

    // compute the true residual

    (*nrm_op)(*vct_dr, dx, *vct_dtmp, *vct_dtmp2);
    double true_res = xmyNorm(db, *vct_dr);
    param.true_res = sqrt(true_res / b2);

    printfQuda("-------- END --------\n");
    printfQuda("MSPCG CONVERGED in %05d iterations(with %02d inner_iterations each) with %03d reliable updates.\n", k,
        inner_iterations, num_reliable_updates);
    printfQuda("True residual/target_r2: %8.4e/%8.4e.\n", true_res, stop);
    printfQuda("Performance precise:        %8.2f TFLOPS in %8.2f secs(%02d%%) with %05d calls.\n",
        precise_tflops / precise_timer.time, precise_timer.time, int(precise_timer.time / param.secs * 100.),
        precise_timer.count);
    printfQuda("Performance sloppy:         %8.2f TFLOPS in %8.2f secs(%02d%%) with %05d calls.\n",
        sloppy_tflops / sloppy_timer.time, sloppy_timer.time, int(sloppy_timer.time / param.secs * 100.),
        sloppy_timer.count);
    printfQuda("Performance preconditioner: %8.2f TFLOPS in %8.2f secs(%02d%%) with %05d calls.\n",
        preconditioner_tflops / prec_time, prec_time, int(prec_time / param.secs * 100.), preconditioner_timer.count);
    for (int i = 0; i < 2; i++) {
      printfQuda("Performance linear algebra [%d]:             in %8.2f secs(%02d%%) with %05d calls.\n", i,
          linalg_timer[i].time, int(linalg_timer[i].time / param.secs * 100.), linalg_timer[i].count);
    }
    printfQuda("Total time %8.2f secs.\n", param.secs);

    // reset the flops counters
    blas::flops = 0;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    deallocate();
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
