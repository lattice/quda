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
#include <madwf_ml.h>

#include <random>
#include <deque>
#include <polynomial.h>

extern quda::cudaGaugeField *gaugePrecondition;
extern quda::cudaGaugeField *gaugePrecise;
extern quda::cudaGaugeField *gaugeSloppy;

namespace quda
{

  using namespace blas;

  static cudaGaugeField *createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile,
                                             bool redundant_comms = false,
                                             QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID)
  {
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

    cudaGaugeField *out = new cudaGaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);

    // now fill up the halos
    profile.TPSTART(QUDA_PROFILE_COMMS);
    out->exchangeExtendedGhost(R, redundant_comms);
    profile.TPSTOP(QUDA_PROFILE_COMMS);

    return out;
  }

  static void set_mobius_dirac_param(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
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

  MSPCG::MSPCG(QudaInvertParam *inv_param, SolverParam &param_, TimeProfile &profile) :
    Solver(param_, profile),
    solver_prec(0),
    solver_prec_param(param_),
    mat(nullptr),
    mat_sloppy(nullptr),
    mat_precondition(nullptr),
    mat_precondition_truncated(nullptr),
    nrm_op(nullptr),
    nrm_op_sloppy(nullptr),
    nrm_op_precondition(nullptr),
    nrm_op_precondition_truncated(nullptr),
    vct_dr(nullptr),
    vct_dp(nullptr),
    vct_dmmp(nullptr),
    vct_dtmp(nullptr),
    vct_dtmp2(nullptr),
    r(nullptr),
    p(nullptr),
    z(nullptr),
    mmp(nullptr),
    tmp(nullptr),
    tmp2(nullptr),
    fr(nullptr),
    fz(nullptr),
    immp(nullptr),
    ip(nullptr),
    ifmmp(nullptr),
    ifp(nullptr),
    iftmp(nullptr),
    ifset(nullptr),
    inner_iterations(param.maxiter_precondition)
  {

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

    for (int i = 0; i < 4; i++) { dirac_param_precondition.commDim[i] = 0; }

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
    for (int i = 0; i < 2; i++) { linalg_timer[i].Reset(fname, cname, __LINE__); }

    printfQuda("MSPCG constructor ends.\n");
  }

  MSPCG::~MSPCG()
  {
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

  void MSPCG::inner_dslash(ColorSpinorField &out, const ColorSpinorField &in) { (*nrm_op_precondition)(out, in); }

  void MSPCG::calculate_TdATx(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap)
  {

    cudaColorSpinorField tmp(in);
    ColorSpinorParam csParam(in);
    csParam.create = QUDA_NULL_FIELD_CREATE;
    csParam.x[4] = Ls_cheap;
    cudaColorSpinorField truncated_cs_field_out(csParam);

    // A * T * phi
    ATx(truncated_cs_field_out, in, tp);

    // T^ * A * T * phi
    madwf_ml::transfer_5d_hh(out, truncated_cs_field_out, tp, true);

    axpy(mu, tmp, out);
  }

  double MSPCG::calculate_chi(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap)
  {
    bool global_reduce = commGlobalReduction();
    if(global_reduce){ commGlobalReductionSet(false); }
    
    ColorSpinorParam csParam(in);
    cudaColorSpinorField tmp1(csParam);
    cudaColorSpinorField tmp2(csParam);

    // tmp1 = T^ * A * T * phi
    calculate_TdATx(tmp1, in, tp, mu, Ls_cheap);

    inner_dslash(tmp2, tmp1);

    copy(out, in);
    // M * T^ * A * T * phi - phi
    return xmyNorm(tmp2, out);
  
    if(global_reduce){ commGlobalReductionSet(true); }
  }

  void MSPCG::ATx(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp)
  {
    bool global_reduce = commGlobalReduction();
    if(global_reduce){ commGlobalReductionSet(false); }

    int Ls_cheap = out.X(4);
    ColorSpinorParam csParam(in);
    csParam.x[4] = Ls_cheap;
    csParam.create = QUDA_NULL_FIELD_CREATE;

    cudaColorSpinorField truncated_cs_field_in(csParam);
    cudaColorSpinorField ip(csParam);
    cudaColorSpinorField immp(csParam);

    // T * phi
    madwf_ml::transfer_5d_hh(truncated_cs_field_in, in, tp, false);
    
    // A * T * phi
    blas::zero(out);
    double rk2 = blas::norm2(truncated_cs_field_in);
    double Mpk2, alpha, beta, rkp12;
    blas::copy(ip, truncated_cs_field_in);
    for (int local_loop_count = 0; local_loop_count < inner_iterations; local_loop_count++) {
      (*nrm_op_precondition_truncated)(immp, ip);
      Mpk2 = reDotProduct(ip, immp);
      alpha = rk2 / Mpk2;
      rkp12 = axpyNorm(-alpha, immp, truncated_cs_field_in);
      beta = rkp12 / rk2;
      rk2 = rkp12;
      axpyZpbx(alpha, ip, out, truncated_cs_field_in, beta);
    }
    
    if(global_reduce){ commGlobalReductionSet(true); }
  }

  void fill_random(std::vector<float> &v)
  {
    static std::random_device rd;
    // the good rng
    static std::mt19937 rng(23ul * comm_rank());
    // The gaussian distribution
    static std::normal_distribution<double> n(0., 1.);

    for (auto &x : v) { x = 1e-1 * n(rng); }
  }

  void MSPCG::train_param(const std::vector<ColorSpinorField *> &in, std::vector<float> &tp, const double mu, int Ls_cheap)
  {
    commGlobalReductionSet(false);

    constexpr int color_spin_dim = 12;

    int Ls_in = in[0]->X(4);
#if 1
    size_t param_size = Ls_in * Ls_cheap * 16 * 2;

    if (tp.size() != param_size) { errorQuda("wrong param size.\n"); }
#endif
    ColorSpinorParam csParam(*in[0]);
    cudaColorSpinorField chi(csParam);
    cudaColorSpinorField tmp(csParam);
    cudaColorSpinorField theta(csParam);
    cudaColorSpinorField lambda(csParam);

    cudaColorSpinorField Mchi(csParam);

    double ref = 0.0;
    int count = 0;
    for (const auto &phi : in) {
      ref += norm2(*phi);
      printfQuda("reference dslash norm %03d = %8.4e\n", count, norm2(*phi));
      count++;
    }
    printfQuda("reference dslash norm = %8.4e\n", ref);

    csParam.x[4] = Ls_cheap;
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    cudaColorSpinorField ATchi(csParam);
    cudaColorSpinorField ATphi(csParam);
    cudaColorSpinorField ADphi(csParam);

    cudaColorSpinorField ATMchi(csParam);

    fill_random(tp);
    madwf_ml::TrainingParameter<float> T(tp);

    madwf_ml::TrainingParameter<float> d1(tp.size());
    madwf_ml::TrainingParameter<float> d2(tp.size());

    madwf_ml::TrainingParameter<float> P(tp.size());
     
    madwf_ml::TrainingParameter<float> D_old(tp.size());

    // double pmu = 0.0;

    // double old_chi2 = 0.0;
    float alpha;
    float b = 0.8;
    printfQuda("beta = %f\n", b);
    printfQuda("training mu   = %f\n", mu);
    for (int iteration = 0; iteration < 1600; iteration++) {

      madwf_ml::TrainingParameter<float> D(tp.size());
      // double dmu = 0.0;

      double chi2 = 0.0;
      std::vector<double> a(5, 0.0);

      for (const auto &phi : in) {
        chi2 += calculate_chi(chi, *phi, T, mu, Ls_cheap);

        ATx(ATphi, *phi, T);
        inner_dslash(Mchi, chi);
        ATx(ATMchi, Mchi, T);

        madwf_ml::tensor_5d_hh(ATphi, Mchi, d1);
        madwf_ml::tensor_5d_hh(ATMchi, *phi, d2);

        madwf_ml::axpby(D, 2.0f, d1, 2.0f, d2);
        // dmu += 2.0 * reDotProduct(Mchi, *phi);
      }

#if 1
      madwf_ml::axpby(P, (b - 1), P, (1 - b), D);
#else
      if(iteration == 0){
        P.copy(D);
      }else{
        // double den = madwf_ml::inner_product(D_old, D_old);
        double den = madwf_ml::inner_product(P, D_old) - madwf_ml::inner_product(P, D);
        double num = madwf_ml::inner_product(D, D) - madwf_ml::inner_product(D, D_old);
        b = std::max(0.0, num / den);
        madwf_ml::axpby(P, b, P, 1.0f, D);
      }
      D_old.copy(D);

      printfQuda("beta = %8.4e ", b);
#endif
      // pmu = b * pmu + (1-b) * dmu;

      chi2 = 0.0;
      // line search
      for (const auto &phi : in) {

        double ind_chi2 = calculate_chi(chi, *phi, T, mu, Ls_cheap);
        chi2 += ind_chi2;

        ATx(ATphi, *phi, T);
        // D' * A * T * phi
        madwf_ml::transfer_5d_hh(theta, ATphi, P, true);

        ATx(ADphi, *phi, P);
        // T' * A * D * phi
        madwf_ml::transfer_5d_hh(tmp, ADphi, T, true);
        // theta
        axpy(1.0, theta, tmp);
        // axpy(pmu, *phi, tmp);

        inner_dslash(theta, tmp);

        // lambda = D' * A * D * phi
        madwf_ml::transfer_5d_hh(tmp, ADphi, P, true);

        inner_dslash(lambda, tmp);

        std::vector<ColorSpinorField *> lhs {&chi, &theta, &lambda};
        std::vector<ColorSpinorField *> rhs {&chi, &theta, &lambda};
        Complex dot[9];
        cDotProduct(dot, lhs, rhs);

        a[0] += dot[0].real();
        a[1] += -2.0 * dot[1].real();
        a[2] += dot[4].real() + 2.0 * dot[2].real();
        a[3] += -2.0 * dot[5].real();
        a[4] += dot[8].real();
      }

      double r[3] = {0.0, 0.0, 0.0};
      solve_deg3(4.0 * a[4], 3.0 * a[3], 2.0 * a[2], a[1], r[0], r[1], r[2]);

      // try the three roots
      double try_root[3];
      for (int i = 0; i < 3; i++) { try_root[i] = eval_deg4(a[4], a[3], a[2], a[1], a[0], r[i]); }

      if (try_root[0] < try_root[1] && try_root[0] < try_root[2]) {
        alpha = r[0];
      } else if (try_root[1] < try_root[2]) {
        alpha = r[1];
      } else {
        alpha = r[2];
      }
      madwf_ml::axpby(T, 0.0f, T, -alpha, P);
      // mu -= alpha * pmu;

      printfQuda("grad min iter %03d: %04d chi2 = %8.4e, chi2 %% = %8.4e, alpha = %+8.4e, mu = %+8.4e\n", comm_rank(),
                 iteration, chi2, chi2 / ref, alpha, mu);
    
      // if((chi2 - old_chi2) * (chi2 - old_chi2) / (ref * ref) / (old_chi2*old_chi2 / (ref*ref)) < 1e-10){ break; }
      // old_chi2 = chi2;
    }

    printfQuda("Training finished ...\n");
    count = 0;
    for (const auto &phi : in) {
      double ind_chi2 = calculate_chi(chi, *phi, T, mu, Ls_cheap);
      double phi2 = norm2(*phi);
      printfQuda("chi2 %03d %% = %8.4e, phi2 = %8.4e\n", count, ind_chi2 / phi2, phi2);
      count++;
    }

    tp = T.to_host();

    std::string save_param_path(getenv("QUDA_RESOURCE_PATH"));
    char cstring[512];
    // sprintf(cstring, "/training_param_rank_%03d_ls_%02d_%02d_mu_%.3f.dat", comm_rank(), Ls_in, Ls_cheap, mu);
    sprintf(cstring, "/training_param_rank_%05d_ls_%02d_%02d_mu_%.3f_it_%02d.dat", comm_rank(), Ls_in, Ls_cheap, mu, inner_iterations);
    save_param_path += std::string(cstring);
    FILE *fp = fopen(save_param_path.c_str(), "w");
    size_t fwrite_count = fwrite(tp.data(), sizeof(float), tp.size(), fp);
    fclose(fp);
    if (fwrite_count != tp.size()) {
      errorQuda("Unable to write training params to %s (%lu neq %lu).\n", save_param_path.c_str(), fwrite_count,
                tp.size());
    }
    printfQuda("Training params saved to %s ...\n", save_param_path.c_str());
    
    commGlobalReductionSet(true);
    
    double dummy_for_sync = 0.0;
    reduceDouble(dummy_for_sync);
    
    return;
  }

  void MSPCG::inner_cg(ColorSpinorField &ix, ColorSpinorField &ib)
  {
    commGlobalReductionSet(false);

    blas::zero(ix);
    double rk2 = blas::norm2(ib);
    if (rk2 == 0.0) {
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

    commGlobalReductionSet(true);
    return;
  }

  int MSPCG::outer_cg(ColorSpinorField &dx, ColorSpinorField &db, double quit)
  {
    double Mpk2, alpha, beta, rkp12;
    (*nrm_op)(*vct_dr, dx, *vct_dtmp, *vct_dtmp2); // r = nrm_op * x
    double rk2 = xmyNorm(db, *vct_dr);             // r = b - nrm_op * x

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

      printfQuda("outer_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n", loop_count, rk2, alpha, beta,
                 Mpk2);
    }

    printfQuda("outer_cg: CONVERGED after %04d iterations: r2/target_r2 = %8.4e/%8.4e.\n", loop_count + 1, rk2, quit);

    return loop_count;
  }

  void MSPCG::Minv(ColorSpinorField &out, const ColorSpinorField &in)
  {
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

  void MSPCG::m_inv_trained(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap)
  {
    preconditioner_timer.Start("woo", "hoo", 0);
    if (inner_iterations <= 0) {
      blas::copy(out, in);
    } else {
      blas::copy(*fr, in);
      if (dirac_param_sloppy.gauge->Precision() == dirac_param_precondition.gauge->Precision()) {
        calculate_TdATx(out, *fr, tp, mu, Ls_cheap);
      } else {
        calculate_TdATx(*fz, *fr, tp, mu, Ls_cheap);
        blas::copy(out, *fz);
      }
    }
    preconditioner_timer.Stop("woo", "hoo", 0);
  }

  void MSPCG::allocate(ColorSpinorField &db)
  {

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

  void MSPCG::deallocate()
  {

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

  void MSPCG::mspcg_madwf_ml(ColorSpinorField &dx, ColorSpinorField &db, const bool use_training,
                             const bool perform_training, const int Ls_cheap)
  {

    const char fname[] = "MSPCG::operator()(ColorSpinorField&, ColorSpinorField&)";
    const char cname[] = __FILE__;

    constexpr double target_sample_scale = 5e3;
    // const bool perform_training = true;
    // const bool use_training = true;
    const int training_sample_size = 16;
    const int training_sample_starting_point = 400;

    // const int Ls_cheap = 8;
    const double mu = dirac_param_precondition.mu;

    bool trained = false;
    std::vector<ColorSpinorField *> training_sample(0);

    constexpr int complex_matrix_size = 16; // spin by spin
    const int Ls = db.X(4);
    int training_param_size = Ls * Ls_cheap * complex_matrix_size * 2;
    printfQuda("training parameter size = %d %d \n", Ls, Ls_cheap);
    std::vector<float> host_training_param(training_param_size);
    Tp training_param(training_param_size);

    if (use_training || perform_training) {
      dirac_param_precondition.Ls = Ls_cheap;
      mat_precondition_truncated = new DiracMobiusPC(dirac_param_precondition);
      nrm_op_precondition_truncated = new DiracMdagMLocal(mat_precondition_truncated);
    }

    // If we want to use training without performing it we need to load the params from file.
    if (use_training && !perform_training) {
      printfQuda("inference mu   = %f\n", dirac_param_precondition.mu);
      std::string save_param_path(getenv("QUDA_RESOURCE_PATH"));
      char cstring[512];
      // sprintf(cstring, "/training_param_rank_%03d_ls_%02d_%02d_mu_%.3f.dat", comm_rank(), Ls, Ls_cheap, mu);
      sprintf(cstring, "/training_param_rank_%05d_ls_%02d_%02d_mu_%.3f_it_%02d.dat", 0, Ls, Ls_cheap, mu, inner_iterations);
      save_param_path += std::string(cstring);
      FILE *fp = fopen(save_param_path.c_str(), "rb");
      if (!fp) errorQuda("Unable to open file %s\n", save_param_path.c_str());
      size_t fread_count = fread(host_training_param.data(), sizeof(float), host_training_param.size(), fp);
      fclose(fp);
      if (fread_count != host_training_param.size()) {
        errorQuda("Unable to load training params from %s (%lu neq %lu).\n", save_param_path.c_str(), fread_count,
                  host_training_param.size());
      }
      printfQuda("Training params loaded from %s.\n", save_param_path.c_str());
      training_param.from_host(host_training_param);
      trained = true;
    }

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

    allocate(db);

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
    const double u = param.precision_sloppy == 8 ?
      std::numeric_limits<double>::epsilon() / 2. :
      ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() / 2. : fp16_eps);
    const double uhigh = param.precision == 8 ?
      std::numeric_limits<double>::epsilon() / 2. :
      ((param.precision == 4) ? std::numeric_limits<float>::epsilon() / 2. : fp16_eps);
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

    if (use_training && trained) {
      m_inv_trained(*z, *r, training_param, mu, Ls_cheap);
    } else {
      Minv(*z, *r);
    }

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
      std::vector<ColorSpinorField *> lhs {p, tmp2, r};
      std::vector<ColorSpinorField *> rhs {p, tmp2, z};
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

      axpy(alpha, *p, *x);              // x_k+1 = x_k + alpha * p_k
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
        r2 = xmyNorm(db, *vct_dr);                     // r = b - MdagM * x

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

      if (perform_training && !trained && training_sample.size() < training_sample_size) {
        // store the r vector for training.
        if (!trained && k > training_sample_starting_point) {
          cudaColorSpinorField *p = new cudaColorSpinorField(*r);
          ax(target_sample_scale / sqrt(norm2(*p)), *p);
          training_sample.push_back(p);
        }

        if (training_sample.size() == training_sample_size) {
          // perform training.
          train_param(training_sample, host_training_param, mu, Ls_cheap);
          training_param.from_host(host_training_param);
          trained = true;
          break;
        }
      }

      if (use_training && trained) {
        m_inv_trained(*z, *r, training_param, mu, Ls_cheap);
      } else {
        Minv(*z, *r);
      }

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
    double preconditioner_truncated_tflops = nrm_op_precondition_truncated ? nrm_op_precondition_truncated->flops() * 1e-12 : 0 ;
    double linalg_tflops = blas::flops * 1e-12;

    param.gflops = (preconditioner_tflops + preconditioner_truncated_tflops + sloppy_tflops + precise_tflops + linalg_tflops) * 1e3;

    reduceDouble(precise_tflops);
    reduceDouble(sloppy_tflops);
    reduceDouble(preconditioner_tflops);
    reduceDouble(preconditioner_truncated_tflops);
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
               (preconditioner_tflops + preconditioner_truncated_tflops)/ prec_time, prec_time, int(prec_time / param.secs * 100.),
               preconditioner_timer.count);
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

    for (auto p : training_sample) { delete p; }

    if(mat_precondition_truncated){
      delete mat_precondition_truncated;
      delete nrm_op_precondition_truncated;
    }

    return;
  }

} // namespace quda
