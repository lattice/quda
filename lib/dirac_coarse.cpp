#include <string.h>
#include <multigrid.h>
#include <algorithm>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param, bool gpu_setup, bool mapped) :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(param.transfer),
    dirac(param.dirac),
    need_bidirectional(param.need_bidirectional),
    Y_h(nullptr),
    X_h(nullptr),
    Xinv_h(nullptr),
    Yhat_h(nullptr),
    Y_d(nullptr),
    X_d(nullptr),
    Xinv_d(nullptr),
    Yhat_d(nullptr),
    enable_gpu(false),
    enable_cpu(false),
    gpu_setup(gpu_setup),
    init_gpu(gpu_setup),
    init_cpu(!gpu_setup),
    mapped(mapped)
  {
    initializeCoarse();
  }

  DiracCoarse::DiracCoarse(const DiracParam &param, cpuGaugeField *Y_h, cpuGaugeField *X_h, cpuGaugeField *Xinv_h,
                           cpuGaugeField *Yhat_h, // cpu link fields
                           cudaGaugeField *Y_d, cudaGaugeField *X_d, cudaGaugeField *Xinv_d,
                           cudaGaugeField *Yhat_d) // gpu link field
    :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(nullptr),
    dirac(nullptr),
    need_bidirectional(false),
    Y_h(Y_h),
    X_h(X_h),
    Xinv_h(Xinv_h),
    Yhat_h(Yhat_h),
    Y_d(Y_d),
    X_d(X_d),
    Xinv_d(Xinv_d),
    Yhat_d(Yhat_d),
    enable_gpu(Y_d ? true : false),
    enable_cpu(Y_h ? true : false),
    gpu_setup(true),
    init_gpu(enable_gpu ? false : true),
    init_cpu(enable_cpu ? false : true),
    mapped(Y_d->MemType() == QUDA_MEMORY_MAPPED)
  {

  }

  DiracCoarse::DiracCoarse(const DiracCoarse &dirac, const DiracParam &param) :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(param.transfer),
    dirac(param.dirac),
    need_bidirectional(param.need_bidirectional),
    Y_h(dirac.Y_h),
    X_h(dirac.X_h),
    Xinv_h(dirac.Xinv_h),
    Yhat_h(dirac.Yhat_h),
    Y_d(dirac.Y_d),
    X_d(dirac.X_d),
    Xinv_d(dirac.Xinv_d),
    Yhat_d(dirac.Yhat_d),
    enable_gpu(dirac.enable_gpu),
    enable_cpu(dirac.enable_cpu),
    gpu_setup(dirac.gpu_setup),
    init_gpu(enable_gpu ? false : true),
    init_cpu(enable_cpu ? false : true),
    mapped(dirac.mapped)
  {

  }

  DiracCoarse::~DiracCoarse()
  {
    if (init_cpu) {
      if (Y_h) delete Y_h;
      if (X_h) delete X_h;
      if (Xinv_h) delete Xinv_h;
      if (Yhat_h) delete Yhat_h;
    }
    if (init_gpu) {
      if (Y_d) delete Y_d;
      if (X_d) delete X_d;
      if (Xinv_d) delete Xinv_d;
      if (Yhat_d) delete Yhat_d;
    }
  }

  void DiracCoarse::createY(bool gpu, bool mapped) const
  {
    int ndim = transfer->Vectors().Ndim();
    // FIXME MRHS NDIM hack
    if (ndim == 5 && transfer->Vectors().Nspin() != 4) ndim = 4; // forced case for staggered, coarsened staggered
    int x[QUDA_MAX_DIM];
    const int *geo_bs = transfer->Geo_bs(); // Number of coarse sites.
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];
    int Nc_c = transfer->nvec(); // Coarse Color
    // Coarse Spin
    int Ns_c = (transfer->Spin_bs() == 0) ? 2 : transfer->Vectors().Nspin() / transfer->Spin_bs();
    GaugeFieldParam gParam;
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = gpu ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    // use null-space precision for coarse links on gpu
    gParam.setPrecision( transfer->NullPrecision(gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION) );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;
    if (mapped) gParam.mem_type = QUDA_MEMORY_MAPPED;

    int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
    gParam.pad = gpu ? gParam.nFace * pad * 2 : 0; // factor of 2 since we have to store bi-directional ghost zone

    if (gpu) Y_d = new cudaGaugeField(gParam);
    else     Y_h = new cpuGaugeField(gParam);

    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    if (gpu) X_d = new cudaGaugeField(gParam);
    else     X_h = new cpuGaugeField(gParam);
  }

  void DiracCoarse::createYhat(bool gpu) const
  {
    int ndim = transfer->Vectors().Ndim();
    if (ndim == 5 && transfer->Vectors().Nspin() != 4) ndim = 4; // forced case for staggered, coarsened staggered
    int x[QUDA_MAX_DIM];
    const int *geo_bs = transfer->Geo_bs(); // Number of coarse sites.
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];
    int Nc_c = transfer->nvec();     // Coarse Color
    int Ns_c = (transfer->Spin_bs() == 0) ? 2 : transfer->Vectors().Nspin() / transfer->Spin_bs();

    GaugeFieldParam gParam;
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = gpu ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    // use null-space precision for preconditioned links on gpu
    gParam.setPrecision( transfer->NullPrecision(gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION) );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;

    int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
    gParam.pad = gpu ? gParam.nFace * pad * 2 : 0; // factor of 2 since we have to store bi-directional ghost zone

    if (gpu) Yhat_d = new cudaGaugeField(gParam);
    else     Yhat_h = new cpuGaugeField(gParam);

    gParam.setPrecision(gpu ? X_d->Precision() : X_h->Precision());
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    if (gpu) Xinv_d = new cudaGaugeField(gParam);
    else     Xinv_h = new cpuGaugeField(gParam);
  }

  void DiracCoarse::initializeCoarse()
  {
    createY(gpu_setup, mapped);

    if (gpu_setup) dirac->createCoarseOp(*Y_d,*X_d,*transfer,kappa,mass,Mu(),MuFactor());
    else dirac->createCoarseOp(*Y_h,*X_h,*transfer,kappa,mass,Mu(),MuFactor());

    // save the intermediate tunecache after the UV and VUV tune
    saveTuneCache();

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to build the preconditioned coarse clover\n");

    createYhat(gpu_setup);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished building the preconditioned coarse clover\n");
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to create the preconditioned coarse op\n");

    if (gpu_setup) createPreconditionedCoarseOp(*Yhat_d,*Xinv_d,*Y_d,*X_d);
    else createPreconditionedCoarseOp(*Yhat_h,*Xinv_h,*Y_h,*X_h);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished creating the preconditioned coarse op\n");

    // save the intermediate tunecache after the Yhat tune
    saveTuneCache();

    if (gpu_setup) {
      enable_gpu = true;
      init_gpu = true;
    } else {
      enable_cpu = true;
      init_cpu = true;
    }
  }

  // we only copy to host or device lazily on demand
  void DiracCoarse::initializeLazy(QudaFieldLocation location) const
  {
    if (!enable_cpu && !enable_gpu) errorQuda("Neither CPU or GPU coarse fields initialized");
    switch(location) {
    case QUDA_CUDA_FIELD_LOCATION:
      if (enable_gpu) return;
      createY(true, mapped);
      createYhat(true);
      Y_d->copy(*Y_h);
      Yhat_d->copy(*Yhat_h);
      X_d->copy(*X_h);
      Xinv_d->copy(*Xinv_h);
      enable_gpu = true;
      init_gpu = true;
      break;
    case QUDA_CPU_FIELD_LOCATION:
      if (enable_cpu) return;
      createY(false);
      createYhat(false);
      Y_h->copy(*Y_d);
      Yhat_h->copy(*Yhat_d);
      X_h->copy(*X_d);
      Xinv_h->copy(*Xinv_d);
      enable_cpu = true;
      init_cpu = true;
      break;
    default:
      errorQuda("Unknown location");
    }
  }

  void DiracCoarse::createPreconditionedCoarseOp(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X) {
    calculateYhat(Yhat, Xinv, Y, X);
  }

  void DiracCoarse::Clover(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if (&in == &out) errorQuda("Fields cannot alias");
    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, parity, false, true, dagger, commDim);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, false, true, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*n*n-2*n)*(long long)in.VolumeCB();
  }

  void DiracCoarse::CloverInv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if (&in == &out) errorQuda("Fields cannot alias");
    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if ( location  == QUDA_CUDA_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_d, *Xinv_d, kappa, parity, false, true, dagger, commDim);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *Xinv_h, kappa, parity, false, true, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*n*n-2*n)*(long long)in.VolumeCB();
  }

  void DiracCoarse::Dslash(ColorSpinorField &out, const ColorSpinorField &in,
			   const QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, parity, true, false, dagger, commDim, halo_precision);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, true, false, dagger, commDim, halo_precision);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*(8*n*n)-2*n)*(long long)in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarse::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in,
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    if (k!=1.0) errorQuda("%s not supported for k!=1.0", __func__);

    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      ApplyCoarse(out, in, x, *Y_d, *X_d, kappa, parity, true, true, dagger, commDim, halo_precision);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, x, *Y_h, *X_h, kappa, parity, true, true, dagger, commDim, halo_precision);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (9*(8*n*n)-2*n)*(long long)in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim, halo_precision);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim, halo_precision);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (9*(8*n*n)-2*n)*(long long)in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarse::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp1, in);
    if (tmp1->SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Temporary vector is not full-site vector");

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset1);
  }

  void DiracCoarse::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b,
			    const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracCoarse::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				const QudaSolutionType solType) const
  {
    /* do nothing */
  }

  //Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarse::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double mass, double mu, double mu_factor) const
  {
    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    if (checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
      initializeLazy(QUDA_CPU_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Y_h), *(this->X_h), *(this->Xinv_h), kappa, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional);
    } else {
      initializeLazy(QUDA_CUDA_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Y_d), *(this->X_d), *(this->Xinv_d), kappa, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional);
    }
  }

  void DiracCoarse::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    if (Y_d) Y_d->prefetch(mem_space, stream);
    if (X_d) X_d->prefetch(mem_space, stream);
  }

  DiracCoarsePC::DiracCoarsePC(const DiracParam &param, bool gpu_setup) : DiracCoarse(param, gpu_setup)
  {
    /* do nothing */
  }

  DiracCoarsePC::DiracCoarsePC(const DiracCoarse &dirac, const DiracParam &param) : DiracCoarse(dirac, param)
  {
    /* do nothing */
  }

  DiracCoarsePC::~DiracCoarsePC() { }

  void DiracCoarsePC::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out,in);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Yhat_d, *X_d, kappa, parity, true, false, dagger, commDim, halo_precision);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Yhat_h, *X_h, kappa, parity, true, false, dagger, commDim, halo_precision);
    }

    int n = in.Nspin()*in.Ncolor();
    flops += (8*(8*n*n)-2*n)*in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarsePC::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
				 const ColorSpinorField &x, const double &k) const
  {
    // emulated for now
    Dslash(out, in, parity);
    blas::xpay(const_cast<ColorSpinorField&>(x), k, out);

    int n = in.Nspin()*in.Ncolor();
    flops += (8*(8*n*n)-2*n)*in.VolumeCB(); // blas flops counted separately so only need to count dslash flops
  }

  void DiracCoarsePC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp1, in);

    if (in.SiteSubset() == QUDA_FULL_SITE_SUBSET || out.SiteSubset() == QUDA_FULL_SITE_SUBSET ||
	tmp1->SiteSubset() == QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot apply preconditioned operator to full field (subsets = %d %d %d)",
		in.SiteSubset(), out.SiteSubset(), tmp1->SiteSubset());

    if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // DiracCoarsePC::Dslash applies A^{-1}Dslash
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      // DiracCoarse::DslashXpay applies (A - D) // FIXME this ignores the -1
      DiracCoarse::Dslash(out, *tmp1, QUDA_EVEN_PARITY);
      Clover(*tmp1, in, QUDA_EVEN_PARITY);
      blas::xpay(*tmp1, -1.0, out);
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // DiracCoarsePC::Dslash applies A^{-1}Dslash
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      // DiracCoarse::DslashXpay applies (A - D) // FIXME this ignores the -1
      DiracCoarse::Dslash(out, *tmp1, QUDA_ODD_PARITY);
      Clover(*tmp1, in, QUDA_ODD_PARITY);
      blas::xpay(*tmp1, -1.0, out);
    } else if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, -1.0);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, -1.0);
    } else {
      errorQuda("MatPCType %d not valid for DiracCoarsePC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracCoarsePC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset1);
  }

  void DiracCoarsePC::prepare(ColorSpinorField* &src, ColorSpinorField* &sol, ColorSpinorField &x, ColorSpinorField &b,
			      const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
      return;
    }

    bool reset = newTmp(&tmp1, b.Even());

    // we desire solution to full system
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      // src = A_ee^-1 (b_e - D_eo A_oo^-1 b_o)
      src = &(x.Odd());
#if 0
      CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
      DiracCoarse::Dslash(*tmp1, *src, QUDA_EVEN_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Even()), -1.0, *tmp1);
      CloverInv(*src, *tmp1, QUDA_EVEN_PARITY);
#endif
      // src = A_ee^{-1} b_e - (A_ee^{-1} D_eo) A_oo^{-1} b_o
      CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
      Dslash(*tmp1, *src, QUDA_EVEN_PARITY);
      CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
      blas::axpy(-1.0, *tmp1, *src);

      sol = &(x.Even());
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // src = A_oo^-1 (b_o - D_oe A_ee^-1 b_e)
      src = &(x.Even());
#if 0
      CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
      DiracCoarse::Dslash(*tmp1, *src, QUDA_ODD_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Odd()), -1.0, *tmp1);
      CloverInv(*src, *tmp1, QUDA_ODD_PARITY);
#endif
      // src = A_oo^{-1} b_o - (A_oo^{-1} D_oe) A_ee^{-1} b_e
      CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
      Dslash(*tmp1, *src, QUDA_ODD_PARITY);
      CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
      blas::axpy(-1.0, *tmp1, *src);

      sol = &(x.Odd());
    } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // src = b_e - D_eo A_oo^-1 b_o
      src = &(x.Odd());
      CloverInv(*tmp1, b.Odd(), QUDA_ODD_PARITY);
      DiracCoarse::Dslash(*src, *tmp1, QUDA_EVEN_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Even()), -1.0, *src);
      sol = &(x.Even());
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // src = b_o - D_oe A_ee^-1 b_e
      src = &(x.Even());
      CloverInv(*tmp1, b.Even(), QUDA_EVEN_PARITY);
      DiracCoarse::Dslash(*src, *tmp1, QUDA_ODD_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Odd()), -1.0, *src);
      sol = &(x.Odd());
    } else {
      errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
    }

    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }

  void DiracCoarsePC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }

    checkFullSpinor(x, b);

    bool reset = newTmp(&tmp1, b.Even());

    // create full solution

    if (matpcType == QUDA_MATPC_EVEN_EVEN ||
	matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
#if 0
      // x_o = A_oo^-1 (b_o - D_oe x_e)
      DiracCoarse::Dslash(*tmp1, x.Even(), QUDA_ODD_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Odd()), -1.0, *tmp1);
      CloverInv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
#endif
      // x_o = A_oo^{-1} b_o - (A_oo^{-1} D_oe) x_e
      Dslash(*tmp1, x.Even(), QUDA_ODD_PARITY);
      CloverInv(x.Odd(), b.Odd(), QUDA_ODD_PARITY);
      blas::axpy(-1.0, const_cast<ColorSpinorField &>(*tmp1), x.Odd());

    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
#if 0
      // x_e = A_ee^-1 (b_e - D_eo x_o)
      DiracCoarse::Dslash(*tmp1, x.Odd(), QUDA_EVEN_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Even()), -1.0, *tmp1);
      CloverInv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
#endif
      // x_e = A_ee^{-1} b_e - (A_ee^{-1} D_eo) x_o
      Dslash(*tmp1, x.Odd(), QUDA_EVEN_PARITY);
      CloverInv(x.Even(), b.Even(), QUDA_EVEN_PARITY);
      blas::axpy(-1.0, const_cast<ColorSpinorField &>(*tmp1), x.Even());

    } else {
      errorQuda("MatPCType %d not valid for DiracCoarsePC", matpcType);
    }

    deleteTmp(&tmp1, reset);
  }

  //Make the coarse operator one level down.  For the preconditioned
  //operator we are coarsening the Yhat links, not the Y links.  We
  //pass the fine clover fields, though they are actually ignored.
  void DiracCoarsePC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double mass, double mu, double mu_factor) const
  {
    double a = -2.0 * kappa * mu * T.Vectors().TwistFlavor();
    if (checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
      initializeLazy(QUDA_CPU_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Yhat_h), *(this->X_h), *(this->Xinv_h), kappa, a, -mu_factor, QUDA_COARSEPC_DIRAC,
                     matpcType, true);
    } else {
      initializeLazy(QUDA_CUDA_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Yhat_d), *(this->X_d), *(this->Xinv_d), kappa, a, -mu_factor, QUDA_COARSEPC_DIRAC,
                     matpcType, true);
    }
  }

  void DiracCoarsePC::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    if (Xinv_d) Xinv_d->prefetch(mem_space, stream);
    if (Yhat_d) Yhat_d->prefetch(mem_space, stream);
  }
}
