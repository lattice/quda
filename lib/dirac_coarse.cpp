#include <string.h>
#include <multigrid.h>
#include <algorithm>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param, bool enable_gpu, bool gpu_setup)
    : Dirac(param), mu(param.mu), mu_factor(param.mu_factor), transfer(param.transfer), dirac(param.dirac),
      Y_h(nullptr), X_h(nullptr), Xinv_h(nullptr), Yhat_h(nullptr),
      Y_d(nullptr), X_d(nullptr), Xinv_d(nullptr), Yhat_d(nullptr),
      enable_gpu(enable_gpu), gpu_setup(gpu_setup), init(true)
  {
    initializeCoarse();
  }

  DiracCoarse::DiracCoarse(const DiracParam &param,
			   cpuGaugeField *Y_h, cpuGaugeField *X_h, cpuGaugeField *Xinv_h, cpuGaugeField *Yhat_h,   // cpu link fields
			   cudaGaugeField *Y_d, cudaGaugeField *X_d, cudaGaugeField *Xinv_d, cudaGaugeField *Yhat_d) // gpu link field
    : Dirac(param), mu(param.mu), mu_factor(param.mu_factor), transfer(nullptr), dirac(nullptr),
      Y_h(Y_h), X_h(X_h), Xinv_h(Xinv_h), Yhat_h(Yhat_h),
      Y_d(Y_d), X_d(X_d), Xinv_d(Xinv_d), Yhat_d(Yhat_d),
      enable_gpu(Y_d && X_d && Xinv_d), gpu_setup(true), init(false)
  {

  }

  DiracCoarse::DiracCoarse(const DiracCoarse &dirac, const DiracParam &param)
    : Dirac(param), mu(param.mu), mu_factor(param.mu_factor), transfer(param.transfer), dirac(param.dirac),
      Y_h(dirac.Y_h), X_h(dirac.X_h), Xinv_h(dirac.Xinv_h), Yhat_h(dirac.Yhat_h),
      Y_d(dirac.Y_d), X_d(dirac.X_d), Xinv_d(dirac.Xinv_d), Yhat_d(dirac.Yhat_d),
      enable_gpu(dirac.enable_gpu), gpu_setup(dirac.gpu_setup), init(false)
  {

  }

  DiracCoarse::~DiracCoarse()
  {
    if (init) {
      if (Y_h) delete Y_h;
      if (X_h) delete X_h;
      if (Xinv_h) delete Xinv_h;
      if (Yhat_h) delete Yhat_h;
      if (Y_d) delete Y_d;
      if (X_d) delete X_d;
      if (Xinv_d) delete Xinv_d;
      if (Yhat_d) delete Yhat_d;
    }
  }

  void DiracCoarse::initializeCoarse()
  {
    if (!init) errorQuda("Cannot call from this reference instance");

    QudaPrecision prec = transfer->Vectors().Precision();
    int ndim = transfer->Vectors().Ndim();
    if (ndim == 5 && transfer->Vectors().Nspin() == 1)
      ndim = 4; // forced case for staggered
    int x[QUDA_MAX_DIM];
    //Number of coarse sites.
    const int *geo_bs = transfer->Geo_bs();
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];

    //Coarse Color
    int Nc_c = transfer->nvec();

    //Coarse Spin
    int Ns_c = (transfer->Spin_bs() == 0) ? 2 : transfer->Vectors().Nspin()/transfer->Spin_bs();

    GaugeFieldParam gParam;
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.precision = prec;
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;

    gParam.geometry = QUDA_COARSE_GEOMETRY;

    Y_h = new cpuGaugeField(gParam);

    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;

    X_h = new cpuGaugeField(gParam);

    if (enable_gpu) {
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      gParam.nFace = 1;
      gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
      gParam.geometry = QUDA_COARSE_GEOMETRY;
      int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
      gParam.pad = gParam.nFace * pad * 2; // factor of 2 since we have to store bi-directional ghost zone
      Y_d = new cudaGaugeField(gParam);

      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.nFace = 0;
      gParam.pad = 0;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      X_d = new cudaGaugeField(gParam);
    }

    if (enable_gpu && gpu_setup) dirac->createCoarseOp(*Y_d,*X_d,*transfer,kappa,mass,Mu(),MuFactor());
    else dirac->createCoarseOp(*Y_h,*X_h,*transfer,kappa,mass,Mu(),MuFactor());

    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;
    Yhat_h = new cpuGaugeField(gParam);

    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    Xinv_h = new cpuGaugeField(gParam);

    if (enable_gpu) {
      // use the null-space precision for the preconditioned links
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      gParam.nFace = 1;
      gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
      gParam.geometry = QUDA_COARSE_GEOMETRY;
      gParam.precision = transfer->Vectors(QUDA_CUDA_FIELD_LOCATION).Precision();
      int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
      gParam.pad = gParam.nFace * pad * 2; // factor of 2 since we have to store bi-directional ghost zone
      Yhat_d = new cudaGaugeField(gParam);

      gParam.precision = prec;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.nFace = 0;
      gParam.pad = 0;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      Xinv_d = new cudaGaugeField(gParam);
    }

    if (enable_gpu && gpu_setup) createPreconditionedCoarseOp(*Yhat_d,*Xinv_d,*Y_d,*X_d);
    else createPreconditionedCoarseOp(*Yhat_h,*Xinv_h,*Y_h,*X_h);

    if (enable_gpu) {
      if (gpu_setup) {
	Y_h->copy(*Y_d);
	Yhat_h->copy(*Yhat_d);
	X_h->copy(*X_d);
	Xinv_h->copy(*Xinv_d);
      } else {
	Y_d->copy(*Y_h);
	Yhat_d->copy(*Yhat_h);
	X_d->copy(*X_h);
	Xinv_d->copy(*Xinv_h);
      }
    }

  }

  void DiracCoarse::createPreconditionedCoarseOp(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X) {
    calculateYhat(Yhat, Xinv, Y, X);
  }

  void DiracCoarse::Clover(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if (&in == &out) errorQuda("Fields cannot alias");
    if (checkLocation(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, parity, false, true, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, false, true, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*n*n-2*n)*(long long)in.VolumeCB();
  }

  void DiracCoarse::CloverInv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    if (&in == &out) errorQuda("Fields cannot alias");
    if (checkLocation(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Y_d, *Xinv_d, kappa, parity, false, true, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *Xinv_h, kappa, parity, false, true, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*n*n-2*n)*(long long)in.VolumeCB();
  }

  void DiracCoarse::Dslash(ColorSpinorField &out, const ColorSpinorField &in,
			   const QudaParity parity) const
  {
    if (checkLocation(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, parity, true, false, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, true, false, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (8*(8*n*n)-2*n)*(long long)in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarse::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in,
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const
  {
    if (k!=1.0) errorQuda("%s not supported for k!=1.0", __func__);

    if (checkLocation(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, x, *Y_d, *X_d, kappa, parity, true, true, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, x, *Y_h, *X_h, kappa, parity, true, true, dagger, commDim);
    }
    int n = in.Nspin()*in.Ncolor();
    flops += (9*(8*n*n)-2*n)*(long long)in.VolumeCB()*in.SiteSubset();
  }

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( checkLocation(out, in) == QUDA_CUDA_FIELD_LOCATION ) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim);
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
      CoarseCoarseOp(Y, X, T, *(this->Y_h), *(this->X_h), *(this->Xinv_h), kappa, a, mu_factor, QUDA_COARSE_DIRAC, QUDA_MATPC_INVALID);
    } else {
      CoarseCoarseOp(Y, X, T, *(this->Y_d), *(this->X_d), *(this->Xinv_d), kappa, a, mu_factor, QUDA_COARSE_DIRAC, QUDA_MATPC_INVALID);
    }
  }

  DiracCoarsePC::DiracCoarsePC(const DiracParam &param, bool enable_gpu, bool gpu_setup) : DiracCoarse(param, enable_gpu, gpu_setup)
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
    if (checkLocation(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Yhat_d, *X_d, kappa, parity, true, false, dagger, commDim);
    } else if ( checkLocation(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Yhat_h, *X_h, kappa, parity, true, false, dagger, commDim);
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
      CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
      DiracCoarse::Dslash(*tmp1, *src, QUDA_EVEN_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Even()), -1.0, *tmp1);
      CloverInv(*src, *tmp1, QUDA_EVEN_PARITY);
      sol = &(x.Even());
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      // src = A_oo^-1 (b_o - D_oe A_ee^-1 b_e)
      src = &(x.Even());
      CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
      DiracCoarse::Dslash(*tmp1, *src, QUDA_ODD_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Odd()), -1.0, *tmp1);
      CloverInv(*src, *tmp1, QUDA_ODD_PARITY);
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
      // x_o = A_oo^-1 (b_o - D_oe x_e)
      DiracCoarse::Dslash(*tmp1, x.Even(), QUDA_ODD_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Odd()), -1.0, *tmp1);
      CloverInv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
    } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	       matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // x_e = A_ee^-1 (b_e - D_eo x_o)
      DiracCoarse::Dslash(*tmp1, x.Odd(), QUDA_EVEN_PARITY);
      blas::xpay(const_cast<ColorSpinorField&>(b.Even()), -1.0, *tmp1);
      CloverInv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
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
      CoarseCoarseOp(Y, X, T, *(this->Yhat_h), *(this->X_h), *(this->Xinv_h), kappa, a, -mu_factor, QUDA_COARSEPC_DIRAC, matpcType);
    } else {
      CoarseCoarseOp(Y, X, T, *(this->Yhat_d), *(this->X_d), *(this->Xinv_d), kappa, a, -mu_factor, QUDA_COARSEPC_DIRAC, matpcType);
    }
  }

}
