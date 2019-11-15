#include <deflation.h>
#include <qio_field.h>
#include <string.h>

#include <memory>

#ifdef MAGMA_LIB
#include <blas_magma.h>
#endif

#include <Eigen/Dense>

namespace quda
{

  using namespace blas;
  using namespace Eigen;
  using DynamicStride = Stride<Dynamic, Dynamic>;

  static auto pinned_allocator = [] (size_t bytes ) { return static_cast<Complex*>(pool_pinned_malloc(bytes)); };
  static auto pinned_deleter   = [] (Complex *hptr) { pool_pinned_free(hptr); };

  Deflation::Deflation(DeflationParam &param, TimeProfile &profile) :
    param(param),
    profile(profile),
    r(nullptr),
    Av(nullptr),
    r_sloppy(nullptr),
    Av_sloppy(nullptr)
  {
    // for reporting level 1 is the fine level but internally use level 0 for indexing
    printfQuda("Creating deflation space of %d vectors.\n", param.tot_dim);

    if (param.eig_global.import_vectors) loadVectors(param.RV); // whether to load eigenvectors
    // create aux fields
    ColorSpinorParam csParam(param.RV->Component(0));
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.location = param.location;
    csParam.mem_type = QUDA_MEMORY_DEVICE;
    csParam.setPrecision(QUDA_DOUBLE_PRECISION);//accum fields always full precision

    if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
      // all coarse GPU vectors use FLOAT2 ordering
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
      if(csParam.nSpin != 1) csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    }

    r  = ColorSpinorField::Create(csParam);
    Av = ColorSpinorField::Create(csParam);

    if (param.eig_global.cuda_prec_ritz != QUDA_DOUBLE_PRECISION) { // allocate sloppy fields
      csParam.setPrecision(param.eig_global.cuda_prec_ritz);//accum fields always full precision
      if (csParam.location==QUDA_CUDA_FIELD_LOCATION) csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;

      r_sloppy  = ColorSpinorField::Create(csParam);
      Av_sloppy = ColorSpinorField::Create(csParam);
    } else {
      r_sloppy  = r;
      Av_sloppy = Av;
    }

    printfQuda("Deflation space setup completed\n");
    // now we can run through the verification if requested
    if (param.eig_global.run_verify && param.eig_global.import_vectors) verify();
    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  Deflation::~Deflation()
  {
    if( param.eig_global.cuda_prec_ritz != QUDA_DOUBLE_PRECISION ) {
      if (r_sloppy) delete r_sloppy;
      if (Av_sloppy) delete Av_sloppy;
    }

    if (r)  delete r;
    if (Av) delete Av;

    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  double Deflation::flops() const
  {
    double flops = 0;//Do we need to report this?
    //compute total flops for deflation application. Not sure we really need this.
    return flops;
  }

  /**
     Verification that the computed approximate eigenvectors are (not) valid
  */
  void Deflation::verify()
  {
    const int nevs_to_print = param.cur_dim;
    if (nevs_to_print == 0) errorQuda("Incorrect size of current deflation space");

    std::unique_ptr<Complex, decltype(pinned_deleter) > projm( pinned_allocator(param.ld*param.cur_dim * sizeof(Complex)), pinned_deleter);

    if (param.eig_global.extlib_type == QUDA_MAGMA_EXTLIB) {
#ifdef MAGMA_LIB
      memcpy(projm.get(), param.matProj, param.ld*param.cur_dim*sizeof(Complex));
      std::unique_ptr<double[] > evals(new double[param.ld]);
      magma_Xheev(projm.get(), param.cur_dim, param.ld, evals.get(), sizeof(Complex));
#else
      errorQuda("MAGMA library was not built");
#endif
    } else if( param.eig_global.extlib_type == QUDA_EIGEN_EXTLIB ) {
      Map<MatrixXcd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
      Map<MatrixXcd, Unaligned, DynamicStride> evecs_(projm.get(), param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));

      SelfAdjointEigenSolver<MatrixXcd> es_projm( projm_ );
      evecs_.block(0, 0, param.cur_dim, param.cur_dim) = es_projm.eigenvectors();
    } else {
      errorQuda("Library type %d is currently not supported", param.eig_global.extlib_type);
    }

    std::vector<ColorSpinorField*> rv(param.RV->Components().begin(), param.RV->Components().begin() + param.cur_dim);
    std::vector<ColorSpinorField*> res;
    res.push_back(r);

    for (int i = 0; i < nevs_to_print; i++) {
      zero(*r);
      blas::caxpy(&projm.get()[i * param.ld], rv, res); // multiblas
      *r_sloppy = *r;
      param.matDeflation(*Av_sloppy, *r_sloppy);
      double3 dotnorm = cDotProductNormA(*r_sloppy, *Av_sloppy);
      double eval = dotnorm.x / dotnorm.z;
      blas::xpay(*Av_sloppy, -eval, *r_sloppy);
      double relerr = sqrt(norm2(*r_sloppy) / dotnorm.z);
      printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", i + 1, eval, relerr);
    }
  }

  void Deflation::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.eig_global.invert_param->inv_type != QUDA_EIGCG_INVERTER
        && param.eig_global.invert_param->inv_type != QUDA_INC_EIGCG_INVERTER)
      errorQuda("Method is not implemented for %d inverter type", param.eig_global.invert_param->inv_type);

    if(param.cur_dim == 0) return;//nothing to do

    std::unique_ptr<Complex[] > vec(new Complex[param.ld]);

    double check_nrm2 = norm2(b);

    printfQuda("\nSource norm (gpu): %1.15e, curr deflation space dim = %d\n", sqrt(check_nrm2), param.cur_dim);

    ColorSpinorField *b_sloppy = param.RV->Precision() != b.Precision() ? r_sloppy : &b;
    *b_sloppy = b;

    std::vector<ColorSpinorField*> rv_(param.RV->Components().begin(), param.RV->Components().begin()+param.cur_dim);
    std::vector<ColorSpinorField*> in_;
    in_.push_back(static_cast<ColorSpinorField*>(b_sloppy));

    blas::cDotProduct(vec.get(), rv_, in_);//<i, b>

    if (!param.use_inv_ritz) {
      if (param.eig_global.extlib_type == QUDA_MAGMA_EXTLIB) {
#ifdef MAGMA_LIB
        magma_Xgesv(vec.get(), param.ld, param.cur_dim, param.matProj, param.ld, sizeof(Complex));
#else
        errorQuda("MAGMA library was not built");
#endif
      } else if (param.eig_global.extlib_type == QUDA_EIGEN_EXTLIB) {
        Map<MatrixXcd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim,
                                                        DynamicStride(param.ld, 1));
        Map<VectorXcd, Unaligned> vec_(vec.get(), param.cur_dim);

        VectorXcd vec2_(param.cur_dim);
        vec2_ = projm_.fullPivHouseholderQr().solve(vec_);
        vec_ = vec2_;
      } else {
        errorQuda("Library type %d is currently not supported", param.eig_global.extlib_type);
      }
    } else {
      for(int i = 0; i < param.cur_dim; i++) vec[i] *= param.invRitzVals[i];
    }

    std::vector<ColorSpinorField*> out_;
    out_.push_back(&x);

    blas::caxpy(vec.get(), rv_, out_); //multiblas

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));
  }

  void Deflation::increment(ColorSpinorField &Vm, int nev)
  {
    if (param.eig_global.invert_param->inv_type != QUDA_EIGCG_INVERTER
        && param.eig_global.invert_param->inv_type != QUDA_INC_EIGCG_INVERTER)
      errorQuda("Method is not implemented for %d inverter type", param.eig_global.invert_param->inv_type);

    if( nev == 0 ) return; //nothing to do

    const int first_idx = param.cur_dim;

    if (param.RV->CompositeDim() < (first_idx + nev) || param.tot_dim < (first_idx + nev)) {
      warningQuda("\nNot enough space to add %d vectors. Keep deflation space unchanged.\n", nev);
      return;
    }

    for(int i = 0; i < nev; i++) blas::copy(param.RV->Component(first_idx+i), Vm.Component(i));

    printfQuda("\nConstruct projection matrix..\n");

    // Block MGS orthogonalization
    // The degree to which we interpolate between modified GramSchmidt and GramSchmidt (performance vs stability)
    const int cdot_pipeline_length  = 4;

    for (int i = first_idx; i < (first_idx + nev); i++) {
      std::unique_ptr<Complex[]> alpha(new Complex[i]);

      ColorSpinorField *accum = param.eig_global.cuda_prec_ritz != QUDA_DOUBLE_PRECISION ? r : &param.RV->Component(i);
      *accum = param.RV->Component(i);

      int offset = 0;
      while (offset < i) {
        const int local_length = (i - offset) > cdot_pipeline_length ? cdot_pipeline_length : (i - offset);

        std::vector<ColorSpinorField *> vj_(param.RV->Components().begin() + offset,
                                            param.RV->Components().begin() + offset + local_length);
        std::vector<ColorSpinorField *> vi_;
        vi_.push_back(accum);

        blas::cDotProduct(alpha.get(), vj_, vi_);
        for (int j = 0; j < local_length; j++) alpha[j] = -alpha[j];
        blas::caxpy(alpha.get(), vj_, vi_); // i-<j,i>j

        offset += cdot_pipeline_length;
      }

      alpha[0] = blas::norm2(*accum);

      if (alpha[0].real() > 1e-16)
        blas::ax(1.0 / sqrt(alpha[0].real()), *accum);
      else
        errorQuda("Cannot orthogonalize %dth vector", i);

      param.RV->Component(i) = *accum;

      param.matDeflation(*Av_sloppy, param.RV->Component(i)); // precision must match!
      // load diagonal:
      *Av = *Av_sloppy;
      param.matProj[i * param.ld + i] = cDotProduct(*accum, *Av);

      if (i > 0) {
        std::vector<ColorSpinorField *> vj_(param.RV->Components().begin(), param.RV->Components().begin() + i);
        std::vector<ColorSpinorField *> av_;
        av_.push_back(Av_sloppy);

        blas::cDotProduct(alpha.get(), vj_, av_);

        for (int j = 0; j < i; j++) {
          param.matProj[i * param.ld + j] = alpha[j];
          param.matProj[j * param.ld + i] = conj(alpha[j]); // conj
        }
      }
    }

    param.cur_dim += nev;

    printfQuda("\nNew curr deflation space dim = %d\n", param.cur_dim);
  }

  void Deflation::reduce(double tol, int max_nev)
  {
    if (param.cur_dim < max_nev) {
      printf("\nToo big number of eigenvectors was requested, switched to maximum available number %d\n", param.cur_dim);
      max_nev = param.cur_dim;
    }

    std::unique_ptr<double[]> evals(new double[param.cur_dim]);
    std::unique_ptr<Complex, decltype(pinned_deleter)> projm(
      pinned_allocator(param.ld * param.cur_dim * sizeof(Complex)), pinned_deleter);

    memcpy(projm.get(), param.matProj, param.ld * param.cur_dim * sizeof(Complex));

    if (param.eig_global.extlib_type == QUDA_MAGMA_EXTLIB) {
#ifdef MAGMA_LIB
      magma_Xheev(projm.get(), param.cur_dim, param.ld, evals.get(), sizeof(Complex));
#else
      errorQuda("MAGMA library was not built");
#endif
    } else if (param.eig_global.extlib_type == QUDA_EIGEN_EXTLIB) {
      Map<MatrixXcd, Unaligned, DynamicStride> projm_(projm.get(), param.cur_dim, param.cur_dim,
                                                      DynamicStride(param.ld, 1));
      Map<VectorXd, Unaligned> evals_(evals.get(), param.cur_dim);
      SelfAdjointEigenSolver<MatrixXcd> es(projm_);
      projm_ = es.eigenvectors();
      evals_ = es.eigenvalues();
    } else {
      errorQuda("Library type %d is currently not supported", param.eig_global.extlib_type);
    }

    // reset projection matrix, now we will use inverse ritz values when deflate an initial guess:
    param.use_inv_ritz = true;
    for (int i = 0; i < param.cur_dim; i++) {
      if (fabs(evals[i]) > 1e-16) {
        param.invRitzVals[i] = 1.0 / evals[i];
      } else {
        errorQuda("Cannot invert Ritz value");
      }
    }

    ColorSpinorParam csParam(param.RV->Component(0));
    // Create an eigenvector set:
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // csParam.setPrecision(search_space_prec);//eigCG internal search space precision: must be adjustable.
    csParam.is_composite = true;
    csParam.composite_dim = max_nev;

    csParam.mem_type = QUDA_MEMORY_MAPPED;
    std::unique_ptr<ColorSpinorField> buff(ColorSpinorField::Create(csParam));

    int idx = 0;
    double relerr = 0.0;
    bool do_residual_check = (tol != 0.0);

    while ((relerr < tol) && (idx < max_nev)) {
      std::vector<ColorSpinorField *> rv(param.RV->Components().begin(), param.RV->Components().begin() + param.cur_dim);
      std::vector<ColorSpinorField *> res;
      res.push_back(r);

      blas::zero(*r);
      blas::caxpy(&projm.get()[idx * param.ld], rv, res); // multiblas
      blas::copy(buff->Component(idx), *r);

      if (do_residual_check) { // if tol=0.0 then disable relative residual norm check
        *r_sloppy = *r;
        param.matDeflation(*Av_sloppy, *r_sloppy);
        double3 dotnorm = cDotProductNormA(*r_sloppy, *Av_sloppy);
        double eval = dotnorm.x / dotnorm.z;
        blas::xpay(*Av_sloppy, -eval, *r_sloppy);
        relerr = sqrt(norm2(*r_sloppy) / dotnorm.z);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Eigenvalue: %1.12e Residual: %1.12e\n", eval, relerr);
      }

      idx++;
    }

    printfQuda("\nReserved eigenvectors: %d\n", idx);
    // copy all the stuff to cudaRitzVectors set:
    for (int i = 0; i < idx; i++) blas::copy(param.RV->Component(i), buff->Component(i));

    // reset current dimension:
    param.cur_dim = idx; // idx never exceeds cur_dim.
    param.tot_dim = idx;
  }

  //supports seperate reading or single file read
  void Deflation::loadVectors(ColorSpinorField *RV)
  {
    if (RV->IsComposite()) errorQuda("Not a composite field");

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_IO);

    std::string vec_infile(param.eig_global.vec_infile);
    std::vector<ColorSpinorField *> &B = RV->Components();

    const int Nvec = B.size();
    printfQuda("Start loading %d vectors from %s\n", Nvec, vec_infile.c_str());

    void **V = new void*[Nvec];
    for (int i = 0; i < Nvec; i++) {
      V[i] = B[i]->V();
      if (V[i] == NULL) {
	printfQuda("Could not allocate V[%d]\n", i);
      }
    }

    if (strcmp(vec_infile.c_str(),"")!=0) {
      auto parity = (B[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_INVALID_PARITY : QUDA_EVEN_PARITY);
      read_spinor_field(vec_infile.c_str(), &V[0], B[0]->Precision(), B[0]->X(), B[0]->SiteSubset(), parity,
                        B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0, (char **)0);
    } else {
      errorQuda("No eigenspace file defined");
    }

    printfQuda("Done loading vectors\n");
    profile.TPSTOP(QUDA_PROFILE_IO);
    profile.TPSTART(QUDA_PROFILE_INIT);
  }

  void Deflation::saveVectors(ColorSpinorField *RV)
  {
    if (RV->IsComposite()) errorQuda("Not a composite field");

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_IO);

    std::string vec_outfile(param.eig_global.vec_outfile);
    std::vector<ColorSpinorField*> &B = RV->Components();

    if (strcmp(param.eig_global.vec_outfile,"")!=0) {
      const int Nvec = B.size();
      printfQuda("Start saving %d vectors to %s\n", Nvec, vec_outfile.c_str());

      void **V = static_cast<void**>(safe_malloc(Nvec*sizeof(void*)));
      for (int i=0; i<Nvec; i++) {
	V[i] = B[i]->V();
	if (V[i] == NULL) {
	  printfQuda("Could not allocate V[%d]\n", i);
	}
      }

      // assumes even parity if a single-parity field...
      auto parity = (B[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_INVALID_PARITY : QUDA_EVEN_PARITY);
      write_spinor_field(vec_outfile.c_str(), &V[0], B[0]->Precision(), B[0]->X(), B[0]->SiteSubset(), parity,
                         B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0, (char **)0);

      host_free(V);
      printfQuda("Done saving vectors\n");
    }

    profile.TPSTOP(QUDA_PROFILE_IO);
    profile.TPSTART(QUDA_PROFILE_INIT);
  }

} // namespace quda
