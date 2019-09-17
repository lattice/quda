#include <deflation.h>
#include <qio_field.h>
#include <string.h>

#include <memory>



#ifdef MAGMA_LIB
#include <blas_magma.h>
#endif

#include <Eigen/Dense>


namespace quda {

  using namespace blas;

  using namespace Eigen;

  using DynamicStride = Stride<Dynamic, Dynamic>;

  //static bool debug = false;

  Deflation::Deflation(DeflationParam &param, TimeProfile &profile)
    : param(param),   profile(profile){

    // for reporting level 1 is the fine level but internally use level 0 for indexing
    printfQuda("Creating deflation space of %d vectors.\n", param.tot_dim);

    printfQuda("Deflation space setup completed\n");
    // now we can run through the verification if requested
    if (param.eig_global.run_verify && param.eig_global.import_vectors) verify();
    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  Deflation::~Deflation() { if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print(); }

  double Deflation::flops() const {
			 double flops = 0;//Do we need to report this?

    //compute total flops for deflation application. Not sure we really need this.
    return flops;
  }

  /**
     Verification that the computed approximate eigenvectors are (not) valid
   */

  void Deflation::verify() {
    const int nevs_to_print = param.cur_dim;
    if(nevs_to_print == 0) errorQuda("\nIncorrect size of current deflation space. \n");

    //std::unique_ptr<double, decltype(pinned_deleter) > projm( pinned_allocator(param.ld*param.cur_dim * sizeof(double)), pinned_deleter);
    std::unique_ptr<double[] > projm(new double[param.ld*param.cur_dim]);

    Map<MatrixXd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
    Map<MatrixXd, Unaligned, DynamicStride> evecs_(projm.get(), param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));

    SelfAdjointEigenSolver<MatrixXd> es_projm( projm_ );
    evecs_.block(0, 0, param.cur_dim, param.cur_dim) = es_projm.eigenvectors();

    ColorSpinorParam csParam(*param.RV[0]);
    csParam.create   = QUDA_ZERO_FIELD_CREATE;
    csParam.location = param.location;
    csParam.mem_type = QUDA_MEMORY_DEVICE;
    csParam.setPrecision(QUDA_DOUBLE_PRECISION);

    if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
      if(csParam.nSpin != 1) csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    }

    std::unique_ptr<ColorSpinorField> r(ColorSpinorField::Create(csParam));

    csParam.setPrecision(param.eig_global.cuda_prec_ritz);//accum fields always full precision
    if (csParam.location==QUDA_CUDA_FIELD_LOCATION) csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;

    std::unique_ptr<ColorSpinorField> r_sloppy(ColorSpinorField::Create(csParam));
    std::unique_ptr<ColorSpinorField> Av_sloppy(ColorSpinorField::Create(csParam));

    std::vector<ColorSpinorField*> rv_(param.RV.begin(), param.RV.begin() + param.cur_dim);
    std::vector<ColorSpinorField*> res_{r.get()};

    for(int i = 0; i < nevs_to_print; i++)
    {
       zero(*r);
       blas::axpy(&projm.get()[i*param.ld], rv_, res_);
       *r_sloppy = *r;

       param.matDeflation(*Av_sloppy, *r_sloppy);

       double rdotAv = blas::reDotProduct(*r_sloppy, *Av_sloppy);
       double norm2r = blas::norm2(*r_sloppy);

       double eval = rdotAv / norm2r;

       blas::xpay(*Av_sloppy, -eval, *r_sloppy );

       double relerr = sqrt( norm2(*r_sloppy) / norm2r );

       printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", i+1, eval, relerr);
    }

    return;
  }

  void Deflation::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if(param.cur_dim == 0) return;//nothing to do

    std::unique_ptr<double[] > vec(new double[param.ld]);

    double check_nrm2 = norm2(b);

    printfQuda("\nSource norm (gpu): %1.15e, curr deflation space dim = %d\n", sqrt(check_nrm2), param.cur_dim);

    std::vector<ColorSpinorField*> rv_(param.RV.begin(), param.RV.begin() + param.cur_dim);
    std::vector<ColorSpinorField*> in_{static_cast<ColorSpinorField*>(&b)};

    blas::reDotProduct(vec.get(), rv_, in_);//<i, b>
    //for(int j = 0; j < param.cur_dim; j++) vec[j] = blas::reDotProduct(*param.RV[j], b);

    if(!param.use_inv_ritz)
    {
      Map<MatrixXd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
      Map<VectorXd, Unaligned> vec_ (vec.get(), param.cur_dim);

      VectorXd  vec2_(param.cur_dim);
      vec2_ = projm_.fullPivHouseholderQr().solve(vec_);

      vec_  = vec2_;
    } else {
      for(int i = 0; i < param.cur_dim; i++) vec[i] *= param.invRitzVals[i];
    }

    std::vector<ColorSpinorField*> out_{&x};
    blas::axpy(vec.get(), rv_, out_); //multiblas

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));

    return;
  }
}
