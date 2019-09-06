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

  static auto pinned_allocator = [] (size_t bytes ) { return static_cast<double*>(pool_pinned_malloc(bytes)); };
  static auto pinned_deleter   = [] (double *hptr)  { pool_pinned_free(hptr); };


  //static bool debug = false;

  Deflation::Deflation(DeflationParam &param, TimeProfile &profile)
    : param(param),   profile(profile),
      r(nullptr), Av(nullptr), r_sloppy(nullptr), Av_sloppy(nullptr) {


    // for reporting level 1 is the fine level but internally use level 0 for indexing
    printfQuda("Creating deflation space of %d vectors.\n", param.tot_dim);

    //if( param.eig_global.import_vectors ) loadVectors(param.RV);//whether to load eigenvectors
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

    if(param.eig_global.cuda_prec_ritz != QUDA_DOUBLE_PRECISION ) { //allocate sloppy fields

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

  Deflation::~Deflation() {

    if( param.eig_global.cuda_prec_ritz != QUDA_DOUBLE_PRECISION ) {
      if (r_sloppy) delete r_sloppy;
      if (Av_sloppy) delete Av_sloppy;
    }

    if (r)  delete r;
    if (Av) delete Av;

    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

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

    ColorSpinorFieldSet &rv = *param.RV;

    std::vector<ColorSpinorField*> rv_(rv(0, param.cur_dim));
    std::vector<ColorSpinorField*> res_;
    res_.push_back(r);

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
//    if(param.eig_global.invert_param->inv_type != QUDA_EIGCG_INVERTER && param.eig_global.invert_param->inv_type != QUDA_INC_EIGCG_INVERTER && param.eig_global.invert_param->inv_type != QUDA_CG_INVERTER)
//       errorQuda("\nMethod is not implemented for %d inverter type.\n", param.eig_global.invert_param->inv_type);

    if(param.cur_dim == 0) return;//nothing to do

    std::unique_ptr<double[] > vec(new double[param.ld]);

    double check_nrm2 = norm2(b);

    ColorSpinorFieldSet &rv = *param.RV;

    printfQuda("\nSource norm (gpu): %1.15e, curr deflation space dim = %d\n", sqrt(check_nrm2), param.cur_dim);

    ColorSpinorField *b_sloppy = rv.Precision() != b.Precision() ? r_sloppy : &b;
    *b_sloppy = b;

    std::vector<ColorSpinorField*> rv_(rv(0,param.cur_dim));
    std::vector<ColorSpinorField*> in_;
    in_.push_back(static_cast<ColorSpinorField*>(b_sloppy));

    //blas::reDotProduct(vec.get(), rv_, in_);//<i, b>
		for(int j = 0; j < param.cur_dim; j++) vec[j] = blas::reDotProduct(rv[j], *b_sloppy);

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

    std::vector<ColorSpinorField*> out_;
    out_.push_back(&x);
    blas::axpy(vec.get(), rv_, out_); //multiblas

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));

    return;
  }
}
