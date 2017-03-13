#include <deflation.h>
#include <qio_field.h>
#include <string.h>

#ifdef MAGMA_LIB
#include <blas_magma.h>
#endif
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace quda {  

  using namespace blas;
  using namespace Eigen;

  using DynamicStride = Stride<Dynamic, Dynamic>;

  //static bool debug = false;

  Deflation::Deflation(DeflationParam &param, TimeProfile &profile)
    : param(param),   profile(profile),
      r(nullptr), Av(nullptr) {

    // for reporting level 1 is the fine level but internally use level 0 for indexing
    printfQuda("Creating deflation space of %d vectors\n", param.tot_dim);

    if( param.eig_global.import_vectors ) loadVectors(param.RV);//whether to load eigenvectors
    // create residual vector
    {
      ColorSpinorParam csParam(param.RV->Component(0));
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.location = param.location;
      if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
	// all coarse GPU vectors use FLOAT2 ordering
	csParam.fieldOrder = (csParam.precision == QUDA_DOUBLE_PRECISION ) ?  QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
	csParam.setPrecision(csParam.precision);
	csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      }

      r  = ColorSpinorField::Create(csParam);
      Av = ColorSpinorField::Create(csParam);
    }

    printfQuda("setup completed\n");
    // now we can run through the verification if requested
    if (param.eig_global.run_verify && param.eig_global.import_vectors) verify();
    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  Deflation::~Deflation() {
    if (r)  delete r;
    if (Av) delete Av;
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  double Deflation::flops() const {
    double flops = 0;
    
    //compute flops
    return flops;
  }

  /**
     Verification that the computed approximate eigenvectors are (not) valid
   */
  void Deflation::verify() {

    const int nevs_to_print = param.cur_dim;
    if(nevs_to_print == 0) errorQuda("\nIncorrect size of current deflation space. \n"); 

    Complex *projm  = new Complex [param.ld*param.cur_dim];

#ifdef MAGMA_LIB 
    memcpy(projm, param.matProj, param.ld*param.cur_dim*sizeof(Complex));
    double *evals = new double[param.ld];

    magma_Xheev(projm, param.ld, param.cur_dim, evals, sizeof(Complex));

    delete [] evals;
#else
    Map<MatrixXcd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
    Map<MatrixXcd, Unaligned, DynamicStride> evecs_(projm, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));

    SelfAdjointEigenSolver<MatrixXcd> es_projm( projm_ );
    evecs_.block(0, 0, param.cur_dim, param.cur_dim) = es_projm.eigenvectors();//??   
#endif

    std::vector<ColorSpinorField*> rv(param.RV->Components().begin(), param.RV->Components().begin() + param.cur_dim);
    std::vector<ColorSpinorField*> res;
    res.push_back(r);

    for(int i = 0; i < nevs_to_print; i++)
    {
       blas::caxpy(&projm[i*param.ld], rv, res);//use multiblas here

       param.matDeflation(*Av, *r);

       double3 dotnorm = cDotProductNormA(*r, *Av);

       double eval = dotnorm.x / dotnorm.z;

       blas::xpay(*Av, -eval, *r );

       double relerr = sqrt( norm2(*r) / dotnorm.z );

       zero(*r);

       printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", i+1, eval, relerr);
    }

    delete [] projm;

    return;
  }

  void Deflation::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if(param.eig_global.invert_param->inv_type != QUDA_EIGCG_INVERTER && param.eig_global.invert_param->inv_type != QUDA_INC_EIGCG_INVERTER) 
       errorQuda("\nMethod is not implemented for %d inverter type.\n", param.eig_global.invert_param->inv_type);

    if(param.cur_dim == 0) return;//nothing to do

    Complex  *vec   = new Complex[param.ld];

    double check_nrm2 = norm2(b);
    printfQuda("\nSource norm (gpu): %1.15e\n", sqrt(check_nrm2));

    if(param.RV->Precision() != x.Precision() || param.RV->Precision() != b.Precision() ) errorQuda("\nMixing precisions is not supported yet.\n");

    const int cdot_length  = 4;

    int offset = 0;

    //Warning! this won't work with arbitrary param.cur_dim, so pipelining is needed, also must be generalized for CPU fields
    do{
      const int local_length = (param.cur_dim - offset) > cdot_length  ? cdot_length : (param.cur_dim - offset) ;

      std::vector<cudaColorSpinorField*> rv_;
      std::vector<cudaColorSpinorField*> in_;

      rv_.reserve(local_length);
      in_.reserve(local_length);

      for(int i = 0; i < local_length; i++)
      {
        rv_.push_back(static_cast<cudaColorSpinorField*>(&param.RV->Component(offset+i)));
        in_.push_back(static_cast<cudaColorSpinorField*>(&b));
      }

      //Warning! this won't work with arbitrary param.cur_dim. Pipelining is needed.
      blas::cDotProduct(&vec[offset], rv_, in_);//<i, b>

      offset += cdot_length;

    } while (offset < param.cur_dim);

    if(!param.use_inv_ritz) 
    {
#ifdef MAGMA_LIB
      magma_Xgesv(vec, param.ld, param.cur_dim, param.matProj, param.ld, sizeof(Complex));
#else
      Map<MatrixXcd, Unaligned, DynamicStride> projm_(param.matProj, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
      Map<VectorXcd, Unaligned, DynamicStride> vec_ (vec, param.cur_dim);

      VectorXcd  vec2_(param.cur_dim);

      vec2_ = projm_.colPivHouseholderQr().solve(vec_);
      vec_  = vec2_ 
#endif
    }
    else
    {
      for(int i = 0; i < param.cur_dim; i++) vec[i] *= param.invRitzVals[i];
    }

    std::vector<ColorSpinorField*> rv_(param.RV->Components().begin(), param.RV->Components().begin()+param.cur_dim);
    std::vector<ColorSpinorField*> out_;
    out_.push_back(&x);

    blas::caxpy(vec, rv_, out_); //a*i+x

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));

    delete [] vec;

    return;
  }

  void Deflation::increment(ColorSpinorField &Vm, int nev) {

    if(param.eig_global.invert_param->inv_type != QUDA_EIGCG_INVERTER && param.eig_global.invert_param->inv_type != QUDA_INC_EIGCG_INVERTER) 
       errorQuda("\nMethod is not implemented for %d inverter type.\n", param.eig_global.invert_param->inv_type);

    if( nev == 0 ) return; //nothing to do

    const int first_idx = param.cur_dim;

    if(param.RV->CompositeDim() < (first_idx+nev) || param.tot_dim < (first_idx+nev)) errorQuda("\nNot enough space to copy %d vectors..\n", param.tot_dim);

    for(int i = 0; i < nev; i++) blas::copy(param.RV->Component(first_idx+i), Vm.Component(i));

    printfQuda("\nConstruct projection matrix..\n");

    //Block MGS orthogonalization
    const int cdot_pipeline_length  = 4;

    for(int i = first_idx; i < (first_idx + nev); i++)
    {
      Complex *alpha = new Complex[cdot_pipeline_length];

      int offset = 0;

      //Warning! this won't work with arbitrary param.cur_dim, so pipelining is needed, also must be generalized for CPU fields
      while (offset < i) {
        
        const int local_length = (i - offset) > cdot_pipeline_length  ? cdot_pipeline_length : (i - offset);

        std::vector<cudaColorSpinorField*> vj_local;
        std::vector<cudaColorSpinorField*> vi_local;

        vj_local.reserve(local_length);
        vi_local.reserve(local_length);

        for(int j = 0; j < local_length; j++)
        {
          vj_local.push_back(static_cast<cudaColorSpinorField*>(&param.RV->Component(offset+j)));
          vi_local.push_back(static_cast<cudaColorSpinorField*>(&param.RV->Component(i)));
          alpha[j] = 0.0;
        }

        //Warning! this won't work with arbitrary param.cur_dim. Pipelining is needed.
        blas::cDotProduct(alpha, vj_local, vi_local);

        std::vector<ColorSpinorField*> vj_global(param.RV->Components().begin()+offset, param.RV->Components().begin()+offset+local_length);
        std::vector<ColorSpinorField*> vi_global;
        vi_global.push_back(&param.RV->Component(i));

        for(int j = 0; j < local_length; j++) alpha[j] = -alpha[j]; 

        blas::caxpy(alpha, vj_global, vi_global); //i-<j,i>j

        offset += cdot_pipeline_length;
      }

      alpha[0] = blas::norm2(param.RV->Component(i));

      if(alpha[0].real() > 1e-16) blas::ax(1.0 /sqrt(alpha[0].real()), param.RV->Component(i));
      else                        errorQuda("\nCannot orthogonalize %dth vector\n", i);

      param.matDeflation(*Av, param.RV->Component(i));//precision must match!
      //load diagonal:
      param.matProj[i*param.ld+i] = cDotProduct(param.RV->Component(i), *Av);

      //off-diagonal (use multiblas):
#if 0
      for (int j = 0; j < i; j++)//row id
      {
        //
        alpha[0] = cDotProduct(param.RV->Component(j), *Av)
        param.matProj[i*param.ld+j] = alpha[0];
        param.matProj[j*param.ld+i] = conj(alpha[0]);//conj
      }
#endif
      offset = 0;

      while (offset < i){
        const int local_length = (i - offset) > cdot_pipeline_length  ? cdot_pipeline_length : (i - offset) ;

        std::vector<cudaColorSpinorField*> vj_local;
        std::vector<cudaColorSpinorField*> av_local;

        vj_local.reserve(local_length);
        av_local.reserve(local_length);

        for(int j = 0; j < local_length; j++)
        {
          vj_local.push_back(static_cast<cudaColorSpinorField*>(&param.RV->Component(offset+j)));
          av_local.push_back(static_cast<cudaColorSpinorField*>(Av));
          alpha[j] = 0.0;
        }

        //Warning! this won't work with arbitrary param.cur_dim. Pipelining is needed.
        blas::cDotProduct(alpha, vj_local, av_local);

        for (int j = 0; j < local_length; j++)//row id
        {
          param.matProj[i*param.ld+(j+offset)] = alpha[j];
          param.matProj[(j+offset)*param.ld+i] = conj(alpha[j]);//conj
        }

        offset += cdot_pipeline_length;

      }

      delete [] alpha;
    }

    param.cur_dim += nev;

    return;
  }

  void Deflation::reduce(double tol, int max_nev) {

     if(param.cur_dim < max_nev)
     {
        printf("\nToo big number of eigenvectors was requested, switched to maximum available number %d\n", param.cur_dim);
        max_nev = param.cur_dim;
     }

     double *evals   = (double*)calloc(param.cur_dim, sizeof(double));//WARNING: Ritz values always in double.

     Complex *projm  = (Complex*)mapped_malloc(param.ld*param.cur_dim * sizeof(Complex));
     memcpy(projm, param.matProj, param.ld*param.cur_dim*sizeof(Complex));

#ifdef MAGMA_LIB
     cudaHostRegister(static_cast<void *>(projm), param.ld*param.cur_dim*sizeof(Complex),  cudaHostRegisterDefault);
     magma_Xheev(projm, param.cur_dim, param.ld, evals, sizeof(Complex));
     cudaHostUnregister(projm);
#else
     Map<MatrixXcd, Unaligned, DynamicStride> projm_(projm, param.cur_dim, param.cur_dim, DynamicStride(param.ld, 1));
     Map<VectorXcd, Unaligned, DynamicStride> evals_(eval, param.cur_dim);

     SelfAdjointEigenSolver<MatrixXcd> es(projm_);

     evals = es.eigenvalues();
#endif

     //reset projection matrix:
     for(int i = 0; i < param.cur_dim; i++)
     {
       if(fabs(evals[i]) > 1e-16)
       {
         param.invRitzVals[i] = 1.0 / evals[i];
       }
       else
       {
         errorQuda("\nCannot invert Ritz value.\n");
       }
     }

     ColorSpinorParam csParam(param.RV->Component(0));
     //Create an eigenvector set:
     csParam.create   = QUDA_ZERO_FIELD_CREATE;
     //csParam.setPrecision(search_space_prec);//eigCG internal search space precision: must be adjustable.
     csParam.is_composite  = true;
     csParam.composite_dim = max_nev;

     ColorSpinorField *Vm = ColorSpinorField::Create(csParam); //search space for Ritz vectors


     int idx       = 0;
     double relerr = 0.0;

     while ((relerr < tol) && (idx < max_nev))//newnev
     {
       std::vector<ColorSpinorField*> rv(param.RV->Components().begin(), param.RV->Components().begin() + param.cur_dim);
       std::vector<ColorSpinorField*> res;
       res.push_back(r);

       zero(*r);

       blas::caxpy(&projm[idx*param.ld], rv, res);

       blas::copy(Vm->Component(idx), *r);

       param.matDeflation(*Av, *r);

       double3 dotnorm = cDotProductNormA(*r, *Av);

       double eval = dotnorm.x / dotnorm.z;

       blas::xpay(*Av, -eval, *r );

       double relerr = sqrt( norm2(*r) / dotnorm.z );

       if(getVerbosity() >= QUDA_VERBOSE) printfQuda("Eigenvalue: %1.12e Residual: %1.12e\n", eval, relerr);

       idx += 1;
     }

     param.ReshapeRitzVectors(idx, param.RV->Location());//

     //copy all the stuff to cudaRitzVectors set:
     for(int i = 0; i < idx; i++) blas::copy(param.RV->Component(i), Vm->Component(i));

     //reset current dimension:
     printfQuda("\nUsed eigenvectors: %d\n", idx);

     param.cur_dim = idx;//idx never exceeds cur_dim.

     free(evals);
     host_free(projm);

     return;
  }

  //supports seperate reading or single file read
  void Deflation::loadVectors(ColorSpinorField *RV) {

    if(RV->IsComposite()) errorQuda("\nNot a composite field.\n");

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_IO);

    std::string vec_infile(param.eig_global.vec_infile);

    std::vector<ColorSpinorField*> &B = RV->Components(); 

    const int Nvec = B.size();
    printfQuda("Start loading %d vectors from %s\n", Nvec, vec_infile.c_str());

    void **V = new void*[Nvec];
    for (int i=0; i<Nvec; i++) { 
      V[i] = B[i]->V();
      if (V[i] == NULL) {
	printfQuda("Could not allocate V[%d]\n", i);
      }
    }

    if (strcmp(vec_infile.c_str(),"")!=0) {
      read_spinor_field(vec_infile.c_str(), &V[0], B[0]->Precision(), B[0]->X(),
			B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);
    } else {
      errorQuda("No eigenspace file defined.");
    }

    printfQuda("Done loading vectors\n");
    profile.TPSTOP(QUDA_PROFILE_IO);
    profile.TPSTART(QUDA_PROFILE_INIT);

    return;
  }

  void Deflation::saveVectors(ColorSpinorField *RV) {
    if(RV->IsComposite()) errorQuda("\nNot a composite field.\n");

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

      write_spinor_field(vec_outfile.c_str(), &V[0], B[0]->Precision(), B[0]->X(),
			 B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);

      host_free(V);
      printfQuda("Done saving vectors\n");
    }

    profile.TPSTOP(QUDA_PROFILE_IO);
    profile.TPSTART(QUDA_PROFILE_INIT);

    return;
  }

}
