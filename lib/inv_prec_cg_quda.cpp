#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <face_quda.h>
#include <domain_decomposition.h>
#include <resize_quda.h>

// I need to add functionality to the conjugate-gradient solver.
// Should I do this by simple inheritance, or should I use a decorator? 
// Is it even possible to use a decorator?

namespace quda {

  // set the required parameters for the inner solver
  void fillInnerCGInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer) {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.reliable_delta = 1e-20; // no reliable updates within the inner solver

    inner.cuda_prec = outer.cuda_prec_precondition; // preconditioners are uni-precision solvers
    inner.cuda_prec_sloppy = outer.cuda_prec_precondition;

    inner.verbosity = outer.verbosity;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_CG_INVERTER; // used to tell the inner solver it is an inner solver

    if (outer.inv_type == QUDA_CG_INVERTER && outer.cuda_prec_sloppy != outer.cuda_prec_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES; // What does this mean?

  } // copied from inv_gcr_quda.cpp



  PreconCG::PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrec, QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrec), K(NULL)
  {
    Kparam = newQudaInvertParam();

    for(int dir=0; dir<4; ++dir) Kparam.domain_overlap[dir] = invParam.domain_overlap[dir];
    fillInnerCGInvertParam(Kparam, invParam);


    Kparam.dslash_type   = invParam.dslash_type;
    Kparam.inv_type      = invParam.inv_type;
    Kparam.solution_type = invParam.solution_type;
    Kparam.solve_type    = invParam.solve_type;
    Kparam.matpc_type    = invParam.matpc_type;
    Kparam.dirac_order   = invParam.dirac_order;


    Kparam.input_location  = invParam.input_location;
    Kparam.output_location = invParam.output_location;
    Kparam.mass = invParam.mass;
    Kparam.dagger = invParam.dagger;
    Kparam.mass_normalization = invParam.mass_normalization;
    Kparam.preserve_source = invParam.preserve_source;

    Kparam.cpu_prec = invParam.cpu_prec;
    Kparam.cuda_prec = invParam.cuda_prec_precondition;
    Kparam.cuda_prec_sloppy = invParam.cuda_prec_precondition;


    K = new CG(matPrecon, matPrecon, Kparam, profile);
  }


  PreconCG::~PreconCG() {
    if (K) delete K;
  }

  void PreconCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b)
  {
    printfQuda("Calling preconditioned solver\n");
    int k=0;
    int rUpdate;

    cudaColorSpinorField* minvrPre_ptr;
    cudaColorSpinorField* rPre_ptr;
    cudaColorSpinorField* minvr_ptr;
    cudaColorSpinorField* p_ptr;

    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    int max_overlap = Kparam.domain_overlap[0];
    for(int dir=1; dir<4; ++dir){
      if(Kparam.domain_overlap[dir] > max_overlap){ 
        max_overlap = Kparam.domain_overlap[dir];
      }
    }

    printfQuda("max_overlap = %d\n", max_overlap);
    fflush(stdout);

    int X[4]; // smaller sublattice dimensions
    int Y[4]; // extended subdomain dimensions
    X[0] = b.X(0)*2; // assume QUDA_PARITY_SITE_SUBSET
    X[1] = b.X(1);
    X[2] = b.X(2);
    X[3] = b.X(3);
    for(int dir=0; dir<4; ++dir) Y[dir] = X[dir] + 2*Kparam.domain_overlap[dir];


    printfQuda("Y = %d %d %d %d\n", Y[0], Y[1], Y[2], Y[3]);
    fflush(stdout);

    DecompParam dparam;
    initDecompParam(&dparam,X,Y);
    DecompParam param2;
    initDecompParam(&param2,X,X);
    int domain_overlap[4];
    for(int dir=0; dir<4; ++dir) domain_overlap[dir] = invParam.domain_overlap[dir];

    printfQuda("Calling ColorSpinorParam param(b)\n");
    fflush(stdout);
    ColorSpinorParam param(b);
    param.nFace  = max_overlap;
    param.create = QUDA_COPY_FIELD_CREATE; 
    printfQuda("Calling cudaColorSpinorField r(b,param)\n");
    fflush(stdout);
    cudaColorSpinorField r(b,param);
    printfQuda("Call to cudaColorSpinorField r(b,param) complete\n");

    printfQuda("Calling Extender constructor\n");
    fflush(stdout);
    Extender extendCuda(r); // function object used to implement overlapping domains
    printfQuda("Call to Extender constructor completed\n");
    fflush(stdout);

    param.nFace  = b.Nface();
    param.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b,param);
    minvr_ptr = new cudaColorSpinorField(b,param);




    MPI_Barrier(MPI_COMM_WORLD);
    printfQuda("Calling test mat(r,x,y)\n");
    fflush(stdout);
    mat(r, b, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    //		cudaColorSpinorField& tmp);
    //
    // => r = A*x;
    MPI_Barrier(MPI_COMM_WORLD);

    printfQuda("Call to mat(r,x,y)\n");
    fflush(stdout);
//    MPI_Finalize();

    double r2 = xmyNormCuda(b,r);
    rUpdate++;

    param.precision = invParam.cuda_prec_sloppy;
    cudaColorSpinorField Ap(x,param);
    cudaColorSpinorField tmp(x,param);


    ColorSpinorParam prec_param(x);
    prec_param.create = QUDA_COPY_FIELD_CREATE;
    prec_param.precision = invParam.cuda_prec_precondition;

    if(K){
      prec_param.create     = QUDA_ZERO_FIELD_CREATE;
      prec_param.precision  = invParam.cuda_prec_precondition;
      prec_param.nColor     = 3;
      prec_param.nDim       = 4;
      prec_param.pad        = 0; // Not sure if this will cause a problem
      // prec_param.pad        = r.Pad(); 
      prec_param.nSpin      = 1;
      prec_param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      prec_param.siteOrder  = QUDA_EVEN_ODD_SITE_ORDER;
      prec_param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
      for(int dir=0; dir<4; ++dir) prec_param.x[dir] = Y[dir];
      prec_param.x[0] /= 2; // since QUDA_PARITY_SITE_SUBSET

      rPre_ptr = new cudaColorSpinorField(prec_param);

      // HACK!!!
      int domain_overlap[4];
      for(int dir=0; dir<4; ++dir) domain_overlap[dir] = invParam.domain_overlap[dir];

      printfQuda("About to check max_overlap\n");
      fflush(stdout);
      //      if(max_overlap > 0){
      printfQuda("Calling extendCuda\n");
      fflush(stdout);

      double norm2_r = norm2(r);
      extendCuda(*rPre_ptr,b,dparam,domain_overlap);
      printfQuda("Call to extendCuda complete\n");
      cropCuda(*minvr_ptr,*rPre_ptr,dparam);
      //cropCuda(*minvr_ptr,r,param2);

      double norm2_mr = norm2(*minvr_ptr);

      printfQuda("norm2_mr = %lf, norm2_r = %lf\n", norm2_mr, norm2_r);

      fflush(stdout);
      //      }else{
      //        printfQuda("Calling *rPre_ptr = r\n");
      //        fflush(stdout);
      //        printfQuda("rPre_ptr->Length() = %d\n", rPre_ptr->Length());
      //        printfQuda("rPre_ptr->RealLength() = %d\n", rPre_ptr->RealLength());
      //        printfQuda("rPre_ptr->Pad() = %d\n", rPre_ptr->Pad());
      //        printfQuda("rPre_ptr->Stride() = %d\n", rPre_ptr->Stride());
      //        printfQuda("r.Length() = %d\n",r.Length());
      //        printfQuda("r.RealLength() = %d\n", r.RealLength());
      //        printfQuda("r.Pad() = %d\n", r.Pad());
      //        printfQuda("r.Stride() = %d\n", r.Stride());
      //        fflush(stdout);

      //        *rPre_ptr = r;
      //        printfQuda("Call to *rPre_ptr complete\n");
      //      }

      // Create minvrPre_ptr 
      minvrPre_ptr = new cudaColorSpinorField(*rPre_ptr);
      globalReduce = false;
      printfQuda("About to call K->operator()(*minvrPre_ptr, *rPre_ptr)");
      fflush(stdout);
      K->operator()(*minvrPre_ptr, *rPre_ptr);  
      globalReduce = true;


      if(max_overlap){
        cropCuda(*minvr_ptr, *minvrPre_ptr, dparam);
      }else{
        *minvr_ptr = *minvrPre_ptr;
      }

      p_ptr = new cudaColorSpinorField(*minvr_ptr);
    }else{
      p_ptr = new cudaColorSpinorField(r);
    }




    // FIX THIS!!
    if(K){ // These are only needed if preconditioning is used
      delete minvrPre_ptr;
      delete rPre_ptr;
    }
    delete p_ptr;
    delete minvr_ptr;

    return;
  }


  } // namespace quda
