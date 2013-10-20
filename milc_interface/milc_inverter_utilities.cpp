#include "include/milc_inverter_utilities.h"
#include <quda.h>
#include <color_spinor_field.h>


using namespace quda;

#define MAX(a,b) ((a)>(b)?(a):(b))


namespace milc_interface {

  // set the params for the single mass solver
  void setInvertParams(const int dim[4],
      QudaPrecision cpu_prec,
      QudaPrecision cuda_prec,
      QudaPrecision cuda_prec_sloppy,
      QudaPrecision cuda_prec_precondition,
      double mass,
      double target_residual, 
      int maxiter,
      double reliable_delta,
      QudaParity parity,
      QudaVerbosity verbosity,
      QudaInverterType inverter,
      QudaInvertParam *invertParam)
  {
    invertParam->verbosity = verbosity;
    invertParam->mass = mass;
    invertParam->tol = target_residual;
    invertParam->num_offset = 0;

    invertParam->inv_type = inverter;
    invertParam->maxiter = maxiter;
    invertParam->reliable_delta = reliable_delta;



    invertParam->mass_normalization = QUDA_MASS_NORMALIZATION;
    invertParam->cpu_prec = cpu_prec;
    invertParam->cuda_prec = cuda_prec;
    invertParam->cuda_prec_sloppy = cuda_prec_sloppy;

    invertParam->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    invertParam->solve_type = QUDA_NORMEQ_PC_SOLVE; 
    invertParam->preserve_source = QUDA_PRESERVE_SOURCE_YES;
    invertParam->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
    invertParam->dirac_order = QUDA_DIRAC_ORDER;

    invertParam->dslash_type = QUDA_ASQTAD_DSLASH;
    invertParam->tune = QUDA_TUNE_YES;
    invertParam->gflops = 0.0;

    invertParam->input_location = QUDA_CPU_FIELD_LOCATION;
    invertParam->output_location = QUDA_CPU_FIELD_LOCATION;


    if(parity == QUDA_EVEN_PARITY){ // even parity
      invertParam->matpc_type = QUDA_MATPC_EVEN_EVEN;
    }else if(parity == QUDA_ODD_PARITY){
      invertParam->matpc_type = QUDA_MATPC_ODD_ODD;
    }else{
      errorQuda("Invalid parity\n");
      exit(1);
    }

    invertParam->dagger = QUDA_DAG_NO;
    invertParam->sp_pad = dim[0]*dim[1]*dim[2]/2;
    invertParam->use_init_guess = QUDA_USE_INIT_GUESS_YES; 

/*
#ifdef MULTI_GPU
    invertParam->nface = 3;
#else
    invertParam->nface = 0;
#endif
*/
    // for the preconditioner
    invertParam->inv_type_precondition = QUDA_CG_INVERTER;
    invertParam->tol_precondition = 1e-1;
    invertParam->maxiter_precondition = 2;
    invertParam->verbosity_precondition = QUDA_SILENT;
    invertParam->cuda_prec_precondition = cuda_prec_precondition;


    return;
  }




  // Set params for the multi-mass solver.
  void setInvertParams(const int dim[4],
      QudaPrecision cpu_prec,
      QudaPrecision cuda_prec,
      QudaPrecision cuda_prec_sloppy,
      QudaPrecision cuda_prec_precondition,
      int num_offset,
      const double offset[],
      const double target_residual_offset[],
      const double target_residual_hq_offset[],
      int maxiter,
      double reliable_delta,
      QudaParity parity,
      QudaVerbosity verbosity,
      QudaInverterType inverter,
      QudaInvertParam *invertParam)
  {

    const double null_mass = -1;
    const double null_residual = -1;


    setInvertParams(dim, cpu_prec, cuda_prec, cuda_prec_sloppy, cuda_prec_precondition,
        null_mass, null_residual, maxiter, reliable_delta, parity, verbosity, inverter, invertParam);

    invertParam->num_offset = num_offset;
    for(int i=0; i<num_offset; ++i){
      invertParam->offset[i] = offset[i];
      invertParam->tol_offset[i] = target_residual_offset[i];
      if(invertParam->residual_type == QUDA_HEAVY_QUARK_RESIDUAL){
        invertParam->tol_hq_offset[i] = target_residual_hq_offset[i];
      }
    }
    return;
  }


  void setGaugeParams(const int dim[4],
      QudaPrecision cpu_prec,
      QudaPrecision cuda_prec,
      QudaPrecision cuda_prec_sloppy,
      QudaPrecision cuda_prec_precondition,
      const double tadpole,
      QudaGaugeParam *gaugeParam)   
  {

    for(int dir=0; dir<4; ++dir){
      gaugeParam->X[dir] = dim[dir];
    }

    gaugeParam->cpu_prec = cpu_prec;
    gaugeParam->cuda_prec = cuda_prec;
    gaugeParam->cuda_prec_sloppy = cuda_prec_sloppy;
    gaugeParam->reconstruct = QUDA_RECONSTRUCT_NO;
    gaugeParam->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

    gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
    gaugeParam->anisotropy = 1.0;
    gaugeParam->tadpole_coeff = tadpole;
    gaugeParam->t_boundary = QUDA_PERIODIC_T; // anti-periodic boundary conditions are built into the gauge field
    gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER; 
    gaugeParam->ga_pad = dim[0]*dim[1]*dim[2]/2;
    gaugeParam->scale = -1.0/(24.0*gaugeParam->tadpole_coeff*gaugeParam->tadpole_coeff);
    

    // preconditioning...
    gaugeParam->cuda_prec_precondition = cuda_prec_precondition;
    gaugeParam->reconstruct_precondition = QUDA_RECONSTRUCT_NO;

    return;
  }



  void setColorSpinorParams(const int dim[4],
      QudaPrecision precision,
      ColorSpinorParam* param)
  {

    param->nColor = 3;
    param->nSpin = 1;
    param->nDim = 4;

    for(int dir=0; dir<4; ++dir){
      param->x[dir] = dim[dir];
    }
    param->x[0] /= 2; // Why this particular direction? J.F.

    param->precision = precision;
    param->pad = 0;
    param->siteSubset = QUDA_PARITY_SITE_SUBSET;
    param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
    param->create = QUDA_ZERO_FIELD_CREATE;

    return;
  } 


  int getFatLinkPadding(const int dim[4])
  {
    int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
    padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
    padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);

    return padding;
  }



  size_t getColorVectorOffset(QudaParity local_parity, bool even_odd_exchange, int volume)
  {
    size_t offset;
    if(local_parity == QUDA_EVEN_PARITY){
      offset = even_odd_exchange ? volume*6/2 : 0;
    }else{
      offset = even_odd_exchange ? 0 : volume*6/2;
    }
    return offset;
  }



} // namespace milc_interface

#undef MAX
