#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
// quda-specific headers
#include <quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "../tests/blas_reference.h" // needed for norm_2
#include "external_headers/quda_milc_interface.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/milc_utilities.h"

extern int V;
extern int Vh;
extern int Vsh;
extern int Vs_t;
extern int Vsh_t;
extern int faceVolume[4]; 
extern int Z[4];

using namespace quda;


namespace milc_interface {

static void setDims(int *X){
  V = 1;
  for(int dir=0; dir<4; ++dir){ 
    V *= X[dir];
    Z[dir] = X[dir];
  }

  for(int dir=0; dir<4; ++dir){
    faceVolume[dir] = V/X[dir];
  }
  Vh = V/2;
  Vs_t  = Z[0]*Z[1]*Z[2];
  Vsh_t = Vs_t/2;

  return;
}


static void
setColorSpinorParams(const int dim[4],
                     QudaPrecision precision,
		     ColorSpinorParam* param)
{

  param->nColor = 3;
  param->nSpin = 4;
  param->nDim = 4;

  for(int dir=0; dir<4; ++dir){
   param->x[dir] = dim[dir];
  }

  param->precision = precision;
  param->pad = 0;
  param->siteSubset = QUDA_FULL_SITE_SUBSET;
  param->siteOrder  = QUDA_EVEN_ODD_SITE_ORDER;
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; 
  param->create     = QUDA_ZERO_FIELD_CREATE;

  return;
}


} // namespace milc_interface


void qudaCloverInvert(int external_precision, 
		      int quda_precision,
		      double kappa,
		      QudaInvertArgs_t inv_args,
		      double target_residual,
		      double target_fermilab_residual,
		      const void* link,
		      void* clover, // could be stored in Milc format
		      void* cloverInverse,
		      void* source,
		      void* solution,
		      double* const final_residual, 
		      double* const final_fermilab_residual,
		      int* num_iters)
{
  using namespace milc_interface;

  if(target_fermilab_residual !=0 && target_residual != 0){
    errorQuda("qudaCloverInvert: conflicting residuals requested\n");
    exit(1);
  }else if(target_fermilab_residual == 0 && target_residual == 0){
    errorQuda("qudaCloverInvert: requesting zero residual\n");
    exit(1);
  }
  
  PersistentData pd;
  static const QudaVerbosity verbosity = pd.getVerbosity();

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision_sloppy = (inv_args.mixed_precision) ? QUDA_SINGLE_PRECISION : device_precision;


  QudaGaugeParam gaugeParam   = newQudaGaugeParam();
  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = (target_residual != 0) ? QUDA_L2_RELATIVE_RESIDUAL : QUDA_HEAVY_QUARK_RESIDUAL;
  invertParam.tol = (target_residual != 0) ? target_residual : target_fermilab_residual;

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = Z[dir];

  gaugeParam.anisotropy               = 1.0;
  gaugeParam.type                     = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order              = QUDA_MILC_GAUGE_ORDER; 

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial 
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;	

  if(trivial_phase){
    gaugeParam.t_boundary               = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_12; 
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_12;
  }else{
    gaugeParam.t_boundary               = QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_NO;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_NO;
  }
  
  gaugeParam.cpu_prec                 = host_precision;
  gaugeParam.cuda_prec                = device_precision;
  gaugeParam.cuda_prec_sloppy         = device_precision_sloppy;
  gaugeParam.cuda_prec_precondition   = device_precision_sloppy;
  gaugeParam.gauge_fix                = QUDA_GAUGE_FIXED_NO;
  gaugeParam.ga_pad 		      = 0;

  invertParam.dslash_type             = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa                   = kappa;

  // solution types
  invertParam.solution_type      = QUDA_MAT_SOLUTION;
  invertParam.solve_type         = QUDA_DIRECT_PC_SOLVE;
  invertParam.inv_type           = QUDA_BICGSTAB_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_ODD_ODD;


  invertParam.dagger             = QUDA_DAG_NO;
  invertParam.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov	 			 = 30; // unnecessary
  invertParam.reliable_delta     = 1e-1; 
  invertParam.maxiter            = inv_args.max_iter;

#ifdef MULTI_GPU
  int x_face_size = gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int y_face_size = gaugeParam.X[0]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int z_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[3]/2;
  int t_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif // MULTI_GPU
  invertParam.cuda_prec_precondition             = device_precision_sloppy;
  invertParam.verbosity_precondition        = QUDA_SILENT;
  invertParam.cpu_prec 		            = host_precision;
  invertParam.cuda_prec		            = device_precision;
  invertParam.cuda_prec_sloppy	            = device_precision_sloppy;
  invertParam.preserve_source               = QUDA_PRESERVE_SOURCE_NO;
  invertParam.gamma_basis 	            = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order		    = QUDA_DIRAC_ORDER;
  invertParam.tune	            	    = QUDA_TUNE_YES;
  invertParam.sp_pad		            = 0;
  invertParam.cl_pad 		            = 0;
  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH){
    invertParam.clover_cpu_prec               = host_precision;
    invertParam.clover_cuda_prec              = device_precision;
    invertParam.clover_cuda_prec_sloppy       = device_precision_sloppy;
    invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
    invertParam.clover_order		      = QUDA_PACKED_CLOVER_ORDER;
  }
  invertParam.verbosity			   = verbosity;

  
  const size_t gSize = getRealSize(gaugeParam.cpu_prec);
  int volume = 1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
    
  loadGaugeQuda(const_cast<void*>(link), &gaugeParam);

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover,cloverInverse, &invertParam);

  invertQuda(solution, source, &invertParam); 
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;
  
  freeGaugeQuda();

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) freeCloverQuda();
  
  return;
} // qudaCloverInvert


void qudaCloverMultishiftInvert(int external_precision, 
				int quda_precision,
				int num_offsets,
				double* const offset,
				double kappa,
				QudaInvertArgs_t inv_args,
				const double* target_residual_offset,
				const void* milc_link,
				void* milc_clover, 
				void* milc_clover_inv,
				void* source,
				void** solutionArray,
				double* const final_residual, 
				int* num_iters)
{
  using namespace milc_interface;

  for(int i=0; i<num_offsets; ++i){
    if(target_residual_offset[i] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  }

  PersistentData pd;
  static const QudaVerbosity verbosity = pd.getVerbosity();

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision_sloppy = (inv_args.mixed_precision) ? QUDA_SINGLE_PRECISION : device_precision;


  QudaGaugeParam gaugeParam   = newQudaGaugeParam();
  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = QUDA_L2_RELATIVE_RESIDUAL;

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = Z[dir];

  gaugeParam.anisotropy               = 1.0;
  gaugeParam.type                     = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order              = QUDA_MILC_GAUGE_ORDER; 

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial 
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;	

  if(trivial_phase){
    gaugeParam.t_boundary               = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_12; 
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_12;
  }else{
    gaugeParam.t_boundary               = QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_NO;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_NO;
  }
  
  gaugeParam.cpu_prec                 = host_precision;
  gaugeParam.cuda_prec                = device_precision;
  gaugeParam.cuda_prec_sloppy         = device_precision_sloppy;
  gaugeParam.cuda_prec_precondition   = device_precision_sloppy;
  gaugeParam.gauge_fix                = QUDA_GAUGE_FIXED_NO;
  gaugeParam.ga_pad 		      = 0;

  invertParam.dslash_type             = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa                   = kappa;

  // solution types
  invertParam.solution_type      = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam.solve_type         = QUDA_NORMOP_PC_SOLVE;
  invertParam.inv_type           = QUDA_CG_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;


  invertParam.dagger             = QUDA_DAG_NO;
  invertParam.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov	 = 30; // unnecessary
  invertParam.reliable_delta     = 1e-1; 
  invertParam.maxiter            = inv_args.max_iter;

#ifdef MULTI_GPU
  int x_face_size = gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int y_face_size = gaugeParam.X[0]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int z_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[3]/2;
  int t_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif // MULTI_GPU
  invertParam.cuda_prec_precondition             = device_precision_sloppy;
  invertParam.verbosity_precondition        = QUDA_SILENT;
  invertParam.cpu_prec 		            = host_precision;
  invertParam.cuda_prec		            = device_precision;
  invertParam.cuda_prec_sloppy	            = device_precision_sloppy;
  invertParam.preserve_source               = QUDA_PRESERVE_SOURCE_NO;
  invertParam.gamma_basis 	            = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order		    = QUDA_DIRAC_ORDER;
  invertParam.tune	            	    = QUDA_TUNE_YES;
  invertParam.sp_pad		            = 0;
  invertParam.cl_pad 		            = 0;
  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH){
    invertParam.clover_cpu_prec               = host_precision;
    invertParam.clover_cuda_prec              = device_precision;
    invertParam.clover_cuda_prec_sloppy       = device_precision_sloppy;
    invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
    invertParam.clover_order		      = QUDA_PACKED_CLOVER_ORDER;
  }
  invertParam.verbosity			   = verbosity;

  invertParam.num_offset = num_offsets;
  for(int i=0; i<num_offsets; ++i){
    invertParam.offset[i] = offset[i];
    invertParam.tol_offset[i] = target_residual_offset[i];
  }
  invertParam.tol = target_residual_offset[0];

  int volume = 1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
    
  loadGaugeQuda(const_cast<void*>(milc_link), &gaugeParam);

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
    loadCloverQuda(milc_clover,milc_clover_inv, &invertParam);

  invertMultiShiftQuda(solutionArray, source, &invertParam); 
  
  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i){
    final_residual[i] = invertParam.true_res_offset[i];
  } // end loop over number of offsets


  freeGaugeQuda();

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) freeCloverQuda();
  
  return;
} // qudaCloverMultishiftInvert

