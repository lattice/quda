#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <test_util.h>
#include "../tests/blas_reference.h"
#include "../tests/staggered_dslash_reference.h"
#include <quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <sys/time.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "external_headers/quda_milc_interface.h"
#include "include/milc_timer.h"
#include "include/milc_inverter_utilities.h"

#ifdef MULTI_GPU
#include <face_quda.h>
#include <comm_quda.h> // for comm_coord()
#endif


#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/milc_utilities.h"

namespace milc_interface {

  static bool invalidate_quda_gauge = true;

  void invalidateGaugeQuda() { 
    freeGaugeQuda();
    invalidate_quda_gauge = true; 
  }

  // Need to figure out if this is needed anymore
  static void
    setDimConstants(const int X[4])
    {
      V = 1;
      for (int d=0; d< 4; d++) {
        V *= X[d];
        Z[d] = X[d];
      }
      Vh = V/2;

      Vs_x = X[1]*X[2]*X[3];
      Vs_y = X[0]*X[2]*X[3];
      Vs_z = X[0]*X[1]*X[3];
      Vs_t = X[0]*X[1]*X[2];


      Vsh_x = Vs_x/2;
      Vsh_y = Vs_y/2;
      Vsh_z = Vs_z/2;
      Vsh_t = Vs_t/2;

      return;
    }


  static 
    bool doEvenOddExchange(const int local_dim[4], const int logical_coord[4])
    {
      bool exchange = 0;
      for(int dir=0; dir<4; ++dir){
        if(local_dim[dir] % 2 == 1 && logical_coord[dir] % 2 == 1){
          exchange = 1-exchange;
        }
      }
      return exchange ? true : false;
    }



} // namespace milc_interface


void qudaMultishiftInvert(int external_precision, 
    int quda_precision,
    int num_offsets,
    double* const offset,
    QudaInvertArgs_t inv_args,
    const double target_residual[], 
    const double target_fermilab_residual[],
    const void* const fatlink,
    const void* const longlink,
    const double tadpole,
    void* source,
    void** solutionArray,
    double* const final_residual,
    double* const final_fermilab_residual,
    int *num_iters)
{

  using namespace milc_interface;
  for(int i=0; i<num_offsets; ++i){
    if(target_residual[i] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  }

  milc_interface::Timer timer("qudaMultishiftInvert"); 
#ifndef TIME_INTERFACE
  timer.mute();
#endif

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));


  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;
  QudaPrecision device_precision_sloppy = (use_mixed_precision) ? QUDA_SINGLE_PRECISION :   device_precision;
  QudaPrecision device_precision_precondition = device_precision_sloppy;



  PersistentData pd;
  static const QudaVerbosity verbosity = pd.getVerbosity();
  //static const QudaVerbosity verbosity = QUDA_VERBOSE;

  if(verbosity >= QUDA_VERBOSE){ 
    if(quda_precision == 2){
      printfQuda("Using %s double-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else if(quda_precision == 1){
      printfQuda("Using %s single-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else{
      errorQuda("Unrecognised precision\n");
      exit(1);
    }
  }


  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition, tadpole, &gaugeParam);

  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = (target_fermilab_residual[0] != 0) ? QUDA_HEAVY_QUARK_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;

  const double ignore_mass = 1.0;
#ifdef MULTI_GPU
  int logical_coord[4];
  for(int dir=0; dir<4; ++dir){
    logical_coord[dir] = comm_coord(dir); // used MPI
  }
  const bool even_odd_exchange = false;	
#else // serial code
  const bool even_odd_exchange = false;	
#endif

  QudaParity local_parity = inv_args.evenodd;
  {
    const double reliable_delta = 1e-1;

    setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition,
        num_offsets, offset, target_residual, target_fermilab_residual, 
        inv_args.max_iter, reliable_delta, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);

  }  

  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);

  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  if (milc_interface::invalidate_quda_gauge) {
#ifdef MULTI_GPU
    const int fat_pad  = getFatLinkPadding(local_dim);
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad;  // don't know if this is correct
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 
    
    const int long_pad = 3*fat_pad;
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; // don't know if this will work
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam);
    
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
    gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#endif
    milc_interface::invalidate_quda_gauge = false;
  }

  int volume=1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];

  void** sln_pointer = (void**)malloc(num_offsets*sizeof(void*));
  int quark_offset = getColorVectorOffset(local_parity, false, volume);
  void* src_pointer;

  if(host_precision == QUDA_SINGLE_PRECISION){
    src_pointer = (float*)source + quark_offset;
    for(int i=0; i<num_offsets; ++i) sln_pointer[i] = (float*)solutionArray[i] + quark_offset;
  }else{
    src_pointer = (double*)source + quark_offset;
    for(int i=0; i<num_offsets; ++i) sln_pointer[i] = (double*)solutionArray[i] + quark_offset;
  }

  timer.check("Setup and data load");
  invertMultiShiftQuda(sln_pointer, src_pointer, &invertParam);
  timer.check("invertMultiShiftQuda");
  timer.check();

  free(sln_pointer); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i){
    final_residual[i] = invertParam.true_res_offset[i];
    final_fermilab_residual[i] = invertParam.true_res_hq_offset[i];
  } // end loop over number of offsets

  //freeGaugeQuda(); // free up the gauge-field objects allocated
  return;
} // qudaMultiShiftInvert




void qudaInvert(int external_precision,
    int quda_precision,
    double mass,
    QudaInvertArgs_t inv_args,
    double target_residual, 
    double target_fermilab_residual,
    const void* const fatlink,
    const void* const longlink,
    const double tadpole,
    void* source,
    void* solution,
    double* const final_residual,
    double* const final_fermilab_residual,
    int* num_iters)
{

  using namespace milc_interface;
  if(target_fermilab_residual && target_residual){
    errorQuda("qudaInvert: conflicting residuals requested\n");
    exit(1);
  }else if(target_fermilab_residual == 0 && target_residual == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }


  milc_interface::Timer timer("qudaInvert");
#ifndef TIME_INTERFACE
  timer.mute();
#endif

  Layout layout;

  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));

  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;
  PersistentData pd;
  //static const QudaVerbosity verbosity = pd.getVerbosity();
  static const QudaVerbosity verbosity = QUDA_SUMMARIZE;


  if(verbosity >= QUDA_VERBOSE){
    if(use_mixed_precision){
      if(quda_precision == 2){
        printfQuda("Using mixed double-precision CG inverter\n");
      }else if(quda_precision == 2){
        printfQuda("Using mixed single-precision CG inverter\n");
      }
    }else if(quda_precision == 2){
      printfQuda("Using double-precision CG inverter\n");
    }else if(quda_precision == 1){
      printfQuda("Using single-precision CG inverter\n");
    }else{
      errorQuda("Unrecognised precision\n");
      exit(1);
    }
  }

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = (use_mixed_precision) ? QUDA_SINGLE_PRECISION : device_precision;
  QudaPrecision device_precision_precondition = device_precision_sloppy;

  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition, tadpole, &gaugeParam);

  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = (target_residual != 0) ? QUDA_L2_RELATIVE_RESIDUAL : QUDA_HEAVY_QUARK_RESIDUAL;
  QudaParity local_parity;

#ifdef MULTI_GPU 
  int logical_coord[4];
  for(int dir=0; dir<4; ++dir) logical_coord[dir] = comm_coord(dir);
  const bool even_odd_exchange = doEvenOddExchange(local_dim, logical_coord);
#else // single gpu 
  const bool even_odd_exchange = false;
#endif

  if(even_odd_exchange){
    local_parity = (inv_args.evenodd==QUDA_EVEN_PARITY) ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  }else{
    local_parity = inv_args.evenodd;
  }

  double& target_res = (invertParam.residual_type == QUDA_L2_RELATIVE_RESIDUAL) ? target_residual : target_fermilab_residual;

  setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition,
      mass, target_res, inv_args.max_iter, 1e-1, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);


  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);



  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 

  const int fat_pad  = getFatLinkPadding(local_dim);
  const int long_pad = 3*fat_pad;

  // No mixed precision here, it seems
  if (milc_interface::invalidate_quda_gauge) {
#ifdef MULTI_GPU
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad; 
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 
    
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; 
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam);
    
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
    gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#endif
    milc_interface::invalidate_quda_gauge = false;
  }

  int volume=1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
  int quark_offset = getColorVectorOffset(local_parity, false, volume);
  void* src_pointer;
  void* sln_pointer; 

  if(host_precision == QUDA_SINGLE_PRECISION){
    src_pointer = (float*)source + quark_offset;
    sln_pointer = (float*)solution + quark_offset;
  }else{
    src_pointer = (double*)source + quark_offset;
    sln_pointer = (double*)solution + quark_offset;
  }


  timer.check("Set up and data load");
  invertQuda(sln_pointer, src_pointer, &invertParam); 
  timer.check("invertQuda");


  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  //freeGaugeQuda(); // free up the gauge-field objects allocated
  // in loadGaugeQuda        

  return;
} // qudaInvert

#undef MAX
