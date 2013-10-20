#include "include/milc_utilities.h"
#include <quda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "external_headers/quda_milc_interface.h"
#include <quda_internal.h>


//extern TimeProfile profileAsqtadForceInterface;

void qudaInit(QudaInitArgs_t input)
{
  using namespace milc_interface;

  static bool initialized = false;
  if(initialized) return;

  // set verbosity here
  setVerbosityQuda(input.verbosity, "", stdout);
  PersistentData persistent_data;
  persistent_data.setVerbosity(input.verbosity);

  qudaSetLayout(input.layout);

  initialized = true;
}


/**
 * Implements a lexicographical mapping of node coordinates to ranks,
 * with t varying fastest.
 */
static int rankFromCoords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = dims[i] * rank + coords[i];
  }
  return rank;
}


void qudaSetLayout(QudaLayout_t input)
{
  using namespace milc_interface;

  int local_dim[4];
  for(int dir=0; dir<4; ++dir){ local_dim[dir] = input.latsize[dir]; }
#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir){ local_dim[dir] /= input.machsize[dir]; }
#endif
  for(int dir=0; dir<4; ++dir){  
    if(local_dim[dir]%2 != 0){
      printf("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

  Layout layout;
  layout.setLocalDim(local_dim);

#ifdef MULTI_GPU
  layout.setGridDim(input.machsize);
  const int* grid_size = layout.getGridDim();
  initCommsGridQuda(4, grid_size, rankFromCoords, (void *)(grid_size));

  static int device = -1;
#else
  static int device = input.device;
#endif

  initQuda(device);
}


void qudaFinalize()
{
  endQuda();

  extern quda::TimeProfile profileAsqtadForceInterface;

  if(getVerbosity() >= QUDA_SUMMARIZE){
#if BUILD_HISQ_FORCE
    profileAsqtadForceInterface.Print();
#endif
  }

  return;
}


// initialization routines for hisq
#include <dslash_quda.h> 
#include <fat_force_quda.h>
#include <hisq_force_quda.h>
#include <gauge_field.h>

#ifdef GPU_UNITARIZE
#include <hisq_links_quda.h>
#endif

void qudaHisqParamsInit(QudaHisqParams_t params)
{

  static bool initialized = false;

  if(initialized) return;


  const bool reunit_allow_svd = (params.reunit_allow_svd) ? true : false;
  const bool reunit_svd_only  = (params.reunit_svd_only) ? true : false;


  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;

#ifdef GPU_HISQ_FORCE
  quda::fermion_force::setUnitarizeForceConstants(unitarize_eps,
                                		     params.force_filter,
                                		     max_error,
                                		     reunit_allow_svd,
                                		     reunit_svd_only,
                                		     params.reunit_svd_rel_error,
                                		     params.reunit_svd_abs_error);
#endif

#ifdef GPU_UNITARIZE
  quda::setUnitarizeLinksConstants(unitarize_eps,
				   max_error,
				   reunit_allow_svd,
				   reunit_svd_only,
				   params.reunit_svd_rel_error,
				   params.reunit_svd_abs_error);
#endif // UNITARIZE_GPU				   

  initialized = true;

  return;
}


