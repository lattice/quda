#pragma once

#include <quda.h>
#include <quda_fortran.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <unitarization_links.h>
#include <algorithm>
#include <staggered_oprod.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>
#include <gauge_update_quda.h>
#include <gauge_tools.h>
#include <contract_quda.h>
#include <momentum.h>
#ifdef GPU_GAUGE_FORCE
#include <gauge_force_quda.h>
#endif
#include <multigrid.h>
#include <deflation.h>
#include <random_quda.h>
#include <mpi_comm_handle.h>
#include <blas_cublas.h>
#include <check_params.h>

#ifdef NUMA_NVML
#include <numa_affinity.h>
#endif

#ifdef QUDA_NVML
#include <nvml.h>
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>
//for MAGMA lib:
#include <blas_magma.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

#define MAX_GPU_NUM_PER_NODE 16

using namespace quda;

extern cudaGaugeField *gaugePrecise;
extern cudaGaugeField *gaugeSloppy;
extern cudaGaugeField *gaugePrecondition;
extern cudaGaugeField *gaugeRefinement;
extern cudaGaugeField *gaugeExtended;

extern cudaGaugeField *gaugeFatPrecise;
extern cudaGaugeField *gaugeFatSloppy;
extern cudaGaugeField *gaugeFatPrecondition;
extern cudaGaugeField *gaugeFatRefinement;
extern cudaGaugeField *gaugeFatExtended;

extern cudaGaugeField *gaugeLongExtended;
extern cudaGaugeField *gaugeLongPrecise;
extern cudaGaugeField *gaugeLongSloppy;
extern cudaGaugeField *gaugeLongPrecondition;
extern cudaGaugeField *gaugeLongRefinement;

extern cudaGaugeField *gaugeSmeared;

extern cudaCloverField *cloverPrecise;
extern cudaCloverField *cloverSloppy;
extern cudaCloverField *cloverPrecondition;
extern cudaCloverField *cloverRefinement;

extern cudaGaugeField *momResident;
extern cudaGaugeField *extendedGaugeResident;

// vector of spinors used for forecasting solutions in HMC
#define QUDA_MAX_CHRONO 12

//!< Profiler for initQuda
extern TimeProfile profileInit;

//!< Profile for loadGaugeQuda / saveGaugeQuda
extern TimeProfile profileGauge;

//!< Profile for loadCloverQuda
extern TimeProfile profileClover;

//!< Profiler for dslashQuda
extern TimeProfile profileDslash;

//!< Profiler for invertQuda
extern TimeProfile profileInvert;

//!< Profiler for invertMultiShiftQuda
extern TimeProfile profileMulti;

//!< Profiler for eigensolveQuda
extern TimeProfile profileEigensolve;

//!< Profiler for computeFatLinkQuda
extern TimeProfile profileFatLink;

//!< Profiler for computeGaugeForceQuda
extern TimeProfile profileGaugeForce;

//!<Profiler for updateGaugeFieldQuda
extern TimeProfile profileGaugeUpdate;

//!<Profiler for createExtendedGaugeField
extern TimeProfile profileExtendedGauge;

//!<Profiler for computeCloverForceQuda
extern TimeProfile profileCloverForce;

//!<Profiler for computeStaggeredForceQuda
extern TimeProfile profileStaggeredForce;

//!<Profiler for computeHISQForceQuda
extern TimeProfile profileHISQForce;

//!<Profiler for plaqQuda
extern TimeProfile profilePlaq;

//!< Profiler for wuppertalQuda
extern TimeProfile profileWuppertal;

//!<Profiler for gaussQuda
extern TimeProfile profileGauss;

//!< Profiler for gaugeObservableQuda
extern TimeProfile profileGaugeObs;

//!< Profiler for APEQuda
extern TimeProfile profileAPE;

//!< Profiler for STOUTQuda
extern TimeProfile profileSTOUT;

//!< Profiler for OvrImpSTOUTQuda
extern TimeProfile profileOvrImpSTOUT;

//!< Profiler for wFlowQuda
extern TimeProfile profileWFlow;

//!< Profiler for projectSU3Quda
extern TimeProfile profileProject;

//!< Profiler for staggeredPhaseQuda
extern TimeProfile profilePhase;

//!< Profiler for contractions
extern TimeProfile profileContract;

//!< Profiler for covariant derivative
extern TimeProfile profileCovDev;

//!< Profiler for momentum action
extern TimeProfile profileMomAction;

//!< Profiler for endQuda
extern TimeProfile profileEnd;

//!< Profiler for endQuda
extern TimeProfile profileCuBLAS;

//!< Profiler for GaugeFixing
extern TimeProfile GaugeFixFFTQuda;
extern TimeProfile GaugeFixOVRQuda;

//!< Profiler for toal time spend between init and end
extern TimeProfile profileInit2End;


