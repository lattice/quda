
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <limits>
#include <cmath>

#include <iostream>
#include <Eigen/Dense>
#include <cuda_profiler_api.h>

namespace quda{


// define this to use multi-functions, otherwise it'll
// do loops over dot products.
// this is more here for development convenience.
#ifdef BLOCKSOLVER
  #define BLOCKSOLVER_MULTIFUNCTIONS
  #define BLOCKSOLVE_DSLASH5D
#endif
//#define BLOCKSOLVER_VERBOSE

// Run algorithm with Q in high precision.
#define BLOCKSOLVER_PRECISE_Q

// Mathias' testing area for Pollock-Ribiere or however it's spelled.
//#define BLOCKSOLVER_ALTERNATIVE_BETA

// Explicitly reorthogonalize Q^\dagger P on reliable update.
//#define BLOCKSOLVER_EXPLICIT_QP_ORTHO
// Explicitly make pAp Hermitian every time it is computed.
//#define BLOCKSOLVER_EXPLICIT_PAP_HERMITIAN

// If defined, trigger a reliable updated whenever _any_ residual
// becomes small enough. Otherwise, trigger a reliable update
// when _all_ residuals become small enough (which is consistent with
// the algorithm stopping condition). Ultimately, this is using a
// min function versus a max function, so it's not a hard swap.
// #define BLOCKSOLVER_RELIABLE_POLICY_MIN

#ifdef BLOCKSOLVER_NVTX
#include "nvToolsExt.h"
static const uint32_t cg_nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static constexpr int cg_nvtx_num_colors = sizeof(cg_nvtx_colors)/sizeof(uint32_t);
#define PUSH_RANGE(name,cid) { \
    static int color_id = cid; \
    color_id = color_id%cg_nvtx_num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = cg_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.category = cid;\
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


  using Eigen::Matrix;
  using Eigen::Map;
  using Eigen::RowMajor;
  using Eigen::Dynamic; 
  using Eigen::MatrixXcd;

// Matrix printing functions

template<typename Matrix>
inline void printmat(const char* label, const Matrix& mat)
{
#ifdef BLOCKSOLVER_VERBOSE
  printfQuda("\n%s\n", label);
  std::cout << mat;
  printfQuda("\n");
#endif
}


int *malloc_int(const int n) {
  auto *ptr = static_cast<int*>(safe_malloc(n*sizeof(int)));
  return ptr; 
}

void free_int(void *ptr) {
  host_free(ptr);
}

// compute the fine-to-coarse site map
void EKS_PCG::CreateSourceBlock(ColorSpinorField& b) {

  int x [QUDA_MAX_DIM];


  //index_map = static_cast<int*>(safe_malloc(b.Volume()*sizeof(int)));//store it for the next RHS?
  //index_map = std::move(std::shared_ptr<int>(new int[b.Volume()], std::default_delete<int[]>()));
  //index_map = std::move(std::shared_ptr<int>(malloc_int(b.Volume()), free_int));
  if(index_map == nullptr) index_map = std::move(std::shared_ptr<int>(static_cast<int*>(safe_malloc(b.Volume()*sizeof(int))), [](void *ptr) {host_free(ptr);} ));

  // compute the coarse grid point for every site (assuming parity ordering currently)
  for (int i = 0; i < b.Volume(); i++) {
    // compute the lattice-site index for this offset index
    b.LatticeIndex(x, i);
      
    //printfQuda("fine idx %d = fine (%d,%d,%d,%d), ", i, x[0], x[1], x[2], x[3]);
    // compute the corresponding coarse-grid index given the block size
    for (int d = 0; d < b.Ndim(); d++) x[d] /= latt_bs[d];

    // compute the coarse-offset index and store in fine_to_coarse
    int k;
    int parity  = 0;
    int lattbs0 = latt_bs[0]; 

    if (b.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      for (int d = 0; d < b.Ndim(); d++) parity += x[d];
      parity = parity & 1;
      latt_bs[0]  /= 2;
    }

    k = parity;
    for (int d = b.Ndim()-1; d >= 0; d--) k = latt_bs[d]*k + x[d];

    index_map.get()[i] = k;

    latt_bs[0] = lattbs0;

    //printfQuda("block after (%d,%d,%d,%d), block idx %d\n", x[0], x[1], x[2], x[3], k);
  }

  return;
}


  template <int nsrc>
  void EKS_PCG::solve_n(ColorSpinorField& x, ColorSpinorField& b) {

    profile.TPSTART(QUDA_PROFILE_INIT);

     ColorSpinorParam csParam(x);

    csParam.is_composite  = true;
    csParam.composite_dim = nsrc;
    csParam.nDim = 5;
    csParam.x[4] = 1;


    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }


  EKS_PCG::EKS_PCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    BlockSolver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), yp(nullptr), rp(nullptr), App(nullptr), tmpp(nullptr),
    B(nullptr), latt_bs{{1,1,1,1,1}}, index_map(nullptr), init(false) {
#ifndef BLOCKSOLVER
    warningQuda("QUDA_BLOCKSOLVER not built with MULTI-BLAS / Dslash options Performance will be slow.");
#endif 
  }

  EKS_PCG::~EKS_PCG() {

    if ( init ) {
      for (auto pi : p) delete pi;
      init = false;
    }
  }

  void EKS_PCG::operator()(ColorSpinorField& x, ColorSpinorField& b) {

    if (param.num_src > QUDA_MAX_BLOCK_SRC) //that looks missleading but okay for the moment
      errorQuda("Requested number of right-hand sides %d exceeds max %d\n", param.num_src, QUDA_MAX_BLOCK_SRC);

    switch (param.num_src) {
      case  1: solve_n< 1>(x, b); break;
      case  2: solve_n< 2>(x, b); break;
      case  4: solve_n< 4>(x, b); break;
      case  8: solve_n< 8>(x, b); break;
      case 12: solve_n<12>(x, b); break;
      case 16: solve_n<16>(x, b); break;
      case 24: solve_n<24>(x, b); break;
      case 32: solve_n<32>(x, b); break;
      case 48: solve_n<48>(x, b); break; 
      case 64: solve_n<64>(x, b); break;
      default:
      errorQuda("EKSCG with dimension %d not supported", param.num_src);
    }

    return; 
  }

}//quda namespace


