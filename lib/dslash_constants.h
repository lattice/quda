#ifndef DSLASH_CONSTANTS_H
#define DSLASH_CONSTANTS_H


#include <dimension_constants.h>


#define MAX(a,b) ((a)>(b) ? (a):(b))



typedef struct fat_force_stride_s {
  int fatlinkStride;
  int sitelinkStride;
  int stapleStride;
  int mom_ga_stride;
  int path_max_length;
  int color_matrix_stride;
} fat_force_const_t;

struct StaggeredConstants {
  int fatlinkStride;
  int longlinkStride;
  int fatlinkMax;
};




__constant__ StaggeredConstants staggeredConstants;
__constant__ StaggeredConstants ddStaggeredConstants; // for domain decomposition

__constant__ int ghostFace[QUDA_MAX_DIM];
  
__constant__ int sp_stride;
__constant__ int ga_stride;
__constant__ int cl_stride;

__constant__ int fatlinkStride;
__constant__ int longlinkStride;
__constant__ float fatlinkMax;

__constant__ int gauge_fixed;


// single precision constants
__constant__ float anisotropy_f;
__constant__ float coeff_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;
__constant__ double coeff;

__constant__ float2 An2;
__constant__ float2 TB2;
__constant__ float2 No2;

// Are we processor 0 in time?
__constant__ bool Pt0;
// Are we processor Nt-1 in time?
__constant__ bool PtNm1;

// factor of 2 (or 1) for T-dimensional spin projection
__constant__ double tProjScale;
__constant__ float tProjScale_f;


__constant__ fat_force_const_t fl; //fatlink
__constant__ fat_force_const_t gf; //gauge force
__constant__ fat_force_const_t hf; //hisq force


void initLatticeConstants(const LatticeField &lat)
{
  checkCudaError();
  // Set the subdomain dimensions
  initDimensionConstants(lat);

  int L1 = lat.X()[0];
  int L2 = lat.X()[1];
  int L3 = lat.X()[2];
  int L4 = lat.X()[3];
  int ghostFace_h[4];
  ghostFace_h[0] = L2*L3*L4/2;
  ghostFace_h[1] = L1*L3*L4/2;
  ghostFace_h[2] = L1*L2*L4/2;
  ghostFace_h[3] = L1*L2*L3/2;
  constants.ghostFace[0] = ghostFace_h[0];
  constants.ghostFace[1] = ghostFace_h[1];
  constants.ghostFace[2] = ghostFace_h[2];
  constants.ghostFace[3] = ghostFace_h[3];
  cudaMemcpyToSymbol(ghostFace, ghostFace_h, 4*sizeof(int));  

#ifdef MULTI_GPU
  bool first_node_in_t = (commCoords(3) == 0);
  bool last_node_in_t = (commCoords(3) == commDim(3)-1);
#else
  bool first_node_in_t = true;
  bool last_node_in_t = true;
#endif

  cudaMemcpyToSymbol(Pt0, &(first_node_in_t), sizeof(bool)); 
  cudaMemcpyToSymbol(PtNm1, &(last_node_in_t), sizeof(bool)); 

  checkCudaError();
}


void initGaugeConstants(const cudaGaugeField &gauge) 
{
  int ga_stride_h = gauge.Stride();
  cudaMemcpyToSymbol(ga_stride, &ga_stride_h, sizeof(int));  

  int gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES);
  cudaMemcpyToSymbol(gauge_fixed, &(gf), sizeof(int));

  double anisotropy_ = gauge.Anisotropy();
  cudaMemcpyToSymbol(anisotropy, &(anisotropy_), sizeof(double));

  double t_bc = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol(t_boundary, &(t_bc), sizeof(double));

  double coeff_h = -24.0*gauge.Tadpole()*gauge.Tadpole();
  cudaMemcpyToSymbol(coeff, &(coeff_h), sizeof(double));

  float anisotropy_fh = gauge.Anisotropy();
  cudaMemcpyToSymbol(anisotropy_f, &(anisotropy_fh), sizeof(float));

  float t_bc_f = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol(t_boundary_f, &(t_bc_f), sizeof(float));

  float coeff_fh = -24.0*gauge.Tadpole()*gauge.Tadpole();
  cudaMemcpyToSymbol(coeff_f, &(coeff_fh), sizeof(float));

  // constants used by the READ_GAUGE() macros in read_gauge.h
  float2 An2_h = make_float2(gauge.Anisotropy(), 1.0 / (gauge.Anisotropy()*gauge.Anisotropy()));
  cudaMemcpyToSymbol(An2, &(An2_h), sizeof(float2));
  float2 TB2_h = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol(TB2, &(TB2_h), sizeof(float2));
  float2 No2_h = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol(No2, &(No2_h), sizeof(float2));

  checkCudaError();
}


/**
 * This routine gets called often, so be sure not to set any constants unnecessarily
 * or introduce synchronization (e.g., via checkCudaError()).
 */
void initSpinorConstants(const cudaColorSpinorField &spinor)
{
  static int last_sp_stride = -1;
  static int last_Ls = -1;

  int sp_stride_h = spinor.Stride();
  if (sp_stride_h != last_sp_stride) {
    cudaMemcpyToSymbol(sp_stride, &sp_stride_h, sizeof(int));
    constants.sp_stride = sp_stride_h;
    cudaMemcpyToSymbol(cudaConstants, &constants, sizeof(LatticeConstants));
    checkCudaError();
    last_sp_stride = sp_stride_h;
  }
  
  // for domain wall:
  if (spinor.Ndim() == 5) {
    int Ls_h = spinor.X(4);
    if (Ls_h != last_Ls) {
      cudaMemcpyToSymbol(Ls, &Ls_h, sizeof(int));  
      dslashConstants.Ls = Ls_h; // needed by tuneLaunch()
      checkCudaError();
      last_Ls = Ls_h;
    }
  }
}


void initDslashConstants()
{
  float pi_f_h = M_PI;
  cudaMemcpyToSymbol(pi_f, &pi_f_h, sizeof(float));

  // temporary additions (?) for checking Ron's T-packing kernel with old multi-gpu kernel

  double tProjScale_h = (kernelPackT ? 1.0 : 2.0);
  cudaMemcpyToSymbol(tProjScale, &tProjScale_h, sizeof(double));

  float tProjScale_fh = (float)tProjScale_h;
  cudaMemcpyToSymbol(tProjScale_f, &tProjScale_fh, sizeof(float));

  checkCudaError();
}


void initCloverConstants (const cudaCloverField &clover)
{
  int cl_stride_h = clover.Stride();
  cudaMemcpyToSymbol(cl_stride, &cl_stride_h, sizeof(int));  

  checkCudaError();
}



void initStaggeredConstants(const cudaGaugeField &fatGauge, const cudaGaugeField &longGauge)
{
  StaggeredConstants hostConstants;
  hostConstants.fatlinkStride = fatGauge.Stride();
  hostConstants.longlinkStride = longGauge.Stride();
  hostConstants.fatlinkMax = fatGauge.LinkMax();

  cudaMemcpyToSymbol(staggeredConstants, &hostConstants, sizeof(StaggeredConstants));

  checkCudaError();
}

void initDDStaggeredConstants(const cudaGaugeField &fatGauge, const cudaGaugeField &longGauge)
{
  StaggeredConstants hostConstants;
  hostConstants.fatlinkStride = fatGauge.Stride();
  hostConstants.longlinkStride = longGauge.Stride();
  hostConstants.fatlinkMax = fatGauge.LinkMax();

  cudaMemcpyToSymbol(ddStaggeredConstants, &hostConstants, sizeof(StaggeredConstants));

  checkCudaError();
}





#endif // DSLASH_CONSTANTS_H


