#ifndef _KERNEL_PARAMS_H
#define _KERNEL_PARAMS_H

#include<quda_internal.h>
#include <color_spinor_field.h>

namespace quda {

  struct KernelParams
  {
    // Lattice Dimensions
    unsigned int X1, X2, X3, X4;
    unsigned int X1h, X2h;
    unsigned int X1_3, X2_3, X3_3, X4_3;
    unsigned int X2X1, X3X1, X3X2;

    unsigned int X3X2X1, X4X3X1, X4X3X2, X4X2X1;
    unsigned int X3X2X1h, X4X3X1h, X4X3X2h, X4X2X1h;

    unsigned int X2X1_3, X3X2X1_3;

    unsigned int X1m1, X2m1, X3m1, X4m1;
    unsigned int X1m3, X2m3, X3m3, X4m3;

    unsigned int X2X1mX1, X3X2X1mX2X1;
    unsigned int X4X3X2X1mX3X2X1;
    unsigned int X4X3X2X1hmX3X2X1h;

    unsigned int X2X1m3X1, X3X2X1m3X2X1;
    unsigned int X4X3X2X1hm3X3X2X1h; 

    // Volumes
    unsigned int Vh;
    unsigned int Vsh;
    unsigned int Vh_2d_max;

    // Ghost zones
    unsigned int ghostFace[QUDA_MAX_DIM];

    // Params set in initSpinorParams
    unsigned int sp_stride;
    unsigned int Ls; // for domain wall

    // Params set in initGaugeParams
    unsigned int ga_stride;
    unsigned int gf; // gauge fixed?
    double anisotropy, t_bc, coeff;
    float anisotropy_f, t_bc_f, coeff_f;

    unsigned int fatlinkStride;
    unsigned int longlinkStride;
    float fatlinkMax;

  };

  class LatticeField;
  class cudaGaugeField;
  //class cudaColorSpinorField;

  void initDimensionParams(KernelParams* params, const LatticeField &lat);
  void initLatticeParams(KernelParams* params, const LatticeField &lat);
  void initGaugeParams(KernelParams* params, const cudaGaugeField& gauge);
  void initSpinorParams(KernelParams* params, const cudaColorSpinorField& spinor);
  void initStaggeredParams(KernelParams* params, const cudaGaugeField &fatGauge, const cudaGaugeField &longGauge);

} // namespace

#endif // _KERNEL_PARAMS_H
