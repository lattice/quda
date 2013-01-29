#include <lattice_field.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <kernel_params.h>

#define MAX(a,b) ((a)>(b) ? (a):(b))


namespace quda {

  void initLatticeParams(KernelParams* params, const LatticeField &lat)
  {
    params->Vh = lat.VolumeCB();
    params->Vsh = lat.X()[0]*lat.X()[1]*lat.X()[2]/2;

    params->X1 = lat.X()[0];
    params->X2 = lat.X()[1];
    params->X3 = lat.X()[2];
    params->X4 = lat.X()[3];

    params->X1h = params->X1/2;
    params->X2h = params->X2/2;

    params->X1_3 = params->X1*3;
    params->X2_3 = params->X2*3;
    params->X3_3 = params->X3*3;
    params->X4_3 = params->X4*3;

    // 2D volumes (AKA areas :) )
    params->X2X1 = params->X2*params->X1;
    params->X3X1 = params->X3*params->X1;
    params->X3X2 = params->X3*params->X2;


    // 3D volumes
    params->X3X2X1 = params->X3*params->X2*params->X1;
    params->X3X2X1h = params->X3*params->X2*params->X1h;

    params->X4X2X1  = params->X4*params->X2*params->X1;
    params->X4X2X1h = params->X4*params->X2*params->X1h;

    params->X4X3X1  = params->X4*params->X3*params->X1;
    params->X4X3X1h = params->X4*params->X3*params->X1h;

    params->X4X3X2  = params->X4*params->X3*params->X2;
    params->X4X3X2h = params->X4*params->X3*params->X2h;


    params->X2X1_3 = 3*params->X2*params->X1;
    params->X3X2X1_3 = params->X3*params->X2X1_3; 


    params->X1m1 = params->X1 - 1;
    params->X2m1 = params->X2 - 1;
    params->X3m1 = params->X3 - 1;
    params->X4m1 = params->X4 - 1;

    // for improved staggerd fermions...
    params->X1m3 = params->X1 - 3;
    params->X2m3 = params->X2 - 3;
    params->X3m3 = params->X3 - 3;
    params->X4m3 = params->X4 - 3;


    params->X2X1mX1 = params->X2X1 - params->X1;

    params->X3X2X1mX2X1 = params->X3X2X1 - params->X2X1;

    params->X4X3X2X1mX3X2X1 = (params->X4-1)*params->X3X2X1;

    params->X4X3X2X1hmX3X2X1h = (params->X4-1)*params->X3*params->X2*params->X1h;

    params->X2X1m3X1 = params->X3X2X1 - 3*params->X1;

    params->X3X2X1m3X2X1 = (params->X3-3)*params->X2X1;

    params->X4X3X2X1hm3X3X2X1h = (params->X4-3)*params->X3*params->X2*params->X1h;

    params->Vh_2d_max = MAX(params->X1*params->X2/2, params->X1*params->X3/2); 
    params->Vh_2d_max = MAX(params->Vh_2d_max, params->X1*params->X4/2); 
    params->Vh_2d_max = MAX(params->Vh_2d_max, params->X2*params->X3/2); 
    params->Vh_2d_max = MAX(params->Vh_2d_max, params->X2*params->X4/2); 
    params->Vh_2d_max = MAX(params->Vh_2d_max, params->X3*params->X4/2); 


    // ghost zones
    for(int i=0; i<4; ++i){
      params->ghostFace[i] = 1;
      for(int j=0; j<4; ++j){
        if(i == j) continue;
        params->ghostFace[i] *= lat.X()[j]; 
      }
      params->ghostFace[i] /= 2;
    }

    return;
  }

  void initSpinorParams(KernelParams* params, const cudaColorSpinorField& spinor)
  {
    params->sp_stride = spinor.Stride();
    if(spinor.Ndim() == 5){
      params->Ls = spinor.X(4);
    }
    return;
  }


  void initGaugeParams(KernelParams* params, const cudaGaugeField& gauge)
  {
    params->ga_stride = gauge.Stride();

    params->gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES);

    params->anisotropy = gauge.Anisotropy();

    params->t_bc = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;

    params->coeff = -24.0*gauge.Tadpole()*gauge.Tadpole();

    // single-precision variants
    params->anisotropy_f = params->anisotropy;
    params->t_bc_f = params->t_bc;
    params->coeff_f = params->coeff;

    return;
  }

} // namespace quda

#undef MAX
