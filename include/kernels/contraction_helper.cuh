#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <kernel.h>

namespace quda {

  using spinor_array = array<double2, 4>;
  using spinor_matrix = array<double2, 16>;
  
  template <int reduction_dim, class T> __device__ int idx_from_t_xyz(int t, int xyz, T X[4])
  {
    int x[4];
#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != reduction_dim) {
	x[d] = xyz % X[d];
	xyz /= X[d];
      }
    }    
    x[reduction_dim] = t;    
    return (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]);
  }
  
  template <int reduction_dim, class T> __device__ int* sink_from_t_xyz(int t, int xyz, T X[4])
  {
    static int sink[4];
#pragma unroll
    for (int d = 0; d < 4; d++) {
      if (d != reduction_dim) {
        sink[d] = xyz % X[d];
        xyz /= X[d];
      }
    }
    sink[reduction_dim] = t;    
    return sink;
  }
  
  template <class T> __device__ int idx_from_sink(T X[4], int* sink) { return ((sink[3] * X[2] + sink[2]) * X[1] + sink[1]) * X[0] + sink[0]; }

  template <typename real> class DRGammaMatrix
  {

  public:
    // Stores gamma matrix column index for non-zero complex value.
    // This is shared by g5gm, gmg5.
    int gm_i[16][4] {};

    // Stores gamma matrix non-zero complex value for the corresponding g5gm_i
    complex<real> g5gm_z[16][4];
    
    // use tr[Gamma*Prop*Gamma*g5*conj(Prop)*g5] = 
    //     tr[g5*Gamma*Prop*g5*Gamma*(-1)^{?}*conj(Prop)].
    // the possible minus sign will be taken care of in the main function
    // Constructor
    DRGammaMatrix()
    {      
      const complex<real> i(0., 1.);
      // VECTORS
      // G_idx = 1: \gamma_1
      gm_i[0][0] = 3;
      gm_i[0][1] = 2;
      gm_i[0][2] = 1;
      gm_i[0][3] = 0;

      g5gm_z[0][0] = i;
      g5gm_z[0][1] = i;
      g5gm_z[0][2] = i;
      g5gm_z[0][3] = i;

      // G_idx = 2: \gamma_2
      gm_i[1][0] = 3;
      gm_i[1][1] = 2;
      gm_i[1][2] = 1;
      gm_i[1][3] = 0;

      g5gm_z[1][0] = -1.;
      g5gm_z[1][1] = 1.;
      g5gm_z[1][2] = -1.;
      g5gm_z[1][3] = 1.;

      // G_idx = 3: \gamma_3
      gm_i[2][0] = 2;
      gm_i[2][1] = 3;
      gm_i[2][2] = 0;
      gm_i[2][3] = 1;

      g5gm_z[2][0] = i;
      g5gm_z[2][1] = -i;
      g5gm_z[2][2] = i;
      g5gm_z[2][3] = -i;

      // G_idx = 4: \gamma_4
      gm_i[3][0] = 2;
      gm_i[3][1] = 3;
      gm_i[3][2] = 0;
      gm_i[3][3] = 1;

      g5gm_z[3][0] = 1.;
      g5gm_z[3][1] = 1.;
      g5gm_z[3][2] = -1.;
      g5gm_z[3][3] = -1.;


      // PSEUDO-VECTORS
      // G_idx = 6: \gamma_5\gamma_1
      gm_i[4][0] = 3;
      gm_i[4][1] = 2;
      gm_i[4][2] = 1;
      gm_i[4][3] = 0;

      g5gm_z[4][0] = i;
      g5gm_z[4][1] = i;
      g5gm_z[4][2] = -i;
      g5gm_z[4][3] = -i;

      // G_idx = 7: \gamma_5\gamma_2
      gm_i[5][0] = 3;
      gm_i[5][1] = 2;
      gm_i[5][2] = 1;
      gm_i[5][3] = 0;

      g5gm_z[5][0] = -1.;
      g5gm_z[5][1] = 1.;
      g5gm_z[5][2] = 1.;
      g5gm_z[5][3] = -1.;

      // G_idx = 8: \gamma_5\gamma_3
      gm_i[6][0] = 2;
      gm_i[6][1] = 3;
      gm_i[6][2] = 0;
      gm_i[6][3] = 1;

      g5gm_z[6][0] = i;
      g5gm_z[6][1] = -i;
      g5gm_z[6][2] = -i;
      g5gm_z[6][3] = i;

      // G_idx = 9: \gamma_5\gamma_4
      gm_i[7][0] = 2;
      gm_i[7][1] = 3;
      gm_i[7][2] = 0;
      gm_i[7][3] = 1;

      g5gm_z[7][0] = 1.;
      g5gm_z[7][1] = 1.;
      g5gm_z[7][2] = 1.;
      g5gm_z[7][3] = 1.;

      // SCALAR
      // G_idx = 0: I
      gm_i[8][0] = 0;
      gm_i[8][1] = 1;
      gm_i[8][2] = 2;
      gm_i[8][3] = 3;

      g5gm_z[8][0] = 1.;
      g5gm_z[8][1] = 1.;
      g5gm_z[8][2] = -1.;
      g5gm_z[8][3] = -1.;


      // PSEUDO-SCALAR
      // G_idx = 5: \gamma_5
      gm_i[9][0] = 0;
      gm_i[9][1] = 1;
      gm_i[9][2] = 2;
      gm_i[9][3] = 3;

      g5gm_z[9][0] = 1.;
      g5gm_z[9][1] = 1.;
      g5gm_z[9][2] = 1.;
      g5gm_z[9][3] = 1.;

      // TENSORS
      // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
      gm_i[10][0] = 0;
      gm_i[10][1] = 1;
      gm_i[10][2] = 2;
      gm_i[10][3] = 3;

      g5gm_z[10][0] = 1.;
      g5gm_z[10][1] = -1.;
      g5gm_z[10][2] = -1.;
      g5gm_z[10][3] = 1.;
      
      // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]. this matrix was corrected
      gm_i[11][0] = 1;
      gm_i[11][1] = 0;
      gm_i[11][2] = 3;
      gm_i[11][3] = 2;

      g5gm_z[11][0] = -i;
      g5gm_z[11][1] = i;
      g5gm_z[11][2] = i;
      g5gm_z[11][3] = -i;
      
      // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
      gm_i[12][0] = 1;
      gm_i[12][1] = 0;
      gm_i[12][2] = 3;
      gm_i[12][3] = 2;

      g5gm_z[12][0] = -1.;
      g5gm_z[12][1] = -1.;
      g5gm_z[12][2] = -1.;
      g5gm_z[12][3] = -1.;

      // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
      gm_i[13][0] = 1;
      gm_i[13][1] = 0;
      gm_i[13][2] = 3;
      gm_i[13][3] = 2;

      g5gm_z[13][0] = 1.;
      g5gm_z[13][1] = 1.;
      g5gm_z[13][2] = -1.;
      g5gm_z[13][3] = -1.;
      // G_idx = 14: (i/2) * [\gamma_2, \gamma_4]
      gm_i[14][0] = 1;
      gm_i[14][1] = 0;
      gm_i[14][2] = 3;
      gm_i[14][3] = 2;

      g5gm_z[14][0] = -i;
      g5gm_z[14][1] = i;
      g5gm_z[14][2] = -i;
      g5gm_z[14][3] = i;
      
      // G_idx = 15: (i/2) * [\gamma_3, \gamma_4]. this matrix was corrected
      gm_i[15][0] = 0;
      gm_i[15][1] = 1;
      gm_i[15][2] = 2;
      gm_i[15][3] = 3;

      g5gm_z[15][0] = -1.;
      g5gm_z[15][1] = 1.;
      g5gm_z[15][2] = -1.;
      g5gm_z[15][3] = 1.;
    };
  };
}
