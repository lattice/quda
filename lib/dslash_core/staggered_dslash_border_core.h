#include "staggered_overlap_utilities.h"




#define MAT_MUL_V_APPEND(VOUT, M, V)            \
  VOUT##0_re += M##00_re * V##0_re;            \
  VOUT##0_re -= M##00_im * V##0_im;            \
  VOUT##0_re += M##01_re * V##1_re;            \
  VOUT##0_re -= M##01_im * V##1_im;            \
  VOUT##0_re += M##02_re * V##2_re;            \
  VOUT##0_re -= M##02_im * V##2_im;            \
  VOUT##0_im += M##00_re * V##0_im; \
  VOUT##0_im += M##00_im * V##0_re;            \
  VOUT##0_im += M##01_re * V##1_im;            \
  VOUT##0_im += M##01_im * V##1_re;            \
  VOUT##0_im += M##02_re * V##2_im;            \
  VOUT##0_im += M##02_im * V##2_re;            \
  VOUT##1_re += M##10_re * V##0_re; \
  VOUT##1_re -= M##10_im * V##0_im;            \
  VOUT##1_re += M##11_re * V##1_re;            \
  VOUT##1_re -= M##11_im * V##1_im;            \
  VOUT##1_re += M##12_re * V##2_re;            \
  VOUT##1_re -= M##12_im * V##2_im;            \
  VOUT##1_im += M##10_re * V##0_im; \
  VOUT##1_im += M##10_im * V##0_re;            \
  VOUT##1_im += M##11_re * V##1_im;            \
  VOUT##1_im += M##11_im * V##1_re;            \
  VOUT##1_im += M##12_re * V##2_im;            \
  VOUT##1_im += M##12_im * V##2_re;            \
  VOUT##2_re += M##20_re * V##0_re; \
  VOUT##2_re -= M##20_im * V##0_im;            \
  VOUT##2_re += M##21_re * V##1_re;            \
  VOUT##2_re -= M##21_im * V##1_im;            \
  VOUT##2_re += M##22_re * V##2_re;            \
  VOUT##2_re -= M##22_im * V##2_im;            \
  VOUT##2_im += M##20_re * V##0_im; \
  VOUT##2_im += M##20_im * V##0_re;            \
  VOUT##2_im += M##21_re * V##1_im;            \
  VOUT##2_im += M##21_im * V##1_re;            \
  VOUT##2_im += M##22_re * V##2_im;            \
  VOUT##2_im += M##22_im * V##2_re;

#define MINUS_ADJ_MAT_MUL_V_APPEND(VOUT, M, V)               \
  VOUT##0_re -= M##00_re * V##0_re; \
  VOUT##0_re -= M##00_im * V##0_im;            \
  VOUT##0_re -= M##10_re * V##1_re;            \
  VOUT##0_re -= M##10_im * V##1_im;            \
  VOUT##0_re -= M##20_re * V##2_re;            \
  VOUT##0_re -= M##20_im * V##2_im;            \
  VOUT##0_im -= M##00_re * V##0_im; \
  VOUT##0_im += M##00_im * V##0_re;            \
  VOUT##0_im -= M##10_re * V##1_im;            \
  VOUT##0_im += M##10_im * V##1_re;            \
  VOUT##0_im -= M##20_re * V##2_im;            \
  VOUT##0_im += M##20_im * V##2_re;            \
  VOUT##1_re -= M##01_re * V##0_re; \
  VOUT##1_re -= M##01_im * V##0_im;            \
  VOUT##1_re -= M##11_re * V##1_re;            \
  VOUT##1_re -= M##11_im * V##1_im;            \
  VOUT##1_re -= M##21_re * V##2_re;            \
  VOUT##1_re -= M##21_im * V##2_im;            \
  VOUT##1_im -= M##01_re * V##0_im; \
  VOUT##1_im += M##01_im * V##0_re;            \
  VOUT##1_im -= M##11_re * V##1_im;            \
  VOUT##1_im += M##11_im * V##1_re;            \
  VOUT##1_im -= M##21_re * V##2_im;            \
  VOUT##1_im += M##21_im * V##2_re;            \
  VOUT##2_re -= M##02_re * V##0_re; \
  VOUT##2_re -= M##02_im * V##0_im;            \
  VOUT##2_re -= M##12_re * V##1_re;            \
  VOUT##2_re -= M##12_im * V##1_im;            \
  VOUT##2_re -= M##22_re * V##2_re;            \
  VOUT##2_re -= M##22_im * V##2_im;            \
  VOUT##2_im -= M##02_re * V##0_im; \
  VOUT##2_im += M##02_im * V##0_re;            \
  VOUT##2_im -= M##12_re * V##1_im;            \
  VOUT##2_im += M##12_im * V##1_re;            \
  VOUT##2_im -= M##22_re * V##2_im;            \
  VOUT##2_im += M##22_im * V##2_re;


#define A0_re  A0.x
#define A0_im  A0.y
#define A1_re  A1.x
#define A1_im  A1.y
#define A2_re  A2.x
#define A2_im  A2.y

#define I0_re I0.x
#define I0_im I0.x
#define I1_re I1.x
#define I1_im I1.y
#define I2_re I2.x
#define I2_im I2.x


#define o0_re o0.x
#define o0_im o0.y
#define o1_re o1.x
#define o1_im o1.y
#define o2_re o2.x
#define o2_im o2.y


#define fat00_re FAT0.x
#define fat00_im FAT0.y
#define fat01_re FAT1.x
#define fat01_im FAT1.y
#define fat02_re FAT2.x
#define fat02_im FAT2.y
#define fat10_re FAT3.x
#define fat10_im FAT3.y
#define fat11_re FAT4.x
#define fat11_im FAT4.y
#define fat12_re FAT5.x
#define fat12_im FAT5.y
#define fat20_re FAT6.x
#define fat20_im FAT6.y
#define fat21_re FAT7.x
#define fat21_im FAT7.y
#define fat22_re FAT8.x
#define fat22_im FAT8.y

#if (DD_PREC==0 || DD_RECON==2) // double precision

#define long00_re LONG0.x
#define long00_im LONG0.y
#define long01_re LONG1.x
#define long01_im LONG1.y
#define long02_re LONG2.x
#define long02_im LONG2.y
#define long10_re LONG3.x
#define long10_im LONG3.y
#define long11_re LONG4.x
#define long11_im LONG4.y
#define long12_re LONG5.x
#define long12_im LONG5.y
#define long20_re LONG6.x
#define long20_im LONG6.y
#define long21_re LONG7.x
#define long21_im LONG7.y
#define long22_re LONG8.x
#define long22_im LONG8.y

#else // Not double precision 

#define long00_re LONG0.x
#define long00_im LONG0.y
#define long01_re LONG0.z
#define long01_im LONG0.w
#define long02_re LONG1.x
#define long02_im LONG1.y
#define long10_re LONG1.z
#define long10_im LONG1.w
#define long11_re LONG2.x
#define long11_im LONG2.y
#define long12_re LONG2.z
#define long12_im LONG2.w
#define long20_re LONG3.x
#define long20_im LONG3.y
#define long21_re LONG3.z
#define long21_im LONG3.w
#define long22_re LONG4.x
#define long22_im LONG4.y

#endif


// Kernel body
{

#if (DD_PREC==0)
#define Real double
#define Real2 double2
#else
#define Real float
#define Real2 float2
#endif

  // Quark variables
  Real2 A0, A1, A2;
  Real2 I0, I1, I2;
  Real2 o0, o1, o2;
  A0_re = A0_im = 0.0;
  A1_re = A1_im = 0.0;
  A2_re = A2_im = 0.0;
  
  int cb_index = threadIdx.x + blockIdx.x*blockDim.x; // checkerboard index

  // x1, x2, x3, x4 denote quark site coordinates in the ghost zone
  // y1, y2, y3, y4 are gluon site coordinates in the extended domain
  int x1, x2, x3, x4; 
  int y1, y2, y3, y4; 
  
  // compute the quark-field coordinates in the ghost zone.
  getCoordinates<Dir>(&x1, &x2, &x3, &x4, cb_index, param.parity);

  getGluonCoordsFromGhostCoords<Dir,Nface>(&y1, &y2, &y3, &y4, 
					    x1, x2, x3, x4, param.parity);


  int spinor_stride;

  int spinor_neighbor_index; // N.B. spinor_index >= 0 implies the neighbor index is in the active region
  { // First consider +ve displacements only 
    const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + y2*Y1 + y1) >> 1;
    // +X
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward<0>(x1,x2,x3,x4,param);

    if(spinor_neighbor_index >= 0){  
#if (DD_PREC == 2) // half precision
      const int norm_idx = getNormIndex<0,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);

      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 0, gluon_index, ddStaggeredConstants.fatlinkStride); // Link-variable elements are declared in this macro
      MAT_MUL_V_APPEND(A, fat, I);
    }

    // +Y 
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward<1>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){

#if (DD_PREC == 2)
      const int norm_idx = getNormIndex<1,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 1, gluon_index, ddStaggeredConstants.fatlinkStride);
      MAT_MUL_V_APPEND(A, fat, I);
    }

    // +Z 
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward<2>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){

#if (DD_PREC == 2)
      const int norm_idx = getNormIndex<2,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 2, gluon_index, ddStaggeredConstants.fatlinkStride);
      MAT_MUL_V_APPEND(A, fat, I);
    }

    // +T
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward<3>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
      const int norm_idx = getNormIndex<3,Nface>(spinor_neighbor_index,3);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 3, gluon_index, ddStaggeredConstants.fatlinkStride);
      MAT_MUL_V_APPEND(A, fat, I);
    }
  } // +ve displacements



  // Need to be careful about this. Need to find neighboring gluon index.
  { // -ve displacements
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back<0>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
      const int norm_idx = getNormIndex<0,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + y2*Y1 + y1-1) >> 1;
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 0, gluon_index, ddStaggeredConstants.fatlinkStride);
      MINUS_ADJ_MAT_MUL_V_APPEND(A, fat, I); // Need to change this!
    }
  
    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back<1>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2) 
      const int norm_idx = getNormIndex<1,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + (y2-1)*Y1 + y1) >> 1;
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 1, gluon_index, ddStaggeredConstants.fatlinkStride);
      MINUS_ADJ_MAT_MUL_V_APPEND(A, fat, I); // Need to change this!
    }

    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back<2>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
      const int norm_idx = getNormIndex<2,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      const int gluon_index = (y4*Y3Y2Y1 + (y3-1)*Y2Y1 + Y1 + y1) >> 1;
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 2, gluon_index, ddStaggeredConstants.fatlinkStride);
      MINUS_ADJ_MAT_MUL_V_APPEND(A, fat, I); // Need to change this!
    }

    spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back<3>(x1,x2,x3,x4,param);
    if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)      
      const int norm_idx = getNormIndex<3,Nface>(spinor_neighbor_index,param);
#endif
      spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
      const int gluon_index = ((y4-1)*Y3Y2Y1 + Y2Y1 + Y1 + y1) >> 1; // divide by 2 since checkerboard index
      READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
      READ_FAT_MATRIX(FAT, FATLINK0TEX, 3, gluon_index, ddStaggeredConstants.fatlinkStride);
      MINUS_ADJ_MAT_MUL_V_APPEND(A, fat, I); // Need to change this!
    }
  } // -ve displacements
  

  // Now apply the three-hop term...
  if(hasNaik){
    { // First consider +ve displacements only 
      const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + y2*Y1 + y1) >> 1;
      // +X
      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward_three<0>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){  
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<0,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 0, gluon_index, ddStaggeredConstants.longlinkStride);
        MAT_MUL_V_APPEND(A, long, I);
      }

      // +Y 
      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward_three<1>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<1,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 1, gluon_index, ddStaggeredConstants.longlinkStride);
        MAT_MUL_V_APPEND(A, long, I);
      }

      // +Z 
      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward_three<2>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<2,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 2, gluon_index, ddStaggeredConstants.longlinkStride);
        MAT_MUL_V_APPEND(A, long, I);
      }

      // +T
      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template forward_three<3>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<3,Nface>(spinor_neighbor_index);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 3, gluon_index, ddStaggeredConstants.longlinkStride);
        MINUS_ADJ_MAT_MUL_V_APPEND(A, long, I);
      }
    } // +ve displacements



    // Need to be careful about this. Need to find neighboring gluon index.
    { // -ve displacements
      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back_three<0>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<0,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + y2*Y1 + y1-3) >> 1;
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 0, gluon_index, ddStaggeredConstants.longlinkStride);
        MINUS_ADJ_MAT_MUL_V_APPEND(A, long , I); // Need to change this!
      }

      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back_three<1>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<1,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        const int gluon_index = (y4*Y3Y2Y1 + y3*Y2Y1 + (y2-3)*Y1 + y1) >> 1;
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 1, gluon_index, ddStaggeredConstants.longlinkStride);
        MINUS_ADJ_MAT_MUL_V_APPEND(A, long, I); // Need to change this!
      }

      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back_three<2>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<2,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        const int gluon_index = (y4*Y3Y2Y1 + (y3-3)*Y2Y1 + Y1 + y1) >> 1;
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 2, gluon_index, ddStaggeredConstants.longlinkStride);
        MINUS_ADJ_MAT_MUL_V_APPEND(A, long, I); // Need to change this!
      }

      spinor_neighbor_index = NeighborIndex<Dir,Nface>::template back_three<3>(x1,x2,x3,x4,param);
      if(spinor_neighbor_index >= 0){
#if (DD_PREC == 2)
        const int norm_idx = getNormIndex<3,Nface>(spinor_neighbor_index,param);
#endif
        spinor_stride = getSpinorStride<Dir,Nface>(spinor_neighbor_index);
        const int gluon_index = ((y4-3)*Y3Y2Y1 + Y2Y1 + Y1 + y1) >> 1; // divide by 2 since checkerboard index
        READ_KS_NBR_SPINOR(I, SPINORTEX, spinor_neighbor_index, spinor_stride);
        READ_LONG_MATRIX(LONG, LONGLINK0TEX, 3, gluon_index, ddStaggeredConstants.longlinkStride);
        MINUS_ADJ_MAT_MUL_V_APPEND(A, long, I); // Need to change this!
      }
    } // -ve displacements

  } // if(hasNaik)


  // Write the result back to device memory
#if (DD_DAG == 1)
{
  o0_re = -A0_re;
  o0_im = -A0_im;
  o1_re = -A1_re;
  o1_im = -A1_im;
  o2_re = -A2_re;
  o2_im = -A2_im;
}
#else
{
  o0_re = A0_re;
  o0_im = A0_im;
  o1_re = A1_re;
  o1_im = A1_im;
  o2_re = A2_re;
  o2_im = A2_im;
}
#endif

#ifdef DSLASH_AXPY
  READ_ACCUM(ACCUMTEX,cb_index,Nface*ghostFace[Dir]);
  o0_re = -o0_re + a*accum0.x;
  o0_im = -o0_im + a*accum0.y;
  o1_re = -o1_re + a*accum1.x;
  o1_im = -o1_im + a*accum1.y;
  o2_re = -o2_re + a*accum2.x;
  o2_im = -o2_im + a*accum2.y;
#endif // DSLASH_AXPY

  // Need to change WRITE_SPINOR
  WRITE_SPINOR(out, cb_index, Nface*ghostFace[Dir]); // Nface*ghostFace[Dir] is the stride in the boundary region
  return;
} // end kernel body 

// Clear quark-field macros
#undef A0_re
#undef A0_im
#undef A1_re
#undef A1_im
#undef A2_re
#undef A2_im 

#undef I0_re
#undef I0_im 
#undef I1_re
#undef I1_im 
#undef I2_re
#undef I2_im

#undef o0_re
#undef o0_im
#undef o1_re
#undef o1_im 
#undef o2_re
#undef o2_im



// Clear gauge-field macros
#undef fat00_re
#undef fat00_im
#undef fat01_re
#undef fat01_im
#undef fat02_re
#undef fat02_im
#undef fat10_re
#undef fat10_im
#undef fat11_re
#undef fat11_im 
#undef fat12_re
#undef fat12_im
#undef fat20_re
#undef fat20_im 
#undef fat21_re
#undef fat21_im 
#undef fat22_re
#undef fat22_im


#undef long00_re
#undef long00_im
#undef long01_re
#undef long01_im
#undef long02_re
#undef long02_im
#undef long10_re
#undef long10_im 
#undef long11_re
#undef long11_im 
#undef long12_re
#undef long12_im
#undef long20_re
#undef long20_im
#undef long21_re
#undef long21_im 
#undef long22_re
#undef long22_im

#undef Real
#undef Real2


//      - NEED to set the strides  				- Done
//      - NEED to code up the matrix multiplication		- Done - well, should probably turn these into functions	
//      - NEED to work out the norm index for half precision    - Done
// TODO - NEED to replace the macros with functions             - Probably don't need to do this
//      - NEED to test it all!
//      - First, test that it compiles with double precision   
//      - Need to specify Y1, Y2, Y3, Y4, etc.
//      - test tomorrow.
  

