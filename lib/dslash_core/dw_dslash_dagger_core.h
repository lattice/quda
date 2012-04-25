// *** CUDA DSLASH DAGGER ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0

// input spinor
#ifdef SPINOR_DOUBLE
#define spinorFloat double
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y
#define i10_re I3.x
#define i10_im I3.y
#define i11_re I4.x
#define i11_im I4.y
#define i12_re I5.x
#define i12_im I5.y
#define i20_re I6.x
#define i20_im I6.y
#define i21_re I7.x
#define i21_im I7.y
#define i22_re I8.x
#define i22_im I8.y
#define i30_re I9.x
#define i30_im I9.y
#define i31_re I10.x
#define i31_im I10.y
#define i32_re I11.x
#define i32_im I11.y

#else
#define spinorFloat float
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I0.z
#define i01_im I0.w
#define i02_re I1.x
#define i02_im I1.y
#define i10_re I1.z
#define i10_im I1.w
#define i11_re I2.x
#define i11_im I2.y
#define i12_re I2.z
#define i12_im I2.w
#define i20_re I3.x
#define i20_im I3.y
#define i21_re I3.z
#define i21_im I3.w
#define i22_re I4.x
#define i22_im I4.y
#define i30_re I4.z
#define i30_im I4.w
#define i31_re I5.x
#define i31_im I5.y
#define i32_re I5.z
#define i32_im I5.w
#endif // SPINOR_DOUBLE

// gauge link
#ifdef GAUGE_FLOAT2
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G1.x
#define g01_im G1.y
#define g02_re G2.x
#define g02_im G2.y
#define g10_re G3.x
#define g10_im G3.y
#define g11_re G4.x
#define g11_im G4.y
#define g12_re G5.x
#define g12_im G5.y
#define g20_re G6.x
#define g20_im G6.y
#define g21_re G7.x
#define g21_im G7.y
#define g22_re G8.x
#define g22_im G8.y
// temporaries
#define A_re G9.x
#define A_im G9.y

#else
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G0.z
#define g01_im G0.w
#define g02_re G1.x
#define g02_im G1.y
#define g10_re G1.z
#define g10_im G1.w
#define g11_re G2.x
#define g11_im G2.y
#define g12_re G2.z
#define g12_im G2.w
#define g20_re G3.x
#define g20_im G3.y
#define g21_re G3.z
#define g21_im G3.w
#define g22_re G4.x
#define g22_im G4.y
// temporaries
#define A_re G4.z
#define A_im G4.w

#endif // GAUGE_DOUBLE

// conjugated gauge link
#define gT00_re (+g00_re)
#define gT00_im (-g00_im)
#define gT01_re (+g10_re)
#define gT01_im (-g10_im)
#define gT02_re (+g20_re)
#define gT02_im (-g20_im)
#define gT10_re (+g01_re)
#define gT10_im (-g01_im)
#define gT11_re (+g11_re)
#define gT11_im (-g11_im)
#define gT12_re (+g21_re)
#define gT12_im (-g21_im)
#define gT20_re (+g02_re)
#define gT20_im (-g02_im)
#define gT21_re (+g12_re)
#define gT21_im (-g12_im)
#define gT22_re (+g22_re)
#define gT22_im (-g22_im)

// output spinor
volatile spinorFloat o00_re;
volatile spinorFloat o00_im;
volatile spinorFloat o01_re;
volatile spinorFloat o01_im;
volatile spinorFloat o02_re;
volatile spinorFloat o02_im;
volatile spinorFloat o10_re;
volatile spinorFloat o10_im;
volatile spinorFloat o11_re;
volatile spinorFloat o11_im;
volatile spinorFloat o12_re;
volatile spinorFloat o12_im;
volatile spinorFloat o20_re;
volatile spinorFloat o20_im;
volatile spinorFloat o21_re;
volatile spinorFloat o21_im;
volatile spinorFloat o22_re;
volatile spinorFloat o22_im;
volatile spinorFloat o30_re;
volatile spinorFloat o30_im;
volatile spinorFloat o31_re;
volatile spinorFloat o31_im;
volatile spinorFloat o32_re;
volatile spinorFloat o32_im;

#ifdef SPINOR_DOUBLE
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#endif
#else
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
#endif

//J  Go up for these.  At this point hoping that no changes
//J  will be needed for the gauge/clover code.  But need to check.
#include "read_gauge.h"
//J  Change to a dwf version which sets N and Vh to dwf values.
#include "io_spinor.h"

//J --- INDEX LOGIC ---
//J  BLOCK_DIM is defined in dslash_quda.h. There it is
//J  set to 64.  Why?  GRID_DIM
//J  is also defined there.  Where is that used?
//J  Introduce global thread index sid.  Each thread is
//J  associated with a pair of sites.  They are paired
//J  in the x1 direction.  They are ordered red/black or
//J  black/red, depending on the number of boundary crossings.
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

#if (defined MULTI_GPU) && (DD_PREC==2) // half precision
int sp_norm_idx;
#endif // MULTI_GPU half precision

int boundaryCrossings;
int boundaryCrossings4d;
int X;
int x1, x2, x3, x4, xs;

//NOTE: param is of type DslashParam defined in dslash_quda.cu:
//! struct DslashParam {
//!   int tOffset; // offset into the T dimension (multi gpu only)
//!   int tMul;    // spatial volume distance between the T faces being updated (multi gpu only)
//!   int threads; // the desired number of active threads
//!   int parity;  // Even-Odd or Odd-Even
//!   int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
//!   int ghostDim[QUDA_MAX_DIM]; // Whether a ghost zone has been allocated for a given dimension
//!   int ghostOffset[QUDA_MAX_DIM];
//!   int ghostNormOffset[QUDA_MAX_DIM];
//!   KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T
//! };

//NOTE: (from dslash_quda.cu): first we create tm dslash objects, by calling class constructor we assign parameters and bound textures: 
// note new texture ref for output spinors that will be reused for exterior kernels (accumulation in global buffer)

//NOTE: (from  dslashCuda(...) function in dslash_quda.cu) prior bulk computuations we perform face spinor interchange:
//!    face->exchangeFacesStart(*inSpinor, 1-parity, dagger, dir, streams);
//exchangeFacesStart method defined, e.g., in face_mpi.cpp and does the following
//i)   allocate ghost buffer (in each direction): allocateGhostBuffer() method defined in cuda_color_spinor_field.cpp
//ii)  performs gather operations: packGhost() method defined in cuda_color_spinor_field.cpp
//NOTE to ii) if dim == 3 (T-direction), packGhost simply performs copy, but if dim != 3 it calls packFaceWilson for each subbuffer
//     packFaceWilson - wrapper for packing cuda kernels defined in pack_face_def.h (kernel bodies are found in wilson_pack_face_core.h)        
//iii) calls recv mpi routines for corresponding buffers
//next stage: computation of the interior kernels (dslash.apply(blockDim[0], ...);) Note, that this is also standard dslash kernels for single-GPU kernels) using apply() method for
//tm dslash object
//then we start sending faces for each dimention (exchangeFacesComms(i) that performs MPI communications, no CUDA copy) 
//and scatter the buffer by exchangeFacesWait(*inSpinor, dagger, i) (it unpack host buffer to ghost cuda face spinor, using cuda copy) 
//Finally: calls external dslash kernels for each direction (dslash.apply(blockDim[i], ...);)
//Note that in this case we need to fetch intermediate data accumulated from internal computations

#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

//J  Boundary crossings has to do with the red/black checkerboard.
//J  We keep track of how many of those boundaries we cross in going
//J  from the origin to the location associated with THIS thread.
//J  Add a dimension relative to original code, so we get an extra term here.
//J  Note that X1,..,X4 and X1h are defined in quda.h.

// Will work with 5d checkerboard (for spinors) b/c this is 5DPC code.
// Because of this, some gymnastics are required when accessing
// the gauge links.  I.e., they should be thought of as living
// on the xs=0 slice, when determining their parity.

  boundaryCrossings = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h) + sid/(X4*X3*X2*X1h);
// Gauge fields have 4d parity.  E.g., if xs is odd, we need to grab
// opposite parity link if we're supposed to use U_\mu(x).
  boundaryCrossings4d = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h);
// We will use the difference between boundaryCrossings and boundaryCrossings4d to determine
// which gaugetex is used in the operations below.

//J  Define the linear index X to checkerboard sites on the lattice.
  X = 2*sid + (boundaryCrossings + param.parity) % 2;
//J  Coordinates of the checkerboard (sublattice) site.
//J  This is for the output spinor.
  xs = X/(X4*X3*X2*X1);
  x4 = (X/(X3*X2*X1)) % X4;
  x3 = (X/(X2*X1)) % X3;
  x2 = (X/X1) % X2;
  x1 = X % X1;

// Initialize output spinor to zero.
  o00_re = o00_im = 0;
  o01_re = o01_im = 0;
  o02_re = o02_im = 0;
  o10_re = o10_im = 0;
  o11_re = o11_im = 0;
  o12_re = o12_im = 0;
  o20_re = o20_im = 0;
  o21_re = o21_im = 0;
  o22_re = o22_im = 0;
  o30_re = o30_im = 0;
  o31_re = o31_im = 0;
  o32_re = o32_im = 0;

#ifdef MULTI_GPU
} else { // exterior DW kernel

  const int dim = static_cast<int>(kernel_type);
  const int face_volume = (param.threads >> 1);           // volume of one face
  const int face_num = (sid >= face_volume);              // is this thread updating face 0 or 1
  face_idx = sid - face_num*face_volume;        // index into the respective face

  // ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
  // face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
  //sp_idx = face_idx + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];
#endif
//NOTE: face_idx is mapped to true lattice coordinates (X, sid, as a cb coordinate, and x#)
//WARNING: for DW fermions we are dealing with 4d faces
//NOTE: sid will be mapped to lattice (parity spinor) coordinate (?)
//NOTE: do we compute here x1,x2,x3,x4,xs, or better to get X and then reconstruct x#?
  coordsFromDWFaceIndex<1>(sid, x1, x2, x3, x4, xs, face_idx, face_volume, dim, face_num, param.parity);
  
  boundaryCrossings = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h) + sid/(X4*X3*X2*X1h);
// Gauge fields have 4d parity.  E.g., if xs is odd, we need to grab
// opposite parity link if we're supposed to use U_\mu(x).
  boundaryCrossings4d = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h); 
  
//J  Define the linear index X to checkerboard sites on the lattice.
  X = 2*sid + (boundaryCrossings + param.parity) % 2;  
  
//NOTE:intermediate spinor is what was 'pre-computed' from interior kernels and accumulated in the global buffer (this is output of 
//interior kernel).

  READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid, sid);

  o00_re = i00_re;  o00_im = i00_im;
  o01_re = i01_re;  o01_im = i01_im;
  o02_re = i02_re;  o02_im = i02_im;
  o10_re = i10_re;  o10_im = i10_im;
  o11_re = i11_re;  o11_im = i11_im;
  o12_re = i12_re;  o12_im = i12_im;
  o20_re = i20_re;  o20_im = i20_im;
  o21_re = i21_re;  o21_im = i21_im;
  o22_re = i22_re;  o22_im = i22_im;
  o30_re = i30_re;  o30_im = i30_im;
  o31_re = i31_re;  o31_im = i31_im;
  o32_re = i32_re;  o32_im = i32_im;
}
#endif // MULTI_GPU


#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[0] || x1<X1m1)) ||
     (kernel_type == EXTERIOR_KERNEL_X && x1==X1m1) )
#endif
{
  // Projector P0+
  // 1 0 0 i 
  // 0 1 i 0 
  // 0 -i 1 0 
  // -i 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x1==X1m1 ? X-X1m1 : X+1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x1==X1m1 ? X-X1m1 : X+1) >> 1;
#endif
  
  const int ga_idx = sid % Vh;
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 0, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i30_im;
    a0_im = +i00_im+i30_re;
    a1_re = +i01_re-i31_im;
    a1_im = +i01_im+i31_re;
    a2_re = +i02_re-i32_im;
    a2_im = +i02_im+i32_re;
    b0_re = +i10_re-i20_im;
    b0_im = +i10_im+i20_re;
    b1_re = +i11_re-i21_im;
    b1_im = +i11_im+i21_re;
    b2_re = +i12_re-i22_im;
    b2_im = +i12_im+i22_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(0);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += B0_im;
  o20_im -= B0_re;
  o30_re += A0_im;
  o30_im -= A0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += B1_im;
  o21_im -= B1_re;
  o31_re += A1_im;
  o31_im -= A1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += B2_im;
  o22_im -= B2_re;
  o32_re += A2_im;
  o32_im -= A2_re;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 0, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i30_im;
    a0_im = +i00_im+i30_re;
    a1_re = +i01_re-i31_im;
    a1_im = +i01_im+i31_re;
    a2_re = +i02_re-i32_im;
    a2_im = +i02_im+i32_re;
    b0_re = +i10_re-i20_im;
    b0_im = +i10_im+i20_re;
    b1_re = +i11_re-i21_im;
    b1_im = +i11_im+i21_re;
    b2_re = +i12_re-i22_im;
    b2_im = +i12_im+i22_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(0);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += B0_im;
  o20_im -= B0_re;
  o30_re += A0_im;
  o30_im -= A0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += B1_im;
  o21_im -= B1_re;
  o31_re += A1_im;
  o31_im -= A1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += B2_im;
  o22_im -= B2_re;
  o32_re += A2_im;
  o32_im -= A2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[0] || x1>0)) ||
     (kernel_type == EXTERIOR_KERNEL_X && x1==0) )
#endif
{
  // Projector P0-
  // 1 0 0 -i 
  // 0 1 -i 0 
  // 0 i 1 0 
  // i 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x1==0 ? X+X1m1 : X-1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x1==0 ? X+X1m1 : X-1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx % Vh : Vh+(face_idx % ghostFace[static_cast<int>(kernel_type)]));
#else
  const int ga_idx = sp_idx % Vh;
#endif
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 1, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i30_im;
    a0_im = +i00_im-i30_re;
    a1_re = +i01_re+i31_im;
    a1_im = +i01_im-i31_re;
    a2_re = +i02_re+i32_im;
    a2_im = +i02_im-i32_re;
    b0_re = +i10_re+i20_im;
    b0_im = +i10_im-i20_re;
    b1_re = +i11_re+i21_im;
    b1_im = +i11_im-i21_re;
    b2_re = +i12_re+i22_im;
    b2_im = +i12_im-i22_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(1);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= B0_im;
  o20_im += B0_re;
  o30_re -= A0_im;
  o30_im += A0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= B1_im;
  o21_im += B1_re;
  o31_re -= A1_im;
  o31_im += A1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= B2_im;
  o22_im += B2_re;
  o32_re -= A2_im;
  o32_im += A2_re;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 1, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i30_im;
    a0_im = +i00_im-i30_re;
    a1_re = +i01_re+i31_im;
    a1_im = +i01_im-i31_re;
    a2_re = +i02_re+i32_im;
    a2_im = +i02_im-i32_re;
    b0_re = +i10_re+i20_im;
    b0_im = +i10_im-i20_re;
    b1_re = +i11_re+i21_im;
    b1_im = +i11_im-i21_re;
    b2_re = +i12_re+i22_im;
    b2_im = +i12_im-i22_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(1);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= B0_im;
  o20_im += B0_re;
  o30_re -= A0_im;
  o30_im += A0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= B1_im;
  o21_im += B1_re;
  o31_re -= A1_im;
  o31_im += A1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= B2_im;
  o22_im += B2_re;
  o32_re -= A2_im;
  o32_im += A2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[1] || x2<X2m1)) ||
     (kernel_type == EXTERIOR_KERNEL_Y && x2==X2m1) )
#endif
{
  // Projector P1+
  // 1 0 0 1 
  // 0 1 -1 0 
  // 0 -1 1 0 
  // 1 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x2==X2m1 ? X-X2X1mX1 : X+X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x2==X2m1 ? X-X2X1mX1 : X+X1) >> 1;
#endif
  
  const int ga_idx = sid % Vh;
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 2, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i30_re;
    a0_im = +i00_im+i30_im;
    a1_re = +i01_re+i31_re;
    a1_im = +i01_im+i31_im;
    a2_re = +i02_re+i32_re;
    a2_im = +i02_im+i32_im;
    b0_re = +i10_re-i20_re;
    b0_im = +i10_im-i20_im;
    b1_re = +i11_re-i21_re;
    b1_im = +i11_im-i21_im;
    b2_re = +i12_re-i22_re;
    b2_im = +i12_im-i22_im;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(2);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= B0_re;
  o20_im -= B0_im;
  o30_re += A0_re;
  o30_im += A0_im;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= B1_re;
  o21_im -= B1_im;
  o31_re += A1_re;
  o31_im += A1_im;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= B2_re;
  o22_im -= B2_im;
  o32_re += A2_re;
  o32_im += A2_im;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 2, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i30_re;
    a0_im = +i00_im+i30_im;
    a1_re = +i01_re+i31_re;
    a1_im = +i01_im+i31_im;
    a2_re = +i02_re+i32_re;
    a2_im = +i02_im+i32_im;
    b0_re = +i10_re-i20_re;
    b0_im = +i10_im-i20_im;
    b1_re = +i11_re-i21_re;
    b1_im = +i11_im-i21_im;
    b2_re = +i12_re-i22_re;
    b2_im = +i12_im-i22_im;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(2);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= B0_re;
  o20_im -= B0_im;
  o30_re += A0_re;
  o30_im += A0_im;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= B1_re;
  o21_im -= B1_im;
  o31_re += A1_re;
  o31_im += A1_im;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= B2_re;
  o22_im -= B2_im;
  o32_re += A2_re;
  o32_im += A2_im;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[1] || x2>0)) ||
     (kernel_type == EXTERIOR_KERNEL_Y && x2==0) )
#endif
{
  // Projector P1-
  // 1 0 0 -1 
  // 0 1 1 0 
  // 0 1 1 0 
  // -1 0 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x2==0 ? X+X2X1mX1 : X-X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x2==0 ? X+X2X1mX1 : X-X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx % Vh : Vh+(face_idx % ghostFace[static_cast<int>(kernel_type)]));
#else
  const int ga_idx = sp_idx % Vh;
#endif
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 3, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i30_re;
    a0_im = +i00_im-i30_im;
    a1_re = +i01_re-i31_re;
    a1_im = +i01_im-i31_im;
    a2_re = +i02_re-i32_re;
    a2_im = +i02_im-i32_im;
    b0_re = +i10_re+i20_re;
    b0_im = +i10_im+i20_im;
    b1_re = +i11_re+i21_re;
    b1_im = +i11_im+i21_im;
    b2_re = +i12_re+i22_re;
    b2_im = +i12_im+i22_im;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(3);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += B0_re;
  o20_im += B0_im;
  o30_re -= A0_re;
  o30_im -= A0_im;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += B1_re;
  o21_im += B1_im;
  o31_re -= A1_re;
  o31_im -= A1_im;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += B2_re;
  o22_im += B2_im;
  o32_re -= A2_re;
  o32_im -= A2_im;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 3, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i30_re;
    a0_im = +i00_im-i30_im;
    a1_re = +i01_re-i31_re;
    a1_im = +i01_im-i31_im;
    a2_re = +i02_re-i32_re;
    a2_im = +i02_im-i32_im;
    b0_re = +i10_re+i20_re;
    b0_im = +i10_im+i20_im;
    b1_re = +i11_re+i21_re;
    b1_im = +i11_im+i21_im;
    b2_re = +i12_re+i22_re;
    b2_im = +i12_im+i22_im;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(3);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += B0_re;
  o20_im += B0_im;
  o30_re -= A0_re;
  o30_im -= A0_im;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += B1_re;
  o21_im += B1_im;
  o31_re -= A1_re;
  o31_im -= A1_im;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += B2_re;
  o22_im += B2_im;
  o32_re -= A2_re;
  o32_im -= A2_im;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[2] || x3<X3m1)) ||
     (kernel_type == EXTERIOR_KERNEL_Z && x3==X3m1) )
#endif
{
  // Projector P2+
  // 1 0 i 0 
  // 0 1 0 -i 
  // -i 0 1 0 
  // 0 i 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x3==X3m1 ? X-X3X2X1mX2X1 : X+X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x3==X3m1 ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
#endif
  
  const int ga_idx = sid % Vh;
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 4, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i20_im;
    a0_im = +i00_im+i20_re;
    a1_re = +i01_re-i21_im;
    a1_im = +i01_im+i21_re;
    a2_re = +i02_re-i22_im;
    a2_im = +i02_im+i22_re;
    b0_re = +i10_re+i30_im;
    b0_im = +i10_im-i30_re;
    b1_re = +i11_re+i31_im;
    b1_im = +i11_im-i31_re;
    b2_re = +i12_re+i32_im;
    b2_im = +i12_im-i32_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(4);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += A0_im;
  o20_im -= A0_re;
  o30_re -= B0_im;
  o30_im += B0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += A1_im;
  o21_im -= A1_re;
  o31_re -= B1_im;
  o31_im += B1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += A2_im;
  o22_im -= A2_re;
  o32_re -= B2_im;
  o32_im += B2_re;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 4, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re-i20_im;
    a0_im = +i00_im+i20_re;
    a1_re = +i01_re-i21_im;
    a1_im = +i01_im+i21_re;
    a2_re = +i02_re-i22_im;
    a2_im = +i02_im+i22_re;
    b0_re = +i10_re+i30_im;
    b0_im = +i10_im-i30_re;
    b1_re = +i11_re+i31_im;
    b1_im = +i11_im-i31_re;
    b2_re = +i12_re+i32_im;
    b2_im = +i12_im-i32_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(4);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += g00_re * a0_re;
  A0_re -= g00_im * a0_im;
  A0_re += g01_re * a1_re;
  A0_re -= g01_im * a1_im;
  A0_re += g02_re * a2_re;
  A0_re -= g02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += g00_re * a0_im;
  A0_im += g00_im * a0_re;
  A0_im += g01_re * a1_im;
  A0_im += g01_im * a1_re;
  A0_im += g02_re * a2_im;
  A0_im += g02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += g00_re * b0_re;
  B0_re -= g00_im * b0_im;
  B0_re += g01_re * b1_re;
  B0_re -= g01_im * b1_im;
  B0_re += g02_re * b2_re;
  B0_re -= g02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += g00_re * b0_im;
  B0_im += g00_im * b0_re;
  B0_im += g01_re * b1_im;
  B0_im += g01_im * b1_re;
  B0_im += g02_re * b2_im;
  B0_im += g02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += g10_re * a0_re;
  A1_re -= g10_im * a0_im;
  A1_re += g11_re * a1_re;
  A1_re -= g11_im * a1_im;
  A1_re += g12_re * a2_re;
  A1_re -= g12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += g10_re * a0_im;
  A1_im += g10_im * a0_re;
  A1_im += g11_re * a1_im;
  A1_im += g11_im * a1_re;
  A1_im += g12_re * a2_im;
  A1_im += g12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += g10_re * b0_re;
  B1_re -= g10_im * b0_im;
  B1_re += g11_re * b1_re;
  B1_re -= g11_im * b1_im;
  B1_re += g12_re * b2_re;
  B1_re -= g12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += g10_re * b0_im;
  B1_im += g10_im * b0_re;
  B1_im += g11_re * b1_im;
  B1_im += g11_im * b1_re;
  B1_im += g12_re * b2_im;
  B1_im += g12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += g20_re * a0_re;
  A2_re -= g20_im * a0_im;
  A2_re += g21_re * a1_re;
  A2_re -= g21_im * a1_im;
  A2_re += g22_re * a2_re;
  A2_re -= g22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += g20_re * a0_im;
  A2_im += g20_im * a0_re;
  A2_im += g21_re * a1_im;
  A2_im += g21_im * a1_re;
  A2_im += g22_re * a2_im;
  A2_im += g22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += g20_re * b0_re;
  B2_re -= g20_im * b0_im;
  B2_re += g21_re * b1_re;
  B2_re -= g21_im * b1_im;
  B2_re += g22_re * b2_re;
  B2_re -= g22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += g20_re * b0_im;
  B2_im += g20_im * b0_re;
  B2_im += g21_re * b1_im;
  B2_im += g21_im * b1_re;
  B2_im += g22_re * b2_im;
  B2_im += g22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re += A0_im;
  o20_im -= A0_re;
  o30_re -= B0_im;
  o30_im += B0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re += A1_im;
  o21_im -= A1_re;
  o31_re -= B1_im;
  o31_im += B1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re += A2_im;
  o22_im -= A2_re;
  o32_re -= B2_im;
  o32_im += B2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[2] || x3>0)) ||
     (kernel_type == EXTERIOR_KERNEL_Z && x3==0) )
#endif
{
  // Projector P2-
  // 1 0 -i 0 
  // 0 1 0 i 
  // i 0 1 0 
  // 0 -i 0 1 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x3==0 ? X+X3X2X1mX2X1 : X-X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x3==0 ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx % Vh : Vh+(face_idx % ghostFace[static_cast<int>(kernel_type)]));
#else
  const int ga_idx = sp_idx % Vh;
#endif
  
  if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE1TEX, 5, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i20_im;
    a0_im = +i00_im-i20_re;
    a1_re = +i01_re+i21_im;
    a1_im = +i01_im-i21_re;
    a2_re = +i02_re+i22_im;
    a2_im = +i02_im-i22_re;
    b0_re = +i10_re-i30_im;
    b0_im = +i10_im+i30_re;
    b1_re = +i11_re-i31_im;
    b1_im = +i11_im+i31_re;
    b2_re = +i12_re-i32_im;
    b2_im = +i12_im+i32_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(5);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= A0_im;
  o20_im += A0_re;
  o30_re += B0_im;
  o30_im -= B0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= A1_im;
  o21_im += A1_re;
  o31_re += B1_im;
  o31_im -= B1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= A2_im;
  o22_im += A2_re;
  o32_re += B2_im;
  o32_im -= B2_re;
  
  }
  else{
  // read gauge matrix from device memory
  READ_GAUGE_MATRIX(G, GAUGE0TEX, 5, ga_idx, ga_stride);
  
  spinorFloat a0_re, a0_im;
  spinorFloat a1_re, a1_im;
  spinorFloat a2_re, a2_im;
  spinorFloat b0_re, b0_im;
  spinorFloat b1_re, b1_im;
  spinorFloat b2_re, b2_im;
  
#ifdef MULTI_GPU
  if (kernel_type == INTERIOR_KERNEL) {
#endif
  
    // read spinor from device memory
    READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
    // project spinor into half spinors
    a0_re = +i00_re+i20_im;
    a0_im = +i00_im-i20_re;
    a1_re = +i01_re+i21_im;
    a1_im = +i01_im-i21_re;
    a2_re = +i02_re+i22_im;
    a2_im = +i02_im-i22_re;
    b0_re = +i10_re-i30_im;
    b0_im = +i10_im+i30_re;
    b1_re = +i11_re-i31_im;
    b1_im = +i11_im+i31_re;
    b2_re = +i12_re-i32_im;
    b2_im = +i12_im+i32_re;
  
#ifdef MULTI_GPU
  } else {
  
    const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
    
    // read half spinor from device memory
    READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
    
    a0_re = i00_re;  a0_im = i00_im;
    a1_re = i01_re;  a1_im = i01_im;
    a2_re = i02_re;  a2_im = i02_im;
    b0_re = i10_re;  b0_im = i10_im;
    b1_re = i11_re;  b1_im = i11_im;
    b2_re = i12_re;  b2_im = i12_im;
    
  }
#endif // MULTI_GPU
  
  // reconstruct gauge matrix
  RECONSTRUCT_GAUGE_MATRIX(5);
  
  // multiply row 0
  spinorFloat A0_re = 0;
  A0_re += gT00_re * a0_re;
  A0_re -= gT00_im * a0_im;
  A0_re += gT01_re * a1_re;
  A0_re -= gT01_im * a1_im;
  A0_re += gT02_re * a2_re;
  A0_re -= gT02_im * a2_im;
  spinorFloat A0_im = 0;
  A0_im += gT00_re * a0_im;
  A0_im += gT00_im * a0_re;
  A0_im += gT01_re * a1_im;
  A0_im += gT01_im * a1_re;
  A0_im += gT02_re * a2_im;
  A0_im += gT02_im * a2_re;
  spinorFloat B0_re = 0;
  B0_re += gT00_re * b0_re;
  B0_re -= gT00_im * b0_im;
  B0_re += gT01_re * b1_re;
  B0_re -= gT01_im * b1_im;
  B0_re += gT02_re * b2_re;
  B0_re -= gT02_im * b2_im;
  spinorFloat B0_im = 0;
  B0_im += gT00_re * b0_im;
  B0_im += gT00_im * b0_re;
  B0_im += gT01_re * b1_im;
  B0_im += gT01_im * b1_re;
  B0_im += gT02_re * b2_im;
  B0_im += gT02_im * b2_re;
  
  // multiply row 1
  spinorFloat A1_re = 0;
  A1_re += gT10_re * a0_re;
  A1_re -= gT10_im * a0_im;
  A1_re += gT11_re * a1_re;
  A1_re -= gT11_im * a1_im;
  A1_re += gT12_re * a2_re;
  A1_re -= gT12_im * a2_im;
  spinorFloat A1_im = 0;
  A1_im += gT10_re * a0_im;
  A1_im += gT10_im * a0_re;
  A1_im += gT11_re * a1_im;
  A1_im += gT11_im * a1_re;
  A1_im += gT12_re * a2_im;
  A1_im += gT12_im * a2_re;
  spinorFloat B1_re = 0;
  B1_re += gT10_re * b0_re;
  B1_re -= gT10_im * b0_im;
  B1_re += gT11_re * b1_re;
  B1_re -= gT11_im * b1_im;
  B1_re += gT12_re * b2_re;
  B1_re -= gT12_im * b2_im;
  spinorFloat B1_im = 0;
  B1_im += gT10_re * b0_im;
  B1_im += gT10_im * b0_re;
  B1_im += gT11_re * b1_im;
  B1_im += gT11_im * b1_re;
  B1_im += gT12_re * b2_im;
  B1_im += gT12_im * b2_re;
  
  // multiply row 2
  spinorFloat A2_re = 0;
  A2_re += gT20_re * a0_re;
  A2_re -= gT20_im * a0_im;
  A2_re += gT21_re * a1_re;
  A2_re -= gT21_im * a1_im;
  A2_re += gT22_re * a2_re;
  A2_re -= gT22_im * a2_im;
  spinorFloat A2_im = 0;
  A2_im += gT20_re * a0_im;
  A2_im += gT20_im * a0_re;
  A2_im += gT21_re * a1_im;
  A2_im += gT21_im * a1_re;
  A2_im += gT22_re * a2_im;
  A2_im += gT22_im * a2_re;
  spinorFloat B2_re = 0;
  B2_re += gT20_re * b0_re;
  B2_re -= gT20_im * b0_im;
  B2_re += gT21_re * b1_re;
  B2_re -= gT21_im * b1_im;
  B2_re += gT22_re * b2_re;
  B2_re -= gT22_im * b2_im;
  spinorFloat B2_im = 0;
  B2_im += gT20_re * b0_im;
  B2_im += gT20_im * b0_re;
  B2_im += gT21_re * b1_im;
  B2_im += gT21_im * b1_re;
  B2_im += gT22_re * b2_im;
  B2_im += gT22_im * b2_re;
  
  o00_re += A0_re;
  o00_im += A0_im;
  o10_re += B0_re;
  o10_im += B0_im;
  o20_re -= A0_im;
  o20_im += A0_re;
  o30_re += B0_im;
  o30_im -= B0_re;
  
  o01_re += A1_re;
  o01_im += A1_im;
  o11_re += B1_re;
  o11_im += B1_im;
  o21_re -= A1_im;
  o21_im += A1_re;
  o31_re += B1_im;
  o31_im -= B1_re;
  
  o02_re += A2_re;
  o02_im += A2_im;
  o12_re += B2_re;
  o12_im += B2_im;
  o22_re -= A2_im;
  o22_im += A2_re;
  o32_re += B2_im;
  o32_im -= B2_re;
  
  }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[3] || x4<X4m1)) ||
     (kernel_type == EXTERIOR_KERNEL_T && x4==X4m1) )
#endif
{
  // Projector P3+
  // 2 0 0 0 
  // 0 2 0 0 
  // 0 0 0 0 
  // 0 0 0 0 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x4==X4m1 ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x4==X4m1 ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
#endif
  
  const int ga_idx = sid % Vh;
  
  if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)
  {
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i00_re;
      a0_im = +2*i00_im;
      a1_re = +2*i01_re;
      a1_im = +2*i01_im;
      a2_re = +2*i02_re;
      a2_im = +2*i02_im;
      b0_re = +2*i10_re;
      b0_im = +2*i10_im;
      b1_re = +2*i11_re;
      b1_im = +2*i11_im;
      b2_re = +2*i12_re;
      b2_im = +2*i12_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    
  } else {
    if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 6, ga_idx, ga_stride);
    
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i00_re;
      a0_im = +2*i00_im;
      a1_re = +2*i01_re;
      a1_im = +2*i01_im;
      a2_re = +2*i02_re;
      a2_im = +2*i02_im;
      b0_re = +2*i10_re;
      b0_im = +2*i10_im;
      b1_re = +2*i11_re;
      b1_im = +2*i11_im;
      b2_re = +2*i12_re;
      b2_im = +2*i12_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(6);
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += g00_re * a0_re;
    A0_re -= g00_im * a0_im;
    A0_re += g01_re * a1_re;
    A0_re -= g01_im * a1_im;
    A0_re += g02_re * a2_re;
    A0_re -= g02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += g00_re * a0_im;
    A0_im += g00_im * a0_re;
    A0_im += g01_re * a1_im;
    A0_im += g01_im * a1_re;
    A0_im += g02_re * a2_im;
    A0_im += g02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += g00_re * b0_re;
    B0_re -= g00_im * b0_im;
    B0_re += g01_re * b1_re;
    B0_re -= g01_im * b1_im;
    B0_re += g02_re * b2_re;
    B0_re -= g02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += g00_re * b0_im;
    B0_im += g00_im * b0_re;
    B0_im += g01_re * b1_im;
    B0_im += g01_im * b1_re;
    B0_im += g02_re * b2_im;
    B0_im += g02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += g10_re * a0_re;
    A1_re -= g10_im * a0_im;
    A1_re += g11_re * a1_re;
    A1_re -= g11_im * a1_im;
    A1_re += g12_re * a2_re;
    A1_re -= g12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += g10_re * a0_im;
    A1_im += g10_im * a0_re;
    A1_im += g11_re * a1_im;
    A1_im += g11_im * a1_re;
    A1_im += g12_re * a2_im;
    A1_im += g12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += g10_re * b0_re;
    B1_re -= g10_im * b0_im;
    B1_re += g11_re * b1_re;
    B1_re -= g11_im * b1_im;
    B1_re += g12_re * b2_re;
    B1_re -= g12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += g10_re * b0_im;
    B1_im += g10_im * b0_re;
    B1_im += g11_re * b1_im;
    B1_im += g11_im * b1_re;
    B1_im += g12_re * b2_im;
    B1_im += g12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += g20_re * a0_re;
    A2_re -= g20_im * a0_im;
    A2_re += g21_re * a1_re;
    A2_re -= g21_im * a1_im;
    A2_re += g22_re * a2_re;
    A2_re -= g22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += g20_re * a0_im;
    A2_im += g20_im * a0_re;
    A2_im += g21_re * a1_im;
    A2_im += g21_im * a1_re;
    A2_im += g22_re * a2_im;
    A2_im += g22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += g20_re * b0_re;
    B2_re -= g20_im * b0_im;
    B2_re += g21_re * b1_re;
    B2_re -= g21_im * b1_im;
    B2_re += g22_re * b2_re;
    B2_re -= g22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += g20_re * b0_im;
    B2_im += g20_im * b0_re;
    B2_im += g21_re * b1_im;
    B2_im += g21_im * b1_re;
    B2_im += g22_re * b2_im;
    B2_im += g22_im * b2_re;
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    
  }else{
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 6, ga_idx, ga_stride);
    
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i00_re;
      a0_im = +2*i00_im;
      a1_re = +2*i01_re;
      a1_im = +2*i01_im;
      a2_re = +2*i02_re;
      a2_im = +2*i02_im;
      b0_re = +2*i10_re;
      b0_im = +2*i10_im;
      b1_re = +2*i11_re;
      b1_im = +2*i11_im;
      b2_re = +2*i12_re;
      b2_im = +2*i12_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(6);
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += g00_re * a0_re;
    A0_re -= g00_im * a0_im;
    A0_re += g01_re * a1_re;
    A0_re -= g01_im * a1_im;
    A0_re += g02_re * a2_re;
    A0_re -= g02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += g00_re * a0_im;
    A0_im += g00_im * a0_re;
    A0_im += g01_re * a1_im;
    A0_im += g01_im * a1_re;
    A0_im += g02_re * a2_im;
    A0_im += g02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += g00_re * b0_re;
    B0_re -= g00_im * b0_im;
    B0_re += g01_re * b1_re;
    B0_re -= g01_im * b1_im;
    B0_re += g02_re * b2_re;
    B0_re -= g02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += g00_re * b0_im;
    B0_im += g00_im * b0_re;
    B0_im += g01_re * b1_im;
    B0_im += g01_im * b1_re;
    B0_im += g02_re * b2_im;
    B0_im += g02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += g10_re * a0_re;
    A1_re -= g10_im * a0_im;
    A1_re += g11_re * a1_re;
    A1_re -= g11_im * a1_im;
    A1_re += g12_re * a2_re;
    A1_re -= g12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += g10_re * a0_im;
    A1_im += g10_im * a0_re;
    A1_im += g11_re * a1_im;
    A1_im += g11_im * a1_re;
    A1_im += g12_re * a2_im;
    A1_im += g12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += g10_re * b0_re;
    B1_re -= g10_im * b0_im;
    B1_re += g11_re * b1_re;
    B1_re -= g11_im * b1_im;
    B1_re += g12_re * b2_re;
    B1_re -= g12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += g10_re * b0_im;
    B1_im += g10_im * b0_re;
    B1_im += g11_re * b1_im;
    B1_im += g11_im * b1_re;
    B1_im += g12_re * b2_im;
    B1_im += g12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += g20_re * a0_re;
    A2_re -= g20_im * a0_im;
    A2_re += g21_re * a1_re;
    A2_re -= g21_im * a1_im;
    A2_re += g22_re * a2_re;
    A2_re -= g22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += g20_re * a0_im;
    A2_im += g20_im * a0_re;
    A2_im += g21_re * a1_im;
    A2_im += g21_im * a1_re;
    A2_im += g22_re * a2_im;
    A2_im += g22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += g20_re * b0_re;
    B2_re -= g20_im * b0_im;
    B2_re += g21_re * b1_re;
    B2_re -= g21_im * b1_im;
    B2_re += g22_re * b2_re;
    B2_re -= g22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += g20_re * b0_im;
    B2_im += g20_im * b0_re;
    B2_im += g21_re * b1_im;
    B2_im += g21_im * b1_re;
    B2_im += g22_re * b2_im;
    B2_im += g22_im * b2_re;
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    
  }
 }
}

#ifdef MULTI_GPU
if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim[3] || x4>0)) ||
     (kernel_type == EXTERIOR_KERNEL_T && x4==0) )
#endif
{
  // Projector P3-
  // 0 0 0 0 
  // 0 0 0 0 
  // 0 0 2 0 
  // 0 0 0 2 
  
#ifdef MULTI_GPU
  const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? (x4==0 ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1 :
    face_idx + param.ghostOffset[static_cast<int>(kernel_type)];
#else
  const int sp_idx = (x4==0 ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
#endif
  
#ifdef MULTI_GPU
  const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx % Vh : Vh+(face_idx % ghostFace[static_cast<int>(kernel_type)]));
#else
  const int ga_idx = sp_idx % Vh;
#endif
  
  if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)
  {
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i20_re;
      a0_im = +2*i20_im;
      a1_re = +2*i21_re;
      a1_im = +2*i21_im;
      a2_re = +2*i22_re;
      a2_im = +2*i22_im;
      b0_re = +2*i30_re;
      b0_im = +2*i30_im;
      b1_re = +2*i31_re;
      b1_im = +2*i31_im;
      b2_re = +2*i32_re;
      b2_im = +2*i32_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // identity gauge matrix
    spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
    spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
    spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
    spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
    spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
    spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
    
    o20_re += A0_re;
    o20_im += A0_im;
    o30_re += B0_re;
    o30_im += B0_im;
    
    o21_re += A1_re;
    o21_im += A1_im;
    o31_re += B1_re;
    o31_im += B1_im;
    
    o22_re += A2_re;
    o22_im += A2_im;
    o32_re += B2_re;
    o32_im += B2_im;
    
  } else {
    if(!( (boundaryCrossings-boundaryCrossings4d) % 2)){
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 7, ga_idx, ga_stride);
    
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i20_re;
      a0_im = +2*i20_im;
      a1_re = +2*i21_re;
      a1_im = +2*i21_im;
      a2_re = +2*i22_re;
      a2_im = +2*i22_im;
      b0_re = +2*i30_re;
      b0_im = +2*i30_im;
      b1_re = +2*i31_re;
      b1_im = +2*i31_im;
      b2_re = +2*i32_re;
      b2_im = +2*i32_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(7);
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += gT00_re * a0_re;
    A0_re -= gT00_im * a0_im;
    A0_re += gT01_re * a1_re;
    A0_re -= gT01_im * a1_im;
    A0_re += gT02_re * a2_re;
    A0_re -= gT02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += gT00_re * a0_im;
    A0_im += gT00_im * a0_re;
    A0_im += gT01_re * a1_im;
    A0_im += gT01_im * a1_re;
    A0_im += gT02_re * a2_im;
    A0_im += gT02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += gT00_re * b0_re;
    B0_re -= gT00_im * b0_im;
    B0_re += gT01_re * b1_re;
    B0_re -= gT01_im * b1_im;
    B0_re += gT02_re * b2_re;
    B0_re -= gT02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += gT00_re * b0_im;
    B0_im += gT00_im * b0_re;
    B0_im += gT01_re * b1_im;
    B0_im += gT01_im * b1_re;
    B0_im += gT02_re * b2_im;
    B0_im += gT02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += gT10_re * a0_re;
    A1_re -= gT10_im * a0_im;
    A1_re += gT11_re * a1_re;
    A1_re -= gT11_im * a1_im;
    A1_re += gT12_re * a2_re;
    A1_re -= gT12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += gT10_re * a0_im;
    A1_im += gT10_im * a0_re;
    A1_im += gT11_re * a1_im;
    A1_im += gT11_im * a1_re;
    A1_im += gT12_re * a2_im;
    A1_im += gT12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += gT10_re * b0_re;
    B1_re -= gT10_im * b0_im;
    B1_re += gT11_re * b1_re;
    B1_re -= gT11_im * b1_im;
    B1_re += gT12_re * b2_re;
    B1_re -= gT12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += gT10_re * b0_im;
    B1_im += gT10_im * b0_re;
    B1_im += gT11_re * b1_im;
    B1_im += gT11_im * b1_re;
    B1_im += gT12_re * b2_im;
    B1_im += gT12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += gT20_re * a0_re;
    A2_re -= gT20_im * a0_im;
    A2_re += gT21_re * a1_re;
    A2_re -= gT21_im * a1_im;
    A2_re += gT22_re * a2_re;
    A2_re -= gT22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += gT20_re * a0_im;
    A2_im += gT20_im * a0_re;
    A2_im += gT21_re * a1_im;
    A2_im += gT21_im * a1_re;
    A2_im += gT22_re * a2_im;
    A2_im += gT22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += gT20_re * b0_re;
    B2_re -= gT20_im * b0_im;
    B2_re += gT21_re * b1_re;
    B2_re -= gT21_im * b1_im;
    B2_re += gT22_re * b2_re;
    B2_re -= gT22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += gT20_re * b0_im;
    B2_im += gT20_im * b0_re;
    B2_im += gT21_re * b1_im;
    B2_im += gT21_im * b1_re;
    B2_im += gT22_re * b2_im;
    B2_im += gT22_im * b2_re;
    
    o20_re += A0_re;
    o20_im += A0_im;
    o30_re += B0_re;
    o30_im += B0_im;
    
    o21_re += A1_re;
    o21_im += A1_im;
    o31_re += B1_re;
    o31_im += B1_im;
    
    o22_re += A2_re;
    o22_im += A2_im;
    o32_re += B2_re;
    o32_im += B2_im;
    
  }else{
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 7, ga_idx, ga_stride);
    
    spinorFloat a0_re, a0_im;
    spinorFloat a1_re, a1_im;
    spinorFloat a2_re, a2_im;
    spinorFloat b0_re, b0_im;
    spinorFloat b1_re, b1_im;
    spinorFloat b2_re, b2_im;
    
#ifdef MULTI_GPU
    if (kernel_type == INTERIOR_KERNEL) {
#endif
    
      // read spinor from device memory
      READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      // project spinor into half spinors
      a0_re = +2*i20_re;
      a0_im = +2*i20_im;
      a1_re = +2*i21_re;
      a1_im = +2*i21_im;
      a2_re = +2*i22_re;
      a2_im = +2*i22_im;
      b0_re = +2*i30_re;
      b0_im = +2*i30_im;
      b1_re = +2*i31_re;
      b1_im = +2*i31_im;
      b2_re = +2*i32_re;
      b2_im = +2*i32_im;
    
#ifdef MULTI_GPU
    } else {
    
      const int sp_stride_pad = Ls * ghostFace[static_cast<int>(kernel_type)];
      const int t_proj_scale = TPROJSCALE;
      
      // read half spinor from device memory
      READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);
      
      a0_re = t_proj_scale*i00_re;  a0_im = t_proj_scale*i00_im;
      a1_re = t_proj_scale*i01_re;  a1_im = t_proj_scale*i01_im;
      a2_re = t_proj_scale*i02_re;  a2_im = t_proj_scale*i02_im;
      b0_re = t_proj_scale*i10_re;  b0_im = t_proj_scale*i10_im;
      b1_re = t_proj_scale*i11_re;  b1_im = t_proj_scale*i11_im;
      b2_re = t_proj_scale*i12_re;  b2_im = t_proj_scale*i12_im;
      
    }
#endif // MULTI_GPU
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(7);
    
    // multiply row 0
    spinorFloat A0_re = 0;
    A0_re += gT00_re * a0_re;
    A0_re -= gT00_im * a0_im;
    A0_re += gT01_re * a1_re;
    A0_re -= gT01_im * a1_im;
    A0_re += gT02_re * a2_re;
    A0_re -= gT02_im * a2_im;
    spinorFloat A0_im = 0;
    A0_im += gT00_re * a0_im;
    A0_im += gT00_im * a0_re;
    A0_im += gT01_re * a1_im;
    A0_im += gT01_im * a1_re;
    A0_im += gT02_re * a2_im;
    A0_im += gT02_im * a2_re;
    spinorFloat B0_re = 0;
    B0_re += gT00_re * b0_re;
    B0_re -= gT00_im * b0_im;
    B0_re += gT01_re * b1_re;
    B0_re -= gT01_im * b1_im;
    B0_re += gT02_re * b2_re;
    B0_re -= gT02_im * b2_im;
    spinorFloat B0_im = 0;
    B0_im += gT00_re * b0_im;
    B0_im += gT00_im * b0_re;
    B0_im += gT01_re * b1_im;
    B0_im += gT01_im * b1_re;
    B0_im += gT02_re * b2_im;
    B0_im += gT02_im * b2_re;
    
    // multiply row 1
    spinorFloat A1_re = 0;
    A1_re += gT10_re * a0_re;
    A1_re -= gT10_im * a0_im;
    A1_re += gT11_re * a1_re;
    A1_re -= gT11_im * a1_im;
    A1_re += gT12_re * a2_re;
    A1_re -= gT12_im * a2_im;
    spinorFloat A1_im = 0;
    A1_im += gT10_re * a0_im;
    A1_im += gT10_im * a0_re;
    A1_im += gT11_re * a1_im;
    A1_im += gT11_im * a1_re;
    A1_im += gT12_re * a2_im;
    A1_im += gT12_im * a2_re;
    spinorFloat B1_re = 0;
    B1_re += gT10_re * b0_re;
    B1_re -= gT10_im * b0_im;
    B1_re += gT11_re * b1_re;
    B1_re -= gT11_im * b1_im;
    B1_re += gT12_re * b2_re;
    B1_re -= gT12_im * b2_im;
    spinorFloat B1_im = 0;
    B1_im += gT10_re * b0_im;
    B1_im += gT10_im * b0_re;
    B1_im += gT11_re * b1_im;
    B1_im += gT11_im * b1_re;
    B1_im += gT12_re * b2_im;
    B1_im += gT12_im * b2_re;
    
    // multiply row 2
    spinorFloat A2_re = 0;
    A2_re += gT20_re * a0_re;
    A2_re -= gT20_im * a0_im;
    A2_re += gT21_re * a1_re;
    A2_re -= gT21_im * a1_im;
    A2_re += gT22_re * a2_re;
    A2_re -= gT22_im * a2_im;
    spinorFloat A2_im = 0;
    A2_im += gT20_re * a0_im;
    A2_im += gT20_im * a0_re;
    A2_im += gT21_re * a1_im;
    A2_im += gT21_im * a1_re;
    A2_im += gT22_re * a2_im;
    A2_im += gT22_im * a2_re;
    spinorFloat B2_re = 0;
    B2_re += gT20_re * b0_re;
    B2_re -= gT20_im * b0_im;
    B2_re += gT21_re * b1_re;
    B2_re -= gT21_im * b1_im;
    B2_re += gT22_re * b2_re;
    B2_re -= gT22_im * b2_im;
    spinorFloat B2_im = 0;
    B2_im += gT20_re * b0_im;
    B2_im += gT20_im * b0_re;
    B2_im += gT21_re * b1_im;
    B2_im += gT21_im * b1_re;
    B2_im += gT22_re * b2_im;
    B2_im += gT22_im * b2_re;
    
    o20_re += A0_re;
    o20_im += A0_im;
    o30_re += B0_re;
    o30_im += B0_im;
    
    o21_re += A1_re;
    o21_im += A1_im;
    o31_re += B1_re;
    o31_im += B1_im;
    
    o22_re += A2_re;
    o22_im += A2_im;
    o32_re += B2_re;
    o32_im += B2_im;
    
  }
 }
}

#ifdef MULTI_GPU
if(kernel_type == INTERIOR_KERNEL)
#endif
//J  ----------------------------------
//J  --- DWF code for 5th dimension ---
//J  ----------------------------------
//
//J  Begin scope.
{ 
   //J  TODO  Insert/check handler for s-direction here.

   //J  Decided to not change to chiral basis.  Then:
   // 2 P_+ = 2 P_R =  1  1
   //                  1  1 
   // --- Begin right-handed spinor projection. ---
   {
      //J  We are right-handed, so for the dslash_dagger we hop backwards.  If we are at 
      //J  boundary in s-direction, special
      //J  things will need to be done.  xs is defined in dslash_dagger_core_ante.h.
      //J  See near Line 328.  N is the 4d volume; cf. quda.h. 
      //J  Cf. hand-written notes 8/6/09 for check of logic.
      //J  The logic sets xs to the s-coordinate of the output
      //J  spinor, which is accumulated by this thread.
      //J  I.e., it uses the thread index to determine xs.
      int sp_idx = ((xs==0) ? X+(Ls-1)*2*Vh : X-2*Vh) / 2;
      // --- Read spinor from device memory. ---
      //J  Q.  How does it know which direction to hop in?  
      //J  A.  It uses sp_idx as the origin and picks up 0*Vh_5d ... 5*Vh_5d
      //J      offsets in the READ_SPINOR_UP that is below.
      //J      This has to do with the "concurrency" optimization.
      //J  Q.  Where does Vh_5d get set and does it know about the dwf
      //J      modification?  Does it care?
      //J
      //
      READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
      
      if (xs != 0) {
         //J  OK, now the input spinor should be at:
         //J     0 < s <= Ls-1
         //
         //J  Project spinor into half spinors, i.e., this is the term
         //J     " + 2 P_R psi(s-1) "
                  
         //J  ------------------------------------
         //J  --- Dirac index 0, Colors 0,1,2. ---
         //J  ------------------------------------
         //J  dagger takes P_R instead of P_L
         o00_re += i00_re+i20_re;  //ok
         o00_im += i00_im+i20_im;  //ok
         o01_re += i01_re+i21_re;  //ok  
         o01_im += i01_im+i21_im;  //ok
         o02_re += i02_re+i22_re;  //ok
         o02_im += i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 1, Colors 0,1,2.  ---
         //J  -------------------------------------
         o10_re += i10_re+i30_re;  //ok
         o10_im += i10_im+i30_im;  //ok
         o11_re += i11_re+i31_re;  //ok
         o11_im += i11_im+i31_im;  //ok
         o12_re += i12_re+i32_re;  //ok
         o12_im += i12_im+i32_im;  //ok
         
         //J  ------------------------------------
         //J  --- Dirac index 2, Colors 0,1,2. ---
         //J  ------------------------------------
         o20_re += i00_re+i20_re;  //ok
         o20_im += i00_im+i20_im;  //ok
         o21_re += i01_re+i21_re;  //ok
         o21_im += i01_im+i21_im;  //ok
         o22_re += i02_re+i22_re;  //ok
         o22_im += i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 3, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o30_re += i10_re+i30_re;  //ok
         o30_im += i10_im+i30_im;  //ok
         // color 1 (second index)
         o31_re += i11_re+i31_re;  //ok
         o31_im += i11_im+i31_im;  //ok
         // color 2 (second index)
         o32_re += i12_re+i32_re;  //ok
         o32_im += i12_im+i32_im;  //ok

      } // End (x,0) < (x,s) <= (x,Ls-1).
      else {
         //J  LH boundary s=0, backwards hop to Ls-1.
         //J  Term to add:  -mferm*P_R*psi(x,Ls-1)
         //J  With any luck, sp_idx is linear equiv. to "(x,Ls-1)"
         //J  Above, we set:
         //J     sp_idx= (X+(Ls-1)*X4*X3*X2*X1)/2    (*).
         //J  efs:  do some case examples where xs=0 comes out of
         //J  dslash_ante_core.h procedure, and check that sp_idx is
         //J  really coming out correct (and in permissable range)
         //J  in the operation (*).
         //J  We need mferm to get passed.  A modification
         //J  was made to DD_PARAM2 in the C preprocessing file
         //J  dslash_dwf_def.h, adding
         //J  an extra argument to the kernel declarations.
         //
         //J  --- Dirac index 0, Colors 0,1,2.  ---
         // color 0 (second index)
         o00_re += -mferm*(i00_re+i20_re); //ok
         o00_im += -mferm*(i00_im+i20_im); //ok
         // color 1
         o01_re += -mferm*(i01_re+i21_re); //ok
         o01_im += -mferm*(i01_im+i21_im); //ok
         // color 2
         o02_re += -mferm*(i02_re+i22_re); //ok
         o02_im += -mferm*(i02_im+i22_im); //ok

         //J  --- Dirac index 1, Colors 0,1,2.  ---
         // color 0
         o10_re += -mferm*(i10_re+i30_re); //ok
         o10_im += -mferm*(i10_im+i30_im); //ok
         // color 1
         o11_re += -mferm*(i11_re+i31_re); //ok
         o11_im += -mferm*(i11_im+i31_im); //ok
         // color 2
         o12_re += -mferm*(i12_re+i32_re); //ok
         o12_im += -mferm*(i12_im+i32_im); //ok
         
         //J  --- Dirac index 2, Colors 0,1,2.  ---
         // color 0 (second index)
         o20_re += -mferm*(i00_re+i20_re); //ok
         o20_im += -mferm*(i00_im+i20_im); //ok
         // color 1
         o21_re += -mferm*(i01_re+i21_re); //ok
         o21_im += -mferm*(i01_im+i21_im); //ok
         // color 2
         o22_re += -mferm*(i02_re+i22_re); //ok
         o22_im += -mferm*(i02_im+i22_im); //ok

         //J  --- Dirac index 3, Colors 0,1,2.  ---
         // color 0
         o30_re += -mferm*(i10_re+i30_re); //ok
         o30_im += -mferm*(i10_im+i30_im); //ok
         // color 1
         o31_re += -mferm*(i11_re+i31_re); //ok
         o31_im += -mferm*(i11_im+i31_im); //ok
         // color 2
         o32_re += -mferm*(i12_re+i32_re); //ok
         o32_im += -mferm*(i12_im+i32_im); //ok
                  
      }  // End (x,s)=(x,0)
   }  
   // --- End of right-handed spinor projection. ---

   // In the GPU Dirac matrix basis:
   // 2 P_- = 2 P_L =  1 -1
   //                 -1  1 
   //J  Begin scope for 2 P_L projection of forward-hopped spinor.
   {
      //J  For P_L spinor, dslash_dagger, we hop forwards.
      
      //J  This bit mimics what is done for x4==X4-1 in dslash_core_ante.h.
      //J  
      //J  Checked logic w/ case examples.
      //J  Cf. hand-written notes 8/6/09 for check of logic.
      int sp_idx = ((xs==(Ls-1)) ? X-(Ls-1)*2*Vh : X+2*Vh) / 2;
         
      //J  Read spinor from device memory.
      //
      READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);

      // 
      //
      if ( xs < (Ls-1) ) {
         //J  Case of not at RH boundary.   Then we just do += P_L psi(s+1).
         
         //J  ------------------------------------
         //J  --- Dirac index 0, Colors 0,1,2. ---
         //J  ------------------------------------
         // color 0 (second index)
         o00_re += i00_re-i20_re;  //ok
         o00_im += i00_im-i20_im;  //ok
         // color 1 (second index)
         o01_re += i01_re-i21_re;  //ok
         o01_im += i01_im-i21_im;  //ok
         // color 2 (second index)
         o02_re += i02_re-i22_re;  //ok
         o02_im += i02_im-i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 1, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o10_re += i10_re-i30_re;  //ok
         o10_im += i10_im-i30_im;  //ok
         // color 1 (second index)
         o11_re += i11_re-i31_re;  //ok
         o11_im += i11_im-i31_im;  //ok
         // color 2 (second index)
         o12_re += i12_re-i32_re;  //ok
         o12_im += i12_im-i32_im;  //ok
         
         //J  ------------------------------------
         //J  --- Dirac index 2, Colors 0,1,2. ---
         //J  ------------------------------------
         // color 0 (second index)
         o20_re += -i00_re+i20_re;  //ok
         o20_im += -i00_im+i20_im;  //ok
         // color 1 (second index)
         o21_re += -i01_re+i21_re;  //ok
         o21_im += -i01_im+i21_im;  //ok
         // color 2 (second index)
         o22_re += -i02_re+i22_re;  //ok
         o22_im += -i02_im+i22_im;  //ok
         
         //J  -------------------------------------
         //J  --- Dirac index 3, Colors 0,1,2.  ---
         //J  -------------------------------------
         // color 0 (second index)
         o30_re += -i10_re+i30_re;  //ok
         o30_im += -i10_im+i30_im;  //ok
         // color 1 (second index)
         o31_re += -i11_re+i31_re;  //ok
         o31_im += -i11_im+i31_im;  //ok
         // color 2 (second index)
         o32_re += -i12_re+i32_re;  //ok
         o32_im += -i12_im+i32_im;  //ok

      } // End (x,0) <= (x,s) < (x,Ls-1).
      else {
         //J  RH boundary s=Ls-1, forwards hop to s=0.
         //J  Term to add:  -mferm*P_L*psi(x,0)
         
         //J  --- Dirac index 0, Colors 0,1,2.  ---
         // color 0 (second index)
         o00_re += -mferm*(i00_re-i20_re);  //ok
         o00_im += -mferm*(i00_im-i20_im);  //ok
         // color 1
         o01_re += -mferm*(i01_re-i21_re);  //ok
         o01_im += -mferm*(i01_im-i21_im);  //ok
         // color 2
         o02_re += -mferm*(i02_re-i22_re);  //ok
         o02_im += -mferm*(i02_im-i22_im);  //ok

         //J  --- Dirac index 1, Colors 0,1,2.  ---
         // color 0
         o10_re += -mferm*(i10_re-i30_re);  //ok
         o10_im += -mferm*(i10_im-i30_im);  //ok
         // color 1
         o11_re += -mferm*(i11_re-i31_re);  //ok
         o11_im += -mferm*(i11_im-i31_im);  //ok
         // color 2
         o12_re += -mferm*(i12_re-i32_re);  //ok
         o12_im += -mferm*(i12_im-i32_im);  //ok
         
         //J  --- Dirac index 2, Colors 0,1,2.  ---
         // color 0 (second index)
         o20_re += -mferm*(-i00_re+i20_re);  //ok
         o20_im += -mferm*(-i00_im+i20_im);  //ok
         // color 1
         o21_re += -mferm*(-i01_re+i21_re);  //ok
         o21_im += -mferm*(-i01_im+i21_im);  //ok
         // color 2
         o22_re += -mferm*(-i02_re+i22_re);  //ok
         o22_im += -mferm*(-i02_im+i22_im);  //ok

         //J  --- Dirac index 3, Colors 0,1,2.  ---
         // color 0
         o30_re += -mferm*(-i10_re+i30_re);  //ok
         o30_im += -mferm*(-i10_im+i30_im);  //ok
         // color 1
         o31_re += -mferm*(-i11_re+i31_re);  //ok
         o31_im += -mferm*(-i11_im+i31_im);  //ok
         // color 2
         o32_re += -mferm*(-i12_re+i32_re);  //ok
         o32_im += -mferm*(-i12_im+i32_im);  //ok
         //
      }  // End (x,s)=(x,Ls-1)
   }
   // -----  end dwf s-direction ----
   
}  // end s-direction block


#if (defined MULTI_GPU) && (defined DSLASH_XPAY)

int incomplete = 0; // Have all 8 contributions been computed for this site?

switch(kernel_type) { // intentional fall-through
case INTERIOR_KERNEL:
  incomplete = incomplete || (param.commDim[3] && (x4==0 || x4==X4m1));
case EXTERIOR_KERNEL_T:
  incomplete = incomplete || (param.commDim[2] && (x3==0 || x3==X3m1));
case EXTERIOR_KERNEL_Z:
  incomplete = incomplete || (param.commDim[1] && (x2==0 || x2==X2m1));
case EXTERIOR_KERNEL_Y:
  incomplete = incomplete || (param.commDim[0] && (x1==0 || x1==X1m1));
}

if (!incomplete)
#endif // MULTI_GPU
{
  
#ifdef DSLASH_XPAY
  
  READ_ACCUM(ACCUMTEX, sp_stride)
  
#ifdef SPINOR_DOUBLE
  o00_re = a*o00_re + accum0.x;
  o00_im = a*o00_im + accum0.y;
  o01_re = a*o01_re + accum1.x;
  o01_im = a*o01_im + accum1.y;
  o02_re = a*o02_re + accum2.x;
  o02_im = a*o02_im + accum2.y;
  o10_re = a*o10_re + accum3.x;
  o10_im = a*o10_im + accum3.y;
  o11_re = a*o11_re + accum4.x;
  o11_im = a*o11_im + accum4.y;
  o12_re = a*o12_re + accum5.x;
  o12_im = a*o12_im + accum5.y;
  o20_re = a*o20_re + accum6.x;
  o20_im = a*o20_im + accum6.y;
  o21_re = a*o21_re + accum7.x;
  o21_im = a*o21_im + accum7.y;
  o22_re = a*o22_re + accum8.x;
  o22_im = a*o22_im + accum8.y;
  o30_re = a*o30_re + accum9.x;
  o30_im = a*o30_im + accum9.y;
  o31_re = a*o31_re + accum10.x;
  o31_im = a*o31_im + accum10.y;
  o32_re = a*o32_re + accum11.x;
  o32_im = a*o32_im + accum11.y;
#else
  o00_re = a*o00_re + accum0.x;
  o00_im = a*o00_im + accum0.y;
  o01_re = a*o01_re + accum0.z;
  o01_im = a*o01_im + accum0.w;
  o02_re = a*o02_re + accum1.x;
  o02_im = a*o02_im + accum1.y;
  o10_re = a*o10_re + accum1.z;
  o10_im = a*o10_im + accum1.w;
  o11_re = a*o11_re + accum2.x;
  o11_im = a*o11_im + accum2.y;
  o12_re = a*o12_re + accum2.z;
  o12_im = a*o12_im + accum2.w;
  o20_re = a*o20_re + accum3.x;
  o20_im = a*o20_im + accum3.y;
  o21_re = a*o21_re + accum3.z;
  o21_im = a*o21_im + accum3.w;
  o22_re = a*o22_re + accum4.x;
  o22_im = a*o22_im + accum4.y;
  o30_re = a*o30_re + accum4.z;
  o30_im = a*o30_im + accum4.w;
  o31_re = a*o31_re + accum5.x;
  o31_im = a*o31_im + accum5.y;
  o32_re = a*o32_re + accum5.z;
  o32_im = a*o32_im + accum5.w;
#endif // SPINOR_DOUBLE
  
#endif // DSLASH_XPAY
}

// write spinor field back to device memory
WRITE_SPINOR(sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef A_re
#undef A_im

#undef g00_re
#undef g00_im
#undef g01_re
#undef g01_im
#undef g02_re
#undef g02_im
#undef g10_re
#undef g10_im
#undef g11_re
#undef g11_im
#undef g12_re
#undef g12_im
#undef g20_re
#undef g20_im
#undef g21_re
#undef g21_im
#undef g22_re
#undef g22_im

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im
#undef i10_re
#undef i10_im
#undef i11_re
#undef i11_im
#undef i12_re
#undef i12_im
#undef i20_re
#undef i20_im
#undef i21_re
#undef i21_im
#undef i22_re
#undef i22_im
#undef i30_re
#undef i30_im
#undef i31_re
#undef i31_im
#undef i32_re
#undef i32_im




