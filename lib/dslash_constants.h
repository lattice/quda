__constant__ int X1h;
__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;

__constant__ int X1m1;
__constant__ int X2m1;
__constant__ int X3m1;
__constant__ int X4m1;

__constant__ int X2X1mX1;
__constant__ int X3X2X1mX2X1;
__constant__ int X4X3X2X1mX3X2X1;
__constant__ int X4X3X2X1hmX3X2X1h;

__constant__ int X2X1;
__constant__ int X3X2X1;

__constant__ int Vh;
__constant__ int Vs;
__constant__ int sp_body_stride;
__constant__ int ga_stride;
__constant__ int cl_stride;

__constant__ int gauge_fixed;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;

__constant__ float2 An2;
__constant__ float2 TB2;
__constant__ float2 No2;

// an int to decide what we are doing
// value = 0 -> first Vs sites (TFace 1)
// value = 1..Nt-1 -> body
// value = Nt-1 -> last Tface
__constant__ int site_offset;

// Are we processor 0 in time?
__constant__ bool Pt0;

// Are we processor Nt-1 in time?
__constant__ bool PtNm1;

// Constants required for doing bulk vs faces
//__constant__ short2 tLocate;
//__constant__ int threads;
//__constant__ int4 multi;
