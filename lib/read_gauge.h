#define FAST_INT_DIVIDE(a, b) ( a/b )

// Performs complex addition
#define COMPLEX_ADD_TO(a, b)			\
  a##_re += b##_re,				\
  a##_im += b##_im

#define COMPLEX_PRODUCT(a, b, c)		\
  a##_re = b##_re*c##_re - b##_im*c##_im,	\
  a##_im = b##_re*c##_im + b##_im*c##_re

#define COMPLEX_CONJUGATE_PRODUCT(a, b, c)	\
  a##_re = b##_re*c##_re - b##_im*c##_im,	\
  a##_im = -b##_re*c##_im - b##_im*c##_re

// Performs a complex dot product
#define COMPLEX_DOT_PRODUCT(a, b, c)	        \
  a##_re = b##_re*c##_re + b##_im*c##_im,	\
  a##_im = b##_re*c##_im - b##_im*c##_re

// Performs a complex norm
#define COMPLEX_NORM(a, b)			\
  a = b##_re*b##_re + b##_im*b##_im

#define ACC_COMPLEX_PROD(a, b, c)		\
  a##_re += b##_re*c##_re - b##_im*c##_im,	\
  a##_im += b##_re*c##_im + b##_im*c##_re

// Performs the complex conjugated accumulation: a += b* c*
#define ACC_CONJ_PROD(a, b, c) \
    a##_re += b##_re * c##_re - b##_im * c##_im, \
    a##_im -= b##_re * c##_im + b##_im * c##_re

#define READ_GAUGE_MATRIX_12_DOUBLE(gauge, dir) \
  double2 G0 = fetch_double2((gauge), ga_idx + ((dir/2)*6+0)*ga_stride);	\
  double2 G1 = fetch_double2((gauge), ga_idx + ((dir/2)*6+1)*ga_stride);	\
  double2 G2 = fetch_double2((gauge), ga_idx + ((dir/2)*6+2)*ga_stride);	\
  double2 G3 = fetch_double2((gauge), ga_idx + ((dir/2)*6+3)*ga_stride);	\
  double2 G4 = fetch_double2((gauge), ga_idx + ((dir/2)*6+4)*ga_stride);	\
  double2 G5 = fetch_double2((gauge), ga_idx + ((dir/2)*6+5)*ga_stride);	\
  double2 G6 = make_double2(0,0);					\
  double2 G7 = make_double2(0,0);					\
  double2 G8 = make_double2(0,0);					\
  double2 G9 = make_double2(0,0);				

#define READ_GAUGE_MATRIX_12_SINGLE(gauge, dir) \
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+0)*ga_stride);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+1)*ga_stride);	\
  float4 G2 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+2)*ga_stride);	\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				

#define RECONSTRUCT_MATRIX_12_DOUBLE(dir)				\
  ACC_CONJ_PROD(g20, +g01, +g12);					\
  ACC_CONJ_PROD(g20, -g02, +g11);					\
  ACC_CONJ_PROD(g21, +g02, +g10);					\
  ACC_CONJ_PROD(g21, -g00, +g12);					\
  ACC_CONJ_PROD(g22, +g00, +g11);					\
  ACC_CONJ_PROD(g22, -g01, +g10);					\
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= (X4-1)*X1h*X2*X3 ? t_boundary : 1)); \
  G6.x*=u0; G6.y*=u0; G7.x*=u0; G7.y*=u0; G8.x*=u0; G8.y*=u0;

#define RECONSTRUCT_MATRIX_12_SINGLE(dir)			\
  ACC_CONJ_PROD(g20, +g01, +g12);				\
  ACC_CONJ_PROD(g20, -g02, +g11);				\
  ACC_CONJ_PROD(g21, +g02, +g10);				\
  ACC_CONJ_PROD(g21, -g00, +g12);				\
  ACC_CONJ_PROD(g22, +g00, +g11);				\
  ACC_CONJ_PROD(g22, -g01, +g10);				\
  float u0 = (dir < 6 ? anisotropy_f : (ga_idx >= (X4-1)*X1h*X2*X3 ? t_boundary_f : 1)); \
  G3.x*=u0; G3.y*=u0; G3.z*=u0; G3.w*=u0; G4.x*=u0; G4.y*=u0;

// set A to be last components of G4 (otherwise unused)
#define READ_GAUGE_MATRIX_8_DOUBLE(gauge, dir)				\
  double2 G0 = fetch_double2((gauge), ga_idx + ((dir/2)*4+0)*ga_stride);	\
  double2 G1 = fetch_double2((gauge), ga_idx + ((dir/2)*4+1)*ga_stride);	\
  double2 G2 = fetch_double2((gauge), ga_idx + ((dir/2)*4+2)*ga_stride);	\
  double2 G3 = fetch_double2((gauge), ga_idx + ((dir/2)*4+3)*ga_stride);	\
  double2 G4 = make_double2(0,0);					\
  double2 G5 = make_double2(0,0);					\
  double2 G6 = make_double2(0,0);					\
  double2 G7 = make_double2(0,0);					\
  double2 G8 = make_double2(0,0);					\
  double2 G9 = make_double2(0,0);					\
  g21_re = g00_re;							\
  g21_im = g00_im;

// set A to be last components of G4 (otherwise unused)
#define READ_GAUGE_MATRIX_8_SINGLE(gauge, dir)			\
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*ga_stride);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*ga_stride);	\
  float4 G2 = make_float4(0,0,0,0);				\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				\
  g21_re = g00_re;						\
  g21_im = g00_im;

#define READ_GAUGE_MATRIX_8_HALF(gauge, dir)			\
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*ga_stride);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*ga_stride);	\
  float4 G2 = make_float4(0,0,0,0);				\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				\
  g21_re = pi_f*g00_re;						\
  g21_im = pi_f*g00_im;

#define RECONSTRUCT_MATRIX_8_DOUBLE(dir)				\
  double row_sum = g01_re*g01_re + g01_im*g01_im;			\
  row_sum += g02_re*g02_re + g02_im*g02_im;				\
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= (X4-1)*X1h*X2*X3 ? t_boundary : 1)); \
  double u02_inv = 1.0 / (u0*u0);					\
  double column_sum = u02_inv - row_sum;				\
  double U00_mag = sqrt((column_sum > 0 ? column_sum : 0));		\
  sincos(g21_re, &g00_im, &g00_re);					\
  g00_re *= U00_mag;							\
  g00_im *= U00_mag;							\
  column_sum += g10_re*g10_re;						\
  column_sum += g10_im*g10_im;						\
  sincos(g21_im, &g20_im, &g20_re);					\
  double U20_mag = sqrt(((u02_inv - column_sum) > 0 ? (u02_inv-column_sum) : 0)); \
  g20_re *= U20_mag;							\
  g20_im *= U20_mag;							\
  double r_inv2 = 1.0 / (u0*row_sum);					\
  COMPLEX_DOT_PRODUCT(A, g00, g10);					\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(g11, g20, g02);				\
  ACC_COMPLEX_PROD(g11, A, g01);					\
  g11_re *= -r_inv2;							\
  g11_im *= -r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g12, g20, g01);				\
  ACC_COMPLEX_PROD(g12, -A, g02);					\
  g12_re *= r_inv2;							\
  g12_im *= r_inv2;							\
  COMPLEX_DOT_PRODUCT(A, g00, g20);					\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(g21, g10, g02);				\
  ACC_COMPLEX_PROD(g21, -A, g01);					\
  g21_re *= r_inv2;							\
  g21_im *= r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g22, g10, g01);				\
  ACC_COMPLEX_PROD(g22, A, g02);					\
  g22_re *= -r_inv2;							\
  g22_im *= -r_inv2;	

#define RECONSTRUCT_MATRIX_8_SINGLE(dir)				\
  float row_sum = g01_re*g01_re + g01_im*g01_im;			\
  row_sum += g02_re*g02_re + g02_im*g02_im;				\
  float u0 = (dir < 6 ? anisotropy_f : (ga_idx >= (X4-1)*X1h*X2*X3 ? t_boundary_f : 1)); \
  float u02_inv = __fdividef(1.f, u0*u0);				\
  float column_sum = u02_inv - row_sum;					\
  float U00_mag = column_sum * rsqrtf((column_sum > 0 ? column_sum : 1e14)); \
  __sincosf(g21_re, &g00_im, &g00_re);					\
  g00_re *= U00_mag;							\
  g00_im *= U00_mag;							\
  column_sum += g10_re*g10_re;						\
  column_sum += g10_im*g10_im;						\
  __sincosf(g21_im, &g20_im, &g20_re);					\
  column_sum = u02_inv - column_sum;					\
  float U20_mag = column_sum * rsqrtf((column_sum > 0 ? column_sum : 1e14)); \
  g20_re *= U20_mag;							\
  g20_im *= U20_mag;							\
  float r_inv2 = __fdividef(1.0f, u0*row_sum);				\
  COMPLEX_DOT_PRODUCT(A, g00, g10);					\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(g11, g20, g02);				\
  ACC_COMPLEX_PROD(g11, A, g01);					\
  g11_re *= -r_inv2;							\
  g11_im *= -r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g12, g20, g01);				\
  ACC_COMPLEX_PROD(g12, -A, g02);					\
  g12_re *= r_inv2;							\
  g12_im *= r_inv2;							\
  COMPLEX_DOT_PRODUCT(A, g00, g20);					\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(g21, g10, g02);				\
  ACC_COMPLEX_PROD(g21, -A, g01);					\
  g21_re *= r_inv2;							\
  g21_im *= r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g22, g10, g01);				\
  ACC_COMPLEX_PROD(g22, A, g02);					\
  g22_re *= -r_inv2;							\
  g22_im *= -r_inv2;	
