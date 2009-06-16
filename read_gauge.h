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
  double2 G0 = fetch_double2((gauge), ga_idx + ((dir/2)*6+0)*Nh);	\
  double2 G1 = fetch_double2((gauge), ga_idx + ((dir/2)*6+1)*Nh);	\
  double2 G2 = fetch_double2((gauge), ga_idx + ((dir/2)*6+2)*Nh);	\
  double2 G3 = fetch_double2((gauge), ga_idx + ((dir/2)*6+3)*Nh);	\
  double2 G4 = fetch_double2((gauge), ga_idx + ((dir/2)*6+4)*Nh);	\
  double2 G5 = fetch_double2((gauge), ga_idx + ((dir/2)*6+5)*Nh);	\
  double2 G6 = make_double2(0,0);					\
  double2 G7 = make_double2(0,0);					\
  double2 G8 = make_double2(0,0);					\
  double2 G9 = make_double2(0,0);				

#define READ_GAUGE_MATRIX_12_SINGLE(gauge, dir) \
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+0)*Nh);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+1)*Nh);	\
  float4 G2 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+2)*Nh);	\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				

#define RECONSTRUCT_MATRIX_12_DOUBLE(dir)				\
  ACC_CONJ_PROD(g20, +g01, +g12);					\
  ACC_CONJ_PROD(g20, -g02, +g11);					\
  ACC_CONJ_PROD(g21, +g02, +g10);					\
  ACC_CONJ_PROD(g21, -g00, +g12);					\
  ACC_CONJ_PROD(g22, +g00, +g11);					\
  ACC_CONJ_PROD(g22, -g01, +g10);					\
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= (L4-1)*L1h*L2*L3 ? t_boundary : 1)); \
  G6.x*=u0; G6.y*=u0; G7.x*=u0; G7.y*=u0; G8.x*=u0; G8.y*=u0;

#define RECONSTRUCT_MATRIX_12_SINGLE(dir)			\
  ACC_CONJ_PROD(g20, +g01, +g12);				\
  ACC_CONJ_PROD(g20, -g02, +g11);				\
  ACC_CONJ_PROD(g21, +g02, +g10);				\
  ACC_CONJ_PROD(g21, -g00, +g12);				\
  ACC_CONJ_PROD(g22, +g00, +g11);				\
  ACC_CONJ_PROD(g22, -g01, +g10);				\
  float u0 = (dir < 6 ? anisotropy_f : (ga_idx >= (L4-1)*L1h*L2*L3 ? t_boundary_f : 1)); \
  G3.x*=u0; G3.y*=u0; G3.z*=u0; G3.w*=u0; G4.x*=u0; G4.y*=u0;

// set A to be last components of G4 (otherwise unused)
#define READ_GAUGE_MATRIX_8_DOUBLE(gauge, dir)				\
  double2 G0 = fetch_double2((gauge), ga_idx + ((dir/2)*4+0)*Nh);	\
  double2 G1 = fetch_double2((gauge), ga_idx + ((dir/2)*4+1)*Nh);	\
  double2 G2 = fetch_double2((gauge), ga_idx + ((dir/2)*4+2)*Nh);	\
  double2 G3 = fetch_double2((gauge), ga_idx + ((dir/2)*4+3)*Nh);	\
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
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*Nh);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*Nh);	\
  float4 G2 = make_float4(0,0,0,0);				\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				\
  g21_re = g00_re;						\
  g21_im = g00_im;

#define READ_GAUGE_MATRIX_8_HALF(gauge, dir)			\
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*Nh);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*Nh);	\
  float4 G2 = make_float4(0,0,0,0);				\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				\
  g21_re = pi_f*g00_re;						\
  g21_im = pi_f*g00_im;

#define RECONSTRUCT_MATRIX_8_DOUBLE(dir)				\
  double row_sum = g01_re*g01_re + g01_im*g01_im;			\
  row_sum += g02_re*g02_re + g02_im*g02_im;				\
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= (L4-1)*L1h*L2*L3 ? t_boundary : 1)); \
  double u02_inv = 1.0 / (u0*u0);					\
  double column_sum = u02_inv - row_sum;				\
  double U00_mag = sqrt(column_sum);					\
  sincos(g21_re, &g00_im, &g00_re);					\
  g00_re *= U00_mag;							\
  g00_im *= U00_mag;							\
  column_sum += g10_re*g10_re;						\
  column_sum += g10_im*g10_im;						\
  sincos(g21_im, &g20_im, &g20_re);					\
  double U20_mag = sqrt(u02_inv - column_sum);				\
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
  float u0 = (dir < 6 ? anisotropy_f : (ga_idx >= (L4-1)*L1h*L2*L3 ? t_boundary_f : 1)); \
  float u02_inv = __fdividef(1.f, u0*u0);				\
  float column_sum = u02_inv - row_sum;					\
  float U00_mag = sqrtf(column_sum);					\
  __sincosf(g21_re, &g00_im, &g00_re);					\
  g00_re *= U00_mag;							\
  g00_im *= U00_mag;							\
  column_sum += g10_re*g10_re;						\
  column_sum += g10_im*g10_im;						\
  __sincosf(g21_im, &g20_im, &g20_re);					\
  float U20_mag = sqrtf(u02_inv - column_sum);				\
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

/*// set A to be last components of G4 (otherwise unused)
#define READ_GAUGE_MATRIX_8(gauge, dir)				\
  float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*Nh);	\
  float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*Nh);	\
  float4 G2 = make_float4(0,0,0,0);				\
  float4 G3 = make_float4(0,0,0,0);				\
  float4 G4 = make_float4(0,0,0,0);				\
  float y2_sum = g02_im*g02_im;				       	\
  y2_sum += g02_re*g02_re;					\
  y2_sum += g01_im*g01_im;					\
  y2_sum += g01_re*g01_re;					\
  float r_column = sqrtf(y2_sum);				\
  y2_sum += g00_im*g00_im;					\
  float u0 = (dir < 6 ? anisotropy : (ga_idx >= (L4-1)*L1h*L2*L3 ? t_boundary : 1)); \
  float r0 = __fdividef(1.0,fabsf(u0));				\
  g20_re = g00_re;						\
  g00_re = r0*(1.f - 2.f*y2_sum);				\
  float y_scale = 2.f*r0*sqrtf(1.0 - y2_sum);			\
  g00_im *= y_scale;						\
  g01_re *= y_scale;						\
  g01_im *= y_scale;						\
  g02_re *= y_scale;						\
  g02_im *= y_scale;						\
  r_column *= y_scale;						\
  y2_sum = g20_re*g20_re;					\
  y2_sum += g10_re*g10_re;					\
  y2_sum += g10_im*g10_im;					\
  g20_im = r_column * (1.f - 2.f*y2_sum);			\
  y_scale = 2.f*r_column*sqrtf(1.f - y2_sum);			\
  g20_re *= y_scale;						\
  g10_re *= y_scale;						\
  g10_im *= y_scale;						\
  float r_inv2 = __fdividef(1.0,u0*r_column*r_column);		\
  COMPLEX_DOT_PRODUCT(A, g00, g10);				\
  A_re *= u0; A_im *= u0;					\
  COMPLEX_CONJUGATE_PRODUCT(g11, g20, g02);			\
  ACC_COMPLEX_PROD(g11, A, g01);				\
  g11_re *= -r_inv2;						\
  g11_im *= -r_inv2;						\
  COMPLEX_CONJUGATE_PRODUCT(g12, g20, g01);			\
  ACC_COMPLEX_PROD(g12, -A, g02);				\
  g12_re *= r_inv2;						\
  g12_im *= r_inv2;						\
  COMPLEX_DOT_PRODUCT(A, g00, g20);				\
  A_re *= u0; A_im *= u0;					\
  COMPLEX_CONJUGATE_PRODUCT(g21, g10, g02);			\
  ACC_COMPLEX_PROD(g21, -A, g01);				\
  g21_re *= r_inv2;						\
  g21_im *= r_inv2;						\
  COMPLEX_CONJUGATE_PRODUCT(g22, g10, g01);			\
  ACC_COMPLEX_PROD(g22, A, g02);				\
  g22_re *= -r_inv2;						\
  g22_im *= -r_inv2;						

#define READ_GAUGE_MATRIX_8(gauge, dir) \
    float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+0)*Nh); \
    float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*2+1)*Nh); \
    float4 G2 = make_float4(0,0,0,0); \
    float4 G3 = make_float4(0,0,0,0); \
    float4 G4 = make_float4(0,0,0,0); \
    g12_re = g02_re; \
    g12_im = g02_im; \
    COMPLEX_NORM(g11_re, g00); \
    g20_re = __fdividef(1.f,g11_re); \
    COMPLEX_NORM(g20_im, g01); \
    g21_re = 1.f - (g11_re + g20_im); \
    g21_im = sqrtf(g21_re); \
    __sincosf(g10_re, &g02_im, &g02_re); \
    g02_re *= g21_im; \
    g02_im *= g21_im; \
    g20_im = 1.f + g20_im*g20_re; \
    __sincosf(g10_im, &g22_im, &g22_re); \
    COMPLEX_DOT_PRODUCT(g10, g01, g02); \
    COMPLEX_DOT_PRODUCT(g11, g12, g22); \
    g21_re = g10_re * g11_re; \
    g21_re -= g10_im * g11_im; \
    g11_re = g21_re * g20_re; \
    g11_re *= 2.f; \
    g11_im = g11_re *g11_re; \
    COMPLEX_NORM(g21_re, g02); \
    COMPLEX_NORM(g21_im, g12); \
    g10_re = 1.f + g21_re * g20_re; \
    g21_re = g21_im * g10_re - 1.f; \
    g10_re = g20_im * g21_re; \
    g10_re *= 4.f; \
    g10_im = g11_im - g10_re; \
    g10_re = sqrtf(g10_im); \
    g10_im = g10_re - g11_re; \
    g21_im = __fdividef(g10_im, g20_im); \
    g21_im *= 0.5f; \
    g11_re = g21_im*g22_re; \
    g11_im = g21_im*g22_im; \
    COMPLEX_DOT_PRODUCT(g22, g01, g11); \
    COMPLEX_DOT_PRODUCT(g21, g02, g12); \
    COMPLEX_ADD_TO(g22, g21); \
    COMPLEX_PRODUCT(g10, g22, g00); \
    g10_re *= -g20_re; \
    g10_im *= -g20_re; \
    COMPLEX_CONJUGATE_PRODUCT(g20, g01, g12); \
    ACC_CONJ_PROD(g20, -g02, +g11); \
    COMPLEX_CONJUGATE_PRODUCT(g21, g02, g10); \
    ACC_CONJ_PROD(g21, -g00, +g12); \
    COMPLEX_CONJUGATE_PRODUCT(g22, g00, g11); \
    ACC_CONJ_PROD(g22, -g01, +g10); \
    float u0 = (dir < 6 ? SPATIAL_SCALING : (ga_idx >= (L4-1)*L1h*L2*L3 ? TIME_SYMMETRY : 1)); \
    G3.x*=u0; G3.y*=u0; G3.z*=u0; G3.w*=u0; G4.x*=u0; G4.y*=u0;
*/
