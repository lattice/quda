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
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= X4X3X2X1hmX3X2X1h ? t_boundary : 1)); \
  G6.x*=u0; G6.y*=u0; G7.x*=u0; G7.y*=u0; G8.x*=u0; G8.y*=u0;

#define RECONSTRUCT_MATRIX_12_SINGLE(dir)			\
  ACC_CONJ_PROD(g20, +g01, +g12);				\
  ACC_CONJ_PROD(g20, -g02, +g11);				\
  ACC_CONJ_PROD(g21, +g02, +g10);				\
  ACC_CONJ_PROD(g21, -g00, +g12);				\
  ACC_CONJ_PROD(g22, +g00, +g11);				\
  ACC_CONJ_PROD(g22, -g01, +g10);				\
  float u0 = (dir < 6 ? anisotropy_f : (ga_idx >= X4X3X2X1hmX3X2X1h ? t_boundary_f : 1)); \
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
  double u0 = (dir < 6 ? anisotropy : (ga_idx >= X4X3X2X1hmX3X2X1h ? t_boundary : 1)); \
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





// use __saturate ?
//  float U00_mag = sqrtf(__saturatef(column_sum));			\
//  float U20_mag = sqrtf(__saturatef(column_sum));			\

#define RECONSTRUCT_MATRIX_8_SINGLE(dir)				\
  float row_sum = g01_re*g01_re + g01_im*g01_im;			\
  row_sum += g02_re*g02_re + g02_im*g02_im;				\
  __sincosf(g21_re, &g00_im, &g00_re);					\
  __sincosf(g21_im, &g20_im, &g20_re);					\
  float2 u0_2 = (dir < 6 ? An2 : (ga_idx >= X4X3X2X1hmX3X2X1h ? TB2 : No2)); \
  float column_sum = u0_2.y - row_sum;					\
  float U00_mag = column_sum * rsqrtf((column_sum > 0 ? column_sum : 1e14)); \
  g00_re *= U00_mag;							\
  g00_im *= U00_mag;							\
  column_sum += g10_re*g10_re;						\
  column_sum += g10_im*g10_im;						\
  column_sum = u0_2.y - column_sum;					\
  float U20_mag = column_sum * rsqrtf((column_sum > 0 ? column_sum : 1e14)); \
  g20_re *= U20_mag;							\
  g20_im *= U20_mag;							\
  float r_inv2 = __fdividef(1.0f, u0_2.x*row_sum);			\
  COMPLEX_DOT_PRODUCT(A, g00, g10);					\
  A_re *= u0_2.x; A_im *= u0_2.x;					\
  COMPLEX_CONJUGATE_PRODUCT(g11, g20, g02);				\
  ACC_COMPLEX_PROD(g11, A, g01);					\
  g11_re *= -r_inv2;							\
  g11_im *= -r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g12, g20, g01);				\
  ACC_COMPLEX_PROD(g12, -A, g02);					\
  g12_re *= r_inv2;							\
  g12_im *= r_inv2;							\
  COMPLEX_DOT_PRODUCT(A, g00, g20);					\
  A_re *= u0_2.x; A_im *= u0_2.x;					\
  COMPLEX_CONJUGATE_PRODUCT(g21, g10, g02);				\
  ACC_COMPLEX_PROD(g21, -A, g01);					\
  g21_re *= r_inv2;							\
  g21_im *= r_inv2;							\
  COMPLEX_CONJUGATE_PRODUCT(g22, g10, g01);				\
  ACC_COMPLEX_PROD(g22, A, g02);					\
  g22_re *= -r_inv2;							\
  g22_im *= -r_inv2;	



/************* the following is added for staggered *********/

#define RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE(dir, gauge, idx, sign)	\
  double row_sum = gauge##01_re*gauge##01_re + gauge##01_im*gauge##01_im; \
  row_sum += gauge##02_re*gauge##02_re + gauge##02_im*gauge##02_im;	\
  double u0 = coeff*sign;						\
  double u02_inv = 1.0 / (u0*u0);					\
  double column_sum = u02_inv - row_sum;				\
  double U00_mag = sqrt(column_sum);					\
  sincos(gauge##21_re, &gauge##00_im, &gauge##00_re);			\
  gauge##00_re *= U00_mag;						\
  gauge##00_im *= U00_mag;						\
  column_sum += gauge##10_re*gauge##10_re;				\
  column_sum += gauge##10_im*gauge##10_im;				\
  sincos(gauge##21_im, &gauge##20_im, &gauge##20_re);			\
  double U20_mag = sqrt(u02_inv - column_sum);				\
  gauge##20_re *= U20_mag;						\
  gauge##20_im *= U20_mag;						\
  double r_inv2 = 1.0 / (u0*row_sum);					\
  COMPLEX_DOT_PRODUCT(A, gauge##00, gauge##10);				\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(gauge##11, gauge##20, gauge##02);		\
  ACC_COMPLEX_PROD(gauge##11, A, gauge##01);				\
  gauge##11_re *= -r_inv2;						\
  gauge##11_im *= -r_inv2;						\
  COMPLEX_CONJUGATE_PRODUCT(gauge##12, gauge##20, gauge##01);		\
  ACC_COMPLEX_PROD(gauge##12, -A, gauge##02);				\
  gauge##12_re *= r_inv2;						\
  gauge##12_im *= r_inv2;						\
  COMPLEX_DOT_PRODUCT(A, gauge##00, gauge##20);				\
  A_re *= u0; A_im *= u0;						\
  COMPLEX_CONJUGATE_PRODUCT(gauge##21, gauge##10, gauge##02);		\
  ACC_COMPLEX_PROD(gauge##21, -A, gauge##01);				\
  gauge##21_re *= r_inv2;						\
  gauge##21_im *= r_inv2;						\
  COMPLEX_CONJUGATE_PRODUCT(gauge##22, gauge##10, gauge##01);		\
  ACC_COMPLEX_PROD(gauge##22, A, gauge##02);				\
  gauge##22_re *= -r_inv2;						\
  gauge##22_im *= -r_inv2;


#define RECONSTRUCT_GAUGE_MATRIX_12_SINGLE(dir, gauge, idx, sign)       \
  ACC_CONJ_PROD(gauge##20, +gauge##01, +gauge##12);			\
  ACC_CONJ_PROD(gauge##20, -gauge##02, +gauge##11);			\
  ACC_CONJ_PROD(gauge##21, +gauge##02, +gauge##10);			\
  ACC_CONJ_PROD(gauge##21, -gauge##00, +gauge##12);			\
  ACC_CONJ_PROD(gauge##22, +gauge##00, +gauge##11);			\
  ACC_CONJ_PROD(gauge##22, -gauge##01, +gauge##10);			\
  if (1){float u0 = coeff_f*sign;					\
    gauge##20_re *=u0;gauge##20_im *=u0; gauge##21_re *=u0; gauge##21_im *=u0; \
    gauge##22_re *=u0;gauge##22_im *=u0;}

#define RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE(dir, gauge, idx, sign)	\
  ACC_CONJ_PROD(gauge##20, +gauge##01, +gauge##12);			\
  ACC_CONJ_PROD(gauge##20, -gauge##02, +gauge##11);			\
  ACC_CONJ_PROD(gauge##21, +gauge##02, +gauge##10);			\
  ACC_CONJ_PROD(gauge##21, -gauge##00, +gauge##12);			\
  ACC_CONJ_PROD(gauge##22, +gauge##00, +gauge##11);			\
  ACC_CONJ_PROD(gauge##22, -gauge##01, +gauge##10);			\
  if (1){double u0 = coeff* sign;					\
    gauge##20_re *=u0;gauge##20_im *=u0; gauge##21_re *=u0; gauge##21_im *=u0; \
    gauge##22_re *=u0;gauge##22_im *=u0;}


#define RECONSTRUCT_GAUGE_MATRIX_8_SINGLE(dir, gauge, idx, sign)        { \
    float row_sum = gauge##01_re*gauge##01_re + gauge##01_im*gauge##01_im; \
    row_sum += gauge##02_re*gauge##02_re + gauge##02_im*gauge##02_im;	\
    float u0 = coeff_f*sign;						\
    float u02_inv = __fdividef(1.f, u0*u0);				\
    float column_sum = u02_inv - row_sum;				\
    float U00_mag = sqrtf(column_sum > 0 ?column_sum:0);		\
    __sincosf(gauge##21_re, &gauge##00_im, &gauge##00_re);		\
    gauge##00_re *= U00_mag;						\
    gauge##00_im *= U00_mag;						\
    column_sum += gauge##10_re*gauge##10_re;				\
    column_sum += gauge##10_im*gauge##10_im;				\
    __sincosf(gauge##21_im, &gauge##20_im, &gauge##20_re);		\
    float U20_mag = sqrtf( (u02_inv - column_sum)>0? (u02_inv - column_sum): 0); \
    gauge##20_re *= U20_mag;						\
    gauge##20_im *= U20_mag;						\
    float r_inv2 = __fdividef(1.0f, u0*row_sum);			\
    COMPLEX_DOT_PRODUCT(A, gauge##00, gauge##10);			\
    A_re *= u0; A_im *= u0;						\
    COMPLEX_CONJUGATE_PRODUCT(gauge##11, gauge##20, gauge##02);		\
    ACC_COMPLEX_PROD(gauge##11, A, gauge##01);				\
    gauge##11_re *= -r_inv2;						\
    gauge##11_im *= -r_inv2;						\
    COMPLEX_CONJUGATE_PRODUCT(gauge##12, gauge##20, gauge##01);		\
    ACC_COMPLEX_PROD(gauge##12, -A, gauge##02);				\
    gauge##12_re *= r_inv2;						\
    gauge##12_im *= r_inv2;						\
    COMPLEX_DOT_PRODUCT(A, gauge##00, gauge##20);			\
    A_re *= u0; A_im *= u0;						\
    COMPLEX_CONJUGATE_PRODUCT(gauge##21, gauge##10, gauge##02);		\
    ACC_COMPLEX_PROD(gauge##21, -A, gauge##01);				\
    gauge##21_re *= r_inv2;						\
    gauge##21_im *= r_inv2;						\
    COMPLEX_CONJUGATE_PRODUCT(gauge##22, gauge##10, gauge##01);		\
    ACC_COMPLEX_PROD(gauge##22, A, gauge##02);				\
    gauge##22_re *= -r_inv2;						\
    gauge##22_im *= -r_inv2;}

#ifndef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX_18_SINGLE(gauge, dir, idx)			\
  float2 FAT0 = tex1Dfetch((gauge), idx + ((dir/2)*9+0)*fat_ga_stride);	\
  float2 FAT1 = tex1Dfetch((gauge), idx + ((dir/2)*9+1)*fat_ga_stride);	\
  float2 FAT2 = tex1Dfetch((gauge), idx + ((dir/2)*9+2)*fat_ga_stride);	\
  float2 FAT3 = tex1Dfetch((gauge), idx + ((dir/2)*9+3)*fat_ga_stride);	\
  float2 FAT4 = tex1Dfetch((gauge), idx + ((dir/2)*9+4)*fat_ga_stride);	\
  float2 FAT5 = tex1Dfetch((gauge), idx + ((dir/2)*9+5)*fat_ga_stride);	\
  float2 FAT6 = tex1Dfetch((gauge), idx + ((dir/2)*9+6)*fat_ga_stride);	\
  float2 FAT7 = tex1Dfetch((gauge), idx + ((dir/2)*9+7)*fat_ga_stride);	\
  float2 FAT8 = tex1Dfetch((gauge), idx + ((dir/2)*9+8)*fat_ga_stride);



#define READ_FAT_MATRIX_18_DOUBLE(gauge, dir, idx)			\
  double2 FAT0 = fetch_double2((gauge), idx + ((dir/2)*9+0)*fat_ga_stride); \
  double2 FAT1 = fetch_double2((gauge), idx + ((dir/2)*9+1)*fat_ga_stride); \
  double2 FAT2 = fetch_double2((gauge), idx + ((dir/2)*9+2)*fat_ga_stride); \
  double2 FAT3 = fetch_double2((gauge), idx + ((dir/2)*9+3)*fat_ga_stride); \
  double2 FAT4 = fetch_double2((gauge), idx + ((dir/2)*9+4)*fat_ga_stride); \
  double2 FAT5 = fetch_double2((gauge), idx + ((dir/2)*9+5)*fat_ga_stride); \
  double2 FAT6 = fetch_double2((gauge), idx + ((dir/2)*9+6)*fat_ga_stride); \
  double2 FAT7 = fetch_double2((gauge), idx + ((dir/2)*9+7)*fat_ga_stride); \
  double2 FAT8 = fetch_double2((gauge), idx + ((dir/2)*9+8)*fat_ga_stride);


#else
#define READ_FAT_MATRIX_18_SINGLE(gauge, dir, idx)		\
  float2 FAT0 = gauge[idx + ((dir/2)*9+0)*fat_ga_stride];	\
  float2 FAT1 = gauge[idx + ((dir/2)*9+1)*fat_ga_stride];	\
  float2 FAT2 = gauge[idx + ((dir/2)*9+2)*fat_ga_stride];	\
  float2 FAT3 = gauge[idx + ((dir/2)*9+3)*fat_ga_stride];	\
  float2 FAT4 = gauge[idx + ((dir/2)*9+4)*fat_ga_stride];	\
  float2 FAT5 = gauge[idx + ((dir/2)*9+5)*fat_ga_stride];	\
  float2 FAT6 = gauge[idx + ((dir/2)*9+6)*fat_ga_stride];	\
  float2 FAT7 = gauge[idx + ((dir/2)*9+7)*fat_ga_stride];	\
  float2 FAT8 = gauge[idx + ((dir/2)*9+8)*fat_ga_stride];


#define READ_FAT_MATRIX_18_DOUBLE(gauge, dir, idx)		\
  double2 FAT0 = gauge[idx + ((dir/2)*9+0)*fat_ga_stride];	\
  double2 FAT1 = gauge[idx + ((dir/2)*9+1)*fat_ga_stride];	\
  double2 FAT2 = gauge[idx + ((dir/2)*9+2)*fat_ga_stride];	\
  double2 FAT3 = gauge[idx + ((dir/2)*9+3)*fat_ga_stride];	\
  double2 FAT4 = gauge[idx + ((dir/2)*9+4)*fat_ga_stride];	\
  double2 FAT5 = gauge[idx + ((dir/2)*9+5)*fat_ga_stride];	\
  double2 FAT6 = gauge[idx + ((dir/2)*9+6)*fat_ga_stride];	\
  double2 FAT7 = gauge[idx + ((dir/2)*9+7)*fat_ga_stride];	\
  double2 FAT8 = gauge[idx + ((dir/2)*9+8)*fat_ga_stride];


#endif


#define READ_FAT_MATRIX_18_HALF(gauge, dir, idx)			\
  float2 FAT0 = tex1Dfetch((gauge), idx + ((dir/2)*9+0)*fat_ga_stride);	\
  float2 FAT1 = tex1Dfetch((gauge), idx + ((dir/2)*9+1)*fat_ga_stride);	\
  float2 FAT2 = tex1Dfetch((gauge), idx + ((dir/2)*9+2)*fat_ga_stride);	\
  float2 FAT3 = tex1Dfetch((gauge), idx + ((dir/2)*9+3)*fat_ga_stride);	\
  float2 FAT4 = tex1Dfetch((gauge), idx + ((dir/2)*9+4)*fat_ga_stride);	\
  float2 FAT5 = tex1Dfetch((gauge), idx + ((dir/2)*9+5)*fat_ga_stride);	\
  float2 FAT6 = tex1Dfetch((gauge), idx + ((dir/2)*9+6)*fat_ga_stride);	\
  float2 FAT7 = tex1Dfetch((gauge), idx + ((dir/2)*9+7)*fat_ga_stride);	\
  float2 FAT8 = tex1Dfetch((gauge), idx + ((dir/2)*9+8)*fat_ga_stride);


#ifndef DIRECT_ACCESS_LONG_LINK //longlink access

#define READ_LONG_MATRIX_12_SINGLE(gauge, dir, idx)			\
  float4 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*3+0)*long_ga_stride); \
  float4 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*3+1)*long_ga_stride); \
  float4 LONG2 = tex1Dfetch((gauge), idx + ((dir/2)*3+2)*long_ga_stride); \
  float4 LONG3 = make_float4(0,0,0,0);					\
  float4 LONG4 = make_float4(0,0,0,0);
#define READ_LONG_MATRIX_8_SINGLE(gauge, dir, idx)			\
  float4 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*2+0)*long_ga_stride); \
  float4 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*2+1)*long_ga_stride); \
  float4 LONG2 = make_float4(0,0,0,0);					\
  float4 LONG3 = make_float4(0,0,0,0);					\
  float4 LONG4 = make_float4(0,0,0,0);					\
  long21_re = long00_re;						\
  long21_im = long00_im;
#define READ_LONG_MATRIX_18_SINGLE(gauge, dir, idx)			\
  float2 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*9+0)*long_ga_stride); \
  float2 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*9+1)*long_ga_stride); \
  float2 LONG2 = tex1Dfetch((gauge), idx + ((dir/2)*9+2)*long_ga_stride); \
  float2 LONG3 = tex1Dfetch((gauge), idx + ((dir/2)*9+3)*long_ga_stride); \
  float2 LONG4 = tex1Dfetch((gauge), idx + ((dir/2)*9+4)*long_ga_stride); \
  float2 LONG5 = tex1Dfetch((gauge), idx + ((dir/2)*9+5)*long_ga_stride); \
  float2 LONG6 = tex1Dfetch((gauge), idx + ((dir/2)*9+6)*long_ga_stride); \
  float2 LONG7 = tex1Dfetch((gauge), idx + ((dir/2)*9+7)*long_ga_stride); \
  float2 LONG8 = tex1Dfetch((gauge), idx + ((dir/2)*9+8)*long_ga_stride);

#define READ_LONG_MATRIX_12_DOUBLE(gauge, dir, idx)			\
  double2 LONG0 = fetch_double2((gauge), idx + ((dir/2)*6+0)*long_ga_stride); \
  double2 LONG1 = fetch_double2((gauge), idx + ((dir/2)*6+1)*long_ga_stride); \
  double2 LONG2 = fetch_double2((gauge), idx + ((dir/2)*6+2)*long_ga_stride); \
  double2 LONG3 = fetch_double2((gauge), idx + ((dir/2)*6+3)*long_ga_stride); \
  double2 LONG4 = fetch_double2((gauge), idx + ((dir/2)*6+4)*long_ga_stride); \
  double2 LONG5 = fetch_double2((gauge), idx + ((dir/2)*6+5)*long_ga_stride); \
  double2 LONG6 = make_double2(0,0);					\
  double2 LONG7 = make_double2(0,0);					\
  double2 LONG8 = make_double2(0,0);					\
  double2 LONG9 = make_double2(0,0);

#define READ_LONG_MATRIX_8_DOUBLE(gauge, dir, idx)                      \
  double2 LONG0 = fetch_double2((gauge), idx + ((dir/2)*4+0)*long_ga_stride); \
  double2 LONG1 = fetch_double2((gauge), idx + ((dir/2)*4+1)*long_ga_stride); \
  double2 LONG2 = fetch_double2((gauge), idx + ((dir/2)*4+2)*long_ga_stride); \
  double2 LONG3 = fetch_double2((gauge), idx + ((dir/2)*4+3)*long_ga_stride); \
  double2 LONG4 = make_double2(0,0);					\
  double2 LONG5 = make_double2(0,0);					\
  double2 LONG6 = make_double2(0,0);					\
  double2 LONG7 = make_double2(0,0);					\
  double2 LONG8 = make_double2(0,0);					\
  double2 LONG9 = make_double2(0,0);					\
  long21_re = long00_re;						\
  long21_im = long00_im;

#define READ_LONG_MATRIX_18_DOUBLE(gauge, dir, idx)                     \
  double2 LONG0 = fetch_double2((gauge), idx + ((dir/2)*9+0)*long_ga_stride); \
  double2 LONG1 = fetch_double2((gauge), idx + ((dir/2)*9+1)*long_ga_stride); \
  double2 LONG2 = fetch_double2((gauge), idx + ((dir/2)*9+2)*long_ga_stride); \
  double2 LONG3 = fetch_double2((gauge), idx + ((dir/2)*9+3)*long_ga_stride); \
  double2 LONG4 = fetch_double2((gauge), idx + ((dir/2)*9+4)*long_ga_stride); \
  double2 LONG5 = fetch_double2((gauge), idx + ((dir/2)*9+5)*long_ga_stride); \
  double2 LONG6 = fetch_double2((gauge), idx + ((dir/2)*9+6)*long_ga_stride); \
  double2 LONG7 = fetch_double2((gauge), idx + ((dir/2)*9+7)*long_ga_stride); \
  double2 LONG8 = fetch_double2((gauge), idx + ((dir/2)*9+8)*long_ga_stride);


#else //longlink access

#define READ_LONG_MATRIX_12_SINGLE(gauge, dir, idx)		\
  float4 LONG0 = gauge[idx + ((dir/2)*3+0)*long_ga_stride];	\
  float4 LONG1 = gauge[idx + ((dir/2)*3+1)*long_ga_stride];	\
  float4 LONG2 = gauge[idx + ((dir/2)*3+2)*long_ga_stride];	\
  float4 LONG3 = make_float4(0,0,0,0);				\
  float4 LONG4 = make_float4(0,0,0,0);
#define READ_LONG_MATRIX_8_SINGLE(gauge, dir, idx)		\
  float4 LONG0 = gauge[idx + ((dir/2)*2+0)*long_ga_stride];	\
  float4 LONG1 = gauge[idx + ((dir/2)*2+1)*long_ga_stride];	\
  float4 LONG2 = make_float4(0,0,0,0);				\
  float4 LONG3 = make_float4(0,0,0,0);				\
  float4 LONG4 = make_float4(0,0,0,0);				\
  long21_re = long00_re;					\
  long21_im = long00_im;
#define READ_LONG_MATRIX_18_SINGLE(gauge, dir, idx)		\
  float2 LONG0 = gauge[idx + ((dir/2)*9+0)*long_ga_stride];	\
  float2 LONG1 = gauge[idx + ((dir/2)*9+1)*long_ga_stride];	\
  float2 LONG2 = gauge[idx + ((dir/2)*9+2)*long_ga_stride];	\
  float2 LONG3 = gauge[idx + ((dir/2)*9+3)*long_ga_stride];	\
  float2 LONG4 = gauge[idx + ((dir/2)*9+4)*long_ga_stride];	\
  float2 LONG5 = gauge[idx + ((dir/2)*9+5)*long_ga_stride];	\
  float2 LONG6 = gauge[idx + ((dir/2)*9+6)*long_ga_stride];	\
  float2 LONG7 = gauge[idx + ((dir/2)*9+7)*long_ga_stride];	\
  float2 LONG8 = gauge[idx + ((dir/2)*9+8)*long_ga_stride];

#define READ_LONG_MATRIX_12_DOUBLE(gauge, dir, idx)		\
  double2 LONG0 = gauge[idx + ((dir/2)*6+0)*long_ga_stride];	\
  double2 LONG1 = gauge[idx + ((dir/2)*6+1)*long_ga_stride];	\
  double2 LONG2 = gauge[idx + ((dir/2)*6+2)*long_ga_stride];	\
  double2 LONG3 = gauge[idx + ((dir/2)*6+3)*long_ga_stride];	\
  double2 LONG4 = gauge[idx + ((dir/2)*6+4)*long_ga_stride];	\
  double2 LONG5 = gauge[idx + ((dir/2)*6+5)*long_ga_stride];	\
  double2 LONG6 = make_double2(0,0);				\
  double2 LONG7 = make_double2(0,0);				\
  double2 LONG8 = make_double2(0,0);				\
  double2 LONG9 = make_double2(0,0);

#define READ_LONG_MATRIX_8_DOUBLE(gauge, dir, idx)		\
  double2 LONG0 = gauge[idx + ((dir/2)*4+0)*long_ga_stride];	\
  double2 LONG1 = gauge[idx + ((dir/2)*4+1)*long_ga_stride];	\
  double2 LONG2 = gauge[idx + ((dir/2)*4+2)*long_ga_stride];	\
  double2 LONG3 = gauge[idx + ((dir/2)*4+3)*long_ga_stride];	\
  double2 LONG4 = make_double2(0,0);				\
  double2 LONG5 = make_double2(0,0);				\
  double2 LONG6 = make_double2(0,0);				\
  double2 LONG7 = make_double2(0,0);				\
  double2 LONG8 = make_double2(0,0);				\
  double2 LONG9 = make_double2(0,0);				\
  long21_re = long00_re;					\
  long21_im = long00_im;

#define READ_LONG_MATRIX_18_DOUBLE(gauge, dir, idx)		\
  double2 LONG0 = gauge[idx + ((dir/2)*9+0)*long_ga_stride];	\
  double2 LONG1 = gauge[idx + ((dir/2)*9+1)*long_ga_stride];	\
  double2 LONG2 = gauge[idx + ((dir/2)*9+2)*long_ga_stride];	\
  double2 LONG3 = gauge[idx + ((dir/2)*9+3)*long_ga_stride];	\
  double2 LONG4 = gauge[idx + ((dir/2)*9+4)*long_ga_stride];	\
  double2 LONG5 = gauge[idx + ((dir/2)*9+5)*long_ga_stride];	\
  double2 LONG6 = gauge[idx + ((dir/2)*9+6)*long_ga_stride];	\
  double2 LONG7 = gauge[idx + ((dir/2)*9+7)*long_ga_stride];	\
  double2 LONG8 = gauge[idx + ((dir/2)*9+8)*long_ga_stride];


#endif //longlink access



#define READ_LONG_MATRIX_8_HALF(gauge, dir, idx)                        \
  float4 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*2+0)*long_ga_stride); \
  float4 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*2+1)*long_ga_stride); \
  float4 LONG2 = make_float4(0,0,0,0);					\
  float4 LONG3 = make_float4(0,0,0,0);					\
  float4 LONG4 = make_float4(0,0,0,0);					\
  long00_re=long21_re = pi_f*long00_re;					\
  long00_im=long21_im = pi_f*long00_im;


#define READ_LONG_MATRIX_12_HALF(gauge, dir, idx)			\
  float4 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*3+0)*long_ga_stride); \
  float4 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*3+1)*long_ga_stride); \
  float4 LONG2 = tex1Dfetch((gauge), idx + ((dir/2)*3+2)*long_ga_stride); \
  float4 LONG3 = make_float4(0,0,0,0);					\
  float4 LONG4 = make_float4(0,0,0,0);



#define READ_LONG_MATRIX_18_HALF(gauge, dir, idx)			\
  float2 LONG0 = tex1Dfetch((gauge), idx + ((dir/2)*9+0)*long_ga_stride); \
  float2 LONG1 = tex1Dfetch((gauge), idx + ((dir/2)*9+1)*long_ga_stride); \
  float2 LONG2 = tex1Dfetch((gauge), idx + ((dir/2)*9+2)*long_ga_stride); \
  float2 LONG3 = tex1Dfetch((gauge), idx + ((dir/2)*9+3)*long_ga_stride); \
  float2 LONG4 = tex1Dfetch((gauge), idx + ((dir/2)*9+4)*long_ga_stride); \
  float2 LONG5 = tex1Dfetch((gauge), idx + ((dir/2)*9+5)*long_ga_stride); \
  float2 LONG6 = tex1Dfetch((gauge), idx + ((dir/2)*9+6)*long_ga_stride); \
  float2 LONG7 = tex1Dfetch((gauge), idx + ((dir/2)*9+7)*long_ga_stride); \
  float2 LONG8 = tex1Dfetch((gauge), idx + ((dir/2)*9+8)*long_ga_stride);
