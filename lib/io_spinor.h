#define READ_SPINOR_DOUBLE(spinor)	                          \
  double2 I0 = fetch_double2((spinor), sp_idx + 0*(sp_stride));   \
  double2 I1 = fetch_double2((spinor), sp_idx + 1*(sp_stride));   \
  double2 I2 = fetch_double2((spinor), sp_idx + 2*(sp_stride));   \
  double2 I3 = fetch_double2((spinor), sp_idx + 3*(sp_stride));   \
  double2 I4 = fetch_double2((spinor), sp_idx + 4*(sp_stride));   \
  double2 I5 = fetch_double2((spinor), sp_idx + 5*(sp_stride));   \
  double2 I6 = fetch_double2((spinor), sp_idx + 6*(sp_stride));   \
  double2 I7 = fetch_double2((spinor), sp_idx + 7*(sp_stride));   \
  double2 I8 = fetch_double2((spinor), sp_idx + 8*(sp_stride));   \
  double2 I9 = fetch_double2((spinor), sp_idx + 9*(sp_stride));   \
  double2 I10 = fetch_double2((spinor), sp_idx + 10*(sp_stride)); \
  double2 I11 = fetch_double2((spinor), sp_idx + 11*(sp_stride));

#define READ_SPINOR_DOUBLE_UP(spinor)		                  \
  double2 I0 = fetch_double2((spinor), sp_idx + 0*(sp_stride));   \
  double2 I1 = fetch_double2((spinor), sp_idx + 1*(sp_stride));   \
  double2 I2 = fetch_double2((spinor), sp_idx + 2*(sp_stride));   \
  double2 I3 = fetch_double2((spinor), sp_idx + 3*(sp_stride));   \
  double2 I4 = fetch_double2((spinor), sp_idx + 4*(sp_stride));   \
  double2 I5 = fetch_double2((spinor), sp_idx + 5*(sp_stride));

#define READ_SPINOR_DOUBLE_DOWN(spinor)		                  \
  double2 I6 = fetch_double2((spinor), sp_idx + 6*(sp_stride));   \
  double2 I7 = fetch_double2((spinor), sp_idx + 7*(sp_stride));   \
  double2 I8 = fetch_double2((spinor), sp_idx + 8*(sp_stride));   \
  double2 I9 = fetch_double2((spinor), sp_idx + 9*(sp_stride));   \
  double2 I10 = fetch_double2((spinor), sp_idx + 10*(sp_stride)); \
  double2 I11 = fetch_double2((spinor), sp_idx + 11*(sp_stride));

#define READ_SPINOR_SINGLE(spinor)		              \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*(sp_stride));   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*(sp_stride));   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*(sp_stride));   \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*(sp_stride));   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*(sp_stride));   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*(sp_stride));

#define READ_SPINOR_SINGLE_UP(spinor)		              \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*(sp_stride));   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*(sp_stride));   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*(sp_stride));   \

#define READ_SPINOR_SINGLE_DOWN(spinor)                       \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*(sp_stride));   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*(sp_stride));   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*(sp_stride));

#define READ_SPINOR_HALF(spinor)                              \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*(sp_stride));   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*(sp_stride));   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*(sp_stride));   \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*(sp_stride));   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*(sp_stride));   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*(sp_stride));   \
  float C = tex1Dfetch((spinorTexNorm), sp_idx);              \
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;	              \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;	              \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;                 \
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;	              \
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;	              \
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_SPINOR_HALF_UP(spinor)		              \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*(sp_stride));   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*(sp_stride));   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*(sp_stride));   \
  float C = tex1Dfetch((spinorTexNorm), sp_idx);              \
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;	              \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;	              \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;                 \

#define READ_SPINOR_HALF_DOWN(spinor)		              \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*(sp_stride));   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*(sp_stride));   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*(sp_stride));   \
  float C = tex1Dfetch((spinorTexNorm), sp_idx);              \
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;	              \
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;	              \
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_ACCUM_DOUBLE(spinor)				   \
  double2 accum0 = fetch_double2((spinor), sid + 0*(sp_stride));   \
  double2 accum1 = fetch_double2((spinor), sid + 1*(sp_stride));   \
  double2 accum2 = fetch_double2((spinor), sid + 2*(sp_stride));   \
  double2 accum3 = fetch_double2((spinor), sid + 3*(sp_stride));   \
  double2 accum4 = fetch_double2((spinor), sid + 4*(sp_stride));   \
  double2 accum5 = fetch_double2((spinor), sid + 5*(sp_stride));   \
  double2 accum6 = fetch_double2((spinor), sid + 6*(sp_stride));   \
  double2 accum7 = fetch_double2((spinor), sid + 7*(sp_stride));   \
  double2 accum8 = fetch_double2((spinor), sid + 8*(sp_stride));   \
  double2 accum9 = fetch_double2((spinor), sid + 9*(sp_stride));   \
  double2 accum10 = fetch_double2((spinor), sid + 10*(sp_stride)); \
  double2 accum11 = fetch_double2((spinor), sid + 11*(sp_stride));	

#define READ_ACCUM_SINGLE(spinor)                                  \
  float4 accum0 = tex1Dfetch((spinor), sid + 0*(sp_stride));       \
  float4 accum1 = tex1Dfetch((spinor), sid + 1*(sp_stride));       \
  float4 accum2 = tex1Dfetch((spinor), sid + 2*(sp_stride));       \
  float4 accum3 = tex1Dfetch((spinor), sid + 3*(sp_stride));       \
  float4 accum4 = tex1Dfetch((spinor), sid + 4*(sp_stride));       \
  float4 accum5 = tex1Dfetch((spinor), sid + 5*(sp_stride)); 

#define READ_ACCUM_HALF(spinor)					   \
  float4 accum0 = tex1Dfetch((spinor), sid + 0*(sp_stride));       \
  float4 accum1 = tex1Dfetch((spinor), sid + 1*(sp_stride));       \
  float4 accum2 = tex1Dfetch((spinor), sid + 2*(sp_stride));       \
  float4 accum3 = tex1Dfetch((spinor), sid + 3*(sp_stride));       \
  float4 accum4 = tex1Dfetch((spinor), sid + 4*(sp_stride));       \
  float4 accum5 = tex1Dfetch((spinor), sid + 5*(sp_stride));       \
  float C = tex1Dfetch((accumTexNorm), sid);		           \
  accum0.x *= C; accum0.y *= C;	accum0.z *= C; accum0.w *= C;      \
  accum1.x *= C; accum1.y *= C;	accum1.z *= C; accum1.w *= C;      \
  accum2.x *= C; accum2.y *= C;	accum2.z *= C; accum2.w *= C;      \
  accum3.x *= C; accum3.y *= C;	accum3.z *= C; accum3.w *= C;      \
  accum4.x *= C; accum4.y *= C; accum4.z *= C; accum4.w *= C;      \
  accum5.x *= C; accum5.y *= C;	accum5.z *= C; accum5.w *= C;					     


#define WRITE_SPINOR_DOUBLE2()					   \
  out[0*(sp_stride)+sid] = make_double2(o00_re, o00_im);	   \
  out[1*(sp_stride)+sid] = make_double2(o01_re, o01_im);	   \
  out[2*(sp_stride)+sid] = make_double2(o02_re, o02_im);	   \
  out[3*(sp_stride)+sid] = make_double2(o10_re, o10_im);	   \
  out[4*(sp_stride)+sid] = make_double2(o11_re, o11_im);	   \
  out[5*(sp_stride)+sid] = make_double2(o12_re, o12_im);	   \
  out[6*(sp_stride)+sid] = make_double2(o20_re, o20_im);	   \
  out[7*(sp_stride)+sid] = make_double2(o21_re, o21_im);	   \
  out[8*(sp_stride)+sid] = make_double2(o22_re, o22_im);	   \
  out[9*(sp_stride)+sid] = make_double2(o30_re, o30_im);	   \
  out[10*(sp_stride)+sid] = make_double2(o31_re, o31_im);	   \
  out[11*(sp_stride)+sid] = make_double2(o32_re, o32_im);		 

#define WRITE_SPINOR_FLOAT4()						\
  out[0*(sp_stride)+sid] = make_float4(o00_re, o00_im, o01_re, o01_im); \
  out[1*(sp_stride)+sid] = make_float4(o02_re, o02_im, o10_re, o10_im); \
  out[2*(sp_stride)+sid] = make_float4(o11_re, o11_im, o12_re, o12_im); \
  out[3*(sp_stride)+sid] = make_float4(o20_re, o20_im, o21_re, o21_im); \
  out[4*(sp_stride)+sid] = make_float4(o22_re, o22_im, o30_re, o30_im); \
  out[5*(sp_stride)+sid] = make_float4(o31_re, o31_im, o32_re, o32_im);

#define WRITE_SPINOR_SHORT4()						\
  float c0 = fmaxf(fabsf(o00_re), fabsf(o00_im));			\
  float c1 = fmaxf(fabsf(o01_re), fabsf(o02_im));			\
  float c2 = fmaxf(fabsf(o02_re), fabsf(o01_im));			\
  float c3 = fmaxf(fabsf(o10_re), fabsf(o10_im));			\
  float c4 = fmaxf(fabsf(o11_re), fabsf(o11_im));			\
  float c5 = fmaxf(fabsf(o12_re), fabsf(o12_im));			\
  float c6 = fmaxf(fabsf(o20_re), fabsf(o20_im));			\
  float c7 = fmaxf(fabsf(o21_re), fabsf(o21_im));			\
  float c8 = fmaxf(fabsf(o22_re), fabsf(o22_im));			\
  float c9 = fmaxf(fabsf(o30_re), fabsf(o30_im));			\
  float c10 = fmaxf(fabsf(o31_re), fabsf(o31_im));			\
  float c11 = fmaxf(fabsf(o32_re), fabsf(o32_im));			\
  c0 = fmaxf(c0, c1);							\
  c1 = fmaxf(c2, c3);							\
  c2 = fmaxf(c4, c5);							\
  c3 = fmaxf(c6, c7);							\
  c4 = fmaxf(c8, c9);							\
  c5 = fmaxf(c10, c11);							\
  c0 = fmaxf(c0, c1);							\
  c1 = fmaxf(c2, c3);							\
  c2 = fmaxf(c4, c5);							\
  c0 = fmaxf(c0, c1);							\
  c0 = fmaxf(c0, c2);							\
  outNorm[sid] = c0;							\
  float scale = __fdividef(MAX_SHORT, c0);				\
  o00_re *= scale; o00_im *= scale; o01_re *= scale; o01_im *= scale;	\
  o02_re *= scale; o02_im *= scale; o10_re *= scale; o10_im *= scale;	\
  o11_re *= scale; o11_im *= scale; o12_re *= scale; o12_im *= scale;	\
  o20_re *= scale; o20_im *= scale; o21_re *= scale; o21_im *= scale;	\
  o22_re *= scale; o22_im *= scale; o30_re *= scale; o30_im *= scale;	\
  o31_re *= scale; o31_im *= scale; o32_re *= scale; o32_im *= scale;	\
  out[sid+0*(sp_stride)] = make_short4((short)o00_re, (short)o00_im, (short)o01_re, (short)o01_im); \
  out[sid+1*(sp_stride)] = make_short4((short)o02_re, (short)o02_im, (short)o10_re, (short)o10_im); \
  out[sid+2*(sp_stride)] = make_short4((short)o11_re, (short)o11_im, (short)o12_re, (short)o12_im); \
  out[sid+3*(sp_stride)] = make_short4((short)o20_re, (short)o20_im, (short)o21_re, (short)o21_im); \
  out[sid+4*(sp_stride)] = make_short4((short)o22_re, (short)o22_im, (short)o30_re, (short)o30_im); \
  out[sid+5*(sp_stride)] = make_short4((short)o31_re, (short)o31_im, (short)o32_re, (short)o32_im);

/*
#define WRITE_SPINOR_FLOAT1_SMEM() \
  int t = threadIdx.x; \
  int B = BLOCK_DIM; \
  int b = blockIdx.x; \
  int f = SHARED_FLOATS_PER_THREAD; \
  __syncthreads(); \
  for (int i = 0; i < 6; i++) for (int c = 0; c < 4; c++) \
      ((float*)out)[i*(Vh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];

// the alternative to writing float4's directly: almost as fast, a lot more confusing
#define WRITE_SPINOR_FLOAT1_STAGGERED() \
  int t = threadIdx.x; \
  int B = BLOCK_DIM; \
  int b = blockIdx.x; \
  int f = SHARED_FLOATS_PER_THREAD; \
  __syncthreads(); \
  for (int i = 0; i < 4; i++) for (int c = 0; c < 4; c++) \
      ((float*)out)[i*(Vh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4]; \
  __syncthreads(); \
  s[0] = o22_re; \
  s[1] = o22_im; \
  s[2] = o30_re; \
  s[3] = o30_im; \
  s[4] = o31_re; \
  s[5] = o31_im; \
  s[6] = o32_re; \
  s[7] = o32_im; \
  __syncthreads(); \
  for (int i = 0; i < 2; i++) for (int c = 0; c < 4; c++) \
    ((float*)out)[(i+4)*(Vh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];
*/


/************* the following is used by staggered *****************/

#ifndef DIRECT_ACCESS_SPINOR //spinor access control

#define READ_1ST_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 I0 = tex1Dfetch((spinor), idx + 0*mystride);		\
  float2 I1 = tex1Dfetch((spinor), idx + 1*mystride);		\
  float2 I2 = tex1Dfetch((spinor), idx + 2*mystride);

#define READ_3RD_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 T0 = tex1Dfetch((spinor), idx + 0*mystride);		\
  float2 T1 = tex1Dfetch((spinor), idx + 1*mystride);		\
  float2 T2 = tex1Dfetch((spinor), idx + 2*mystride);

#define READ_1ST_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 I0 = fetch_double2((spinor), idx + 0*mystride);	\
  double2 I1 = fetch_double2((spinor), idx + 1*mystride);	\
  double2 I2 = fetch_double2((spinor), idx + 2*mystride);

#define READ_3RD_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 T0 = fetch_double2((spinor), idx + 0*mystride);	\
  double2 T1 = fetch_double2((spinor), idx + 1*mystride);	\
  double2 T2 = fetch_double2((spinor), idx + 2*mystride);

#else //spinor access control

#define READ_1ST_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 I0 = spinor[idx + 0*mystride];				\
  float2 I1 = spinor[idx + 1*mystride];				\
  float2 I2 = spinor[idx + 2*mystride];

#define READ_3RD_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 T0 = spinor[idx + 0*mystride];				\
  float2 T1 = spinor[idx + 1*mystride];				\
  float2 T2 = spinor[idx + 2*mystride];

#define READ_1ST_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 I0 = spinor[idx + 0*mystride];			\
  double2 I1 = spinor[idx + 1*mystride];			\
  double2 I2 = spinor[idx + 2*mystride];

#define READ_3RD_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 T0 = spinor[idx + 0*mystride];			\
  double2 T1 = spinor[idx + 1*mystride];			\
  double2 T2 = spinor[idx + 2*mystride];

#endif //spinor access control

#define READ_1ST_NBR_SPINOR_HALF(spinor, idx, mystride)			\
  float2 I0 = tex1Dfetch((spinor), idx + 0*mystride);			\
  float2 I1 = tex1Dfetch((spinor), idx + 1*mystride);			\
  float2 I2 = tex1Dfetch((spinor), idx + 2*mystride);			\
  {float C = tex1Dfetch((spinorTexNorm), idx);				\
    I0.x *= C; I0.y *= C;						\
    I1.x *= C; I1.y *= C;						\
    I2.x *= C; I2.y *= C;}

#define READ_3RD_NBR_SPINOR_HALF(spinor, idx, mystride)			\
  float2 T0 = tex1Dfetch((spinor), idx + 0*mystride);			\
  float2 T1 = tex1Dfetch((spinor), idx + 1*mystride);			\
  float2 T2 = tex1Dfetch((spinor), idx + 2*mystride);			\
  {float C = tex1Dfetch((spinorTexNorm), idx);				\
    T0.x *= C; T0.y *= C;						\
    T1.x *= C; T1.y *= C;						\
    T2.x *= C; T2.y *= C;}


#define WRITE_ST_SPINOR_DOUBLE2()				\
  g_out[0*sp_stride+sid] = make_double2(o00_re, o00_im);	\
  g_out[1*sp_stride+sid] = make_double2(o01_re, o01_im);	\
  g_out[2*sp_stride+sid] = make_double2(o02_re, o02_im);

#define WRITE_ST_SPINOR_FLOAT2()			\
  g_out[0*sp_stride+sid] = make_float2(o00_re, o00_im);	\
  g_out[1*sp_stride+sid] = make_float2(o01_re, o01_im);	\
  g_out[2*sp_stride+sid] = make_float2(o02_re, o02_im);

#define WRITE_ST_SPINOR_SHORT2()					\
  float c0 = fmaxf(fabsf(o00_re), fabsf(o00_im));			\
  float c1 = fmaxf(fabsf(o01_re), fabsf(o01_im));			\
  float c2 = fmaxf(fabsf(o02_re), fabsf(o02_im));			\
  c0 = fmaxf(c0, c1);							\
  c0 = fmaxf(c0, c2);							\
  outNorm[sid] = c0;							\
  float scale = __fdividef(MAX_SHORT, c0);				\
  o00_re *= scale; o00_im *= scale; o01_re *= scale; o01_im *= scale;	\
  o02_re *= scale; o02_im *= scale;					\
  g_out[sid+0*sp_stride] = make_short2((short)o00_re, (short)o00_im);	\
  g_out[sid+1*sp_stride] = make_short2((short)o01_re, (short)o01_im);	\
  g_out[sid+2*sp_stride] = make_short2((short)o02_re, (short)o02_im);

#define READ_AND_SUM_ST_SPINOR()					\
  o00_re += g_out[0*sp_stride+sid].x; o00_im += g_out[0*sp_stride+sid].y; \
  o01_re += g_out[1*sp_stride+sid].x; o01_im += g_out[1*sp_stride+sid].y; \
  o02_re += g_out[2*sp_stride+sid].x; o02_im += g_out[2*sp_stride+sid].y; \
  
#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) * 0.5)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))
#define short2float(a) ( __fdividef(a, SCALE_FLOAT) - SHIFT_FLOAT)

#define READ_AND_SUM_ST_SPINOR_HALF()			\
  float C = outNorm[sid];				\
  o00_re += C*short2float(g_out[0*sp_stride + sid].x);	\
  o00_im += C*short2float(g_out[0*sp_stride + sid].y);	\
  o01_re += C*short2float(g_out[1*sp_stride + sid].x);	\
  o01_im += C*short2float(g_out[1*sp_stride + sid].y);	\
  o02_re += C*short2float(g_out[2*sp_stride + sid].x);	\
  o02_im += C*short2float(g_out[2*sp_stride + sid].y);	
  
#define READ_ST_ACCUM_SINGLE(spinor)				\
  float2 accum0 = tex1Dfetch((spinor), sid + 0*sp_stride);	\
  float2 accum1 = tex1Dfetch((spinor), sid + 1*sp_stride);	\
  float2 accum2 = tex1Dfetch((spinor), sid + 2*sp_stride);     

#define READ_ST_SPINOR_HALF(spinor)				\
  float2 I0 = tex1Dfetch((spinor), sp_idx + 0*sp_stride);	\
  float2 I1 = tex1Dfetch((spinor), sp_idx + 1*sp_stride);	\
  float2 I2 = tex1Dfetch((spinor), sp_idx + 2*sp_stride);	\
  float C = tex1Dfetch((spinorTexNorm), sp_idx);		\
  I0.x *= C; I0.y *= C;						\
  I1.x *= C; I1.y *= C;						\
  I2.x *= C; I2.y *= C;                                  


#define READ_ST_ACCUM_HALF(spinor)				\
  float2 accum0 = tex1Dfetch((spinor), sid + 0*sp_stride);	\
  float2 accum1 = tex1Dfetch((spinor), sid + 1*sp_stride);	\
  float2 accum2 = tex1Dfetch((spinor), sid + 2*sp_stride);	\
  float C = tex1Dfetch((accumTexNorm), sid);			\
  accum0.x *= C; accum0.y *= C;					\
  accum1.x *= C; accum1.y *= C;					\
  accum2.x *= C; accum2.y *= C;       
