#define READ_SPINOR_DOUBLE(spinor, stride, sp_idx, norm_idx)	\
  double2 I0 = spinor[sp_idx + 0*(stride)];   \
  double2 I1 = spinor[sp_idx + 1*(stride)];   \
  double2 I2 = spinor[sp_idx + 2*(stride)];   \
  double2 I3 = spinor[sp_idx + 3*(stride)];   \
  double2 I4 = spinor[sp_idx + 4*(stride)];   \
  double2 I5 = spinor[sp_idx + 5*(stride)];   \
  double2 I6 = spinor[sp_idx + 6*(stride)];   \
  double2 I7 = spinor[sp_idx + 7*(stride)];   \
  double2 I8 = spinor[sp_idx + 8*(stride)];   \
  double2 I9 = spinor[sp_idx + 9*(stride)];   \
  double2 I10 = spinor[sp_idx + 10*(stride)]; \
  double2 I11 = spinor[sp_idx + 11*(stride)];

#define READ_SPINOR_DOUBLE_UP(spinor, stride, sp_idx, norm_idx) \
  double2 I0 = spinor[sp_idx + 0*(stride)];   \
  double2 I1 = spinor[sp_idx + 1*(stride)];   \
  double2 I2 = spinor[sp_idx + 2*(stride)];   \
  double2 I3 = spinor[sp_idx + 3*(stride)];   \
  double2 I4 = spinor[sp_idx + 4*(stride)];   \
  double2 I5 = spinor[sp_idx + 5*(stride)];

#define READ_SPINOR_DOUBLE_DOWN(spinor, stride, sp_idx, norm_idx)      \
  double2 I6 = spinor[sp_idx + 6*(stride)];   \
  double2 I7 = spinor[sp_idx + 7*(stride)];   \
  double2 I8 = spinor[sp_idx + 8*(stride)];   \
  double2 I9 = spinor[sp_idx + 9*(stride)];   \
  double2 I10 = spinor[sp_idx + 10*(stride)]; \
  double2 I11 = spinor[sp_idx + 11*(stride)];

#define READ_SPINOR_SINGLE(spinor, stride, sp_idx, norm_idx)	   \
  float4 I0 = spinor[sp_idx + 0*(stride)];   \
  float4 I1 = spinor[sp_idx + 1*(stride)];   \
  float4 I2 = spinor[sp_idx + 2*(stride)];   \
  float4 I3 = spinor[sp_idx + 3*(stride)];   \
  float4 I4 = spinor[sp_idx + 4*(stride)];   \
  float4 I5 = spinor[sp_idx + 5*(stride)];

#define READ_SPINOR_SINGLE_UP(spinor, stride, sp_idx, norm_idx)	   \
  float4 I0 = spinor[sp_idx + 0*(stride)];   \
  float4 I1 = spinor[sp_idx + 1*(stride)];   \
  float4 I2 = spinor[sp_idx + 2*(stride)];   \

#define READ_SPINOR_SINGLE_DOWN(spinor, stride, sp_idx, norm_idx)  \
  float4 I3 = spinor[sp_idx + 3*(stride)];   \
  float4 I4 = spinor[sp_idx + 4*(stride)];   \
  float4 I5 = spinor[sp_idx + 5*(stride)];

#define READ_SPINOR_HALF_(spinor, stride, sp_idx, norm_idx)	   \
  float4 I0 = short42float4(spinor[sp_idx + 0*(stride)]);	   \
  float4 I1 = short42float4(spinor[sp_idx + 1*(stride)]);	   \
  float4 I2 = short42float4(spinor[sp_idx + 2*(stride)]);	   \
  float4 I3 = short42float4(spinor[sp_idx + 3*(stride)]);	   \
  float4 I4 = short42float4(spinor[sp_idx + 4*(stride)]);	   \
  float4 I5 = short42float4(spinor[sp_idx + 5*(stride)]);	   \
  float C = (spinor ## Norm)[norm_idx];			   \
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;			   \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;			   \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;			   \
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;			   \
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;			   \
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_SPINOR_HALF(spinor, stride, sp_idx, norm_idx)	   \
  READ_SPINOR_HALF_(spinor, stride, sp_idx, norm_idx)		   

#define READ_SPINOR_HALF_UP_(spinor, stride, sp_idx, norm_idx)	      \
  float4 I0 = short42float4(spinor[sp_idx + 0*(stride)]);	      \
  float4 I1 = short42float4(spinor[sp_idx + 1*(stride)]);	      \
  float4 I2 = short42float4(spinor[sp_idx + 2*(stride)]);	      \
  float C = (spinor ## Norm)[norm_idx];				      \
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;			      \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;			      \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;			      \

#define READ_SPINOR_HALF_UP(spinor, stride, sp_idx, norm_idx)	\
  READ_SPINOR_HALF_UP_(spinor, stride, sp_idx, norm_idx)		

#define READ_SPINOR_HALF_DOWN_(spinor, stride, sp_idx, norm_idx)	\
  float4 I3 = short42float4(spinor[sp_idx + 3*stride]);			\
  float4 I4 = short42float4(spinor[sp_idx + 4*stride]);			\
  float4 I5 = short42float4(spinor[sp_idx + 5*stride]);			\
  float C = (spinor ## Norm)[norm_idx];					\
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;				\
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;				\
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_SPINOR_HALF_DOWN(spinor, stride, sp_idx, norm_idx)	\
  READ_SPINOR_HALF_DOWN_(spinor, stride, sp_idx, norm_idx)		

#define READ_ACCUM_DOUBLE(spinor, stride)     \
  double2 accum0 = spinor[sid + 0*stride];   \
  double2 accum1 = spinor[sid + 1*stride];   \
  double2 accum2 = spinor[sid + 2*stride];   \
  double2 accum3 = spinor[sid + 3*stride];   \
  double2 accum4 = spinor[sid + 4*stride];   \
  double2 accum5 = spinor[sid + 5*stride];   \
  double2 accum6 = spinor[sid + 6*stride];   \
  double2 accum7 = spinor[sid + 7*stride];   \
  double2 accum8 = spinor[sid + 8*stride];   \
  double2 accum9 = spinor[sid + 9*stride];  \
  double2 accum10 = spinor[sid + 10*stride]; \
  double2 accum11 = spinor[sid + 11*stride];	

#define READ_ACCUM_SINGLE(spinor, stride)			\
  float4 accum0 = spinor[sid + 0*(stride)];       \
  float4 accum1 = spinor[sid + 1*(stride)];       \
  float4 accum2 = spinor[sid + 2*(stride)];       \
  float4 accum3 = spinor[sid + 3*(stride)];       \
  float4 accum4 = spinor[sid + 4*(stride)];       \
  float4 accum5 = spinor[sid + 5*(stride)]; 

#define READ_ACCUM_HALF_(spinor, stride)			   \
  float4 accum0 = short42float4(spinor[sid + 0*stride]);	   \
  float4 accum1 = short42float4(spinor[sid + 1*stride]);	   \
  float4 accum2 = short42float4(spinor[sid + 2*stride]);	   \
  float4 accum3 = short42float4(spinor[sid + 3*stride]);	   \
  float4 accum4 = short42float4(spinor[sid + 4*stride]);	   \
  float4 accum5 = short42float4(spinor[sid + 5*stride]);	   \
  float C = (spinor ## Norm)[sid];				   \
  accum0.x *= C; accum0.y *= C;	accum0.z *= C; accum0.w *= C;      \
  accum1.x *= C; accum1.y *= C;	accum1.z *= C; accum1.w *= C;      \
  accum2.x *= C; accum2.y *= C;	accum2.z *= C; accum2.w *= C;      \
  accum3.x *= C; accum3.y *= C;	accum3.z *= C; accum3.w *= C;      \
  accum4.x *= C; accum4.y *= C; accum4.z *= C; accum4.w *= C;      \
  accum5.x *= C; accum5.y *= C;	accum5.z *= C; accum5.w *= C;					     

#define READ_ACCUM_HALF(spinor, stride) READ_ACCUM_HALF_(spinor, stride)

#define READ_SPINOR_DOUBLE_TEX(spinor, stride, sp_idx, norm_idx)	\
  double2 I0 = fetch_double2((spinor), sp_idx + 0*(stride));   \
  double2 I1 = fetch_double2((spinor), sp_idx + 1*(stride));   \
  double2 I2 = fetch_double2((spinor), sp_idx + 2*(stride));   \
  double2 I3 = fetch_double2((spinor), sp_idx + 3*(stride));   \
  double2 I4 = fetch_double2((spinor), sp_idx + 4*(stride));   \
  double2 I5 = fetch_double2((spinor), sp_idx + 5*(stride));   \
  double2 I6 = fetch_double2((spinor), sp_idx + 6*(stride));   \
  double2 I7 = fetch_double2((spinor), sp_idx + 7*(stride));   \
  double2 I8 = fetch_double2((spinor), sp_idx + 8*(stride));   \
  double2 I9 = fetch_double2((spinor), sp_idx + 9*(stride));   \
  double2 I10 = fetch_double2((spinor), sp_idx + 10*(stride)); \
  double2 I11 = fetch_double2((spinor), sp_idx + 11*(stride));

#define READ_SPINOR_DOUBLE_UP_TEX(spinor, stride, sp_idx, norm_idx) \
  double2 I0 = fetch_double2((spinor), sp_idx + 0*(stride));   \
  double2 I1 = fetch_double2((spinor), sp_idx + 1*(stride));   \
  double2 I2 = fetch_double2((spinor), sp_idx + 2*(stride));   \
  double2 I3 = fetch_double2((spinor), sp_idx + 3*(stride));   \
  double2 I4 = fetch_double2((spinor), sp_idx + 4*(stride));   \
  double2 I5 = fetch_double2((spinor), sp_idx + 5*(stride));

#define READ_SPINOR_DOUBLE_DOWN_TEX(spinor, stride, sp_idx, norm_idx)      \
  double2 I6 = fetch_double2((spinor), sp_idx + 6*(stride));   \
  double2 I7 = fetch_double2((spinor), sp_idx + 7*(stride));   \
  double2 I8 = fetch_double2((spinor), sp_idx + 8*(stride));   \
  double2 I9 = fetch_double2((spinor), sp_idx + 9*(stride));   \
  double2 I10 = fetch_double2((spinor), sp_idx + 10*(stride)); \
  double2 I11 = fetch_double2((spinor), sp_idx + 11*(stride));

#define READ_ACCUM_DOUBLE_TEX(spinor, stride)			\
  double2 accum0 = fetch_double2((spinor), sid + 0*(stride));   \
  double2 accum1 = fetch_double2((spinor), sid + 1*(stride));   \
  double2 accum2 = fetch_double2((spinor), sid + 2*(stride));   \
  double2 accum3 = fetch_double2((spinor), sid + 3*(stride));   \
  double2 accum4 = fetch_double2((spinor), sid + 4*(stride));   \
  double2 accum5 = fetch_double2((spinor), sid + 5*(stride));   \
  double2 accum6 = fetch_double2((spinor), sid + 6*(stride));   \
  double2 accum7 = fetch_double2((spinor), sid + 7*(stride));   \
  double2 accum8 = fetch_double2((spinor), sid + 8*(stride));   \
  double2 accum9 = fetch_double2((spinor), sid + 9*(stride));   \
  double2 accum10 = fetch_double2((spinor), sid + 10*(stride)); \
  double2 accum11 = fetch_double2((spinor), sid + 11*(stride));	

#define READ_SPINOR_SINGLE_TEX(spinor, stride, sp_idx, norm_idx)	\
  float4 I0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 I1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 I2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\
  float4 I3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));	\
  float4 I4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));	\
  float4 I5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));

#define READ_SPINOR_SINGLE_UP_TEX(spinor, stride, sp_idx, norm_idx)	\
  float4 I0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 I1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 I2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\

#define READ_SPINOR_SINGLE_DOWN_TEX(spinor, stride, sp_idx, norm_idx)  \
  float4 I3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));       \
  float4 I4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));       \
  float4 I5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));

#define READ_SPINOR_HALF_TEX_(spinor, stride, sp_idx, norm_idx)	\
  float4 I0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 I1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 I2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\
  float4 I3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));	\
  float4 I4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));	\
  float4 I5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));	\
  float C = TEX1DFETCH(float, (spinor ## Norm), norm_idx);		\
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;	              \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;	              \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;                 \
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;	              \
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;	              \
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_SPINOR_HALF_TEX(spinor, stride, sp_idx, norm_idx)	   \
  READ_SPINOR_HALF_TEX_(spinor, stride, sp_idx, norm_idx)	   \

#define READ_SPINOR_HALF_UP_TEX_(spinor, stride, sp_idx, norm_idx)	\
  float4 I0 = TEX1DFETCH(float4, (spinor), sp_idx + 0*(stride));	\
  float4 I1 = TEX1DFETCH(float4, (spinor), sp_idx + 1*(stride));	\
  float4 I2 = TEX1DFETCH(float4, (spinor), sp_idx + 2*(stride));	\
  float C = TEX1DFETCH(float, (spinor ## Norm), norm_idx);		\
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;	              \
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;	              \
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;                 \

#define READ_SPINOR_HALF_UP_TEX(spinor, stride, sp_idx, norm_idx) \
  READ_SPINOR_HALF_UP_TEX_(spinor, stride, sp_idx, norm_idx)	  \

#define READ_SPINOR_HALF_DOWN_TEX_(spinor, stride, sp_idx, norm_idx)	\
  float4 I3 = TEX1DFETCH(float4, (spinor), sp_idx + 3*(stride));	\
  float4 I4 = TEX1DFETCH(float4, (spinor), sp_idx + 4*(stride));	\
  float4 I5 = TEX1DFETCH(float4, (spinor), sp_idx + 5*(stride));	\
  float C = TEX1DFETCH(float, (spinor ## Norm), norm_idx);		\
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;	              \
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;	              \
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;					     

#define READ_SPINOR_HALF_DOWN_TEX(spinor, stride, sp_idx, norm_idx)	\
  READ_SPINOR_HALF_DOWN_TEX_(spinor, stride, sp_idx, norm_idx)	\

#define READ_ACCUM_SINGLE_TEX(spinor, stride)			\
  float4 accum0 = TEX1DFETCH(float4, (spinor), sid + 0*(stride));	\
  float4 accum1 = TEX1DFETCH(float4, (spinor), sid + 1*(stride));	\
  float4 accum2 = TEX1DFETCH(float4, (spinor), sid + 2*(stride));	\
  float4 accum3 = TEX1DFETCH(float4, (spinor), sid + 3*(stride));	\
  float4 accum4 = TEX1DFETCH(float4, (spinor), sid + 4*(stride));	\
  float4 accum5 = TEX1DFETCH(float4, (spinor), sid + 5*(stride)); 

#define READ_ACCUM_HALF_TEX_(spinor, stride)				\
  float4 accum0 = TEX1DFETCH(float4, (spinor), sid + 0*(stride));	\
  float4 accum1 = TEX1DFETCH(float4, (spinor), sid + 1*(stride));	\
  float4 accum2 = TEX1DFETCH(float4, (spinor), sid + 2*(stride));	\
  float4 accum3 = TEX1DFETCH(float4, (spinor), sid + 3*(stride));	\
  float4 accum4 = TEX1DFETCH(float4, (spinor), sid + 4*(stride));	\
  float4 accum5 = TEX1DFETCH(float4, (spinor), sid + 5*(stride));	\
  float C = TEX1DFETCH(float, (spinor ## Norm), sid);			\
  accum0.x *= C; accum0.y *= C;	accum0.z *= C; accum0.w *= C;		\
  accum1.x *= C; accum1.y *= C;	accum1.z *= C; accum1.w *= C;		\
  accum2.x *= C; accum2.y *= C;	accum2.z *= C; accum2.w *= C;		\
  accum3.x *= C; accum3.y *= C;	accum3.z *= C; accum3.w *= C;		\
  accum4.x *= C; accum4.y *= C; accum4.z *= C; accum4.w *= C;		\
  accum5.x *= C; accum5.y *= C;	accum5.z *= C; accum5.w *= C;					     

#define READ_ACCUM_HALF_TEX(spinor, stride) READ_ACCUM_HALF_TEX_(spinor, stride)


#define WRITE_SPINOR_DOUBLE2(stride)			   \
  out[0*(stride)+sid] = make_double2(o00_re, o00_im);	   \
  out[1*(stride)+sid] = make_double2(o01_re, o01_im);	   \
  out[2*(stride)+sid] = make_double2(o02_re, o02_im);	   \
  out[3*(stride)+sid] = make_double2(o10_re, o10_im);	   \
  out[4*(stride)+sid] = make_double2(o11_re, o11_im);	   \
  out[5*(stride)+sid] = make_double2(o12_re, o12_im);	   \
  out[6*(stride)+sid] = make_double2(o20_re, o20_im);	   \
  out[7*(stride)+sid] = make_double2(o21_re, o21_im);	   \
  out[8*(stride)+sid] = make_double2(o22_re, o22_im);	   \
  out[9*(stride)+sid] = make_double2(o30_re, o30_im);	   \
  out[10*(stride)+sid] = make_double2(o31_re, o31_im);	   \
  out[11*(stride)+sid] = make_double2(o32_re, o32_im);		 

#define WRITE_SPINOR_FLOAT4(stride)				     \
  out[0*(stride)+sid] = make_float4(o00_re, o00_im, o01_re, o01_im); \
  out[1*(stride)+sid] = make_float4(o02_re, o02_im, o10_re, o10_im); \
  out[2*(stride)+sid] = make_float4(o11_re, o11_im, o12_re, o12_im); \
  out[3*(stride)+sid] = make_float4(o20_re, o20_im, o21_re, o21_im); \
  out[4*(stride)+sid] = make_float4(o22_re, o22_im, o30_re, o30_im); \
  out[5*(stride)+sid] = make_float4(o31_re, o31_im, o32_re, o32_im);

#define WRITE_SPINOR_SHORT4(stride)					\
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
  out[sid+0*(stride)] = make_short4((short)o00_re, (short)o00_im, (short)o01_re, (short)o01_im); \
  out[sid+1*(stride)] = make_short4((short)o02_re, (short)o02_im, (short)o10_re, (short)o10_im); \
  out[sid+2*(stride)] = make_short4((short)o11_re, (short)o11_im, (short)o12_re, (short)o12_im); \
  out[sid+3*(stride)] = make_short4((short)o20_re, (short)o20_im, (short)o21_re, (short)o21_im); \
  out[sid+4*(stride)] = make_short4((short)o22_re, (short)o22_im, (short)o30_re, (short)o30_im); \
  out[sid+5*(stride)] = make_short4((short)o31_re, (short)o31_im, (short)o32_re, (short)o32_im);

#if (__COMPUTE_CAPABILITY__ >= 200)
#define WRITE_SPINOR_DOUBLE2_STR(stride)				\
  store_streaming_double2(&out[0*sp_stride+sid], o00_re, o00_im);	\
  store_streaming_double2(&out[1*sp_stride+sid], o01_re, o01_im);	\
  store_streaming_double2(&out[2*sp_stride+sid], o02_re, o02_im);	\
  store_streaming_double2(&out[3*sp_stride+sid], o10_re, o10_im);	\
  store_streaming_double2(&out[4*sp_stride+sid], o11_re, o11_im);	\
  store_streaming_double2(&out[5*sp_stride+sid], o12_re, o12_im);	\
  store_streaming_double2(&out[6*sp_stride+sid], o20_re, o20_im);	\
  store_streaming_double2(&out[7*sp_stride+sid], o21_re, o21_im);	\
  store_streaming_double2(&out[8*sp_stride+sid], o22_re, o22_im);	\
  store_streaming_double2(&out[9*sp_stride+sid], o30_re, o30_im);	\
  store_streaming_double2(&out[10*sp_stride+sid], o31_re, o31_im);	\
  store_streaming_double2(&out[11*sp_stride+sid], o32_re, o32_im);

#define WRITE_SPINOR_FLOAT4_STR(stride)					\
  store_streaming_float4(&out[0*(stride)+sid], o00_re, o00_im, o01_re, o01_im); \
  store_streaming_float4(&out[1*(stride)+sid], o02_re, o02_im, o10_re, o10_im); \
  store_streaming_float4(&out[2*(stride)+sid], o11_re, o11_im, o12_re, o12_im); \
  store_streaming_float4(&out[3*(stride)+sid], o20_re, o20_im, o21_re, o21_im); \
  store_streaming_float4(&out[4*(stride)+sid], o22_re, o22_im, o30_re, o30_im); \
  store_streaming_float4(&out[5*(stride)+sid], o31_re, o31_im, o32_re, o32_im);

#define WRITE_SPINOR_SHORT4_STR(stride)		\
  float c0 = fmaxf(fabsf(o00_re), fabsf(o00_im));	\
  float c1 = fmaxf(fabsf(o01_re), fabsf(o02_im));	\
  float c2 = fmaxf(fabsf(o02_re), fabsf(o01_im));	\
  float c3 = fmaxf(fabsf(o10_re), fabsf(o10_im));	\
  float c4 = fmaxf(fabsf(o11_re), fabsf(o11_im));	\
  float c5 = fmaxf(fabsf(o12_re), fabsf(o12_im));	\
  float c6 = fmaxf(fabsf(o20_re), fabsf(o20_im));	\
  float c7 = fmaxf(fabsf(o21_re), fabsf(o21_im));	\
  float c8 = fmaxf(fabsf(o22_re), fabsf(o22_im));	\
  float c9 = fmaxf(fabsf(o30_re), fabsf(o30_im));	\
  float c10 = fmaxf(fabsf(o31_re), fabsf(o31_im));	\
  float c11 = fmaxf(fabsf(o32_re), fabsf(o32_im));	\
  c0 = fmaxf(c0, c1);					\
  c1 = fmaxf(c2, c3);					\
  c2 = fmaxf(c4, c5);					\
  c3 = fmaxf(c6, c7);					\
  c4 = fmaxf(c8, c9);					\
  c5 = fmaxf(c10, c11);					\
  c0 = fmaxf(c0, c1);					\
  c1 = fmaxf(c2, c3);					\
  c2 = fmaxf(c4, c5);					\
  c0 = fmaxf(c0, c1);					\
  c0 = fmaxf(c0, c2);					\
  outNorm[sid] = c0;					\
  float scale = __fdividef(MAX_SHORT, c0);			    \
  o00_re *= scale; o00_im *= scale; o01_re *= scale; o01_im *= scale;	\
  o02_re *= scale; o02_im *= scale; o10_re *= scale; o10_im *= scale;	\
  o11_re *= scale; o11_im *= scale; o12_re *= scale; o12_im *= scale;	\
  o20_re *= scale; o20_im *= scale; o21_re *= scale; o21_im *= scale;	\
  o22_re *= scale; o22_im *= scale; o30_re *= scale; o30_im *= scale;	\
  o31_re *= scale; o31_im *= scale; o32_re *= scale; o32_im *= scale;	\
  store_streaming_short4(&out[0*(stride)+sid], (short)o00_re, (short)o00_im, (short)o01_re, (short)o01_im); \
  store_streaming_short4(&out[1*(stride)+sid], (short)o02_re, (short)o02_im, (short)o10_re, (short)o10_im); \
  store_streaming_short4(&out[2*(stride)+sid], (short)o11_re, (short)o11_im, (short)o12_re, (short)o12_im); \
  store_streaming_short4(&out[3*(stride)+sid], (short)o20_re, (short)o20_im, (short)o21_re, (short)o21_im); \
  store_streaming_short4(&out[4*(stride)+sid], (short)o22_re, (short)o22_im, (short)o30_re, (short)o30_im); \
  store_streaming_short4(&out[5*(stride)+sid], (short)o31_re, (short)o31_im, (short)o32_re, (short)o32_im);
#else
#define WRITE_SPINOR_DOUBLE2_STR(stride) WRITE_SPINOR_DOUBLE2(stride)
#define WRITE_SPINOR_FLOAT4_STR(stride) WRITE_SPINOR_FLOAT4(stride)
#define WRITE_SPINOR_SHORT4_STR(stride) WRITE_SPINOR_SHORT4(stride)
#endif

// macros used for exterior Wilson Dslash kernels and face packing

#define READ_HALF_SPINOR READ_SPINOR_UP

#define WRITE_HALF_SPINOR_DOUBLE2(stride, sid)	     \
  out[0*(stride)+sid] = make_double2(a0_re, a0_im);  \
  out[1*(stride)+sid] = make_double2(a1_re, a1_im);  \
  out[2*(stride)+sid] = make_double2(a2_re, a2_im);  \
  out[3*(stride)+sid] = make_double2(b0_re, b0_im);  \
  out[4*(stride)+sid] = make_double2(b1_re, b1_im);  \
  out[5*(stride)+sid] = make_double2(b2_re, b2_im);

#define WRITE_HALF_SPINOR_FLOAT4(stride, sid)			  \
  out[0*(stride)+sid] = make_float4(a0_re, a0_im, a1_re, a1_im);  \
  out[1*(stride)+sid] = make_float4(a2_re, a2_im, b0_re, b0_im);  \
  out[2*(stride)+sid] = make_float4(b1_re, b1_im, b2_re, b2_im);

#define WRITE_HALF_SPINOR_SHORT4(stride, sid)				\
  float c0 = fmaxf(fabsf(a0_re), fabsf(a0_im));				\
  float c1 = fmaxf(fabsf(a1_re), fabsf(a1_im));				\
  float c2 = fmaxf(fabsf(a2_re), fabsf(a2_im));				\
  float c3 = fmaxf(fabsf(b0_re), fabsf(b0_im));				\
  float c4 = fmaxf(fabsf(b1_re), fabsf(b1_im));				\
  float c5 = fmaxf(fabsf(b2_re), fabsf(b2_im));				\
  c0 = fmaxf(c0, c1);							\
  c1 = fmaxf(c2, c3);							\
  c2 = fmaxf(c4, c5);							\
  c0 = fmaxf(c0, c1);							\
  c0 = fmaxf(c0, c2);							\
  outNorm[sid] = c0;							\
  float scale = __fdividef(MAX_SHORT, c0);				\
  a0_re *= scale; a0_im *= scale; a1_re *= scale; a1_im *= scale;	\
  a2_re *= scale; a2_im *= scale; b0_re *= scale; b0_im *= scale;	\
  b1_re *= scale; b1_im *= scale; b2_re *= scale; b2_im *= scale;	\
  out[sid+0*(stride)] = make_short4((short)a0_re, (short)a0_im, (short)a1_re, (short)a1_im); \
  out[sid+1*(stride)] = make_short4((short)a2_re, (short)a2_im, (short)b0_re, (short)b0_im); \
  out[sid+2*(stride)] = make_short4((short)b1_re, (short)b1_im, (short)b2_re, (short)b2_im);

//!ndeg tm:
/******************used by non-degenerate twisted mass**********************/
#define WRITE_FLAVOR_SPINOR_DOUBLE2()					   \
  out[0*(sp_stride)+sid] = make_double2(o1_00_re, o1_00_im);	   \
  out[1*(sp_stride)+sid] = make_double2(o1_01_re, o1_01_im);	   \
  out[2*(sp_stride)+sid] = make_double2(o1_02_re, o1_02_im);	   \
  out[3*(sp_stride)+sid] = make_double2(o1_10_re, o1_10_im);	   \
  out[4*(sp_stride)+sid] = make_double2(o1_11_re, o1_11_im);	   \
  out[5*(sp_stride)+sid] = make_double2(o1_12_re, o1_12_im);	   \
  out[6*(sp_stride)+sid] = make_double2(o1_20_re, o1_20_im);	   \
  out[7*(sp_stride)+sid] = make_double2(o1_21_re, o1_21_im);	   \
  out[8*(sp_stride)+sid] = make_double2(o1_22_re, o1_22_im);	   \
  out[9*(sp_stride)+sid] = make_double2(o1_30_re, o1_30_im);	   \
  out[10*(sp_stride)+sid] = make_double2(o1_31_re, o1_31_im);	   \
  out[11*(sp_stride)+sid] = make_double2(o1_32_re, o1_32_im);          \
  out[0*(sp_stride)+sid+fl_stride] = make_double2(o2_00_re, o2_00_im);	   \
  out[1*(sp_stride)+sid+fl_stride] = make_double2(o2_01_re, o2_01_im);	   \
  out[2*(sp_stride)+sid+fl_stride] = make_double2(o2_02_re, o2_02_im);	   \
  out[3*(sp_stride)+sid+fl_stride] = make_double2(o2_10_re, o2_10_im);	   \
  out[4*(sp_stride)+sid+fl_stride] = make_double2(o2_11_re, o2_11_im);	   \
  out[5*(sp_stride)+sid+fl_stride] = make_double2(o2_12_re, o2_12_im);	   \
  out[6*(sp_stride)+sid+fl_stride] = make_double2(o2_20_re, o2_20_im);	   \
  out[7*(sp_stride)+sid+fl_stride] = make_double2(o2_21_re, o2_21_im);	   \
  out[8*(sp_stride)+sid+fl_stride] = make_double2(o2_22_re, o2_22_im);	   \
  out[9*(sp_stride)+sid+fl_stride] = make_double2(o2_30_re, o2_30_im);	   \
  out[10*(sp_stride)+sid+fl_stride] = make_double2(o2_31_re, o2_31_im);	   \
  out[11*(sp_stride)+sid+fl_stride] = make_double2(o2_32_re, o2_32_im);		 
  
  
#define WRITE_FLAVOR_SPINOR_FLOAT4()						\
  out[0*(sp_stride)+sid] = make_float4(o1_00_re, o1_00_im, o1_01_re, o1_01_im); \
  out[1*(sp_stride)+sid] = make_float4(o1_02_re, o1_02_im, o1_10_re, o1_10_im); \
  out[2*(sp_stride)+sid] = make_float4(o1_11_re, o1_11_im, o1_12_re, o1_12_im); \
  out[3*(sp_stride)+sid] = make_float4(o1_20_re, o1_20_im, o1_21_re, o1_21_im); \
  out[4*(sp_stride)+sid] = make_float4(o1_22_re, o1_22_im, o1_30_re, o1_30_im); \
  out[5*(sp_stride)+sid] = make_float4(o1_31_re, o1_31_im, o1_32_re, o1_32_im); \
  out[0*(sp_stride)+sid+fl_stride] = make_float4(o2_00_re, o2_00_im, o2_01_re, o2_01_im); \
  out[1*(sp_stride)+sid+fl_stride] = make_float4(o2_02_re, o2_02_im, o2_10_re, o2_10_im); \
  out[2*(sp_stride)+sid+fl_stride] = make_float4(o2_11_re, o2_11_im, o2_12_re, o2_12_im); \
  out[3*(sp_stride)+sid+fl_stride] = make_float4(o2_20_re, o2_20_im, o2_21_re, o2_21_im); \
  out[4*(sp_stride)+sid+fl_stride] = make_float4(o2_22_re, o2_22_im, o2_30_re, o2_30_im); \
  out[5*(sp_stride)+sid+fl_stride] = make_float4(o2_31_re, o2_31_im, o2_32_re, o2_32_im); 


#define WRITE_FLAVOR_SPINOR_SHORT4()						\
  float c0 = fmaxf(fabsf(o1_00_re), fabsf(o1_00_im));			\
  float c1 = fmaxf(fabsf(o1_01_re), fabsf(o1_02_im));			\
  float c2 = fmaxf(fabsf(o1_02_re), fabsf(o1_01_im));			\
  float c3 = fmaxf(fabsf(o1_10_re), fabsf(o1_10_im));			\
  float c4 = fmaxf(fabsf(o1_11_re), fabsf(o1_11_im));			\
  float c5 = fmaxf(fabsf(o1_12_re), fabsf(o1_12_im));			\
  float c6 = fmaxf(fabsf(o1_20_re), fabsf(o1_20_im));			\
  float c7 = fmaxf(fabsf(o1_21_re), fabsf(o1_21_im));			\
  float c8 = fmaxf(fabsf(o1_22_re), fabsf(o1_22_im));			\
  float c9 = fmaxf(fabsf(o1_30_re), fabsf(o1_30_im));			\
  float c10 = fmaxf(fabsf(o1_31_re), fabsf(o1_31_im));			\
  float c11 = fmaxf(fabsf(o1_32_re), fabsf(o1_32_im));			\
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
  o1_00_re *= scale; o1_00_im *= scale; o1_01_re *= scale; o1_01_im *= scale;	\
  o1_02_re *= scale; o1_02_im *= scale; o1_10_re *= scale; o1_10_im *= scale;	\
  o1_11_re *= scale; o1_11_im *= scale; o1_12_re *= scale; o1_12_im *= scale;	\
  o1_20_re *= scale; o1_20_im *= scale; o1_21_re *= scale; o1_21_im *= scale;	\
  o1_22_re *= scale; o1_22_im *= scale; o1_30_re *= scale; o1_30_im *= scale;	\
  o1_31_re *= scale; o1_31_im *= scale; o1_32_re *= scale; o1_32_im *= scale;	\
  out[sid+0*(sp_stride)] = make_short4((short)o1_00_re, (short)o1_00_im, (short)o1_01_re, (short)o1_01_im); \
  out[sid+1*(sp_stride)] = make_short4((short)o1_02_re, (short)o1_02_im, (short)o1_10_re, (short)o1_10_im); \
  out[sid+2*(sp_stride)] = make_short4((short)o1_11_re, (short)o1_11_im, (short)o1_12_re, (short)o1_12_im); \
  out[sid+3*(sp_stride)] = make_short4((short)o1_20_re, (short)o1_20_im, (short)o1_21_re, (short)o1_21_im); \
  out[sid+4*(sp_stride)] = make_short4((short)o1_22_re, (short)o1_22_im, (short)o1_30_re, (short)o1_30_im); \
  out[sid+5*(sp_stride)] = make_short4((short)o1_31_re, (short)o1_31_im, (short)o1_32_re, (short)o1_32_im); \
  c0 = fmaxf(fabsf(o2_00_re), fabsf(o2_00_im));			\
  c1 = fmaxf(fabsf(o2_01_re), fabsf(o2_02_im));			\
  c2 = fmaxf(fabsf(o2_02_re), fabsf(o2_01_im));			\
  c3 = fmaxf(fabsf(o2_10_re), fabsf(o2_10_im));			\
  c4 = fmaxf(fabsf(o2_11_re), fabsf(o2_11_im));			\
  c5 = fmaxf(fabsf(o2_12_re), fabsf(o2_12_im));			\
  c6 = fmaxf(fabsf(o2_20_re), fabsf(o2_20_im));			\
  c7 = fmaxf(fabsf(o2_21_re), fabsf(o2_21_im));			\
  c8 = fmaxf(fabsf(o2_22_re), fabsf(o2_22_im));			\
  c9 = fmaxf(fabsf(o2_30_re), fabsf(o2_30_im));			\
  c10 = fmaxf(fabsf(o2_31_re), fabsf(o2_31_im));			\
  c11 = fmaxf(fabsf(o2_32_re), fabsf(o2_32_im));			\
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
  outNorm[sid+fl_stride] = c0;							\
  scale = __fdividef(MAX_SHORT, c0);				\
  o2_00_re *= scale; o2_00_im *= scale; o2_01_re *= scale; o2_01_im *= scale;	\
  o2_02_re *= scale; o2_02_im *= scale; o2_10_re *= scale; o2_10_im *= scale;	\
  o2_11_re *= scale; o2_11_im *= scale; o2_12_re *= scale; o2_12_im *= scale;	\
  o2_20_re *= scale; o2_20_im *= scale; o2_21_re *= scale; o2_21_im *= scale;	\
  o2_22_re *= scale; o2_22_im *= scale; o2_30_re *= scale; o2_30_im *= scale;	\
  o2_31_re *= scale; o2_31_im *= scale; o2_32_re *= scale; o2_32_im *= scale;	\
  out[sid+fl_stride+0*(sp_stride)] = make_short4((short)o2_00_re, (short)o2_00_im, (short)o2_01_re, (short)o2_01_im); \
  out[sid+fl_stride+1*(sp_stride)] = make_short4((short)o2_02_re, (short)o2_02_im, (short)o2_10_re, (short)o2_10_im); \
  out[sid+fl_stride+2*(sp_stride)] = make_short4((short)o2_11_re, (short)o2_11_im, (short)o2_12_re, (short)o2_12_im); \
  out[sid+fl_stride+3*(sp_stride)] = make_short4((short)o2_20_re, (short)o2_20_im, (short)o2_21_re, (short)o2_21_im); \
  out[sid+fl_stride+4*(sp_stride)] = make_short4((short)o2_22_re, (short)o2_22_im, (short)o2_30_re, (short)o2_30_im); \
  out[sid+fl_stride+5*(sp_stride)] = make_short4((short)o2_31_re, (short)o2_31_im, (short)o2_32_re, (short)o2_32_im);  


/************* the following is used by staggered *****************/

#define READ_1ST_NBR_SPINOR_DOUBLE_TEX(spinor, idx, mystride)	\
  double2 I0 = fetch_double2((spinor), idx + 0*mystride);	\
  double2 I1 = fetch_double2((spinor), idx + 1*mystride);	\
  double2 I2 = fetch_double2((spinor), idx + 2*mystride);

#define READ_3RD_NBR_SPINOR_DOUBLE_TEX(spinor, idx, mystride)	\
  double2 T0 = fetch_double2((spinor), idx + 0*mystride);	\
  double2 T1 = fetch_double2((spinor), idx + 1*mystride);	\
  double2 T2 = fetch_double2((spinor), idx + 2*mystride);

#define READ_1ST_NBR_SPINOR_SINGLE_TEX(spinor, idx, mystride)	\
  float2 I0 = TEX1DFETCH(float2, (spinor), idx + 0*mystride);	\
  float2 I1 = TEX1DFETCH(float2, (spinor), idx + 1*mystride);	\
  float2 I2 = TEX1DFETCH(float2, (spinor), idx + 2*mystride);

#define READ_3RD_NBR_SPINOR_SINGLE_TEX(spinor, idx, mystride)	\
  float2 T0 = TEX1DFETCH(float2, (spinor), idx + 0*mystride);	\
  float2 T1 = TEX1DFETCH(float2, (spinor), idx + 1*mystride);	\
  float2 T2 = TEX1DFETCH(float2, (spinor), idx + 2*mystride);

#define READ_1ST_NBR_SPINOR_HALF_TEX_(spinor, idx, mystride)		\
  float2 I0 = TEX1DFETCH(float2, (spinor), idx + 0*mystride);		\
  float2 I1 = TEX1DFETCH(float2, (spinor), idx + 1*mystride);		\
  float2 I2 = TEX1DFETCH(float2, (spinor), idx + 2*mystride);		\
  {									\
    float C = TEX1DFETCH(float, (spinor ## Norm), norm_idx1);		\
    I0.x *= C; I0.y *= C;						\
    I1.x *= C; I1.y *= C;						\
    I2.x *= C; I2.y *= C;}

#define READ_1ST_NBR_SPINOR_HALF_TEX(spinor, idx, mystride) \
  READ_1ST_NBR_SPINOR_HALF_TEX_(spinor, idx, mystride)	    

#define READ_3RD_NBR_SPINOR_HALF_TEX_(spinor, idx, mystride)		\
  float2 T0 = TEX1DFETCH(float2, (spinor), idx + 0*mystride);		\
  float2 T1 = TEX1DFETCH(float2, (spinor), idx + 1*mystride);		\
  float2 T2 = TEX1DFETCH(float2, (spinor), idx + 2*mystride);		\
  {									\
    float C = TEX1DFETCH(float, (spinor ## Norm), norm_idx3);		\
    T0.x *= C; T0.y *= C;						\
    T1.x *= C; T1.y *= C;						\
    T2.x *= C; T2.y *= C;}

#define READ_3RD_NBR_SPINOR_HALF_TEX(spinor, idx, mystride)	\
  READ_3RD_NBR_SPINOR_HALF_TEX_(spinor, idx, mystride)

#define READ_1ST_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 I0 = spinor[idx + 0*mystride];			\
  double2 I1 = spinor[idx + 1*mystride];			\
  double2 I2 = spinor[idx + 2*mystride];

#define READ_3RD_NBR_SPINOR_DOUBLE(spinor, idx, mystride)	\
  double2 T0 = spinor[idx + 0*mystride];			\
  double2 T1 = spinor[idx + 1*mystride];			\
  double2 T2 = spinor[idx + 2*mystride];

#define READ_1ST_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 I0 = spinor[idx + 0*mystride];				\
  float2 I1 = spinor[idx + 1*mystride];				\
  float2 I2 = spinor[idx + 2*mystride];

#define READ_3RD_NBR_SPINOR_SINGLE(spinor, idx, mystride)	\
  float2 T0 = spinor[idx + 0*mystride];				\
  float2 T1 = spinor[idx + 1*mystride];				\
  float2 T2 = spinor[idx + 2*mystride];

#define READ_1ST_NBR_SPINOR_HALF(spinor, idx, mystride)			\
  float2 I0, I1, I2;							\
  {									\
    short2 S0 = in[idx + 0*mystride];					\
    short2 S1 = in[idx + 1*mystride];					\
    short2 S2 = in[idx + 2*mystride];					\
    float C = inNorm[idx];						\
    I0.x =C*short2float(S0.x); I0.y =C*short2float(S0.y);		\
    I1.x =C*short2float(S1.x); I1.y =C*short2float(S1.y);		\
    I2.x =C*short2float(S2.x); I2.y =C*short2float(S2.y);		\
  }

#define READ_3RD_NBR_SPINOR_HALF(spinor, idx, mystride)			\
  float2 T0, T1, T2;							\
  {									\
    short2 S0 = in[idx + 0*mystride];					\
    short2 S1 = in[idx + 1*mystride];					\
    short2 S2 = in[idx + 2*mystride];					\
    float C = inNorm[idx];						\
    T0.x =C*short2float(S0.x); T0.y =C*short2float(S0.y);		\
    T1.x =C*short2float(S1.x); T1.y =C*short2float(S1.y);		\
    T2.x =C*short2float(S2.x); T2.y =C*short2float(S2.y);		\
  }


#define WRITE_ST_SPINOR_DOUBLE2(out)				\
  out[0*sp_stride+sid] = make_double2(o00_re, o00_im);	\
  out[1*sp_stride+sid] = make_double2(o01_re, o01_im);	\
  out[2*sp_stride+sid] = make_double2(o02_re, o02_im);

#define WRITE_ST_SPINOR_FLOAT2(out)			\
  out[0*sp_stride+sid] = make_float2(o00_re, o00_im);	\
  out[1*sp_stride+sid] = make_float2(o01_re, o01_im);	\
  out[2*sp_stride+sid] = make_float2(o02_re, o02_im);

#define WRITE_ST_SPINOR_SHORT2(out)					\
  float c0 = fmaxf(fabsf(o00_re), fabsf(o00_im));			\
  float c1 = fmaxf(fabsf(o01_re), fabsf(o01_im));			\
  float c2 = fmaxf(fabsf(o02_re), fabsf(o02_im));			\
  c0 = fmaxf(c0, c1);							\
  c0 = fmaxf(c0, c2);							\
  out ## Norm[sid] = c0;							\
  float scale = __fdividef(MAX_SHORT, c0);				\
  o00_re *= scale; o00_im *= scale; o01_re *= scale; o01_im *= scale;	\
  o02_re *= scale; o02_im *= scale;					\
  out[sid+0*sp_stride] = make_short2((short)o00_re, (short)o00_im);	\
  out[sid+1*sp_stride] = make_short2((short)o01_re, (short)o01_im);	\
  out[sid+2*sp_stride] = make_short2((short)o02_re, (short)o02_im);

// Non-cache writes to minimize cache polution
#if (__COMPUTE_CAPABILITY__ >= 200)

#define WRITE_ST_SPINOR_DOUBLE2_STR(out) \
  store_streaming_double2(&out[0*sp_stride+sid], o00_re, o00_im);	\
  store_streaming_double2(&out[1*sp_stride+sid], o01_re, o01_im);	\
  store_streaming_double2(&out[2*sp_stride+sid], o02_re, o02_im);

#define WRITE_ST_SPINOR_FLOAT2_STR(out)			       \
  store_streaming_float2(&out[0*sp_stride+sid], o00_re, o00_im);	\
  store_streaming_float2(&out[1*sp_stride+sid], o01_re, o01_im);	\
  store_streaming_float2(&out[2*sp_stride+sid], o02_re, o02_im);

#define WRITE_ST_SPINOR_SHORT2_STR(out) \
  float c0 = fmaxf(fabsf(o00_re), fabsf(o00_im));	\
  float c1 = fmaxf(fabsf(o01_re), fabsf(o01_im));	\
  float c2 = fmaxf(fabsf(o02_re), fabsf(o02_im));	\
  c0 = fmaxf(c0, c1);					\
  c0 = fmaxf(c0, c2);					\
  out ## Norm[sid] = c0;				\
  float scale = __fdividef(MAX_SHORT, c0);			    \
  o00_re *= scale; o00_im *= scale; o01_re *= scale; o01_im *= scale;	\
  o02_re *= scale; o02_im *= scale;					\
  store_streaming_short2(&g_out[0*sp_stride+sid], (short)o00_re, (short)o00_im); \
  store_streaming_short2(&g_out[1*sp_stride+sid], (short)o01_re, (short)o01_im); \
  store_streaming_short2(&g_out[2*sp_stride+sid], (short)o02_re, (short)o02_im);
#else

#define WRITE_ST_SPINOR_DOUBLE2_STR() WRITE_ST_SPINOR_DOUBLE2()
#define WRITE_ST_SPINOR_FLOAT4_STR() WRITE_ST_SPINOR_FLOAT4()
#define WRITE_ST_SPINOR_SHORT4_STR() WRITE_ST_SPINOR_SHORT4()

#endif

#define READ_AND_SUM_ST_SPINOR_DOUBLE_TEX(spinor) {			\
  double2 tmp0 = fetch_double2((spinor), sid + 0*(sp_stride));		\
  double2 tmp1 = fetch_double2((spinor), sid + 1*(sp_stride));		\
  double2 tmp2 = fetch_double2((spinor), sid + 2*(sp_stride));		\
  o00_re += tmp0.x; o00_im += tmp0.y;					\
  o01_re += tmp1.x; o01_im += tmp1.y;					\
  o02_re += tmp2.x; o02_im += tmp2.y; }
  
#define READ_AND_SUM_ST_SPINOR_SINGLE_TEX(spinor) {			\
  float2 tmp0 = TEX1DFETCH(float2, (spinor), sid + 0*(sp_stride));	\
  float2 tmp1 = TEX1DFETCH(float2, (spinor), sid + 1*(sp_stride));	\
  float2 tmp2 = TEX1DFETCH(float2, (spinor), sid + 2*(sp_stride));	\
  o00_re += tmp0.x; o00_im += tmp0.y;					\
  o01_re += tmp1.x; o01_im += tmp1.y;					\
  o02_re += tmp2.x; o02_im += tmp2.y; }

#define READ_AND_SUM_ST_SPINOR_HALF_TEX_(spinor) {			\
  float2 tmp0 = TEX1DFETCH(float2, (spinor), sid + 0*sp_stride);	\
  float2 tmp1 = TEX1DFETCH(float2, (spinor), sid + 1*sp_stride);	\
  float2 tmp2 = TEX1DFETCH(float2, (spinor), sid + 2*sp_stride);	\
  float C = TEX1DFETCH(float, (spinor##Norm), sid);			\
  o00_re += C*tmp0.x; o00_im += C*tmp0.y;				\
  o01_re += C*tmp1.x; o01_im += C*tmp1.y;				\
  o02_re += C*tmp2.x; o02_im += C*tmp2.y; }

#define READ_AND_SUM_ST_SPINOR_HALF_TEX(spinor)	\
  READ_AND_SUM_ST_SPINOR_HALF_TEX_(spinor)

#define READ_AND_SUM_ST_SPINOR(spinor)					\
  o00_re += spinor[0*sp_stride+sid].x; o00_im += spinor[0*sp_stride+sid].y; \
  o01_re += spinor[1*sp_stride+sid].x; o01_im += spinor[1*sp_stride+sid].y; \
  o02_re += spinor[2*sp_stride+sid].x; o02_im += spinor[2*sp_stride+sid].y; \
  
#define READ_AND_SUM_ST_SPINOR_HALF_(spinor)			\
  float C = spinor ## Norm[sid];				\
  o00_re += C*short2float(spinor[0*sp_stride + sid].x);		\
  o00_im += C*short2float(spinor[0*sp_stride + sid].y);		\
  o01_re += C*short2float(spinor[1*sp_stride + sid].x);		\
  o01_im += C*short2float(spinor[1*sp_stride + sid].y);		\
  o02_re += C*short2float(spinor[2*sp_stride + sid].x);		\
  o02_im += C*short2float(spinor[2*sp_stride + sid].y);	

#define READ_AND_SUM_ST_SPINOR_HALF(spinor)			\
  READ_AND_SUM_ST_SPINOR_HALF_(spinor)

#define READ_ST_ACCUM_DOUBLE_TEX(spinor)			   \
  double2 accum0 = fetch_double2((spinor), sid + 0*(sp_stride));   \
  double2 accum1 = fetch_double2((spinor), sid + 1*(sp_stride));   \
  double2 accum2 = fetch_double2((spinor), sid + 2*(sp_stride));   

#define READ_ST_ACCUM_SINGLE_TEX(spinor)			\
  float2 accum0 = TEX1DFETCH(float2, (spinor), sid + 0*sp_stride);	\
  float2 accum1 = TEX1DFETCH(float2, (spinor), sid + 1*sp_stride);	\
  float2 accum2 = TEX1DFETCH(float2, (spinor), sid + 2*sp_stride);     

#define READ_ST_ACCUM_HALF_TEX_(spinor)					\
  float2 accum0 = TEX1DFETCH(float2, (spinor), sid + 0*sp_stride);	\
  float2 accum1 = TEX1DFETCH(float2, (spinor), sid + 1*sp_stride);	\
  float2 accum2 = TEX1DFETCH(float2, (spinor), sid + 2*sp_stride);	\
  float C = TEX1DFETCH(float, (spinor ## Norm), sid);			\
  accum0.x *= C; accum0.y *= C;						\
  accum1.x *= C; accum1.y *= C;						\
  accum2.x *= C; accum2.y *= C;       

#define READ_ST_ACCUM_HALF_TEX(spinor) READ_ST_ACCUM_HALF_TEX_(spinor) 

#define READ_ST_ACCUM_DOUBLE(spinor)				   \
  double2 accum0 = spinor[sid + 0*(sp_stride)];			   \
  double2 accum1 = spinor[sid + 1*(sp_stride)];			   \
  double2 accum2 = spinor[sid + 2*(sp_stride)];   

#define READ_ST_ACCUM_SINGLE(spinor)				\
  float2 accum0 = spinor[sid + 0*(sp_stride)];			\
  float2 accum1 = spinor[sid + 1*(sp_stride)];			\
  float2 accum2 = spinor[sid + 2*(sp_stride)];     

#define READ_ST_ACCUM_HALF(spinor)					\
  float2 accum0, accum1, accum2;					\
  {									\
    short2 S0 = x[sid + 0*sp_stride];					\
    short2 S1 = x[sid + 1*sp_stride];					\
    short2 S2 = x[sid + 2*sp_stride];					\
    float C = spinor##Norm[sid];					\
    accum0.x =C*short2float(S0.x); accum0.y =C*short2float(S0.y);	\
    accum1.x =C*short2float(S1.x); accum1.y =C*short2float(S1.y);	\
    accum2.x =C*short2float(S2.x); accum2.y =C*short2float(S2.y);	\
  }

#define WRITE_SPINOR_SHARED_REAL(tx, ty, tz, reg)			\
  extern __shared__ char s_data[];					\
  spinorFloat *sh = (spinorFloat*)s_data + DSLASH_SHARED_FLOATS_PER_THREAD*SHARED_STRIDE* \
    ((tx+blockDim.x*(ty+blockDim.y*tz))/SHARED_STRIDE) + ((tx+blockDim.x*(ty+blockDim.y*tz)) % SHARED_STRIDE); \
  sh[0*SHARED_STRIDE] = reg##00_re;		\
  sh[1*SHARED_STRIDE] = reg##00_im;		\
  sh[2*SHARED_STRIDE] = reg##01_re;		\
  sh[3*SHARED_STRIDE] = reg##01_im;		\
  sh[4*SHARED_STRIDE] = reg##02_re;		\
  sh[5*SHARED_STRIDE] = reg##02_im;		\
  sh[6*SHARED_STRIDE] = reg##10_re;		\
  sh[7*SHARED_STRIDE] = reg##10_im;		\
  sh[8*SHARED_STRIDE] = reg##11_re;		\
  sh[9*SHARED_STRIDE] = reg##11_im;		\
  sh[10*SHARED_STRIDE] = reg##12_re;		\
  sh[11*SHARED_STRIDE] = reg##12_im;		\
  sh[12*SHARED_STRIDE] = reg##20_re;		\
  sh[13*SHARED_STRIDE] = reg##20_im;		\
  sh[14*SHARED_STRIDE] = reg##21_re;		\
  sh[15*SHARED_STRIDE] = reg##21_im;		\
  sh[16*SHARED_STRIDE] = reg##22_re;		\
  sh[17*SHARED_STRIDE] = reg##22_im;		\
  sh[18*SHARED_STRIDE] = reg##30_re;		\
  sh[19*SHARED_STRIDE] = reg##30_im;		\
  sh[20*SHARED_STRIDE] = reg##31_re;		\
  sh[21*SHARED_STRIDE] = reg##31_im;		\
  sh[22*SHARED_STRIDE] = reg##32_re;		\
  sh[23*SHARED_STRIDE] = reg##32_im;

#define WRITE_SPINOR_SHARED_DOUBLE2 WRITE_SPINOR_SHARED_REAL

#define READ_SPINOR_SHARED_DOUBLE2(tx, ty, tz)				\
  extern __shared__ char s_data[];					\
  double *sh = (double*)s_data + DSLASH_SHARED_FLOATS_PER_THREAD*SHARED_STRIDE* \
    ((tx+blockDim.x*(ty+blockDim.y*tz)) / SHARED_STRIDE) + ((tx+blockDim.x*(ty+blockDim.y*tz)) % SHARED_STRIDE); \
  double2 I0 = make_double2(sh[0*SHARED_STRIDE], sh[1*SHARED_STRIDE]);	\
  double2 I1 = make_double2(sh[2*SHARED_STRIDE], sh[3*SHARED_STRIDE]);	\
  double2 I2 = make_double2(sh[4*SHARED_STRIDE], sh[5*SHARED_STRIDE]);	\
  double2 I3 = make_double2(sh[6*SHARED_STRIDE], sh[7*SHARED_STRIDE]);	\
  double2 I4 = make_double2(sh[8*SHARED_STRIDE], sh[9*SHARED_STRIDE]);	\
  double2 I5 = make_double2(sh[10*SHARED_STRIDE], sh[11*SHARED_STRIDE]); \
  double2 I6 = make_double2(sh[12*SHARED_STRIDE], sh[13*SHARED_STRIDE]); \
  double2 I7 = make_double2(sh[14*SHARED_STRIDE], sh[15*SHARED_STRIDE]); \
  double2 I8 = make_double2(sh[16*SHARED_STRIDE], sh[17*SHARED_STRIDE]); \
  double2 I9 = make_double2(sh[18*SHARED_STRIDE], sh[19*SHARED_STRIDE]); \
  double2 I10 = make_double2(sh[20*SHARED_STRIDE], sh[21*SHARED_STRIDE]); \
  double2 I11 = make_double2(sh[22*SHARED_STRIDE], sh[23*SHARED_STRIDE]);

#ifndef SHARED_8_BYTE_WORD_SIZE // 4-byte shared memory access

#define WRITE_SPINOR_SHARED_FLOAT4 WRITE_SPINOR_SHARED_REAL

#define READ_SPINOR_SHARED_FLOAT4(tx, ty, tz)				\
  extern __shared__ char s_data[];					\
  float *sh = (float*)s_data + DSLASH_SHARED_FLOATS_PER_THREAD*SHARED_STRIDE* \
    ((tx+blockDim.x*(ty+blockDim.y*tz)) / SHARED_STRIDE) + ((tx+blockDim.x*(ty+blockDim.y*tz)) % SHARED_STRIDE); \
  float4 I0 = make_float4(sh[0*SHARED_STRIDE], sh[1*SHARED_STRIDE], sh[2*SHARED_STRIDE], sh[3*SHARED_STRIDE]); \
  float4 I1 = make_float4(sh[4*SHARED_STRIDE], sh[5*SHARED_STRIDE], sh[6*SHARED_STRIDE], sh[7*SHARED_STRIDE]); \
  float4 I2 = make_float4(sh[8*SHARED_STRIDE], sh[9*SHARED_STRIDE], sh[10*SHARED_STRIDE], sh[11*SHARED_STRIDE]); \
  float4 I3 = make_float4(sh[12*SHARED_STRIDE], sh[13*SHARED_STRIDE], sh[14*SHARED_STRIDE], sh[15*SHARED_STRIDE]); \
  float4 I4 = make_float4(sh[16*SHARED_STRIDE], sh[17*SHARED_STRIDE], sh[18*SHARED_STRIDE], sh[19*SHARED_STRIDE]); \
  float4 I5 = make_float4(sh[20*SHARED_STRIDE], sh[21*SHARED_STRIDE], sh[22*SHARED_STRIDE], sh[23*SHARED_STRIDE]);

#else // 8-byte shared memory words

#define WRITE_SPINOR_SHARED_FLOAT4(tx, ty, tz, reg)			\
  extern __shared__ char s_data[];					\
  float2 *sh = (float2*)s_data + (DSLASH_SHARED_FLOATS_PER_THREAD/2)*SHARED_STRIDE* \
    ((tx+blockDim.x*(ty+blockDim.y*tz)) / SHARED_STRIDE) + ((tx+blockDim.x*(ty+blockDim.y*tz)) % SHARED_STRIDE); \
  sh[0*SHARED_STRIDE] = make_float2(reg##00_re, reg##00_im);		\
  sh[1*SHARED_STRIDE] = make_float2(reg##01_re, reg##01_im);		\
  sh[2*SHARED_STRIDE] = make_float2(reg##02_re, reg##02_im);		\
  sh[3*SHARED_STRIDE] = make_float2(reg##10_re, reg##10_im);		\
  sh[4*SHARED_STRIDE] = make_float2(reg##11_re, reg##11_im);		\
  sh[5*SHARED_STRIDE] = make_float2(reg##12_re, reg##12_im);		\
  sh[6*SHARED_STRIDE] = make_float2(reg##20_re, reg##20_im);		\
  sh[7*SHARED_STRIDE] = make_float2(reg##21_re, reg##21_im);		\
  sh[8*SHARED_STRIDE] = make_float2(reg##22_re, reg##22_im);		\
  sh[9*SHARED_STRIDE] = make_float2(reg##30_re, reg##30_im);		\
  sh[10*SHARED_STRIDE] = make_float2(reg##31_re, reg##31_im);		\
  sh[11*SHARED_STRIDE] = make_float2(reg##32_re, reg##32_im);

#define READ_SPINOR_SHARED_FLOAT4(tx, ty, tz)				\
  extern __shared__ char s_data[];					\
  float2 *sh = (float2*)s_data + (DSLASH_SHARED_FLOATS_PER_THREAD/2)*SHARED_STRIDE* \
    ((tx+blockDim.x*(ty+blockDim.y*tz)) / SHARED_STRIDE) + ((tx+blockDim.x*(ty+blockDim.y*tz)) % SHARED_STRIDE); \
  float2 tmp1, tmp2;							\
  tmp1 = sh[0*SHARED_STRIDE]; tmp2 = sh[1*SHARED_STRIDE]; float4 I0 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); \
  tmp1 = sh[2*SHARED_STRIDE]; tmp2 = sh[3*SHARED_STRIDE]; float4 I1 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); \
  tmp1 = sh[4*SHARED_STRIDE]; tmp2 = sh[5*SHARED_STRIDE]; float4 I2 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); \
  tmp1 = sh[6*SHARED_STRIDE]; tmp2 = sh[7*SHARED_STRIDE]; float4 I3 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); \
  tmp1 = sh[8*SHARED_STRIDE]; tmp2 = sh[9*SHARED_STRIDE]; float4 I4 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); \
  tmp1 = sh[10*SHARED_STRIDE]; tmp2 = sh[11*SHARED_STRIDE]; float4 I5 = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y); 

#endif


