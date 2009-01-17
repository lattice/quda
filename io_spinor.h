#define READ_SPINOR(spinor)			     \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*Nh);   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*Nh);   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*Nh);   \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*Nh);   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*Nh);   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*Nh);

#define READ_SPINOR_UP(spinor)			     \
  float4 I0 = tex1Dfetch((spinor), sp_idx + 0*Nh);   \
  float4 I1 = tex1Dfetch((spinor), sp_idx + 1*Nh);   \
  float4 I2 = tex1Dfetch((spinor), sp_idx + 2*Nh);   \

#define READ_SPINOR_DOWN(spinor)		     \
  float4 I3 = tex1Dfetch((spinor), sp_idx + 3*Nh);   \
  float4 I4 = tex1Dfetch((spinor), sp_idx + 4*Nh);   \
  float4 I5 = tex1Dfetch((spinor), sp_idx + 5*Nh);

#define WRITE_SPINOR_FLOAT4()					 \
  g_out[0*Nh+sid] = make_float4(o00_re, o00_im, o01_re, o01_im); \
  g_out[1*Nh+sid] = make_float4(o02_re, o02_im, o10_re, o10_im); \
  g_out[2*Nh+sid] = make_float4(o11_re, o11_im, o12_re, o12_im); \
  g_out[3*Nh+sid] = make_float4(o20_re, o20_im, o21_re, o21_im); \
  g_out[4*Nh+sid] = make_float4(o22_re, o22_im, o30_re, o30_im); \
  g_out[5*Nh+sid] = make_float4(o31_re, o31_im, o32_re, o32_im);

#define WRITE_SPINOR_FLOAT1_SMEM() \
  int t = threadIdx.x; \
  int B = BLOCK_DIM; \
  int b = blockIdx.x; \
  int f = SHARED_FLOATS_PER_THREAD; \
  __syncthreads(); \
  for (int i = 0; i < 6; i++) for (int c = 0; c < 4; c++) \
      ((float*)g_out)[i*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];

// the alternative to writing float4's directly: almost as fast, a lot more confusing
#define WRITE_SPINOR_FLOAT1_STAGGERED() \
  int t = threadIdx.x; \
  int B = BLOCK_DIM; \
  int b = blockIdx.x; \
  int f = SHARED_FLOATS_PER_THREAD; \
  __syncthreads(); \
  for (int i = 0; i < 4; i++) for (int c = 0; c < 4; c++) \
      ((float*)g_out)[i*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4]; \
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
    ((float*)g_out)[(i+4)*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];
