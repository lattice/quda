#define READ_CLOVER_DOUBLE(clover_, chi)		    \
  double2* clover = (double2*)clover_;			    \
  double2 C0 = clover[sid + (18*chi+0)*param.cl_stride];    \
  double2 C1 = clover[sid + (18*chi+1)*param.cl_stride];    \
  double2 C2 = clover[sid + (18*chi+2)*param.cl_stride];    \
  double2 C3 = clover[sid + (18*chi+3)*param.cl_stride];    \
  double2 C4 = clover[sid + (18*chi+4)*param.cl_stride];    \
  double2 C5 = clover[sid + (18*chi+5)*param.cl_stride];    \
  double2 C6 = clover[sid + (18*chi+6)*param.cl_stride];    \
  double2 C7 = clover[sid + (18*chi+7)*param.cl_stride];    \
  double2 C8 = clover[sid + (18*chi+8)*param.cl_stride];    \
  double2 C9 = clover[sid + (18*chi+9)*param.cl_stride];    \
  double2 C10 = clover[sid + (18*chi+10)*param.cl_stride];  \
  double2 C11 = clover[sid + (18*chi+11)*param.cl_stride];  \
  double2 C12 = clover[sid + (18*chi+12)*param.cl_stride];  \
  double2 C13 = clover[sid + (18*chi+13)*param.cl_stride];  \
  double2 C14 = clover[sid + (18*chi+14)*param.cl_stride];  \
  double2 C15 = clover[sid + (18*chi+15)*param.cl_stride];  \
  double2 C16 = clover[sid + (18*chi+16)*param.cl_stride];  \
  double2 C17 = clover[sid + (18*chi+17)*param.cl_stride];

#define READ_CLOVER_DOUBLE_STR(clover_, chi)				\
  double2 C0, C1, C2, C3, C4, C5, C6, C7, C8, C9;			\
  double2 C10, C11, C12, C13, C14, C15, C16, C17;			\
  double2* clover = (double2*)clover_;					\
  load_streaming_double2(C0, &clover[sid + (18*chi+0)*param.cl_stride]);	\
  load_streaming_double2(C1, &clover[sid + (18*chi+1)*param.cl_stride]);	\
  load_streaming_double2(C2, &clover[sid + (18*chi+2)*param.cl_stride]);	\
  load_streaming_double2(C3, &clover[sid + (18*chi+3)*param.cl_stride]);	\
  load_streaming_double2(C4, &clover[sid + (18*chi+4)*param.cl_stride]);	\
  load_streaming_double2(C5, &clover[sid + (18*chi+5)*param.cl_stride]);	\
  load_streaming_double2(C6, &clover[sid + (18*chi+6)*param.cl_stride]);	\
  load_streaming_double2(C7, &clover[sid + (18*chi+7)*param.cl_stride]);	\
  load_streaming_double2(C8, &clover[sid + (18*chi+8)*param.cl_stride]);	\
  load_streaming_double2(C9, &clover[sid + (18*chi+9)*param.cl_stride]);	\
  load_streaming_double2(C10, &clover[sid + (18*chi+10)*param.cl_stride]);	\
  load_streaming_double2(C11, &clover[sid + (18*chi+11)*param.cl_stride]);	\
  load_streaming_double2(C12, &clover[sid + (18*chi+12)*param.cl_stride]);	\
  load_streaming_double2(C13, &clover[sid + (18*chi+13)*param.cl_stride]);	\
  load_streaming_double2(C14, &clover[sid + (18*chi+14)*param.cl_stride]);	\
  load_streaming_double2(C15, &clover[sid + (18*chi+15)*param.cl_stride]);	\
  load_streaming_double2(C16, &clover[sid + (18*chi+16)*param.cl_stride]);	\
  load_streaming_double2(C17, &clover[sid + (18*chi+17)*param.cl_stride]);	

#define READ_CLOVER2_DOUBLE_STR(clover_, chi)				\
  double2 C0, C1, C2, C3, C4, C5, C6, C7, C8, C9;			\
  double2 C10, C11, C12, C13, C14, C15, C16, C17;			\
  double2* clover = (double2*)clover_;					\
  load_streaming_double2(C0, &clover[sid + (18*chi+0)*param.cl_stride]); \
  load_streaming_double2(C1, &clover[sid + (18*chi+1)*param.cl_stride]); \
  double diag = 0.5*(C0.x + C1.y);					\
  double diag_inv = 1.0/diag;						\
  C2 = make_double2(diag*(2-C0.y*diag_inv), diag*(2-C1.x*diag_inv));	\
  load_streaming_double2(C3, &clover[sid + (18*chi+3)*param.cl_stride]);	\
  load_streaming_double2(C4, &clover[sid + (18*chi+4)*param.cl_stride]);	\
  load_streaming_double2(C5, &clover[sid + (18*chi+5)*param.cl_stride]);	\
  load_streaming_double2(C6, &clover[sid + (18*chi+6)*param.cl_stride]);	\
  load_streaming_double2(C7, &clover[sid + (18*chi+7)*param.cl_stride]);	\
  load_streaming_double2(C8, &clover[sid + (18*chi+8)*param.cl_stride]);	\
  load_streaming_double2(C9, &clover[sid + (18*chi+9)*param.cl_stride]);	\
  load_streaming_double2(C10, &clover[sid + (18*chi+10)*param.cl_stride]);	\
  load_streaming_double2(C11, &clover[sid + (18*chi+11)*param.cl_stride]);	\
  load_streaming_double2(C12, &clover[sid + (18*chi+12)*param.cl_stride]);	\
  load_streaming_double2(C13, &clover[sid + (18*chi+13)*param.cl_stride]);	\
  load_streaming_double2(C14, &clover[sid + (18*chi+14)*param.cl_stride]); \
  C15 = make_double2(-C3.x,-C3.y);					\
  C16 = make_double2(-C4.x,-C4.y);					\
  C17 = make_double2(-C8.x,-C8.y);					\
  C0.x += param.rho; C0.y += param.rho; C1.x += param.rho;		\
  C1.y += param.rho; C2.x += param.rho; C2.y += param.rho;

#define READ_CLOVER_SINGLE(clover_, chi)				\
  float4 *clover = (float4*)clover_;					\
  float4 C0 = clover[sid + (9*chi+0)*param.cl_stride];	\
  float4 C1 = clover[sid + (9*chi+1)*param.cl_stride];	\
  float4 C2 = clover[sid + (9*chi+2)*param.cl_stride];	\
  float4 C3 = clover[sid + (9*chi+3)*param.cl_stride];	\
  float4 C4 = clover[sid + (9*chi+4)*param.cl_stride];	\
  float4 C5 = clover[sid + (9*chi+5)*param.cl_stride];	\
  float4 C6 = clover[sid + (9*chi+6)*param.cl_stride];	\
  float4 C7 = clover[sid + (9*chi+7)*param.cl_stride];	\
  float4 C8 = clover[sid + (9*chi+8)*param.cl_stride];

#define READ_CLOVER2_SINGLE(clover_, chi)				\
  float4 *clover = (float4*)clover_;					\
  float4 C0 = clover[sid + (9*chi+0)*param.cl_stride];	\
  float4 C1 = clover[sid + (9*chi+1)*param.cl_stride];	\
  float4 C2 = clover[sid + (9*chi+2)*param.cl_stride];	\
  float4 C3 = clover[sid + (9*chi+3)*param.cl_stride];	\
  float4 C4 = clover[sid + (9*chi+4)*param.cl_stride];	\
  float4 C5 = clover[sid + (9*chi+5)*param.cl_stride];	\
  float4 C6 = clover[sid + (9*chi+6)*param.cl_stride];	\
  float4 C7 = clover[sid + (9*chi+7)*param.cl_stride];	\
  float4 C8 = make_float4(-C2.x,-C2.y,-C4.x,-C4.y);			\
  C0.x += param.rho; C0.y += param.rho; C0.z += param.rho;		\
  C0.w += param.rho; C1.x += param.rho; C1.y += param.rho;

#define READ_CLOVER_HALF(clover_, chi)					\
  short4 *clover = (short4*)clover_;					\
  float4 C0 = short42float4(clover[sid + (9*chi+0)*param.cl_stride]);	\
  float4 C1 = short42float4(clover[sid + (9*chi+1)*param.cl_stride]);	\
  float4 C2 = short42float4(clover[sid + (9*chi+2)*param.cl_stride]);	\
  float4 C3 = short42float4(clover[sid + (9*chi+3)*param.cl_stride]);	\
  float4 C4 = short42float4(clover[sid + (9*chi+4)*param.cl_stride]);	\
  float4 C5 = short42float4(clover[sid + (9*chi+5)*param.cl_stride]);	\
  float4 C6 = short42float4(clover[sid + (9*chi+6)*param.cl_stride]);	\
  float4 C7 = short42float4(clover[sid + (9*chi+7)*param.cl_stride]);	\
  float4 C8 = short42float4(clover[sid + (9*chi+8)*param.cl_stride]);	\
  float K = CLOVERTEXNORM[sid + chi*param.cl_stride];		\
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		 

#define READ_CLOVER2_HALF(clover_, chi)					\
  short4 *clover = (short4*)clover_;					\
  float4 C0 = short42float4(clover[sid + (9*chi+0)*param.cl_stride]);	\
  float4 C1 = short42float4(clover[sid + (9*chi+1)*param.cl_stride]);	\
  float4 C2 = short42float4(clover[sid + (9*chi+2)*param.cl_stride]);	\
  float4 C3 = short42float4(clover[sid + (9*chi+3)*param.cl_stride]);	\
  float4 C4 = short42float4(clover[sid + (9*chi+4)*param.cl_stride]);	\
  float4 C5 = short42float4(clover[sid + (9*chi+5)*param.cl_stride]);	\
  float4 C6 = short42float4(clover[sid + (9*chi+6)*param.cl_stride]);	\
  float4 C7 = short42float4(clover[sid + (9*chi+7)*param.cl_stride]);	\
  float K = CLOVERTEXNORM[sid + chi*param.cl_stride];		\
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;			\
  float4 C8 = make_float4(-C2.x, -C2.y, -C4.x, -C4.y);		\
  C0.x += param.rho; C0.y += param.rho; C0.z += param.rho;	\
  C0.w += param.rho; C1.x += param.rho; C1.y += param.rho;

#define READ_CLOVER_DOUBLE_TEX(clover, chi)			       \
  double2 C0 = fetch_double2((clover), sid + (18*chi+0)*param.cl_stride);	\
  double2 C1 = fetch_double2((clover), sid + (18*chi+1)*param.cl_stride);    \
  double2 C2 = fetch_double2((clover), sid + (18*chi+2)*param.cl_stride);    \
  double2 C3 = fetch_double2((clover), sid + (18*chi+3)*param.cl_stride);    \
  double2 C4 = fetch_double2((clover), sid + (18*chi+4)*param.cl_stride);    \
  double2 C5 = fetch_double2((clover), sid + (18*chi+5)*param.cl_stride);    \
  double2 C6 = fetch_double2((clover), sid + (18*chi+6)*param.cl_stride);    \
  double2 C7 = fetch_double2((clover), sid + (18*chi+7)*param.cl_stride);    \
  double2 C8 = fetch_double2((clover), sid + (18*chi+8)*param.cl_stride);    \
  double2 C9 = fetch_double2((clover), sid + (18*chi+9)*param.cl_stride);    \
  double2 C10 = fetch_double2((clover), sid + (18*chi+10)*param.cl_stride);  \
  double2 C11 = fetch_double2((clover), sid + (18*chi+11)*param.cl_stride);  \
  double2 C12 = fetch_double2((clover), sid + (18*chi+12)*param.cl_stride);  \
  double2 C13 = fetch_double2((clover), sid + (18*chi+13)*param.cl_stride);  \
  double2 C14 = fetch_double2((clover), sid + (18*chi+14)*param.cl_stride);  \
  double2 C15 = fetch_double2((clover), sid + (18*chi+15)*param.cl_stride);  \
  double2 C16 = fetch_double2((clover), sid + (18*chi+16)*param.cl_stride);  \
  double2 C17 = fetch_double2((clover), sid + (18*chi+17)*param.cl_stride);

// minimize the reads using the symmetry of the clover matrix
#define READ_CLOVER2_DOUBLE_TEX(clover, chi)			       \
  double2 C0 = fetch_double2((clover), sid + (18*chi+0)*param.cl_stride);    \
  double2 C1 = fetch_double2((clover), sid + (18*chi+1)*param.cl_stride);    \
  double diag = 0.5*(C0.x + C1.y);					\
  double diag_inv = 1.0/diag;						\
  double2 C2 = make_double2(diag*(2-C0.y*diag_inv), diag*(2-C1.x*diag_inv)); \
  double2 C3 = fetch_double2((clover), sid + (18*chi+3)*param.cl_stride);    \
  double2 C4 = fetch_double2((clover), sid + (18*chi+4)*param.cl_stride);    \
  double2 C5 = fetch_double2((clover), sid + (18*chi+5)*param.cl_stride);    \
  double2 C6 = fetch_double2((clover), sid + (18*chi+6)*param.cl_stride);    \
  double2 C7 = fetch_double2((clover), sid + (18*chi+7)*param.cl_stride);    \
  double2 C8 = fetch_double2((clover), sid + (18*chi+8)*param.cl_stride);    \
  double2 C9 = fetch_double2((clover), sid + (18*chi+9)*param.cl_stride);    \
  double2 C10 = fetch_double2((clover), sid + (18*chi+10)*param.cl_stride);  \
  double2 C11 = fetch_double2((clover), sid + (18*chi+11)*param.cl_stride);  \
  double2 C12 = fetch_double2((clover), sid + (18*chi+12)*param.cl_stride);  \
  double2 C13 = fetch_double2((clover), sid + (18*chi+13)*param.cl_stride);  \
  double2 C14 = fetch_double2((clover), sid + (18*chi+14)*param.cl_stride);  \
  double2 C15 = make_double2(-C3.x,-C3.y);				\
  double2 C16 = make_double2(-C4.x,-C4.y);				\
  double2 C17 = make_double2(-C8.x,-C8.y);				\
  C0.x += param.rho; C0.y += param.rho; C1.x += param.rho;		\
  C1.y += param.rho; C2.x += param.rho; C2.y += param.rho;

#define READ_CLOVER_SINGLE_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);	\
  float4 C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);	\
  float4 C8 = TEX1DFETCH(float4, (clover), sid + (9*chi+8)*param.cl_stride);

#define READ_CLOVER2_SINGLE_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);	\
  float4 C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);	\
  float4 C8 = make_float4(-C2.x,-C2.y,-C4.x,-C4.y);			\
  C0.x += param.rho; C0.y += param.rho; C0.z += param.rho;		\
  C0.w += param.rho; C1.x += param.rho; C1.y += param.rho;

#define READ_CLOVER_HALF_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);  \
  float4 C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);  \
  float4 C8 = TEX1DFETCH(float4, (clover), sid + (9*chi+8)*param.cl_stride);  \
  float K = TEX1DFETCH(float, (CLOVERTEXNORM), sid + chi*param.cl_stride); \
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		
 
#define READ_CLOVER2_HALF_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);  \
  float4 C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);  \
  float K = TEX1DFETCH(float, (CLOVERTEXNORM), sid + chi*param.cl_stride); \
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  float4 C8 = make_float4(-C2.x, -C2.y, -C4.x, -C4.y);		\
  C0.x += param.rho; C0.y += param.rho; C0.z += param.rho;	\
  C0.w += param.rho; C1.x += param.rho; C1.y += param.rho;

#define ASSN_CLOVER_DOUBLE(clover, chi)		      \
  C0 = clover[sid + (18*chi+0)*param.cl_stride];    \
  C1 = clover[sid + (18*chi+1)*param.cl_stride];    \
  C2 = clover[sid + (18*chi+2)*param.cl_stride];    \
  C3 = clover[sid + (18*chi+3)*param.cl_stride];    \
  C4 = clover[sid + (18*chi+4)*param.cl_stride];    \
  C5 = clover[sid + (18*chi+5)*param.cl_stride];    \
  C6 = clover[sid + (18*chi+6)*param.cl_stride];    \
  C7 = clover[sid + (18*chi+7)*param.cl_stride];    \
  C8 = clover[sid + (18*chi+8)*param.cl_stride];    \
  C9 = clover[sid + (18*chi+9)*param.cl_stride];    \
  C10 = clover[sid + (18*chi+10)*param.cl_stride];  \
  C11 = clover[sid + (18*chi+11)*param.cl_stride];  \
  C12 = clover[sid + (18*chi+12)*param.cl_stride];  \
  C13 = clover[sid + (18*chi+13)*param.cl_stride];  \
  C14 = clover[sid + (18*chi+14)*param.cl_stride];  \
  C15 = clover[sid + (18*chi+15)*param.cl_stride];  \
  C16 = clover[sid + (18*chi+16)*param.cl_stride];  \
  C17 = clover[sid + (18*chi+17)*param.cl_stride];

#define ASSN_CLOVER_DOUBLE_STR(clover, chi)				\
  load_streaming_double2(C0, &clover[sid + (18*chi+0)*param.cl_stride]);	\
  load_streaming_double2(C1, &clover[sid + (18*chi+1)*param.cl_stride]);	\
  load_streaming_double2(C2, &clover[sid + (18*chi+2)*param.cl_stride]);	\
  load_streaming_double2(C3, &clover[sid + (18*chi+3)*param.cl_stride]);	\
  load_streaming_double2(C4, &clover[sid + (18*chi+4)*param.cl_stride]);	\
  load_streaming_double2(C5, &clover[sid + (18*chi+5)*param.cl_stride]);	\
  load_streaming_double2(C6, &clover[sid + (18*chi+6)*param.cl_stride]);	\
  load_streaming_double2(C7, &clover[sid + (18*chi+7)*param.cl_stride]);	\
  load_streaming_double2(C8, &clover[sid + (18*chi+8)*param.cl_stride]);	\
  load_streaming_double2(C9, &clover[sid + (18*chi+9)*param.cl_stride]);	\
  load_streaming_double2(C10, &clover[sid + (18*chi+10)*param.cl_stride]);	\
  load_streaming_double2(C11, &clover[sid + (18*chi+11)*param.cl_stride]);	\
  load_streaming_double2(C12, &clover[sid + (18*chi+12)*param.cl_stride]);	\
  load_streaming_double2(C13, &clover[sid + (18*chi+13)*param.cl_stride]);	\
  load_streaming_double2(C14, &clover[sid + (18*chi+14)*param.cl_stride]);	\
  load_streaming_double2(C15, &clover[sid + (18*chi+15)*param.cl_stride]);	\
  load_streaming_double2(C16, &clover[sid + (18*chi+16)*param.cl_stride]);	\
  load_streaming_double2(C17, &clover[sid + (18*chi+17)*param.cl_stride]);	

#define ASSN_CLOVER_SINGLE(clover, chi)		  \
  C0 = clover[sid + (9*chi+0)*param.cl_stride];  \
  C1 = clover[sid + (9*chi+1)*param.cl_stride];  \
  C2 = clover[sid + (9*chi+2)*param.cl_stride];  \
  C3 = clover[sid + (9*chi+3)*param.cl_stride];  \
  C4 = clover[sid + (9*chi+4)*param.cl_stride];  \
  C5 = clover[sid + (9*chi+5)*param.cl_stride];  \
  C6 = clover[sid + (9*chi+6)*param.cl_stride];  \
  C7 = clover[sid + (9*chi+7)*param.cl_stride];  \
  C8 = clover[sid + (9*chi+8)*param.cl_stride];

#define ASSN_CLOVER_HALF(clover, chi)				\
  C0 = short42float4(clover[sid + (9*chi+0)*param.cl_stride]);	\
  C1 = short42float4(clover[sid + (9*chi+1)*param.cl_stride]);	\
  C2 = short42float4(clover[sid + (9*chi+2)*param.cl_stride]);	\
  C3 = short42float4(clover[sid + (9*chi+3)*param.cl_stride]);	\
  C4 = short42float4(clover[sid + (9*chi+4)*param.cl_stride]);	\
  C5 = short42float4(clover[sid + (9*chi+5)*param.cl_stride]);	\
  C6 = short42float4(clover[sid + (9*chi+6)*param.cl_stride]);	\
  C7 = short42float4(clover[sid + (9*chi+7)*param.cl_stride]);	\
  C8 = short42float4(clover[sid + (9*chi+8)*param.cl_stride]);	\
  K = TMCLOVERTEXNORM[sid + chi*param.cl_stride];		\
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		 

#define ASSN_CLOVER_DOUBLE_TEX(clover, chi)			       \
  C0 = fetch_double2((clover), sid + (18*chi+0)*param.cl_stride);	\
  C1 = fetch_double2((clover), sid + (18*chi+1)*param.cl_stride);    \
  C2 = fetch_double2((clover), sid + (18*chi+2)*param.cl_stride);    \
  C3 = fetch_double2((clover), sid + (18*chi+3)*param.cl_stride);    \
  C4 = fetch_double2((clover), sid + (18*chi+4)*param.cl_stride);    \
  C5 = fetch_double2((clover), sid + (18*chi+5)*param.cl_stride);    \
  C6 = fetch_double2((clover), sid + (18*chi+6)*param.cl_stride);    \
  C7 = fetch_double2((clover), sid + (18*chi+7)*param.cl_stride);    \
  C8 = fetch_double2((clover), sid + (18*chi+8)*param.cl_stride);    \
  C9 = fetch_double2((clover), sid + (18*chi+9)*param.cl_stride);    \
  C10 = fetch_double2((clover), sid + (18*chi+10)*param.cl_stride);  \
  C11 = fetch_double2((clover), sid + (18*chi+11)*param.cl_stride);  \
  C12 = fetch_double2((clover), sid + (18*chi+12)*param.cl_stride);  \
  C13 = fetch_double2((clover), sid + (18*chi+13)*param.cl_stride);  \
  C14 = fetch_double2((clover), sid + (18*chi+14)*param.cl_stride);  \
  C15 = fetch_double2((clover), sid + (18*chi+15)*param.cl_stride);  \
  C16 = fetch_double2((clover), sid + (18*chi+16)*param.cl_stride);  \
  C17 = fetch_double2((clover), sid + (18*chi+17)*param.cl_stride);

#define ASSN_CLOVER_SINGLE_TEX(clover, chi)			\
  C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);	\
  C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);	\
  C8 = TEX1DFETCH(float4, (clover), sid + (9*chi+8)*param.cl_stride);

#define ASSN_CLOVER_HALF_TEX(clover, chi)			\
  C0 = TEX1DFETCH(float4, (clover), sid + (9*chi+0)*param.cl_stride);  \
  C1 = TEX1DFETCH(float4, (clover), sid + (9*chi+1)*param.cl_stride);  \
  C2 = TEX1DFETCH(float4, (clover), sid + (9*chi+2)*param.cl_stride);  \
  C3 = TEX1DFETCH(float4, (clover), sid + (9*chi+3)*param.cl_stride);  \
  C4 = TEX1DFETCH(float4, (clover), sid + (9*chi+4)*param.cl_stride);  \
  C5 = TEX1DFETCH(float4, (clover), sid + (9*chi+5)*param.cl_stride);  \
  C6 = TEX1DFETCH(float4, (clover), sid + (9*chi+6)*param.cl_stride);  \
  C7 = TEX1DFETCH(float4, (clover), sid + (9*chi+7)*param.cl_stride);  \
  C8 = TEX1DFETCH(float4, (clover), sid + (9*chi+8)*param.cl_stride);  \
  K = TEX1DFETCH(float, (TMCLOVERTEXNORM), sid + chi*param.cl_stride); \
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		
 
#define PACK_CLOVER_DOUBLE(clover, chi)		      \
  double2 C0 = clover[idx + (18*chi+0)*param.cl_stride];    \
  double2 C1 = clover[idx + (18*chi+1)*param.cl_stride];    \
  double2 C2 = clover[idx + (18*chi+2)*param.cl_stride];    \
  double2 C3 = clover[idx + (18*chi+3)*param.cl_stride];    \
  double2 C4 = clover[idx + (18*chi+4)*param.cl_stride];    \
  double2 C5 = clover[idx + (18*chi+5)*param.cl_stride];    \
  double2 C6 = clover[idx + (18*chi+6)*param.cl_stride];    \
  double2 C7 = clover[idx + (18*chi+7)*param.cl_stride];    \
  double2 C8 = clover[idx + (18*chi+8)*param.cl_stride];    \
  double2 C9 = clover[idx + (18*chi+9)*param.cl_stride];    \
  double2 C10 = clover[idx + (18*chi+10)*param.cl_stride];  \
  double2 C11 = clover[idx + (18*chi+11)*param.cl_stride];  \
  double2 C12 = clover[idx + (18*chi+12)*param.cl_stride];  \
  double2 C13 = clover[idx + (18*chi+13)*param.cl_stride];  \
  double2 C14 = clover[idx + (18*chi+14)*param.cl_stride];  \
  double2 C15 = clover[idx + (18*chi+15)*param.cl_stride];  \
  double2 C16 = clover[idx + (18*chi+16)*param.cl_stride];  \
  double2 C17 = clover[idx + (18*chi+17)*param.cl_stride];

#define PACK_CLOVER_SINGLE(clover, chi)		  \
  float4 C0 = clover[idx + (9*chi+0)*param.cl_stride];  \
  float4 C1 = clover[idx + (9*chi+1)*param.cl_stride];  \
  float4 C2 = clover[idx + (9*chi+2)*param.cl_stride];  \
  float4 C3 = clover[idx + (9*chi+3)*param.cl_stride];  \
  float4 C4 = clover[idx + (9*chi+4)*param.cl_stride];  \
  float4 C5 = clover[idx + (9*chi+5)*param.cl_stride];  \
  float4 C6 = clover[idx + (9*chi+6)*param.cl_stride];  \
  float4 C7 = clover[idx + (9*chi+7)*param.cl_stride];  \
  float4 C8 = clover[idx + (9*chi+8)*param.cl_stride];

#define PACK_CLOVER_HALF(clover, chi)				\
  float4 C0 = short42float4(clover[idx + (9*chi+0)*param.cl_stride]);	\
  float4 C1 = short42float4(clover[idx + (9*chi+1)*param.cl_stride]);	\
  float4 C2 = short42float4(clover[idx + (9*chi+2)*param.cl_stride]);	\
  float4 C3 = short42float4(clover[idx + (9*chi+3)*param.cl_stride]);	\
  float4 C4 = short42float4(clover[idx + (9*chi+4)*param.cl_stride]);	\
  float4 C5 = short42float4(clover[idx + (9*chi+5)*param.cl_stride]);	\
  float4 C6 = short42float4(clover[idx + (9*chi+6)*param.cl_stride]);	\
  float4 C7 = short42float4(clover[idx + (9*chi+7)*param.cl_stride]);	\
  float4 C8 = short42float4(clover[idx + (9*chi+8)*param.cl_stride]);	\
  float K = cloverNorm[idx + chi*param.cl_stride];			\
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		 

#define PACK_CLOVER_DOUBLE_TEX(clover, chi)			\
  double2 C0 = fetch_double2((clover), idx + (18*chi+0)*param.cl_stride);	\
  double2 C1 = fetch_double2((clover), idx + (18*chi+1)*param.cl_stride);	\
  double2 C2 = fetch_double2((clover), idx + (18*chi+2)*param.cl_stride);	\
  double2 C3 = fetch_double2((clover), idx + (18*chi+3)*param.cl_stride);	\
  double2 C4 = fetch_double2((clover), idx + (18*chi+4)*param.cl_stride);	\
  double2 C5 = fetch_double2((clover), idx + (18*chi+5)*param.cl_stride);	\
  double2 C6 = fetch_double2((clover), idx + (18*chi+6)*param.cl_stride);	\
  double2 C7 = fetch_double2((clover), idx + (18*chi+7)*param.cl_stride);	\
  double2 C8 = fetch_double2((clover), idx + (18*chi+8)*param.cl_stride);	\
  double2 C9 = fetch_double2((clover), idx + (18*chi+9)*param.cl_stride);	\
  double2 C10 = fetch_double2((clover), idx + (18*chi+10)*param.cl_stride);	\
  double2 C11 = fetch_double2((clover), idx + (18*chi+11)*param.cl_stride);	\
  double2 C12 = fetch_double2((clover), idx + (18*chi+12)*param.cl_stride);	\
  double2 C13 = fetch_double2((clover), idx + (18*chi+13)*param.cl_stride);	\
  double2 C14 = fetch_double2((clover), idx + (18*chi+14)*param.cl_stride);	\
  double2 C15 = fetch_double2((clover), idx + (18*chi+15)*param.cl_stride);	\
  double2 C16 = fetch_double2((clover), idx + (18*chi+16)*param.cl_stride);	\
  double2 C17 = fetch_double2((clover), idx + (18*chi+17)*param.cl_stride);

#define PACK_CLOVER_SINGLE_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), idx + (9*chi+0)*param.cl_stride);	\
  float4 C1 = TEX1DFETCH(float4, (clover), idx + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), idx + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), idx + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), idx + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), idx + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), idx + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), idx + (9*chi+7)*param.cl_stride);	\
  float4 C8 = TEX1DFETCH(float4, (clover), idx + (9*chi+8)*param.cl_stride);

#define PACK_CLOVER_HALF_TEX(clover, chi)			\
  float4 C0 = TEX1DFETCH(float4, (clover), idx + (9*chi+0)*param.cl_stride);  \
  float4 C1 = TEX1DFETCH(float4, (clover), idx + (9*chi+1)*param.cl_stride);  \
  float4 C2 = TEX1DFETCH(float4, (clover), idx + (9*chi+2)*param.cl_stride);  \
  float4 C3 = TEX1DFETCH(float4, (clover), idx + (9*chi+3)*param.cl_stride);  \
  float4 C4 = TEX1DFETCH(float4, (clover), idx + (9*chi+4)*param.cl_stride);  \
  float4 C5 = TEX1DFETCH(float4, (clover), idx + (9*chi+5)*param.cl_stride);  \
  float4 C6 = TEX1DFETCH(float4, (clover), idx + (9*chi+6)*param.cl_stride);  \
  float4 C7 = TEX1DFETCH(float4, (clover), idx + (9*chi+7)*param.cl_stride);  \
  float4 C8 = TEX1DFETCH(float4, (clover), idx + (9*chi+8)*param.cl_stride);  \
  float K = TEX1DFETCH(float, (TMCLOVERTEXNORM), idx + chi*param.cl_stride); \
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		        \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		        \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		        \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		        \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		        \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		        \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		        \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		        \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		
 
