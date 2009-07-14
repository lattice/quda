#define READ_CLOVER_DOUBLE(clover, chi)				\
  double2 C0 = fetch_double2((clover), sid + (18*chi+0)*Vh);	\
  double2 C1 = fetch_double2((clover), sid + (18*chi+1)*Vh);	\
  double2 C2 = fetch_double2((clover), sid + (18*chi+2)*Vh);	\
  double2 C3 = fetch_double2((clover), sid + (18*chi+3)*Vh);	\
  double2 C4 = fetch_double2((clover), sid + (18*chi+4)*Vh);	\
  double2 C5 = fetch_double2((clover), sid + (18*chi+5)*Vh);	\
  double2 C6 = fetch_double2((clover), sid + (18*chi+6)*Vh);	\
  double2 C7 = fetch_double2((clover), sid + (18*chi+7)*Vh);	\
  double2 C8 = fetch_double2((clover), sid + (18*chi+8)*Vh);	\
  double2 c9 = fetch_double2((clover), sid + (18*chi+9)*Vh);	\
  double2 C10 = fetch_double2((clover), sid + (18*chi+10)*Vh);  \
  double2 C11 = fetch_double2((clover), sid + (18*chi+11)*Vh);  \
  double2 C12 = fetch_double2((clover), sid + (18*chi+12)*Vh);  \
  double2 C13 = fetch_double2((clover), sid + (18*chi+13)*Vh);  \
  double2 C14 = fetch_double2((clover), sid + (18*chi+14)*Vh);  \
  double2 C15 = fetch_double2((clover), sid + (18*chi+15)*Vh);  \
  double2 C16 = fetch_double2((clover), sid + (18*chi+16)*Vh);  \
  double2 C17 = fetch_double2((clover), sid + (18*chi+17)*Vh);

#define READ_CLOVER_SINGLE(clover, chi)                  \
  float4 C0 = tex1Dfetch((clover), sid + (9*chi+0)*Vh);  \
  float4 C1 = tex1Dfetch((clover), sid + (9*chi+1)*Vh);  \
  float4 C2 = tex1Dfetch((clover), sid + (9*chi+2)*Vh);  \
  float4 C3 = tex1Dfetch((clover), sid + (9*chi+3)*Vh);  \
  float4 C4 = tex1Dfetch((clover), sid + (9*chi+4)*Vh);  \
  float4 C5 = tex1Dfetch((clover), sid + (9*chi+5)*Vh);  \
  float4 C6 = tex1Dfetch((clover), sid + (9*chi+6)*Vh);  \
  float4 C7 = tex1Dfetch((clover), sid + (9*chi+7)*Vh);  \
  float4 C8 = tex1Dfetch((clover), sid + (9*chi+8)*Vh);

#define READ_CLOVER_HALF(clover, chi)			 \
  float4 C0 = tex1Dfetch((clover), sid + (9*chi+0)*Vh);  \
  float4 C1 = tex1Dfetch((clover), sid + (9*chi+1)*Vh);  \
  float4 C2 = tex1Dfetch((clover), sid + (9*chi+2)*Vh);  \
  float4 C3 = tex1Dfetch((clover), sid + (9*chi+3)*Vh);  \
  float4 C4 = tex1Dfetch((clover), sid + (9*chi+4)*Vh);  \
  float4 C5 = tex1Dfetch((clover), sid + (9*chi+5)*Vh);  \
  float4 C6 = tex1Dfetch((clover), sid + (9*chi+6)*Vh);  \
  float4 C7 = tex1Dfetch((clover), sid + (9*chi+7)*Vh);  \
  float4 C8 = tex1Dfetch((clover), sid + (9*chi+8)*Vh);	 \
  float K = tex1Dfetch((cloverTexNorm), sid+chi*Vh);	 \
  C0.x *= K; C0.y *= K;	C0.z *= K; C0.w *= K;		 \
  C1.x *= K; C1.y *= K;	C1.z *= K; C1.w *= K;		 \
  C2.x *= K; C2.y *= K;	C2.z *= K; C2.w *= K;		 \
  C3.x *= K; C3.y *= K;	C3.z *= K; C3.w *= K;		 \
  C4.x *= K; C4.y *= K;	C4.z *= K; C4.w *= K;		 \
  C5.x *= K; C5.y *= K;	C5.z *= K; C5.w *= K;		 \
  C6.x *= K; C6.y *= K;	C6.z *= K; C6.w *= K;		 \
  C7.x *= K; C7.y *= K;	C7.z *= K; C7.w *= K;		 \
  C8.x *= K; C8.y *= K;	C8.z *= K; C8.w *= K;		 
