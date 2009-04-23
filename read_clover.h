#define READ_CLOVER_SINGLE(clover, chi)                  \
  float4 C0 = tex1Dfetch((clover), sid + (9*chi+0)*Nh);  \
  float4 C1 = tex1Dfetch((clover), sid + (9*chi+1)*Nh);  \
  float4 C2 = tex1Dfetch((clover), sid + (9*chi+2)*Nh);  \
  float4 C3 = tex1Dfetch((clover), sid + (9*chi+3)*Nh);  \
  float4 C4 = tex1Dfetch((clover), sid + (9*chi+4)*Nh);  \
  float4 C5 = tex1Dfetch((clover), sid + (9*chi+5)*Nh);  \
  float4 C6 = tex1Dfetch((clover), sid + (9*chi+6)*Nh);  \
  float4 C7 = tex1Dfetch((clover), sid + (9*chi+7)*Nh);  \
  float4 C8 = tex1Dfetch((clover), sid + (9*chi+8)*Nh);
