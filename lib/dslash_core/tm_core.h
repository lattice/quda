#ifndef _TM_CORE_H
#define _TM_CORE_H

//action of the operator b*(1 + i*a*gamma5)
//used also macros from io_spinor.h

__device__ float4 operator*(const float &x, const float4 &y) 
{
  float4 res;

  res.x = x * y.x;
  res.y = x * y.y;  
  res.z = x * y.z;
  res.w = x * y.w;  

  return res;
}

__device__ double2 operator*(const double &x, const double2 &y) 
{
  double2 res;

  res.x = x * y.x;
  res.y = x * y.y;  

  return res;
}


#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp1.x
#define tmp1_im tmp1.y
#define tmp2_re tmp2.x
#define tmp2_im tmp2.y
#define tmp3_re tmp3.x
#define tmp3_im tmp3.y

#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif

#if (__COMPUTE_CAPABILITY__ >= 130)
__global__ void twistGamma5Kernel(double2 *spinor, float *null, double a, double b, 
				  const double2 *in, const float *null2, DslashParam param)
{
#ifdef GPU_TWISTED_MASS_DIRAC

   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

#ifndef FERMI_NO_DBLE_TEX
   double2 I0  = fetch_double2(SPINORTEX, sid + 0 * sp_stride);   
   double2 I1  = fetch_double2(SPINORTEX, sid + 1 * sp_stride);   
   double2 I2  = fetch_double2(SPINORTEX, sid + 2 * sp_stride);   
   double2 I3  = fetch_double2(SPINORTEX, sid + 3 * sp_stride);   
   double2 I4  = fetch_double2(SPINORTEX, sid + 4 * sp_stride);   
   double2 I5  = fetch_double2(SPINORTEX, sid + 5 * sp_stride);   
   double2 I6  = fetch_double2(SPINORTEX, sid + 6 * sp_stride);   
   double2 I7  = fetch_double2(SPINORTEX, sid + 7 * sp_stride);   
   double2 I8  = fetch_double2(SPINORTEX, sid + 8 * sp_stride);   
   double2 I9  = fetch_double2(SPINORTEX, sid + 9 * sp_stride);   
   double2 I10 = fetch_double2(SPINORTEX, sid + 10 * sp_stride); 
   double2 I11 = fetch_double2(SPINORTEX, sid + 11 * sp_stride);
#else
   double2 I0  = in[sid + 0 * sp_stride];   
   double2 I1  = in[sid + 1 * sp_stride];   
   double2 I2  = in[sid + 2 * sp_stride];   
   double2 I3  = in[sid + 3 * sp_stride];   
   double2 I4  = in[sid + 4 * sp_stride];   
   double2 I5  = in[sid + 5 * sp_stride];   
   double2 I6  = in[sid + 6 * sp_stride];   
   double2 I7  = in[sid + 7 * sp_stride];   
   double2 I8  = in[sid + 8 * sp_stride];   
   double2 I9  = in[sid + 9 * sp_stride];   
   double2 I10 = in[sid + 10 * sp_stride]; 
   double2 I11 = in[sid + 11 * sp_stride];
#endif

   volatile double2 tmp0, tmp1, tmp2, tmp3;
   
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
    //get the 1st color component:
   
   tmp0_re = I0.x - a * I6.y;
   tmp0_im = I0.y + a * I6.x;
   
   tmp2_re = I6.x - a * I0.y;
   tmp2_im = I6.y + a * I0.x;
   
   tmp1_re = I3.x - a * I9.y;
   tmp1_im = I3.y + a * I9.x;
   
   tmp3_re = I9.x - a * I3.y;
   tmp3_im = I9.y + a * I3.x;
   
   I0.x = b * tmp0_re;
   I0.y = b * tmp0_im;
   I3.x = b * tmp1_re;
   I3.y = b * tmp1_im;
   I6.x = b * tmp2_re;
   I6.y = b * tmp2_im;
   I9.x = b * tmp3_re;
   I9.y = b * tmp3_im;
   
   //get the 2nd color component:    
   
   tmp0_re = I1.x - a * I7.y;
   tmp0_im = I1.y + a * I7.x;
   
   tmp2_re = I7.x - a * I1.y;
   tmp2_im = I7.y + a * I1.x;
   
   tmp1_re = I4.x - a * I10.y;
   tmp1_im = I4.y + a * I10.x;
   
   tmp3_re = I10.x - a * I4.y;
   tmp3_im = I10.y + a * I4.x;
   
   I1.x  = b * tmp0_re;
   I1.y  = b * tmp0_im;
   I4.x  = b * tmp1_re;
   I4.y  = b * tmp1_im;
   I7.x  = b * tmp2_re;
   I7.y  = b * tmp2_im;
   I10.x = b * tmp3_re;
   I10.y = b * tmp3_im;
   
   //get the 3d color component:    
   
   tmp0_re = I2.x - a* I8.y;
   tmp0_im = I2.y + a* I8.x;
    
   tmp2_re = I8.x - a* I2.y;
   tmp2_im = I8.y + a* I2.x;
   
   tmp1_re = I5.x - a* I11.y;
   tmp1_im = I5.y + a* I11.x;
   
   tmp3_re = I11.x - a* I5.y;
   tmp3_im = I11.y + a* I5.x;
   
   I2.x  = b * tmp0_re;
   I2.y  = b * tmp0_im;
   I5.x  = b * tmp1_re;
   I5.y  = b * tmp1_im;
   I8.x  = b * tmp2_re;
   I8.y  = b * tmp2_im;
   I11.x = b * tmp3_re;
   I11.y = b * tmp3_im;
      
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   
   spinor[sid + 6  * sp_stride] = I6;   
   spinor[sid + 7  * sp_stride] = I7;   
   spinor[sid + 8  * sp_stride] = I8;   
   spinor[sid + 9  * sp_stride] = I9;   
   spinor[sid + 10 * sp_stride] = I10;   
   spinor[sid + 11 * sp_stride] = I11;
#endif
}


__global__ void twistGamma5Kernel(double2 *spinor, float *null, const double a, const double b, const double c, const double2 *in, const float *null2, DslashParam param)
{
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
   int sid = blockIdx.x * blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;   
   
   //temporal regs:
   double2 accum1_0, accum1_1;
   double2 accum2_0, accum2_1;
   double2 tmp0,     tmp1;
   
   int flv1_idx = sid;
   int flv2_idx = sid + fl_stride; //or simply +flavor_length (volume incl. pad)
   
   //apply (1 - i*a*gamma_5 * tau_3 + b * tau_1) 
   
   //get the first color component for each flavor:   

#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 0 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 6 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 0 * sp_stride];   
   tmp1  = in[flv1_idx + 6 * sp_stride];   
#endif

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   

#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 0 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 6 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 0 * sp_stride];   
   tmp1  = in[flv2_idx + 6 * sp_stride];   
#endif

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 0  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 6  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 0  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 6  * sp_stride] = c * accum2_1;   

#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 3 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 9 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 3 * sp_stride];   
   tmp1  = in[flv1_idx + 9 * sp_stride];   
#endif   

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 3 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 9 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 3 * sp_stride];   
   tmp1  = in[flv2_idx + 9 * sp_stride];   
#endif   

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 3  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 9  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 3  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 9  * sp_stride] = c * accum2_1;   
   //get the second color component for each flavor:   
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 1 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 7 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 1 * sp_stride];   
   tmp1  = in[flv1_idx + 7 * sp_stride];   
#endif   

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 1 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 7 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 1 * sp_stride];   
   tmp1  = in[flv2_idx + 7 * sp_stride];   
#endif   

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 1  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 7  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 1  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 7  * sp_stride] = c * accum2_1; 
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 4 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 10 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 4 * sp_stride];   
   tmp1  = in[flv1_idx + 10 * sp_stride];   
#endif   

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 4 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 10 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 4 * sp_stride];   
   tmp1  = in[flv2_idx + 10 * sp_stride];   
#endif   

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 4  * sp_stride]  = c * accum1_0;   
   spinor[flv1_idx + 10  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 4  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 10  * sp_stride] = c * accum2_1; 
   //get the second color component for each flavor:   
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 2 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 8 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 2 * sp_stride];   
   tmp1  = in[flv1_idx + 8 * sp_stride];   
#endif   

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 2 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 8 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 2 * sp_stride];   
   tmp1  = in[flv2_idx + 8 * sp_stride];   
#endif   

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 2  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 8  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 2  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 8  * sp_stride] = c * accum2_1; 
   
#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv1_idx + 5 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv1_idx + 11 * sp_stride);   
#else
   tmp0  = in[flv1_idx + 5 * sp_stride];   
   tmp1  = in[flv1_idx + 11 * sp_stride];   
#endif   

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;

   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;   

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   

#ifndef FERMI_NO_DBLE_TEX   
   tmp0  = fetch_double2(SPINORTEX, flv2_idx + 5 * sp_stride);   
   tmp1  = fetch_double2(SPINORTEX, flv2_idx + 11 * sp_stride);   
#else
   tmp0  = in[flv2_idx + 5 * sp_stride];   
   tmp1  = in[flv2_idx + 11 * sp_stride];   
#endif   

   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;

   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;   

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   
   //store results back to memory:
   
   spinor[flv1_idx + 5  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 11  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 5  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 11  * sp_stride] = c * accum2_1;
    
#endif
}

#endif // (__COMPUTE_CAPABILITY__ >= 130)

#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp0.z
#define tmp1_im tmp0.w
#define tmp2_re tmp1.x
#define tmp2_im tmp1.y
#define tmp3_re tmp1.z
#define tmp3_im tmp1.w

#undef SPINORTEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif


__global__ void twistGamma5Kernel(float4 *spinor, float *null, float a, float b, 
				  const float4 *in, const float *null2, DslashParam param)
{
#ifdef GPU_TWISTED_MASS_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);

   volatile float4 tmp0, tmp1;
    
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
   //get the 1st color component:(I0.xy, I1.zw, I3.xy, I4.zw)
   
   tmp0_re = I0.x - a * I3.y;
   tmp0_im = I0.y + a * I3.x;
   
   tmp1_re = I1.z - a * I4.w;
   tmp1_im = I1.w + a * I4.z;
   
   tmp2_re = I3.x - a * I0.y;
   tmp2_im = I3.y + a * I0.x;
   
   tmp3_re = I4.z - a * I1.w;
   tmp3_im = I4.w + a * I1.z;
   
   I0.x = b * tmp0_re;
   I0.y = b * tmp0_im;
   I1.z = b * tmp1_re;
   I1.w = b * tmp1_im;
   I3.x = b * tmp2_re;
   I3.y = b * tmp2_im;
   I4.z = b * tmp3_re;
   I4.w = b * tmp3_im;
   
   //get the 2nd color component:(I0.zw, I2.xy, I3.zw, I5.xy)
   
   tmp0_re = I0.z - a * I3.w;
   tmp0_im = I0.w + a * I3.z;
   
   tmp1_re = I2.x - a * I5.y;
   tmp1_im = I2.y + a * I5.x;
   
   tmp2_re = I3.z - a * I0.w;
   tmp2_im = I3.w + a * I0.z;
   
   tmp3_re = I5.x - a * I2.y;
   tmp3_im = I5.y + a * I2.x;
   
   I0.z = b * tmp0_re;
   I0.w = b * tmp0_im;
   I2.x = b * tmp1_re;
   I2.y = b * tmp1_im;
   I3.z = b * tmp2_re;
   I3.w = b * tmp2_im;
   I5.x = b * tmp3_re;
   I5.y = b * tmp3_im;
   
   //get the 3d color component:(I1.xy, I2.zw, I4.xy, I5.zw)
   
   tmp0_re = I1.x - a * I4.y;
   tmp0_im = I1.y + a * I4.x;
   
   tmp1_re = I2.z - a * I5.w;
   tmp1_im = I2.w + a * I5.z;
   
   tmp2_re = I4.x - a * I1.y;
   tmp2_im = I4.y + a * I1.x;
   
   tmp3_re = I5.z - a * I2.w;
   tmp3_im = I5.w + a * I2.z;
   
   I1.x = b * tmp0_re;
   I1.y = b * tmp0_im;
   I2.z = b * tmp1_re;
   I2.w = b * tmp1_im;
   I4.x = b * tmp2_re;
   I4.y = b * tmp2_im;
   I5.z = b * tmp3_re;
   I5.w = b * tmp3_im;
   
   spinor[sid + 0  * sp_stride] = I0;   
   spinor[sid + 1  * sp_stride] = I1;   
   spinor[sid + 2  * sp_stride] = I2;   
   spinor[sid + 3  * sp_stride] = I3;   
   spinor[sid + 4  * sp_stride] = I4;   
   spinor[sid + 5  * sp_stride] = I5;   
#endif 
}

__global__ void twistGamma5Kernel(float4 *spinor, float *null, float a, float b, float c,  const float4 *in,  const float *null2, DslashParam param)
{
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
   int sid = blockIdx.x * blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;
   
   float4 accum1_0, accum1_1;
   float4 accum2_0, accum2_1;
   float4 tmp0, tmp1;

   int flv1_idx = sid;
   int flv2_idx = sid + fl_stride; 
   
   //apply (1 - i*a*gamma_5 * tau_3 + b * tau_1) 
   
   //get the first color component for each flavor: 
   
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 0 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 3 * sp_stride);     

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;
   accum1_0.z = tmp0.z + a * tmp1.w;
   accum1_0.w = tmp0.w - a * tmp1.z;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;
   accum2_0.z = b * tmp0.z;
   accum2_0.w = b * tmp0.w;
  
   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;
   accum1_1.z = tmp1.z + a * tmp0.w;
   accum1_1.w = tmp1.w - a * tmp0.z;

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   accum2_1.z = b * tmp1.z;
   accum2_1.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 0 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 3 * sp_stride);     
   
   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;
   accum2_0.z += tmp0.z - a * tmp1.w;
   accum2_0.w += tmp0.w + a * tmp1.z;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;
   accum1_0.z += b * tmp0.z;
   accum1_0.w += b * tmp0.w;
  
   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;
   accum2_1.z += tmp1.z - a * tmp0.w;
   accum2_1.w += tmp1.w + a * tmp0.z;

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   accum1_1.z += b * tmp1.z;
   accum1_1.w += b * tmp1.w;
   
   spinor[flv1_idx + 0  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 3  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 0  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 3  * sp_stride] = c * accum2_1;   
   
   //get the second color component for each flavor: 
   
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 1 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 4 * sp_stride);     

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;
   accum1_0.z = tmp0.z + a * tmp1.w;
   accum1_0.w = tmp0.w - a * tmp1.z;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;
   accum2_0.z = b * tmp0.z;
   accum2_0.w = b * tmp0.w;
  
   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;
   accum1_1.z = tmp1.z + a * tmp0.w;
   accum1_1.w = tmp1.w - a * tmp0.z;

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   accum2_1.z = b * tmp1.z;
   accum2_1.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 1 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 4 * sp_stride);     
   
   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;
   accum2_0.z += tmp0.z - a * tmp1.w;
   accum2_0.w += tmp0.w + a * tmp1.z;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;
   accum1_0.z += b * tmp0.z;
   accum1_0.w += b * tmp0.w;
  
   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;
   accum2_1.z += tmp1.z - a * tmp0.w;
   accum2_1.w += tmp1.w + a * tmp0.z;

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   accum1_1.z += b * tmp1.z;
   accum1_1.w += b * tmp1.w;
   
   spinor[flv1_idx + 1  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 4  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 1  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 4  * sp_stride] = c * accum2_1; 

   //get the third color component for each flavor: 
   
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 2 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 5 * sp_stride);     

   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;
   accum1_0.z = tmp0.z + a * tmp1.w;
   accum1_0.w = tmp0.w - a * tmp1.z;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;
   accum2_0.z = b * tmp0.z;
   accum2_0.w = b * tmp0.w;
  
   accum1_1.x = tmp1.x + a * tmp0.y;
   accum1_1.y = tmp1.y - a * tmp0.x;
   accum1_1.z = tmp1.z + a * tmp0.w;
   accum1_1.w = tmp1.w - a * tmp0.z;

   accum2_1.x = b * tmp1.x;
   accum2_1.y = b * tmp1.y;
   accum2_1.z = b * tmp1.z;
   accum2_1.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 2 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 5 * sp_stride);     
   
   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;
   accum2_0.z += tmp0.z - a * tmp1.w;
   accum2_0.w += tmp0.w + a * tmp1.z;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;
   accum1_0.z += b * tmp0.z;
   accum1_0.w += b * tmp0.w;
  
   accum2_1.x += tmp1.x - a * tmp0.y;
   accum2_1.y += tmp1.y + a * tmp0.x;
   accum2_1.z += tmp1.z - a * tmp0.w;
   accum2_1.w += tmp1.w + a * tmp0.z;

   accum1_1.x += b * tmp1.x;
   accum1_1.y += b * tmp1.y;
   accum1_1.z += b * tmp1.z;
   accum1_1.w += b * tmp1.w;
   
   spinor[flv1_idx + 2  * sp_stride] = c * accum1_0;   
   spinor[flv1_idx + 5  * sp_stride] = c * accum1_1;   
   spinor[flv2_idx + 2  * sp_stride] = c * accum2_0;   
   spinor[flv2_idx + 5  * sp_stride] = c * accum2_1;

#endif 
}

#undef SPINORTEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#define SPINORTEXNORM param.inTexNorm
#else
#define SPINORTEX spinorTexHalf
#define SPINORTEXNORM spinorTexHalfNorm
#endif


__global__ void twistGamma5Kernel(short4* spinor, float *spinorNorm, float a, float b, 
				  const short4 *in, const float *inNorm, DslashParam param)
{
#ifdef GPU_TWISTED_MASS_DIRAC
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = TEX1DFETCH(float4, SPINORTEX, sid + 0 * sp_stride);   
   float4 I1 = TEX1DFETCH(float4, SPINORTEX, sid + 1 * sp_stride);   
   float4 I2 = TEX1DFETCH(float4, SPINORTEX, sid + 2 * sp_stride);   
   float4 I3 = TEX1DFETCH(float4, SPINORTEX, sid + 3 * sp_stride);   
   float4 I4 = TEX1DFETCH(float4, SPINORTEX, sid + 4 * sp_stride);   
   float4 I5 = TEX1DFETCH(float4, SPINORTEX, sid + 5 * sp_stride);
   
   float C = TEX1DFETCH(float, SPINORTEXNORM, sid);
   
   I0 = C * I0;
   I1 = C * I1;
   I2 = C * I2;
   I3 = C * I3;
   I4 = C * I4;
   I5 = C * I5;    
   
   volatile float4 tmp0, tmp1;
   
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
   //get the 1st color component:(I0.xy, I1.zw, I3.xy, I4.zw)
   
   tmp0_re = I0.x - a * I3.y;
   tmp0_im = I0.y + a * I3.x;
   
   tmp1_re = I1.z - a * I4.w;
   tmp1_im = I1.w + a * I4.z;
   
   tmp2_re = I3.x - a * I0.y;
   tmp2_im = I3.y + a * I0.x;
   
   tmp3_re = I4.z - a * I1.w;
   tmp3_im = I4.w + a * I1.z;
   
   I0.x = b * tmp0_re;
   I0.y = b * tmp0_im;
   I1.z = b * tmp1_re;
   I1.w = b * tmp1_im;
   I3.x = b * tmp2_re;
   I3.y = b * tmp2_im;
   I4.z = b * tmp3_re;
   I4.w = b * tmp3_im;
   
   //get the 2nd color component:(I0.zw, I2.xy, I3.zw, I5.xy)
   
   tmp0_re = I0.z - a * I3.w;
   tmp0_im = I0.w + a * I3.z;
    
   tmp1_re = I2.x - a * I5.y;
   tmp1_im = I2.y + a * I5.x;
   
   tmp2_re = I3.z - a * I0.w;
   tmp2_im = I3.w + a * I0.z;
   
   tmp3_re = I5.x - a * I2.y;
   tmp3_im = I5.y + a * I2.x;
   
   I0.z = b * tmp0_re;
   I0.w = b * tmp0_im;
   I2.x = b * tmp1_re;
   I2.y = b * tmp1_im;
   I3.z = b * tmp2_re;
   I3.w = b * tmp2_im;
   I5.x = b * tmp3_re;
   I5.y = b * tmp3_im;
   
   //get the 3d color component:(I1.xy, I2.zw, I4.xy, I5.zw)
   
   tmp0_re = I1.x - a * I4.y;
   tmp0_im = I1.y + a * I4.x;
   
   tmp1_re = I2.z - a * I5.w;
   tmp1_im = I2.w + a * I5.z;
   
   tmp2_re = I4.x - a * I1.y;
   tmp2_im = I4.y + a * I1.x;
   
   tmp3_re = I5.z - a * I2.w;
   tmp3_im = I5.w + a * I2.z;
   
   I1.x = b * tmp0_re;
   I1.y = b * tmp0_im;
   I2.z = b * tmp1_re;
   I2.w = b * tmp1_im;
   I4.x = b * tmp2_re;
   I4.y = b * tmp2_im;
   I5.z = b * tmp3_re;
   I5.w = b * tmp3_im;
   
   
   float c0  = fmaxf(fabsf(I0.x), fabsf(I0.y));			
   float c1  = fmaxf(fabsf(I0.z), fabsf(I0.w));			
   float c2  = fmaxf(fabsf(I1.x), fabsf(I1.y));			
   float c3  = fmaxf(fabsf(I1.z), fabsf(I1.w));			
   float c4  = fmaxf(fabsf(I2.x), fabsf(I2.y));			
   float c5  = fmaxf(fabsf(I2.z), fabsf(I2.w));			
   float c6  = fmaxf(fabsf(I3.x), fabsf(I3.y));			
   float c7  = fmaxf(fabsf(I3.z), fabsf(I3.w));			
   float c8  = fmaxf(fabsf(I4.x), fabsf(I4.y));			
   float c9  = fmaxf(fabsf(I4.z), fabsf(I4.w));			
   float c10 = fmaxf(fabsf(I5.x), fabsf(I5.y));			
   float c11 = fmaxf(fabsf(I5.z), fabsf(I5.w));			
   c0 = fmaxf(c0, c1);							
   c1 = fmaxf(c2, c3);							
   c2 = fmaxf(c4, c5);							
   c3 = fmaxf(c6, c7);							
   c4 = fmaxf(c8, c9);							
   c5 = fmaxf(c10, c11);							
   c0 = fmaxf(c0, c1);							
   c1 = fmaxf(c2, c3);							
   c2 = fmaxf(c4, c5);							
   c0 = fmaxf(c0, c1);							
   c0 = fmaxf(c0, c2);							
   spinorNorm[sid] = c0;								
   float scale = __fdividef(MAX_SHORT, c0);
   
   I0 = scale * I0; 	
   I1 = scale * I1;
   I2 = scale * I2;
   I3 = scale * I3;
   I4 = scale * I4;
   I5 = scale * I5;
   
   spinor[sid+0*(sp_stride)] = make_short4((short)I0.x, (short)I0.y, (short)I0.z, (short)I0.w); 
   spinor[sid+1*(sp_stride)] = make_short4((short)I1.x, (short)I1.y, (short)I1.z, (short)I1.w); 
   spinor[sid+2*(sp_stride)] = make_short4((short)I2.x, (short)I2.y, (short)I2.z, (short)I2.w); 
   spinor[sid+3*(sp_stride)] = make_short4((short)I3.x, (short)I3.y, (short)I3.z, (short)I3.w); 
   spinor[sid+4*(sp_stride)] = make_short4((short)I4.x, (short)I4.y, (short)I4.z, (short)I4.w); 
   spinor[sid+5*(sp_stride)] = make_short4((short)I5.x, (short)I5.y, (short)I5.z, (short)I5.w);

#endif 
}


__global__ void twistGamma5Kernel(short4* spinor, float *spinorNorm, float a, float b, float c, const short4 *in, const float *inNorm, DslashParam param)
{
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
   int sid = blockIdx.x * blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;
   
   int flv1_idx = sid;
   int flv2_idx = sid + fl_stride; 
   
   float C1 = TEX1DFETCH(float, SPINORTEXNORM, flv1_idx);
   float C2 = TEX1DFETCH(float, SPINORTEXNORM, flv2_idx);
      
   float4 accum1_0, accum1_1, accum1_2, accum1_3, accum1_4, accum1_5;
   float4 accum2_0, accum2_1, accum2_2, accum2_3, accum2_4, accum2_5;
   
   float4 tmp0, tmp1;
   
   C1 *= c, C2 *= c;
   
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 0 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 3 * sp_stride); 
    
   tmp0 = C1 * tmp0; 
   tmp1 = C1 * tmp1;
   
   accum1_0.x = tmp0.x + a * tmp1.y;
   accum1_0.y = tmp0.y - a * tmp1.x;
   accum1_0.z = tmp0.z + a * tmp1.w;
   accum1_0.w = tmp0.w - a * tmp1.z;

   accum2_0.x = b * tmp0.x;
   accum2_0.y = b * tmp0.y;
   accum2_0.z = b * tmp0.z;
   accum2_0.w = b * tmp0.w;
  
   accum1_3.x = tmp1.x + a * tmp0.y;
   accum1_3.y = tmp1.y - a * tmp0.x;
   accum1_3.z = tmp1.z + a * tmp0.w;
   accum1_3.w = tmp1.w - a * tmp0.z;

   accum2_3.x = b * tmp1.x;
   accum2_3.y = b * tmp1.y;
   accum2_3.z = b * tmp1.z;
   accum2_3.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 0 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 3 * sp_stride);     

   tmp0 = C2 * tmp0; 
   tmp1 = C2 * tmp1;
   
   accum2_0.x += tmp0.x - a * tmp1.y;
   accum2_0.y += tmp0.y + a * tmp1.x;
   accum2_0.z += tmp0.z - a * tmp1.w;
   accum2_0.w += tmp0.w + a * tmp1.z;

   accum1_0.x += b * tmp0.x;
   accum1_0.y += b * tmp0.y;
   accum1_0.z += b * tmp0.z;
   accum1_0.w += b * tmp0.w;
  
   accum2_3.x += tmp1.x - a * tmp0.y;
   accum2_3.y += tmp1.y + a * tmp0.x;
   accum2_3.z += tmp1.z - a * tmp0.w;
   accum2_3.w += tmp1.w + a * tmp0.z;

   accum1_3.x += b * tmp1.x;
   accum1_3.y += b * tmp1.y;
   accum1_3.z += b * tmp1.z;
   accum1_3.w += b * tmp1.w;
   
   float c1_0  = fmaxf(fabsf(accum1_0.x), fabsf(accum1_0.y));			
   float c1_1  = fmaxf(fabsf(accum1_0.z), fabsf(accum1_0.w));
   float c1_6  = fmaxf(fabsf(accum1_3.x), fabsf(accum1_3.y));			
   float c1_7  = fmaxf(fabsf(accum1_3.z), fabsf(accum1_3.w));	   

   float c2_0  = fmaxf(fabsf(accum2_0.x), fabsf(accum2_0.y));			
   float c2_1  = fmaxf(fabsf(accum2_0.z), fabsf(accum2_0.w));
   float c2_6  = fmaxf(fabsf(accum2_3.x), fabsf(accum2_3.y));			
   float c2_7  = fmaxf(fabsf(accum2_3.z), fabsf(accum2_3.w));	   
   
   //get the second color component for each flavor: 
   
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 1 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 4 * sp_stride); 
    
   tmp0 = C1 * tmp0; 
   tmp1 = C1 * tmp1;
   
   accum1_1.x = tmp0.x + a * tmp1.y;
   accum1_1.y = tmp0.y - a * tmp1.x;
   accum1_1.z = tmp0.z + a * tmp1.w;
   accum1_1.w = tmp0.w - a * tmp1.z;

   accum2_1.x = b * tmp0.x;
   accum2_1.y = b * tmp0.y;
   accum2_1.z = b * tmp0.z;
   accum2_1.w = b * tmp0.w;
  
   accum1_4.x = tmp1.x + a * tmp0.y;
   accum1_4.y = tmp1.y - a * tmp0.x;
   accum1_4.z = tmp1.z + a * tmp0.w;
   accum1_4.w = tmp1.w - a * tmp0.z;

   accum2_4.x = b * tmp1.x;
   accum2_4.y = b * tmp1.y;
   accum2_4.z = b * tmp1.z;
   accum2_4.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 1 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 4 * sp_stride);     

   tmp0 = C2 * tmp0; 
   tmp1 = C2 * tmp1;
   
   accum2_1.x += tmp0.x - a * tmp1.y;
   accum2_1.y += tmp0.y + a * tmp1.x;
   accum2_1.z += tmp0.z - a * tmp1.w;
   accum2_1.w += tmp0.w + a * tmp1.z;

   accum1_1.x += b * tmp0.x;
   accum1_1.y += b * tmp0.y;
   accum1_1.z += b * tmp0.z;
   accum1_1.w += b * tmp0.w;
  
   accum2_4.x += tmp1.x - a * tmp0.y;
   accum2_4.y += tmp1.y + a * tmp0.x;
   accum2_4.z += tmp1.z - a * tmp0.w;
   accum2_4.w += tmp1.w + a * tmp0.z;

   accum1_4.x += b * tmp1.x;
   accum1_4.y += b * tmp1.y;
   accum1_4.z += b * tmp1.z;
   accum1_4.w += b * tmp1.w;
   
   float c1_2  = fmaxf(fabsf(accum1_1.x), fabsf(accum1_1.y));			
   float c1_3  = fmaxf(fabsf(accum1_1.z), fabsf(accum1_1.w));
   float c1_8  = fmaxf(fabsf(accum1_4.x), fabsf(accum1_4.y));			
   float c1_9  = fmaxf(fabsf(accum1_4.z), fabsf(accum1_4.w));	   

   float c2_2  = fmaxf(fabsf(accum2_1.x), fabsf(accum2_1.y));			
   float c2_3  = fmaxf(fabsf(accum2_1.z), fabsf(accum2_1.w));
   float c2_8  = fmaxf(fabsf(accum2_4.x), fabsf(accum2_4.y));			
   float c2_9  = fmaxf(fabsf(accum2_4.z), fabsf(accum2_4.w));	   

   //get the third color component for each flavor: 
      
   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 2 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv1_idx + 5 * sp_stride); 
    
   tmp0 = C1 * tmp0; 
   tmp1 = C1 * tmp1;
   
   accum1_2.x = tmp0.x + a * tmp1.y;
   accum1_2.y = tmp0.y - a * tmp1.x;
   accum1_2.z = tmp0.z + a * tmp1.w;
   accum1_2.w = tmp0.w - a * tmp1.z;

   accum2_2.x = b * tmp0.x;
   accum2_2.y = b * tmp0.y;
   accum2_2.z = b * tmp0.z;
   accum2_2.w = b * tmp0.w;
  
   accum1_5.x = tmp1.x + a * tmp0.y;
   accum1_5.y = tmp1.y - a * tmp0.x;
   accum1_5.z = tmp1.z + a * tmp0.w;
   accum1_5.w = tmp1.w - a * tmp0.z;

   accum2_5.x = b * tmp1.x;
   accum2_5.y = b * tmp1.y;
   accum2_5.z = b * tmp1.z;
   accum2_5.w = b * tmp1.w;

   tmp0 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 2 * sp_stride);   
   tmp1 = TEX1DFETCH(float4, SPINORTEX, flv2_idx + 5 * sp_stride);     

   tmp0 = C2 * tmp0; 
   tmp1 = C2 * tmp1;
   
   accum2_2.x += tmp0.x - a * tmp1.y;
   accum2_2.y += tmp0.y + a * tmp1.x;
   accum2_2.z += tmp0.z - a * tmp1.w;
   accum2_2.w += tmp0.w + a * tmp1.z;

   accum1_2.x += b * tmp0.x;
   accum1_2.y += b * tmp0.y;
   accum1_2.z += b * tmp0.z;
   accum1_2.w += b * tmp0.w;
  
   accum2_5.x += tmp1.x - a * tmp0.y;
   accum2_5.y += tmp1.y + a * tmp0.x;
   accum2_5.z += tmp1.z - a * tmp0.w;
   accum2_5.w += tmp1.w + a * tmp0.z;

   accum1_5.x += b * tmp1.x;
   accum1_5.y += b * tmp1.y;
   accum1_5.z += b * tmp1.z;
   accum1_5.w += b * tmp1.w;
   
   float c1_4  = fmaxf(fabsf(accum1_2.x), fabsf(accum1_2.y));			
   float c1_5  = fmaxf(fabsf(accum1_2.z), fabsf(accum1_2.w));
   float c1_10  = fmaxf(fabsf(accum1_5.x), fabsf(accum1_5.y));			
   float c1_11  = fmaxf(fabsf(accum1_5.z), fabsf(accum1_5.w));	   

   float c2_4  = fmaxf(fabsf(accum2_2.x), fabsf(accum2_2.y));			
   float c2_5  = fmaxf(fabsf(accum2_2.z), fabsf(accum2_2.w));
   float c2_10  = fmaxf(fabsf(accum2_5.x), fabsf(accum2_5.y));			
   float c2_11  = fmaxf(fabsf(accum2_5.z), fabsf(accum2_5.w));	
   
   
   c1_0 = fmaxf(c1_0, c1_1);							
   c1_1 = fmaxf(c1_2, c1_3);							
   c1_2 = fmaxf(c1_4, c1_5);							
   c1_3 = fmaxf(c1_6, c1_7);							
   c1_4 = fmaxf(c1_8, c1_9);							
   c1_5 = fmaxf(c1_10, c1_11);							
   c1_0 = fmaxf(c1_0, c1_1);							
   c1_1 = fmaxf(c1_2, c1_3);							
   c1_2 = fmaxf(c1_4, c1_5);							
   c1_0 = fmaxf(c1_0, c1_1);							
   c1_0 = fmaxf(c1_0, c1_2);							
   spinorNorm[flv1_idx] = c1_0;								
   float scale = __fdividef(MAX_SHORT, c1_0);
   
   accum1_0 = scale * accum1_0; 	
   accum1_1 = scale * accum1_1;
   accum1_2 = scale * accum1_2;
   accum1_3 = scale * accum1_3;
   accum1_4 = scale * accum1_4;
   accum1_5 = scale * accum1_5;
   
   c2_0 = fmaxf(c2_0, c2_1);							
   c2_1 = fmaxf(c2_2, c2_3);							
   c2_2 = fmaxf(c2_4, c2_5);							
   c2_3 = fmaxf(c2_6, c2_7);							
   c2_4 = fmaxf(c2_8, c2_9);							
   c2_5 = fmaxf(c2_10, c2_11);							
   c2_0 = fmaxf(c2_0, c2_1);							
   c2_1 = fmaxf(c2_2, c2_3);							
   c2_2 = fmaxf(c2_4, c2_5);							
   c2_0 = fmaxf(c2_0, c2_1);							
   c2_0 = fmaxf(c2_0, c2_2);							
   spinorNorm[flv2_idx] = c2_0;								
   scale = __fdividef(MAX_SHORT, c2_0);
   
   accum2_0 = scale * accum2_0; 	
   accum2_1 = scale * accum2_1;
   accum2_2 = scale * accum2_2;
   accum2_3 = scale * accum2_3;
   accum2_4 = scale * accum2_4;
   accum2_5 = scale * accum2_5;   

   
   spinor[flv1_idx+0*(sp_stride)] = make_short4((short)accum1_0.x, (short)accum1_0.y, (short)accum1_0.z, (short)accum1_0.w); 
   spinor[flv1_idx+1*(sp_stride)] = make_short4((short)accum1_1.x, (short)accum1_1.y, (short)accum1_1.z, (short)accum1_1.w); 
   spinor[flv1_idx+2*(sp_stride)] = make_short4((short)accum1_2.x, (short)accum1_2.y, (short)accum1_2.z, (short)accum1_2.w); 
   spinor[flv1_idx+3*(sp_stride)] = make_short4((short)accum1_3.x, (short)accum1_3.y, (short)accum1_3.z, (short)accum1_3.w); 
   spinor[flv1_idx+4*(sp_stride)] = make_short4((short)accum1_4.x, (short)accum1_4.y, (short)accum1_4.z, (short)accum1_4.w); 
   spinor[flv1_idx+5*(sp_stride)] = make_short4((short)accum1_5.x, (short)accum1_5.y, (short)accum1_5.z, (short)accum1_5.w);
   
   spinor[flv2_idx+0*(sp_stride)] = make_short4((short)accum2_0.x, (short)accum2_0.y, (short)accum2_0.z, (short)accum2_0.w); 
   spinor[flv2_idx+1*(sp_stride)] = make_short4((short)accum2_1.x, (short)accum2_1.y, (short)accum2_1.z, (short)accum2_1.w); 
   spinor[flv2_idx+2*(sp_stride)] = make_short4((short)accum2_2.x, (short)accum2_2.y, (short)accum2_2.z, (short)accum2_2.w); 
   spinor[flv2_idx+3*(sp_stride)] = make_short4((short)accum2_3.x, (short)accum2_3.y, (short)accum2_3.z, (short)accum2_3.w); 
   spinor[flv2_idx+4*(sp_stride)] = make_short4((short)accum2_4.x, (short)accum2_4.y, (short)accum2_4.z, (short)accum2_4.w); 
   spinor[flv2_idx+5*(sp_stride)] = make_short4((short)accum2_5.x, (short)accum2_5.y, (short)accum2_5.z, (short)accum2_5.w);  
 
#endif
}

#undef SPINORTEX
#undef SPINORTEXNORM


#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#endif //_TM_CUDA_H


