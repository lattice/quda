#ifndef _TWIST_QUDA_CUH
#define _TWIST_QUDA_CUH

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


#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp1.x
#define tmp1_im tmp1.y
#define tmp2_re tmp2.x
#define tmp2_im tmp2.y
#define tmp3_re tmp3.x
#define tmp3_im tmp3.y

#if (__COMPUTE_CAPABILITY__ >= 130)
__global__ void twistGamma5Kernel(double2 *spinor, float *null, double a, double b, 
				  const double2 *in, const float *null2, DslashParam param)
{

   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

#ifndef FERMI_NO_DBLE_TEX
   double2 I0  = fetch_double2(spinorTexDouble, sid + 0 * sp_stride);   
   double2 I1  = fetch_double2(spinorTexDouble, sid + 1 * sp_stride);   
   double2 I2  = fetch_double2(spinorTexDouble, sid + 2 * sp_stride);   
   double2 I3  = fetch_double2(spinorTexDouble, sid + 3 * sp_stride);   
   double2 I4  = fetch_double2(spinorTexDouble, sid + 4 * sp_stride);   
   double2 I5  = fetch_double2(spinorTexDouble, sid + 5 * sp_stride);   
   double2 I6  = fetch_double2(spinorTexDouble, sid + 6 * sp_stride);   
   double2 I7  = fetch_double2(spinorTexDouble, sid + 7 * sp_stride);   
   double2 I8  = fetch_double2(spinorTexDouble, sid + 8 * sp_stride);   
   double2 I9  = fetch_double2(spinorTexDouble, sid + 9 * sp_stride);   
   double2 I10 = fetch_double2(spinorTexDouble, sid + 10 * sp_stride); 
   double2 I11 = fetch_double2(spinorTexDouble, sid + 11 * sp_stride);
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

   return;  
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

__global__ void twistGamma5Kernel(float4 *spinor, float *null, float a, float b, 
				  const float4 *in, const float *null2, DslashParam param)
{
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = tex1Dfetch(spinorTexSingle, sid + 0 * sp_stride);   
   float4 I1 = tex1Dfetch(spinorTexSingle, sid + 1 * sp_stride);   
   float4 I2 = tex1Dfetch(spinorTexSingle, sid + 2 * sp_stride);   
   float4 I3 = tex1Dfetch(spinorTexSingle, sid + 3 * sp_stride);   
   float4 I4 = tex1Dfetch(spinorTexSingle, sid + 4 * sp_stride);   
   float4 I5 = tex1Dfetch(spinorTexSingle, sid + 5 * sp_stride);

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

   return;  
}


__global__ void twistGamma5Kernel(short4* spinor, float *spinorNorm, float a, float b, 
				  const short4 *in, const float *inNorm, DslashParam param)
{
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

   float4 I0 = tex1Dfetch(spinorTexHalf, sid + 0 * sp_stride);   
   float4 I1 = tex1Dfetch(spinorTexHalf, sid + 1 * sp_stride);   
   float4 I2 = tex1Dfetch(spinorTexHalf, sid + 2 * sp_stride);   
   float4 I3 = tex1Dfetch(spinorTexHalf, sid + 3 * sp_stride);   
   float4 I4 = tex1Dfetch(spinorTexHalf, sid + 4 * sp_stride);   
   float4 I5 = tex1Dfetch(spinorTexHalf, sid + 5 * sp_stride);
   
   float C = tex1Dfetch(spinorTexHalfNorm, sid);
   
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

   return;  
}

#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#endif //_TWIST_QUDA_CUH


