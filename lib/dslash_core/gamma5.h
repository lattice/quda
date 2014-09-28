#ifndef _TWIST_QUDA_G5
#define _TWIST_QUDA_G5

//action of the operator b*(1 + i*a*gamma5)
//used also macros from io_spinor.h

#if ((CUDA_VERSION >= 4010) && (__COMPUTE_CAPABILITY__ >= 200)) // NVVM compiler
#define VOLATILE
#else // Open64 compiler
#define VOLATILE volatile
#endif
// input spinor

#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp1.x
#define tmp1_im tmp1.y
#define tmp2_re tmp2.x
#define tmp2_im tmp2.y
#define tmp3_re tmp3.x
#define tmp3_im tmp3.y

#if (__COMPUTE_CAPABILITY__ >= 130)
#ifdef DIRECT_ACCESS_WILSON_SPINOR
	#define READ_SPINOR READ_SPINOR_DOUBLE
	#define SPINORTEX in
#else
	#define READ_SPINOR READ_SPINOR_DOUBLE_TEX

	#ifdef USE_TEXTURE_OBJECTS
		#define SPINORTEX param.inTex
	#else
		#define SPINORTEX spinorTexDouble
	#endif	// USE_TEXTURE_OBJECTS
#endif

#define	SPINOR_HOP	12
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2_STR

#define o00_re I0.x
#define o00_im I0.y
#define o01_re I1.x
#define o01_im I1.y
#define o02_re I2.x
#define o02_im I2.y
#define o10_re I3.x
#define o10_im I3.y
#define o11_re I4.x
#define o11_im I4.y
#define o12_re I5.x
#define o12_im I5.y
#define o20_re I6.x
#define o20_im I6.y
#define o21_re I7.x
#define o21_im I7.y
#define o22_re I8.x
#define o22_im I8.y
#define o30_re I9.x
#define o30_im I9.y
#define o31_re I10.x
#define o31_im I10.y
#define o32_re I11.x
#define o32_im I11.y

__global__ void gamma5Kernel(double2 *out, float *outNorm, double2 *in, float *inNorm, DslashParam param, int myStride)
{
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;
/*
   // output spinor
   VOLATILE double2 I0;
   VOLATILE double2 I1;
   VOLATILE double2 I2;
   VOLATILE double2 I3;
   VOLATILE double2 I4;
   VOLATILE double2 I5;
   VOLATILE double2 I6;
   VOLATILE double2 I7;
   VOLATILE double2 I8;
   VOLATILE double2 I9;
   VOLATILE double2 I10;
   VOLATILE double2 I11;
*/
   READ_SPINOR			(SPINORTEX, myStride, sid, sid);
/*
#if defined(FERMI_NO_DBLE_TEX) || defined (USE_TEXTURE_OBJECTS)
   double2 I0  = spinor[sid + 0 * sp_stride];   
   double2 I1  = spinor[sid + 1 * sp_stride];   
   double2 I2  = spinor[sid + 2 * sp_stride];   
   double2 I3  = spinor[sid + 3 * sp_stride];   
   double2 I4  = spinor[sid + 4 * sp_stride];   
   double2 I5  = spinor[sid + 5 * sp_stride];   
   double2 I6  = spinor[sid + 6 * sp_stride];   
   double2 I7  = spinor[sid + 7 * sp_stride];   
   double2 I8  = spinor[sid + 8 * sp_stride];   
   double2 I9  = spinor[sid + 9 * sp_stride];   
   double2 I10 = spinor[sid + 10 * sp_stride]; 
   double2 I11 = spinor[sid + 11 * sp_stride];
#else
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
#endif
*/

   volatile double2 tmp0, tmp1, tmp2, tmp3;
   
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
    //get the 1st color component:
   
   tmp0_re = o20_re;
   tmp0_im = o20_im;
   
   tmp2_re = o00_re;
   tmp2_im = o00_im;
   
   tmp1_re = o30_re;
   tmp1_im = o30_im;
   
   tmp3_re = o10_re;
   tmp3_im = o10_im;
   
   o00_re = tmp0_re;
   o00_im = tmp0_im;
   o10_re = tmp1_re;
   o10_im = tmp1_im;
   o20_re = tmp2_re;
   o20_im = tmp2_im;
   o30_re = tmp3_re;
   o30_im = tmp3_im;
  
   //get the 2nd color component:    
   
   tmp0_re = o21_re;
   tmp0_im = o21_im;
   
   tmp2_re = o01_re;
   tmp2_im = o01_im;
   
   tmp1_re = o31_re;
   tmp1_im = o31_im;
   
   tmp3_re = o11_re;
   tmp3_im = o11_im;

   o01_re  = tmp0_re;
   o01_im  = tmp0_im;
   o11_re  = tmp1_re;
   o11_im  = tmp1_im;
   o21_re  = tmp2_re;
   o21_im  = tmp2_im;
   o31_re = tmp3_re;
   o31_im = tmp3_im;

   //get the 3d color component:    
   
   tmp0_re = o22_re;
   tmp0_im = o22_im;
          
   tmp2_re = o02_re;
   tmp2_im = o02_im;
          
   tmp1_re = o32_re;
   tmp1_im = o32_im;
          
   tmp3_re = o12_re;
   tmp3_im = o12_im;

   o02_re  = tmp0_re;
   o02_im  = tmp0_im;
   o12_re  = tmp1_re;
   o12_im  = tmp1_im;
   o22_re  = tmp2_re;
   o22_im  = tmp2_im;
   o32_re = tmp3_re;
   o32_im = tmp3_im;
/*
   spinor[sid +  0 * myStride] = I0;   
   spinor[sid +  1 * myStride] = I1;   
   spinor[sid +  2 * myStride] = I2;   
   spinor[sid +  3 * myStride] = I3;   
   spinor[sid +  4 * myStride] = I4;   
   spinor[sid +  5 * myStride] = I5;   
   spinor[sid +  6 * myStride] = I6;   
   spinor[sid +  7 * myStride] = I7;   
   spinor[sid +  8 * myStride] = I8;   
   spinor[sid +  9 * myStride] = I9;   
   spinor[sid + 10 * myStride] = I10;   
   spinor[sid + 11 * myStride] = I11;
*/

   WRITE_SPINOR(myStride);

   return;  
}
#endif // (__CUDA_ARCH__ >= 130)

#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#undef SPINOR_HOP
#undef READ_SPINOR
#undef SPINORTEX
#undef WRITE_SPINOR

#undef o00_re
#undef o00_im
#undef o01_re
#undef o01_im
#undef o02_re
#undef o02_im
#undef o10_re
#undef o10_im
#undef o11_re
#undef o11_im
#undef o12_re
#undef o12_im
#undef o20_re
#undef o20_im
#undef o21_re
#undef o21_im
#undef o22_re
#undef o22_im
#undef o30_re
#undef o30_im
#undef o31_re
#undef o31_im
#undef o32_re
#undef o32_im

#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp0.z
#define tmp1_im tmp0.w
#define tmp2_re tmp1.x
#define tmp2_im tmp1.y
#define tmp3_re tmp1.z
#define tmp3_im tmp1.w

#ifdef DIRECT_ACCESS_WILSON_SPINOR
	#define READ_SPINOR READ_SPINOR_SINGLE
	#define SPINORTEX in
#else
	#define READ_SPINOR READ_SPINOR_SINGLE_TEX

	#ifdef USE_TEXTURE_OBJECTS
		#define SPINORTEX param.inTex
	#else
		#define SPINORTEX spinorTexSingle
	#endif	// USE_TEXTURE_OBJECTS
#endif

#define	SPINOR_HOP	6
#define WRITE_SPINOR WRITE_SPINOR_FLOAT4_STR

#define o00_re I0.x
#define o00_im I0.y
#define o01_re I0.z
#define o01_im I0.w
#define o02_re I1.x
#define o02_im I1.y
#define o10_re I1.z
#define o10_im I1.w
#define o11_re I2.x
#define o11_im I2.y
#define o12_re I2.z
#define o12_im I2.w
#define o20_re I3.x
#define o20_im I3.y
#define o21_re I3.z
#define o21_im I3.w
#define o22_re I4.x
#define o22_im I4.y
#define o30_re I4.z
#define o30_im I4.w
#define o31_re I5.x
#define o31_im I5.y
#define o32_re I5.z
#define o32_im I5.w

__global__ void gamma5Kernel(float4 *out, float *outNorm, float4 *in, float *inNorm, DslashParam param, int myStride)
{
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;
/*
   // output spinor
   VOLATILE float4 I0;
   VOLATILE float4 I1;
   VOLATILE float4 I2;
   VOLATILE float4 I3;
   VOLATILE float4 I4;
   VOLATILE float4 I5;
/*
#if defined(FERMI_NO_DBLE_TEX) || defined (USE_TEXTURE_OBJECTS)
   float4 I0  = spinor[sid + 0 * sp_stride];
   float4 I1  = spinor[sid + 1 * sp_stride];
   float4 I2  = spinor[sid + 2 * sp_stride];
   float4 I3  = spinor[sid + 3 * sp_stride];
   float4 I4  = spinor[sid + 4 * sp_stride];
   float4 I5  = spinor[sid + 5 * sp_stride]; 
#else
   float4 I0 = tex1Dfetch(spinorTexSingle, sid + 0 * myStride);   
   float4 I1 = tex1Dfetch(spinorTexSingle, sid + 1 * myStride);   
   float4 I2 = tex1Dfetch(spinorTexSingle, sid + 2 * myStride);   
   float4 I3 = tex1Dfetch(spinorTexSingle, sid + 3 * myStride);   
   float4 I4 = tex1Dfetch(spinorTexSingle, sid + 4 * myStride);   
   float4 I5 = tex1Dfetch(spinorTexSingle, sid + 5 * myStride);
#endif
*/
   READ_SPINOR			(SPINORTEX, myStride, sid, sid);

   volatile float4 tmp0, tmp1;
    
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
   //get the 1st color component:(o00_rey, o10_rew, o20_rey, o30_rew)
   
   tmp0_re = o20_re;
   tmp0_im = o20_im;
   
   tmp1_re = o30_re;
   tmp1_im = o30_im;
   
   tmp2_re = o00_re;
   tmp2_im = o00_im;
             
   tmp3_re = o10_re;
   tmp3_im = o10_im;
   
   o00_re = tmp0_re;
   o00_im = tmp0_im;
   o10_re = tmp1_re;
   o10_im = tmp1_im;
   o20_re = tmp2_re;
   o20_im = tmp2_im;
   o30_re = tmp3_re;
   o30_im = tmp3_im;
   
   //get the 2nd color component:(o01_rew, o11_rey, o21_rew, o31_rey)
   
   tmp0_re = o21_re;
   tmp0_im = o21_im;
             
   tmp1_re = o31_re;
   tmp1_im = o31_im;
             
   tmp2_re = o01_re;
   tmp2_im = o01_im;
             
   tmp3_re = o11_re;
   tmp3_im = o11_im;
   
   o01_re = tmp0_re;
   o01_im = tmp0_im;
   o11_re = tmp1_re;
   o11_im = tmp1_im;
   o21_re = tmp2_re;
   o21_im = tmp2_im;
   o31_re = tmp3_re;
   o31_im = tmp3_im;
   
   //get the 3d color component:(o02_rey, o12_rew, o22_rey, o32_rew)
   
   tmp0_re = o22_re;
   tmp0_im = o22_im;
             
   tmp1_re = o32_re;
   tmp1_im = o32_im;
             
   tmp2_re = o02_re;
   tmp2_im = o02_im;
             
   tmp3_re = o12_re;
   tmp3_im = o12_im;
   
   o02_re = tmp0_re;
   o02_im = tmp0_im;
   o12_re = tmp1_re;
   o12_im = tmp1_im;
   o22_re = tmp2_re;
   o22_im = tmp2_im;
   o32_re = tmp3_re;
   o32_im = tmp3_im;
  /* 
   spinor[sid + 0  * myStride] = I0;   
   spinor[sid + 1  * myStride] = I1;   
   spinor[sid + 2  * myStride] = I2;   
   spinor[sid + 3  * myStride] = I3;   
   spinor[sid + 4  * myStride] = I4;   
   spinor[sid + 5  * myStride] = I5;   
*/

   WRITE_SPINOR(myStride);

   return;  
}

/*
__global__ void gamma5Kernel(short4* spinor, float *spinorNorm, DslashParam param, int myStride)
{
   int sid = blockIdx.x*blockDim.x + threadIdx.x;
   if (sid >= param.threads) return;

#if defined(FERMI_NO_DBLE_TEX) || defined (USE_TEXTURE_OBJECTS)
   float4 I0  = spinor[sid + 0 * sp_stride];
   float4 I1  = spinor[sid + 1 * sp_stride];
   float4 I2  = spinor[sid + 2 * sp_stride];
   float4 I3  = spinor[sid + 3 * sp_stride];
   float4 I4  = spinor[sid + 4 * sp_stride];
   float4 I5  = spinor[sid + 5 * sp_stride]; 
#else
   float4 I0 = tex1Dfetch(spinorTexHalf, sid + 0 * myStride);   
   float4 I1 = tex1Dfetch(spinorTexHalf, sid + 1 * myStride);   
   float4 I2 = tex1Dfetch(spinorTexHalf, sid + 2 * myStride);   
   float4 I3 = tex1Dfetch(spinorTexHalf, sid + 3 * myStride);   
   float4 I4 = tex1Dfetch(spinorTexHalf, sid + 4 * myStride);   
   float4 I5 = tex1Dfetch(spinorTexHalf, sid + 5 * myStride);
   
   float C = tex1Dfetch(spinorTexHalfNorm, sid);
#endif
   
   I0 = C * I0;
   I1 = C * I1;
   I2 = C * I2;
   I3 = C * I3;
   I4 = C * I4;
   I5 = C * I5;    
   
   volatile float4 tmp0, tmp1;
   
   //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
   
   //get the 1st color component:(o00_rey, o10_rew, o20_rey, o30_rew)
   
   tmp0_re = o20_re;
   tmp0_im = o20_im;
   
   tmp1_re = o30_re;
   tmp1_im = o30_im;
   
   tmp2_re = o00_re;
   tmp2_im = o00_im;
          
   tmp3_re = o10_re;
   tmp3_im = o10_im;
   
   o00_re = tmp0_re;
   o00_im = tmp0_im;
   o10_re = tmp1_re;
   o10_im = tmp1_im;
   o20_re = tmp2_re;
   o20_im = tmp2_im;
   o30_re = tmp3_re;
   o30_im = tmp3_im;
   
   //get the 2nd color component:(o01_rew, o11_rey, o21_rew, o31_rey)
   
   tmp0_re = o21_re;
   tmp0_im = o21_im;
          
   tmp1_re = o31_re;
   tmp1_im = o31_im;
          
   tmp2_re = o01_re;
   tmp2_im = o01_im;
          
   tmp3_re = o11_re;
   tmp3_im = o11_im;
   
   o01_re = tmp0_re;
   o01_im = tmp0_im;
   o11_re = tmp1_re;
   o11_im = tmp1_im;
   o21_re = tmp2_re;
   o21_im = tmp2_im;
   o31_re = tmp3_re;
   o31_im = tmp3_im;
   
   //get the 3d color component:(o02_rey, o12_rew, o22_rey, o32_rew)
   
   tmp0_re = o22_re;
   tmp0_im = o22_im;
          
   tmp1_re = o32_re;
   tmp1_im = o32_im;
          
   tmp2_re = o02_re;
   tmp2_im = o02_im;
          
   tmp3_re = o12_re;
   tmp3_im = o12_im;
   
   o02_re = tmp0_re;
   o02_im = tmp0_im;
   o12_re = tmp1_re;
   o12_im = tmp1_im;
   o22_re = tmp2_re;
   o22_im = tmp2_im;
   o32_re = tmp3_re;
   o32_im = tmp3_im;
   
   
   float c0  = fmaxf(fabsf(o00_re), fabsf(o00_im));			
   float c1  = fmaxf(fabsf(o01_re), fabsf(o01_im));			
   float c2  = fmaxf(fabsf(o02_re), fabsf(o02_im));			
   float c3  = fmaxf(fabsf(o10_re), fabsf(o10_im));			
   float c4  = fmaxf(fabsf(o11_re), fabsf(o11_im));			
   float c5  = fmaxf(fabsf(o12_re), fabsf(o12_im));			
   float c6  = fmaxf(fabsf(o20_re), fabsf(o20_im));			
   float c7  = fmaxf(fabsf(o21_re), fabsf(o21_im));			
   float c8  = fmaxf(fabsf(o22_re), fabsf(o22_im));			
   float c9  = fmaxf(fabsf(o30_re), fabsf(o30_im));			
   float c10 = fmaxf(fabsf(o31_re), fabsf(o31_im));			
   float c11 = fmaxf(fabsf(o32_re), fabsf(o32_im));			
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
   
   spinor[sid+0*(myStride)] = make_short4((short)o00_re, (short)o00_im, (short)o01_re, (short)o01_im); 
   spinor[sid+1*(myStride)] = make_short4((short)o02_re, (short)o02_im, (short)o10_re, (short)o10_im); 
   spinor[sid+2*(myStride)] = make_short4((short)o11_re, (short)o11_im, (short)o12_re, (short)o12_im); 
   spinor[sid+3*(myStride)] = make_short4((short)o20_re, (short)o20_im, (short)o21_re, (short)o21_im); 
   spinor[sid+4*(myStride)] = make_short4((short)o22_re, (short)o22_im, (short)o30_re, (short)o30_im); 
   spinor[sid+5*(myStride)] = make_short4((short)o31_re, (short)o31_im, (short)o32_re, (short)o32_im);

   return;  
}
*/
#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#endif	//_TWIST_QUDA_G5

