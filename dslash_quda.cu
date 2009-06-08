#include <stdlib.h>
#include <stdio.h>

#include <dslash_quda.h>

// ----------------------------------------------------------------------
// Cuda code

#if (__CUDA_ARCH__ == 130)
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
    int4 v = tex1Dfetch(t,i);
    return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#endif

// Double precision gauge field
texture<int4, 1> gauge0TexDouble;
texture<int4, 1> gauge1TexDouble;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle;

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf;

// Single precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<float, 1, cudaReadModeElementType> spinorTexNorm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTexSingle;

// Half precision accumulate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<float, 1, cudaReadModeElementType> accumTexNorm;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

QudaGaugeParam *gauge_param;
QudaInvertParam *invert_param;

__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;
__constant__ int X1h;

__constant__ float anisotropy;
__constant__ float t_boundary;
__constant__ int gauge_fixed;
__constant__ float pi;

#include <dslash_def.h>

__global__ void spinorHalfPack(float *c, void *half) {

  int sid = BLOCK_DIM*blockIdx.x + threadIdx.x;
  short4 *h = (short4 *)half;

  float4 F0 = tex1Dfetch(spinorTexSingle, sid + 0*Nh);
  float4 F1 = tex1Dfetch(spinorTexSingle, sid + 1*Nh);
  float4 F2 = tex1Dfetch(spinorTexSingle, sid + 2*Nh);
  float4 F3 = tex1Dfetch(spinorTexSingle, sid + 3*Nh);
  float4 F4 = tex1Dfetch(spinorTexSingle, sid + 4*Nh);
  float4 F5 = tex1Dfetch(spinorTexSingle, sid + 5*Nh);
  
  float c0 = fmaxf(fabsf(F0.x), fabsf(F0.y));
  float c1 = fmaxf(fabsf(F0.z), fabsf(F0.w));
  float c2 = fmaxf(fabsf(F1.x), fabsf(F1.y));
  float c3 = fmaxf(fabsf(F1.z), fabsf(F1.w));
  float c4 = fmaxf(fabsf(F2.x), fabsf(F2.y));    
  float c5 = fmaxf(fabsf(F2.z), fabsf(F2.w));
  float c6 = fmaxf(fabsf(F3.x), fabsf(F3.y));
  float c7 = fmaxf(fabsf(F3.z), fabsf(F3.w));
  float c8 = fmaxf(fabsf(F4.x), fabsf(F4.y));
  float c9 = fmaxf(fabsf(F4.z), fabsf(F4.w));
  float c10 = fmaxf(fabsf(F5.x), fabsf(F5.y));
  float c11 = fmaxf(fabsf(F5.z), fabsf(F5.w));
  
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
  c0 = fmaxf(c0, c2); // c0 is now the maximum element
  
  c[sid] = c0;

  float scale = __fdividef(MAX_SHORT, c0);
  
  F0.x *= scale; F0.y *= scale; F0.z *= scale; F0.w *= scale;
  F1.x *= scale; F1.y *= scale; F1.z *= scale; F1.w *= scale;
  F2.x *= scale; F2.y *= scale; F2.z *= scale; F2.w *= scale;
  F3.x *= scale; F3.y *= scale; F3.z *= scale; F3.w *= scale;
  F4.x *= scale; F4.y *= scale; F4.z *= scale; F4.w *= scale;
  F5.x *= scale; F5.y *= scale; F5.z *= scale; F5.w *= scale;
  
  h[sid+0*Nh] = make_short4((short)F0.x, (short)F0.y, (short)F0.z, (short)F0.w);
  h[sid+1*Nh] = make_short4((short)F1.x, (short)F1.y, (short)F1.z, (short)F1.w);
  h[sid+2*Nh] = make_short4((short)F2.x, (short)F2.y, (short)F2.z, (short)F2.w);
  h[sid+3*Nh] = make_short4((short)F3.x, (short)F3.y, (short)F3.z, (short)F3.w);
  h[sid+4*Nh] = make_short4((short)F4.x, (short)F4.y, (short)F4.z, (short)F4.w);
  h[sid+5*Nh] = make_short4((short)F5.x, (short)F5.y, (short)F5.z, (short)F5.w);
}

__global__ void spinorHalfUnpack(ParitySpinor out) {

  float4* out4 = (float4*)out.spinor;

  int sid = BLOCK_DIM*blockIdx.x + threadIdx.x;

  float4 I0 = tex1Dfetch(spinorTexHalf, sid + 0*Nh);
  float4 I1 = tex1Dfetch(spinorTexHalf, sid + 1*Nh);
  float4 I2 = tex1Dfetch(spinorTexHalf, sid + 2*Nh);
  float4 I3 = tex1Dfetch(spinorTexHalf, sid + 3*Nh);
  float4 I4 = tex1Dfetch(spinorTexHalf, sid + 4*Nh);
  float4 I5 = tex1Dfetch(spinorTexHalf, sid + 5*Nh);
  float C = tex1Dfetch(spinorTexNorm, sid);
  I0.x *= C; I0.y *= C;	I0.z *= C; I0.w *= C;
  I1.x *= C; I1.y *= C;	I1.z *= C; I1.w *= C;
  I2.x *= C; I2.y *= C;	I2.z *= C; I2.w *= C;
  I3.x *= C; I3.y *= C;	I3.z *= C; I3.w *= C;
  I4.x *= C; I4.y *= C; I4.z *= C; I4.w *= C;
  I5.x *= C; I5.y *= C;	I5.z *= C; I5.w *= C;

  out4[0*Nh+sid] = I0;
  out4[1*Nh+sid] = I1;
  out4[2*Nh+sid] = I2;
  out4[3*Nh+sid] = I3;
  out4[4*Nh+sid] = I4;
  out4[5*Nh+sid] = I5;
}

void setCudaGaugeParam() {
  cudaMemcpyToSymbol("anisotropy", &(gauge_param->anisotropy), sizeof(float));
  float t_bc = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(float));
  int gf = (gauge_param->gauge_fix == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));
  float h_pi = M_PI;
  cudaMemcpyToSymbol("pi", &(h_pi), sizeof(float));
}

void bindGaugeTex(FullGauge gauge, int oddBit) {
  int reconstruct = (gauge.reconstruct == QUDA_RECONSTRUCT_12) ? 12 : 8;
  int packed_gauge_bytes = 4*Nh*reconstruct;

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    packed_gauge_bytes *= sizeof(double);
    if (oddBit) {
      cudaBindTexture(0, gauge0TexDouble, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexDouble, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexDouble, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexDouble, gauge.odd, packed_gauge_bytes); 
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    packed_gauge_bytes *= sizeof(float);
    if (oddBit) {
      cudaBindTexture(0, gauge0TexSingle, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexSingle, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexSingle, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexSingle, gauge.odd, packed_gauge_bytes); 
    }
  } else {
    packed_gauge_bytes *= sizeof(float)/2;
    if (oddBit) {
      cudaBindTexture(0, gauge0TexHalf, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexHalf, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexHalf, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexHalf, gauge.odd, packed_gauge_bytes); 
    }
  }
}


// ----------------------------------------------------------------------

void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  if (invert_param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    dslashDCuda(out, gauge, in, parity, dagger);
  } else if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    dslashSCuda(out, gauge, in, parity, dagger);
  } else if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    dim3 gridDim(GRID_DIM, 1, 1);
    dim3 blockDim(BLOCK_DIM, 1, 1);

    int spinor_float_bytes = Nh*spinorSiteSize*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, in.spinor, spinor_float_bytes); 
    spinorHalfPack <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>(hSpinor1.spinorNorm, hSpinor1.spinor);
    
    dslashHCuda(hSpinor2, gauge, hSpinor1, parity, dagger);

    int spinor_half_bytes = Nh*spinorSiteSize*sizeof(float)/2;
    cudaBindTexture(0, spinorTexHalf, hSpinor2.spinor, spinor_half_bytes); 
    cudaBindTexture(0, spinorTexNorm, hSpinor2.spinorNorm, spinor_half_bytes/12); 
    spinorHalfUnpack <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>(out);
  }
}


void dslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashDD12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDD8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashDD8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashSD12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSD8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashSD8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashHD12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHD8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashHD8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit);
      }
    }
  }
  
}


void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashDS12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDS8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashDS8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashSS12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSS8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashSS8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashHS12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHS8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashHS8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit);
      }
    }
  }
  
}


void dslashHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDH8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSH8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHH8Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  }
  
}

void dslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, float a) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashDS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashDS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashSS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashSS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
    } else {
	dslashHS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashHS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  }

}

void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, float a) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashDS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashDS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashSS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashSS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
    } else {
	dslashHS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashHS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  }

}

void dslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, float a) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashDH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashDH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashSH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashSH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashHH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashHH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  }

}

int dslashCudaSharedBytes() {
  return SHARED_BYTES_SINGLE;
}


// Apply the even-odd preconditioned Dirac operator
void MatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
	       ParitySpinor tmp, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (invert_param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashDCuda(tmp, gauge, in, 1, 0);
      dslashXpayDCuda(out, gauge, tmp, 0, 0, in, kappa2); 
    } else {
      dslashDCuda(tmp, gauge, in, 0, 0);
      dslashXpayDCuda(out, gauge, tmp, 1, 0, in, kappa2); 
    }
  } else if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashSCuda(tmp, gauge, in, 1, 0);
      dslashXpaySCuda(out, gauge, tmp, 0, 0, in, kappa2); 
    } else {
      dslashSCuda(tmp, gauge, in, 0, 0);
      dslashXpaySCuda(out, gauge, tmp, 1, 0, in, kappa2); 
    }
  } else if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    dim3 gridDim(GRID_DIM, 1, 1);
    dim3 blockDim(BLOCK_DIM, 1, 1);

    int spinor_bytes = Nh*spinorSiteSize*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, in.spinor, spinor_bytes); 
    spinorHalfPack <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>(hSpinor1.spinorNorm, hSpinor1.spinor);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) dslashHCuda(hSpinor2, gauge, hSpinor1, 1, 0);
    else dslashHCuda(hSpinor2, gauge, hSpinor1, 0, 0);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) dslashXpayHCuda(out, gauge, hSpinor2, 0, 0, hSpinor1, kappa2); 
    else dslashXpayHCuda(out, gauge, hSpinor2, 1, 0, hSpinor1, kappa2); 
  }

}

// Apply the even-odd preconditioned Dirac operator
void MatPCDagCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
		  ParitySpinor tmp, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (invert_param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashDCuda(tmp, gauge, in, 1, 1);
      dslashXpayDCuda(out, gauge, tmp, 0, 1, in, kappa2);
    } else {
      dslashDCuda(tmp, gauge, in, 0, 1);
      dslashXpayDCuda(out, gauge, tmp, 1, 1, in, kappa2);
    }
  } else if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashSCuda(tmp, gauge, in, 1, 1);
      dslashXpaySCuda(out, gauge, tmp, 0, 1, in, kappa2);
    } else {
      dslashSCuda(tmp, gauge, in, 0, 1);
      dslashXpaySCuda(out, gauge, tmp, 1, 1, in, kappa2);
    }
  } else {
    dim3 gridDim(GRID_DIM, 1, 1);
    dim3 blockDim(BLOCK_DIM, 1, 1);

    int spinor_bytes = Nh*spinorSiteSize*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, in.spinor, spinor_bytes); 
    spinorHalfPack <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>(hSpinor1.spinorNorm, hSpinor1.spinor);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) dslashHCuda(hSpinor2, gauge, hSpinor1, 1, 1);
    else dslashHCuda(hSpinor2, gauge, hSpinor1, 0, 1);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) dslashXpayHCuda(out, gauge, hSpinor2, 0, 1, hSpinor1, kappa2);
    else dslashXpayHCuda(out, gauge, hSpinor2, 1, 1, hSpinor1, kappa2);
  }

}

void MatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       float kappa, ParitySpinor tmp, MatPCType matpc_type) {
  MatPCCuda(out, gauge, in, kappa, tmp, matpc_type);
  MatPCDagCuda(out, gauge, out, kappa, tmp, matpc_type);
}

// Apply the full operator
void MatCuda(FullSpinor out, FullGauge gauge, FullSpinor in, float kappa) {

  if (invert_param->cuda_prec == QUDA_DOUBLE_PRECISION) {
    dslashXpayDCuda(out.odd, gauge, in.even, 1, 0, in.odd, -kappa);
    dslashXpayDCuda(out.even, gauge, in.odd, 0, 0, in.even, -kappa);
  } else if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    dslashXpaySCuda(out.odd, gauge, in.even, 1, 0, in.odd, -kappa);
    dslashXpaySCuda(out.even, gauge, in.odd, 0, 0, in.even, -kappa);
  } else if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported in MatCuda\n");
    exit(-1);
  }

}

// Apply the full operator dagger
void MatDaggerCuda(FullSpinor out, FullGauge gauge, FullSpinor in, float kappa) {

  if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    dslashXpayDCuda(out.odd, gauge, in.even, 1, 1, in.odd, -kappa);
    dslashXpayDCuda(out.even, gauge, in.odd, 0, 1, in.even, -kappa);
  } else if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    dslashXpaySCuda(out.odd, gauge, in.even, 1, 1, in.odd, -kappa);
    dslashXpaySCuda(out.even, gauge, in.odd, 0, 1, in.even, -kappa);
  } else if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported in MatDaggerCuda\n");
    exit(-1);
  }

}

/*
// Apply the even-odd preconditioned Dirac operator
QudaSumComplex MatPCcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
				ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashSCuda(tmp, gauge, in, 1, 0);
    dslashSCuda(out, gauge, tmp, 0, 0); 
  } else {
    dslashSCuda(tmp, gauge, in, 0, 0);
    dslashSCuda(out, gauge, tmp, 1, 0); 
  }

  // out = in - kappa2*out, dot = (d, out)
  return xpaycDotzyCuda((float2*)in.spinor, kappa2, (float2*)out.spinor, (float2*)d.spinor, Nh*spinorSiteSize/2);
}

// Apply the even-odd preconditioned Dirac operator
QudaSumComplex MatPCDagcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
				  ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashSCuda(tmp, gauge, in, 1, 1);
    dslashSCuda(out, gauge, tmp, 0, 1);
  } else {
    dslashSCuda(tmp, gauge, in, 0, 1);
    dslashSCuda(out, gauge, tmp, 1, 1);
  }

  // out = in - kappa2*out, dot = (d, out)
  return xpaycDotzyCuda((float2*)in.spinor, kappa2, (float2*)out.spinor, (float2*)d.spinor, Nh*spinorSiteSize/2);
  }*/
