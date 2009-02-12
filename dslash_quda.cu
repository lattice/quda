#include <stdlib.h>
#include <stdio.h>

#include <dslash_quda.h>

// ----------------------------------------------------------------------
// Cuda code

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<float, 1, cudaReadModeElementType> spinorTexNorm;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTex;

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

#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
//#define WRITE_SPINOR WRITE_FLOAT1_SMEM
//#define WRITE_SPINOR WRITE_SPINOR_FLOAT1_STAGGERED

#include <dslash_def.h>

__global__ void spinorHalfPack(float *c, short4* half) {

  int sid = BLOCK_DIM*blockIdx.x + threadIdx.x;

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
  
  half[sid+0*Nh] = make_short4((short)F0.x, (short)F0.y, (short)F0.z, (short)F0.w);
  half[sid+1*Nh] = make_short4((short)F1.x, (short)F1.y, (short)F1.z, (short)F1.w);
  half[sid+2*Nh] = make_short4((short)F2.x, (short)F2.y, (short)F2.z, (short)F2.w);
  half[sid+3*Nh] = make_short4((short)F3.x, (short)F3.y, (short)F3.z, (short)F3.w);
  half[sid+4*Nh] = make_short4((short)F4.x, (short)F4.y, (short)F4.z, (short)F4.w);
  half[sid+5*Nh] = make_short4((short)F5.x, (short)F5.y, (short)F5.z, (short)F5.w);
}

void setCudaGaugeParam() {
  cudaMemcpyToSymbol("anisotropy", &(gauge_param->anisotropy), sizeof(float));
  float t_bc = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(float));
  int gf = (gauge_param->gauge_fix == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));
}


// ----------------------------------------------------------------------

void dslashCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		int oddBit, int daggerBit) {

  int packed_gauge_bytes = (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) ?
    PACKED12_GAUGE_BYTES : PACKED8_GAUGE_BYTES;
  if (gauge_param->cuda_prec == QUDA_HALF_PRECISION) packed_gauge_bytes/=2;

  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexSingle, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexSingle, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexSingle, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexSingle, gauge.odd, packed_gauge_bytes); 
    }
  } else {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexHalf, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexHalf, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexHalf, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexHalf, gauge.odd, packed_gauge_bytes); 
    }
  }

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  cudaBindTexture(0, spinorTexSingle, spinor, SPINOR_BYTES); 
  if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    cudaBindTexture(0, spinorTexHalf, spinorHalf, SPINOR_BYTES/2); 
    cudaBindTexture(0, spinorTexNorm, spinorNorm, SPINOR_BYTES/24); 
    spinorHalfPack <<<gridDim, blockDim, SHARED_BYTES>>>(spinorNorm, spinorHalf);
  }
  
  if (invert_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
      if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSS12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashSS12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSS8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashSS8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      }
    } else {
      if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHS12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashHS12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHS8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashHS8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      }
    }
  } else {
    if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
      if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSH12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashSH12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSH8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashSH8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      }
    } else {
      if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHH12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashHH12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHH8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	} else {
	  dslashHH8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	    ((float4 *)res, oddBit);
	}
      }
    }
  }
  
}

void dslashXpayCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, float a) {
  int packed_gauge_bytes = (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) ?
    PACKED12_GAUGE_BYTES : PACKED8_GAUGE_BYTES;
  if (gauge_param->cuda_prec == QUDA_HALF_PRECISION) packed_gauge_bytes/=2;

  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexSingle, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexSingle, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexSingle, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexSingle, gauge.odd, packed_gauge_bytes); 
    }
  } else {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexHalf, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexHalf, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexHalf, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexHalf, gauge.odd, packed_gauge_bytes); 
    }
  }

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  cudaBindTexture(0, spinorTexSingle, spinor, SPINOR_BYTES); 
  if (invert_param->cuda_prec == QUDA_HALF_PRECISION) {
    cudaBindTexture(0, spinorTexHalf, spinorHalf, SPINOR_BYTES/2); 
    cudaBindTexture(0, spinorTexNorm, spinorNorm, SPINOR_BYTES/24); 
    spinorHalfPack <<<gridDim, blockDim, SHARED_BYTES>>>(spinorNorm, spinorHalf);
  }  
  cudaBindTexture(0, accumTex, x, SPINOR_BYTES); 
  
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      } else {
	dslashSS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
    } else if (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
      else {
	dslashSS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
    }
  } else {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
    } else {
	dslashHS12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
    }
    } else if (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHS8XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
      else {
	dslashHS8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
    }
  }

}

int dslashCudaSharedBytes() {
  return SHARED_BYTES;
}

// Apply the even-odd preconditioned Dirac operator
void MatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
	       ParitySpinor tmp, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, 0);
    dslashXpayCuda(out, gauge, tmp, 0, 0, in, kappa2); 
  } else {
    dslashCuda(tmp, gauge, in, 0, 0);
    dslashXpayCuda(out, gauge, tmp, 1, 0, in, kappa2); 
  }
}

// Apply the even-odd preconditioned Dirac operator
void MatPCDagCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
		  ParitySpinor tmp, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, 1);
    dslashXpayCuda(out, gauge, tmp, 0, 1, in, kappa2);
  } else {
    dslashCuda(tmp, gauge, in, 0, 1);
    dslashXpayCuda(out, gauge, tmp, 1, 1, in, kappa2);
  }
}

// Apply the even-odd preconditioned Dirac operator
QudaSumComplex MatPCcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
				ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, 0);
    dslashCuda(out, gauge, tmp, 0, 0); 
  } else {
    dslashCuda(tmp, gauge, in, 0, 0);
    dslashCuda(out, gauge, tmp, 1, 0); 
  }

  // out = in - kappa2*out, dot = (d, out)
  return xpaycDotzyCuda((float2*)in, kappa2, (float2*)out, (float2*)d, Nh*spinorSiteSize/2);
}

// Apply the even-odd preconditioned Dirac operator
QudaSumComplex MatPCDagcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
				  ParitySpinor tmp, ParitySpinor d, MatPCType matpc_type) {
  float kappa2 = -kappa*kappa;

  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, 1);
    dslashCuda(out, gauge, tmp, 0, 1);
  } else {
    dslashCuda(tmp, gauge, in, 0, 1);
    dslashCuda(out, gauge, tmp, 1, 1);
  }

  // out = in - kappa2*out, dot = (d, out)
  return xpaycDotzyCuda((float2*)in, kappa2, (float2*)out, (float2*)d, Nh*spinorSiteSize/2);
}

void MatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       float kappa, ParitySpinor tmp, MatPCType matpc_type) {
  MatPCCuda(out, gauge, in, kappa, tmp, matpc_type);
  MatPCDagCuda(out, gauge, out, kappa, tmp, matpc_type);
}
