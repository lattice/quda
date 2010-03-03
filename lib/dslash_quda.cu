#include <stdlib.h>
#include <stdio.h>

#include <quda_internal.h>
#include <dslash_quda.h>

#include <dslash_textures.h>
#include <dslash_constants.h>

unsigned long long dslash_quda_flops;
unsigned long long dslash_quda_bytes;

#define BLOCK_DIM 64

//#include <dslash_def.h> // Dslash kernel definitions

// kludge to avoid '#include nested too deeply' error
//#define DD_CPREC 3
#define DD_DAG 0
#include <dslash_def.h>
#undef DD_DAG
#define DD_DAG 1
#include <dslash_def.h>
#undef DD_DAG
//#undef DD_CPREC

#include <clover_def.h> // kernels for applying the clover term alone

int dslashCudaSharedBytes(QudaPrecision precision) {
  return BLOCK_DIM*SHARED_FLOATS_PER_THREAD*precision;
}

int initDslash = 0;

void initDslashConstants(FullGauge gauge, int sp_stride, int cl_stride) {
  int Vh = gauge.volume;
  cudaMemcpyToSymbol("Vh", &Vh, sizeof(int));  

  cudaMemcpyToSymbol("sp_stride", &sp_stride, sizeof(int));  

  int ga_stride = gauge.stride;
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  cudaMemcpyToSymbol("cl_stride", &cl_stride, sizeof(int));  

  if (Vh%BLOCK_DIM != 0) {
    errorQuda("Error, Volume not a multiple of the thread block size");
  }

  int X1 = 2*gauge.X[0];
  cudaMemcpyToSymbol("X1", &X1, sizeof(int));  

  int X2 = gauge.X[1];
  cudaMemcpyToSymbol("X2", &X2, sizeof(int));  

  int X3 = gauge.X[2];
  cudaMemcpyToSymbol("X3", &X3, sizeof(int));  

  int X4 = gauge.X[3];
  cudaMemcpyToSymbol("X4", &X4, sizeof(int));  

  int X2X1 = X2*X1;
  cudaMemcpyToSymbol("X2X1", &X2X1, sizeof(int));  

  int X3X2X1 = X3*X2*X1;
  cudaMemcpyToSymbol("X3X2X1", &X3X2X1, sizeof(int));  

  int X1h = X1/2;
  cudaMemcpyToSymbol("X1h", &X1h, sizeof(int));  

  int X1m1 = X1 - 1;
  cudaMemcpyToSymbol("X1m1", &X1m1, sizeof(int));  

  int X2m1 = X2 - 1;
  cudaMemcpyToSymbol("X2m1", &X2m1, sizeof(int));  

  int X3m1 = X3 - 1;
  cudaMemcpyToSymbol("X3m1", &X3m1, sizeof(int));  

  int X4m1 = X4 - 1;
  cudaMemcpyToSymbol("X4m1", &X4m1, sizeof(int));  
  
  int X2X1mX1 = X2X1 - X1;
  cudaMemcpyToSymbol("X2X1mX1", &X2X1mX1, sizeof(int));  

  int X3X2X1mX2X1 = X3X2X1 - X2X1;
  cudaMemcpyToSymbol("X3X2X1mX2X1", &X3X2X1mX2X1, sizeof(int));  

  int X4X3X2X1mX3X2X1 = (X4-1)*X3X2X1;
  cudaMemcpyToSymbol("X4X3X2X1mX3X2X1", &X4X3X2X1mX3X2X1, sizeof(int));  

  int X4X3X2X1hmX3X2X1h = (X4-1)*X3*X2*X1h;
  cudaMemcpyToSymbol("X4X3X2X1hmX3X2X1h", &X4X3X2X1hmX3X2X1h, sizeof(int));  

  int gf = (gauge.gauge_fixed == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  cudaMemcpyToSymbol("anisotropy", &(gauge.anisotropy), sizeof(double));

  double t_bc = (gauge.t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  float anisotropy_f = gauge.anisotropy;
  cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));

  float t_bc_f = (gauge.t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

  float2 An2 = make_float2(gauge.anisotropy, 1.0 / (gauge.anisotropy*gauge.anisotropy));
  cudaMemcpyToSymbol("An2", &(An2), sizeof(float2));
  float2 TB2 = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol("TB2", &(TB2), sizeof(float2));
  float2 No2 = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol("No2", &(No2), sizeof(float2));

  float h_pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));

  checkCudaError();

  initDslash = 1;
}

static void bindGaugeTex(FullGauge gauge, int oddBit) {
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexDouble, gauge.odd, gauge.bytes); 
      cudaBindTexture(0, gauge1TexDouble, gauge.even, gauge.bytes);
    } else {
      cudaBindTexture(0, gauge0TexDouble, gauge.even, gauge.bytes);
      cudaBindTexture(0, gauge1TexDouble, gauge.odd, gauge.bytes); 
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexSingle, gauge.odd, gauge.bytes); 
      cudaBindTexture(0, gauge1TexSingle, gauge.even, gauge.bytes);
    } else {
      cudaBindTexture(0, gauge0TexSingle, gauge.even, gauge.bytes);
      cudaBindTexture(0, gauge1TexSingle, gauge.odd, gauge.bytes); 
    }
  } else {
    if (oddBit) {
      cudaBindTexture(0, gauge0TexHalf, gauge.odd, gauge.bytes); 
      cudaBindTexture(0, gauge1TexHalf, gauge.even, gauge.bytes);
    } else {
      cudaBindTexture(0, gauge0TexHalf, gauge.even, gauge.bytes);
      cudaBindTexture(0, gauge1TexHalf, gauge.odd, gauge.bytes); 
    }
  }
}

void dslashDCuda(double2 *out, FullGauge gauge, double2 *in, int oddBit, int daggerBit, 
		 int volume, int length) {
  
#if (__CUDA_ARCH__ == 130)

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  
}


void dslashSCuda(float4 *out, FullGauge gauge, float4 *in, int oddBit, int daggerBit, 
		 int volume, int length) {
  
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      } else {
	dslashHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
      }
    }
  }
  
}


void dslashHCuda(short4 *out, float *outNorm, FullGauge gauge, short4 *in, float *inNorm,
		 int oddBit, int daggerBit, int volume, int length) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, in, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, inNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      } else {
	dslashHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
      }
    }
  }
  
}

void dslashXpayDCuda(double2 *out, FullGauge gauge, double2 *in, 
		     int oddBit, int daggerBit, double2 *x, double a,
		     int volume, int length) {

#if (__CUDA_ARCH__ == 130)

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, in, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      } else {
	dslashDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      } else {
	dslashSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
    } else {
	dslashHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif

}

void dslashXpaySCuda(float4 *out, FullGauge gauge, float4 *in, 
		     int oddBit, int daggerBit, float4 *x, double a, int volume, int length) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, in, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      } else {
	dslashDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      } else {
	dslashSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
    } else {
	dslashHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
      else {
	dslashHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
      }
    }
  }

}

void dslashXpayHCuda(short4 *out, float *outNorm, FullGauge gauge, short4 *in, float *inNorm, 
		     int oddBit, int daggerBit, short4 *x, float *xNorm, double a,
		     int volume, int length) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, in, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, inNorm, spinor_bytes/12); 
  cudaBindTexture(0, accumTexHalf, x, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, xNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, oddBit, a);
      } else {
	dslashDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, oddBit, a);
      } else {
	dslashDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      } else {
	dslashSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      } else {
	dslashSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      } else {
	dslashHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      } else {
	dslashHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, oddBit, a);
      }
    }
  }

}

static void bindCloverTex(ParityClover clover) {
  if (clover.precision == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, cloverTexDouble, clover.clover, clover.bytes); 
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, cloverTexSingle, clover.clover, clover.bytes); 
  } else {
    cudaBindTexture(0, cloverTexHalf, clover.clover, clover.bytes); 
    cudaBindTexture(0, cloverTexNorm, clover.cloverNorm, clover.bytes/18);
  }
}

void cloverDslashDCuda(double2 *out, FullGauge gauge, FullClover cloverInv, 
		       double2 *in, int oddBit, int daggerBit, int volume, int length)
{

#if (__CUDA_ARCH__ == 130)

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif

}

void cloverDslashSCuda(float4 *out, FullGauge gauge, FullClover cloverInv,
		       float4 *in, int oddBit, int daggerBit, int volume, int length)
{  
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashDSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashSSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	} else {
	  dslashHSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
	}
      }
    }
  }
}

void cloverDslashHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover cloverInv,
		       short4 *in, float *inNorm, int oddBit, int daggerBit, int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, in, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, inNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHD12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHD8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHS12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHS8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashDHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashSHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHH12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHH8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	} else {
	  dslashHHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
	}
      }
    }
  }
}

void cloverDslashXpayDCuda(double2 *out, FullGauge gauge, FullClover cloverInv, double2 *in, 
			   int oddBit, int daggerBit, double2 *x, double a, int volume, int length)
{
#if (__CUDA_ARCH__ == 130)

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, in, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
}

void cloverDslashXpaySCuda(float4 *out, FullGauge gauge, FullClover cloverInv, float4 *in, 
			   int oddBit, int daggerBit, float4 *x, double a, int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, in, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashDSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashSSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	} else {
	  dslashHSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit, a);
	}
      }
    }
  }
}

void cloverDslashXpayHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover cloverInv, 
			   short4 *in, float *inNorm, int oddBit, int daggerBit, 
			   short4 *x, float *xNorm, double a, int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, in, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, inNorm, spinor_bytes/12); 
  cudaBindTexture(0, accumTexHalf, x, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, xNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    (out, outNorm, oddBit, a);
	} else {
	  dslashDHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashSHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	} else {
	  dslashHHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    (out, outNorm, oddBit, a);
	}
      }
    }
  } 
}

void cloverDCuda(double2 *out, FullGauge gauge, FullClover clover, double2 *in, int oddBit,
		 int volume, int length)
{
#if (__CUDA_ARCH__ == 130)

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    cloverDDKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverDSKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
  } else {
    cloverDHKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
}

void cloverSCuda(float4 *out, FullGauge gauge, FullClover clover, float4 *in, int oddBit,
		 int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, in, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverSDKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverSSKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
  } else {
    cloverSHKernel <<<gridDim, blockDim, shared_bytes>>> (out, oddBit);
  }
}

void cloverHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover clover,
		 short4 *in, float *inNorm, int oddBit, int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  QudaPrecision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, in, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, inNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverHDKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverHSKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
  } else {
    cloverHHKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, oddBit);
  }
}
