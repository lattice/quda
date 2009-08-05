#include <stdlib.h>
#include <stdio.h>

#include <dslash_quda.h>

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

// Double precision input spinor field
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

// Double precision clover term
texture<int4, 1> cloverTexDouble;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

// Half precision clover term
texture<short4, 1, cudaReadModeNormalizedFloat> cloverTexHalf;
texture<float, 1, cudaReadModeElementType> cloverTexNorm;

QudaGaugeParam *gauge_param;
QudaInvertParam *invert_param;

__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;

__constant__ int X1h;
__constant__ int X2X1h;
__constant__ int X3X2X1h;
__constant__ int X4X3X2X1h;
__constant__ int X2X1;
__constant__ int X3X2X1;
__constant__ int X4X3X2X1;

__constant__ float X1h_inv;
__constant__ float X2X1h_inv;
__constant__ float X3X2X1h_inv;

__constant__ float X1_inv;
__constant__ float X2_inv;
__constant__ float X3_inv;
__constant__ float X2X1_inv;
__constant__ float X3X2X1_inv;

__constant__ int Vh;

__constant__ int gauge_fixed;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;

#include <dslash_def.h>

void initDslashCuda(FullGauge gauge) {
  int Vh = gauge.volume;
  cudaMemcpyToSymbol("Vh", &Vh, sizeof(int));  

  if (Vh%BLOCK_DIM !=0) {
    printf("Sorry, volume is not a multiple of number of threads %d\n", BLOCK_DIM);
    exit(-1);
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

  int X4X3X2X1 = X4*X3*X2*X1;
  cudaMemcpyToSymbol("X4X3X2X1", &X4X3X2X1, sizeof(int));  

  int X1h = X1/2;
  cudaMemcpyToSymbol("X1h", &X1h, sizeof(int));  

  int X2X1h = X2*X1h;
  cudaMemcpyToSymbol("X2X1h", &X2X1h, sizeof(int));  

  int X3X2X1h = X3*X2*X1h;
  cudaMemcpyToSymbol("X3X2X1h", &X3X2X1h, sizeof(int));  

  int X4X3X2X1h = X4*X3*X2*X1h;
  cudaMemcpyToSymbol("X4X3X2X1h", &X4X3X2X1h, sizeof(int));  

  float X1h_inv = 1.0 / X1h;
  cudaMemcpyToSymbol("X1h_inv", &X1h_inv, sizeof(float));  

  float X2X1h_inv = 1.0 / X2X1h;
  cudaMemcpyToSymbol("X2X1h_inv", &X2X1h_inv, sizeof(float));  

  float X3X2X1h_inv = 1.0 / X3X2X1h;
  cudaMemcpyToSymbol("X3X2X1h_inv", &X3X2X1h_inv, sizeof(float));  

  float X1_inv = 1.0 / X1;
  cudaMemcpyToSymbol("X1_inv", &X1_inv, sizeof(float));  

  float X2_inv = 1.0 / X2;
  cudaMemcpyToSymbol("X2_inv", &X2_inv, sizeof(float));  

  float X3_inv = 1.0 / X3;
  cudaMemcpyToSymbol("X3_inv", &X3_inv, sizeof(float));  

  float X2X1_inv = 1.0 / X2X1;
  cudaMemcpyToSymbol("X2X1_inv", &X2X1_inv, sizeof(float));  

  float X3X2X1_inv = 1.0 / X3X2X1;
  cudaMemcpyToSymbol("X3X2X1_inv", &X3X2X1_inv, sizeof(float));  

  int gf = (gauge_param->gauge_fix == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  cudaMemcpyToSymbol("anisotropy", &(gauge_param->anisotropy), sizeof(double));

  double t_bc = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  float anisotropy_f = gauge_param->anisotropy;
  cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));

  float t_bc_f = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

  float h_pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));
}

void bindGaugeTex(FullGauge gauge, int oddBit) {
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


// ----------------------------------------------------------------------

void checkSpinor(ParitySpinor out, ParitySpinor in) {
  if (in.precision != out.precision) {
    printf("Error in dslash quda: input and out spinor precision's don't match\n");
    exit(-1);
  }

#if (__CUDA_ARCH__ != 130)
  if (in.precision == QUDA_DOUBLE_PRECISION) {
    printf("Double precision not supported on this GPU\n");
    exit(-1);    
  }
#endif
}

void checkGaugeSpinor(ParitySpinor spinor, FullGauge gauge) {
  if (spinor.volume != gauge.volume) {
    printf("Error, spinor volume %d doesn't match gauge volume %d\n", spinor.volume, gauge.volume);
    exit(-1);
  }

#if (__CUDA_ARCH__ != 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    printf("Double precision not supported on this GPU\n");
    exit(-1);    
  }
#endif
}

void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  initDslashCuda(gauge);
  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    dslashDCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    dslashSCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    dslashHCuda(out, gauge, in, parity, dagger);
  }
}


void dslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

#if (__CUDA_ARCH__ == 130)
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
#endif
  
}


void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
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
#endif
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

  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
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
#endif
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

void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		    ParitySpinor x, double a) {
  initDslashCuda(gauge);
  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    dslashXpayDCuda(out, gauge, in, parity, dagger, x, a);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    dslashXpaySCuda(out, gauge, in, parity, dagger, x, a);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    dslashXpayHCuda(out, gauge, in, parity, dagger, x, a);
  }
}


void dslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		     int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
#if (__CUDA_ARCH__ == 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      } else {
	dslashDD12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDD8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashDD8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      } else {
	dslashSD12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSD8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashSD8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
    } else {
	dslashHD12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHD8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashHD8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  }
#endif

}

void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
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
#endif
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
		    int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(res.volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHH8XpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  }

}

int dslashCudaSharedBytes() {
  return SHARED_BYTES_SINGLE;
}

// Apply the even-odd preconditioned Dirac operator
void MatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, double kappa, 
	       ParitySpinor tmp, MatPCType matpc_type, int dagger) {

  double kappa2 = -kappa*kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, dagger);
    dslashXpayCuda(out, gauge, tmp, 0, dagger, in, kappa2); 
  } else {
    dslashCuda(tmp, gauge, in, 0, dagger);
    dslashXpayCuda(out, gauge, tmp, 1, dagger, in, kappa2); 
  }
}

void MatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, MatPCType matpc_type) {
  MatPCCuda(out, gauge, in, kappa, tmp, matpc_type, 0);
  MatPCCuda(out, gauge, out, kappa, tmp, matpc_type, 1);
}

// Apply the full operator
void MatCuda(FullSpinor out, FullGauge gauge, FullSpinor in, double kappa, int dagger) {
  dslashXpayCuda(out.odd, gauge, in.even, 1, dagger, in.odd, -kappa);
  dslashXpayCuda(out.even, gauge, in.odd, 0, dagger, in.even, -kappa);
}
