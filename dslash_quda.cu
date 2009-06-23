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

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

QudaGaugeParam *gauge_param;
QudaInvertParam *invert_param;

__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;
__constant__ int X1h;

__constant__ int gauge_fixed;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;

#include <dslash_def.h>

void setCudaGaugeParam() {
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

void checkSpinor(ParitySpinor out, ParitySpinor in) {
  if (in.precision != out.precision) {
    printf("Error in dslash quda: input and out spinor precision's don't match\n");
    exit(-1);
  }
}



void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  checkSpinor(in, out);

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

void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		    ParitySpinor x, double a) {
  checkSpinor(in, out);

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

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh*spinorSiteSize*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
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

}

void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, double a) {

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
		    int oddBit, int daggerBit, ParitySpinor x, double a) {

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
	       ParitySpinor tmp, MatPCType matpc_type) {

  checkSpinor(in, out);
  checkSpinor(in, tmp);

  double kappa2 = -kappa*kappa;

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashDCuda(tmp, gauge, in, 1, 0);
      dslashXpayDCuda(out, gauge, tmp, 0, 0, in, kappa2); 
    } else {
      dslashDCuda(tmp, gauge, in, 0, 0);
      dslashXpayDCuda(out, gauge, tmp, 1, 0, in, kappa2); 
    }
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashSCuda(tmp, gauge, in, 1, 0);
      dslashXpaySCuda(out, gauge, tmp, 0, 0, in, kappa2); 
    } else {
      dslashSCuda(tmp, gauge, in, 0, 0);
      dslashXpaySCuda(out, gauge, tmp, 1, 0, in, kappa2); 
    }
  } else if (in.precision == QUDA_HALF_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashHCuda(tmp, gauge, in, 1, 0);
      dslashXpayHCuda(out, gauge, tmp, 0, 0, in, kappa2); 
    } else {
      dslashHCuda(tmp, gauge, in, 0, 0);
      dslashXpayHCuda(out, gauge, tmp, 1, 0, in, kappa2); 
    }
  }

}

// Apply the even-odd preconditioned Dirac operator
void MatPCDagCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, double kappa, 
		  ParitySpinor tmp, MatPCType matpc_type) {

  checkSpinor(in, out);
  checkSpinor(in, tmp);

  double kappa2 = -kappa*kappa;

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashDCuda(tmp, gauge, in, 1, 1);
      dslashXpayDCuda(out, gauge, tmp, 0, 1, in, kappa2);
    } else {
      dslashDCuda(tmp, gauge, in, 0, 1);
      dslashXpayDCuda(out, gauge, tmp, 1, 1, in, kappa2);
    }
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashSCuda(tmp, gauge, in, 1, 1);
      dslashXpaySCuda(out, gauge, tmp, 0, 1, in, kappa2);
    } else {
      dslashSCuda(tmp, gauge, in, 0, 1);
      dslashXpaySCuda(out, gauge, tmp, 1, 1, in, kappa2);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashHCuda(tmp, gauge, in, 1, 1);
      dslashXpayHCuda(out, gauge, tmp, 0, 1, in, kappa2);
    } else {
      dslashHCuda(tmp, gauge, in, 0, 1);
      dslashXpayHCuda(out, gauge, tmp, 1, 1, in, kappa2);
    }
  }

}

void MatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, MatPCType matpc_type) {
  MatPCCuda(out, gauge, in, kappa, tmp, matpc_type);
  MatPCDagCuda(out, gauge, out, kappa, tmp, matpc_type);
}

// Apply the full operator
void MatCuda(FullSpinor out, FullGauge gauge, FullSpinor in, double kappa) {
  checkSpinor(in.even, out.even);

  if (in.even.precision == QUDA_DOUBLE_PRECISION) {
    dslashXpayDCuda(out.odd, gauge, in.even, 1, 0, in.odd, -kappa);
    dslashXpayDCuda(out.even, gauge, in.odd, 0, 0, in.even, -kappa);
  } else if (in.even.precision == QUDA_SINGLE_PRECISION) {
    dslashXpaySCuda(out.odd, gauge, in.even, 1, 0, in.odd, -kappa);
    dslashXpaySCuda(out.even, gauge, in.odd, 0, 0, in.even, -kappa);
  } else if (in.even.precision == QUDA_HALF_PRECISION) {
    dslashXpayHCuda(out.odd, gauge, in.even, 1, 0, in.odd, -kappa);
    dslashXpayHCuda(out.even, gauge, in.odd, 0, 0, in.even, -kappa);
  }

}

// Apply the full operator dagger
void MatDaggerCuda(FullSpinor out, FullGauge gauge, FullSpinor in, double kappa) {
  checkSpinor(in.even, out.even);

  if (in.even.precision == QUDA_SINGLE_PRECISION) {
    dslashXpayDCuda(out.odd, gauge, in.even, 1, 1, in.odd, -kappa);
    dslashXpayDCuda(out.even, gauge, in.odd, 0, 1, in.even, -kappa);
  } else if (in.even.precision == QUDA_SINGLE_PRECISION) {
    dslashXpaySCuda(out.odd, gauge, in.even, 1, 1, in.odd, -kappa);
    dslashXpaySCuda(out.even, gauge, in.odd, 0, 1, in.even, -kappa);
  } else if (in.even.precision == QUDA_HALF_PRECISION) {
    dslashXpayHCuda(out.odd, gauge, in.even, 1, 1, in.odd, -kappa);
    dslashXpayHCuda(out.even, gauge, in.odd, 0, 1, in.even, -kappa);
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
