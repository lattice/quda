#include <stdlib.h>
#include <stdio.h>

#include <quda_internal.h>
#include <dslash_quda.h>
#include <spinor_quda.h> // not needed once call to allocateParitySpinor() is removed
#include <face_quda.h>
#include <dslash_textures.h>
#include <dslash_constants.h>

unsigned long long dslash_quda_flops;
unsigned long long dslash_quda_bytes;

#define BLOCK_DIM 64

//#include <dslash_def.h> // Dslash kernel definitions

// kludge to avoid '#include nested too deeply' error

#define DD_DAG 0
#include <dslash_def.h>
#undef DD_DAG
#define DD_DAG 1
#include <dslash_def.h>
#undef DD_DAG

#ifdef BUILD_3D_DSLASH
#define DD3D_DAG 0
#include <dslash_3d_def.h>
#undef DD3D_DAG
#define DD3D_DAG 1
#include <dslash_3d_def.h>
#undef DD3D_DAG
#endif

bool qudaPt0=true;   // Single core versions always to Boundary
bool qudaPtNm1=true;

#include <clover_def.h> // kernels for applying the clover term alone

int dslashCudaSharedBytes(Precision precision) {
  return BLOCK_DIM*SHARED_FLOATS_PER_THREAD*precision;
}

#include <dslash_common.h>

static int initDslash = 0;
static int gridVolume;

void initDslashConstants(FullGauge gauge, int sp_body_stride, int cl_stride) {
  int Vh = gauge.volume;

  cudaMemcpyToSymbol("Vh", &Vh, sizeof(int));  

  int Vs = gauge.X[0]*gauge.X[1]*gauge.X[2];
  cudaMemcpyToSymbol("Vs", &Vs, sizeof(int));

  cudaMemcpyToSymbol("sp_body_stride", &sp_body_stride, sizeof(int));  

  int ga_stride = gauge.stride;
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  cudaMemcpyToSymbol("cl_stride", &cl_stride, sizeof(int));  

  /*  if (Vh%BLOCK_DIM != 0) {
    errorQuda("Volume not a multiple of the thread block size");
    }*/

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

  cudaMemcpyToSymbol("Pt0", &(qudaPt0), sizeof(bool)); 
  cudaMemcpyToSymbol("PtNm1", &(qudaPtNm1), sizeof(bool)); 

  checkCudaError();

  initDslash = 1;
}

void initTLocation(int toffset, int tmul, int threads) {

  short2 tLocate = make_short2((short)toffset, (short)tmul);
  cudaMemcpyToSymbol("tLocate", &(tLocate), sizeof(short2));
  cudaMemcpyToSymbol("threads", &(threads), sizeof(threads));

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

// ----------------------------------------------------------------------
// plain Wilson Dslash:

void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  if (!initDslash) {
    initDslashConstants(gauge, in.stride, 0);
    //faceBufferPrecise=allocateFaceBuffer(gauge.X[0]*gauge.X[1]*gauge.X[2], gauge.volume, in.stride, in.precision); 
  }    

  // This gathers from source spinors and starts comms
  exchangeFacesStart(gauge.faces, in, dagger);

  // This waits for comms to finish, and sprays into the 
  // pads of the SOURCE spinor
  exchangeFacesWait(gauge.faces, in, dagger);

  // printf("Freeing Face Buffer");
  //freeFaceBuffer(faceBufferPrecise);

  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  int Vs = gauge.X[0] * gauge.X[1] * gauge.X[2];
  
  // do body
  {
    int tOffset = 1;
    int tMul = 1;
    int threads = in.volume - 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
        
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      dslashDCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      dslashSCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      dslashHCuda(out, gauge, in, parity, dagger);
    }
    checkCudaError();
    }
    
  // do faces
  {
    int tOffset = 0;
    int tMul = gauge.X[3] - 1;
    int threads = 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      dslashDCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      dslashSCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      dslashHCuda(out, gauge, in, parity, dagger);
    }
    checkCudaError();
  }

  // do all
  /*{
    int tOffset = 0;
    int tMul = 1;
    int threads = in.volume;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
        
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      dslashDCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      dslashSCuda(out, gauge, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      dslashHCuda(out, gauge, in, parity, dagger);
    }
    checkCudaError();
    }*/

  dslash_quda_flops += 1320*in.volume;
}

#ifdef BUILD_3D_DSLASH
void dslash3DCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    dslash3DDCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    dslash3DSCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    dslash3DHCuda(out, gauge, in, parity, dagger);
  }
  checkCudaError();

  dslash_quda_flops += 1320*in.volume;
}
#endif // BUILD_3D_DSLASH

void dslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslashHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  
}


void dslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  //
  

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = spinor.bytes+spinor.tface_bytes;
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslashHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
  }
  
}


void dslashHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {

  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  int spinor_norm_bytes = spinor.bytes/12 + spinor.tface_bytes/6;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_norm_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  }
  
}

void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		    ParitySpinor x, double a) {
//  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  if (!initDslash) {
    initDslashConstants(gauge, in.stride, 0);
    // faceBufferPrecise=allocateFaceBuffer(gauge.X[0]*gauge.X[1]*gauge.X[2], gauge.volume, in.stride, in.precision); 
  }    

 
  // This gathers from source spinors and starts comms
  exchangeFacesStart(gauge.faces, in, dagger);

  // This waits for comms to finish, and sprays into the 
  // pads of the SOURCE spinor
  exchangeFacesWait(gauge.faces, in, dagger);

  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  int Vs = gauge.X[0] * gauge.X[1] * gauge.X[2];

  // do body
  {
    int tOffset = 1;
    int tMul = 1;
    int threads = in.volume - 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      dslashXpayDCuda(out, gauge, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      dslashXpaySCuda(out, gauge, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      dslashXpayHCuda(out, gauge, in, parity, dagger, x, a);
    }
    checkCudaError();
  }

  // do faces
  {
    int tOffset = 0;
    int tMul = gauge.X[3] - 1;
    int threads = 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      dslashXpayDCuda(out, gauge, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      dslashXpaySCuda(out, gauge, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      dslashXpayHCuda(out, gauge, in, parity, dagger, x, a);
    }
    checkCudaError();
  }

  dslash_quda_flops += (1320+48)*in.volume;
}


void dslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		     int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = spinor.bytes+spinor.tface_bytes;
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      } else {
	dslashDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      } else {
	dslashSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
    } else {
	dslashHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
      else {
	dslashHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif

}

void dslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		     int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);


  int spinor_bytes = spinor.bytes+spinor.tface_bytes;
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      } else {
	dslashSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
    } else {
	dslashHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
      else {
	dslashHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
      }
    }
  }

}

void dslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		     int oddBit, int daggerBit, ParitySpinor x, double a) {

  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  int spinor_norm_bytes = spinor.bytes/12 + spinor.tface_bytes/6;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_norm_bytes); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, spinor_norm_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  }

}

// Apply the even-odd preconditioned Dirac operator
void MatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, double kappa, 
	       ParitySpinor tmp, MatPCType matpc_type, int dagger) {

  double kappa2 = -kappa*kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, dagger);
    dslashXpayCuda(out, gauge, tmp, 0, dagger, in, kappa2); 
  } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
    dslashCuda(tmp, gauge, in, 0, dagger);
    dslashXpayCuda(out, gauge, tmp, 1, dagger, in, kappa2); 
  } else {
    errorQuda("matpc_type not valid for plain Wilson");
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

// ----------------------------------------------------------------------
// clover-improved Wilson Dslash
//
// apply hopping term, then clover: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (A_ee^-1 D^dagger_eo) or (A_oo^-1 D^dagger_oe)

void cloverDslashCuda(ParitySpinor out, FullGauge gauge, FullClover cloverInv,
		      ParitySpinor in, int parity, int dagger)
{
//  if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);

  if (!initDslash) {
    initDslashConstants(gauge, in.stride, cloverInv.even.stride);
    // faceBufferPrecise=allocateFaceBuffer(gauge.X[0]*gauge.X[1]*gauge.X[2], gauge.volume, in.stride, in.precision); 
  }    

 
 // This gathers from source spinors and starts comms
  exchangeFacesStart(gauge.faces, in, dagger);

  // This waits for comms to finish, and sprays into the 
  // pads of the SOURCE spinor
  exchangeFacesWait(gauge.faces, in, dagger);

  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);
  checkCloverSpinor(in, cloverInv);

  int Vs = gauge.X[0] * gauge.X[1] * gauge.X[2];

  // do body
  {
    int tOffset = 1;
    int tMul = 1;
    int threads = in.volume - 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      cloverDslashDCuda(out, gauge, cloverInv, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      cloverDslashSCuda(out, gauge, cloverInv, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      cloverDslashHCuda(out, gauge, cloverInv, in, parity, dagger);
    }
    checkCudaError();
  }

  // do faces
  {
    int tOffset = 0;
    int tMul = gauge.X[3] - 1;
    int threads = 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      cloverDslashDCuda(out, gauge, cloverInv, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      cloverDslashSCuda(out, gauge, cloverInv, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      cloverDslashHCuda(out, gauge, cloverInv, in, parity, dagger);
    }
    checkCudaError();
  }

  dslash_quda_flops += (1320+504)*in.volume;
}

void cloverDslashDCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv,
		       ParitySpinor spinor, int oddBit, int daggerBit)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    }
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashDDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashSDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	} else {
	  dslashHDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
	}
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif

}

void cloverDslashSCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv,
		       ParitySpinor spinor, int oddBit, int daggerBit)
{  
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes+spinor.tface_bytes;
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
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
	  dslashDSS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashDSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashSSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	} else {
	  dslashHSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
	}
      }
    }
  }
}

void cloverDslashHCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv,
		       ParitySpinor spinor, int oddBit, int daggerBit)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  int spinor_norm_bytes = spinor.bytes/12 + spinor.tface_bytes/6;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_norm_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
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
	  dslashDHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashDHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashDHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashSHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashSHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      } else {
	if (!daggerBit) {
	  dslashHHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	} else {
	  dslashHHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
	}
      }
    }
  }
}

void cloverDslashXpayCuda(ParitySpinor out, FullGauge gauge, FullClover cloverInv, ParitySpinor in,
			  int parity, int dagger, ParitySpinor x, double a)
{
  //if (!initDslash) initDslashConstants(gauge, in.stride, cloverInv.even.stride);
 
  if (!initDslash) {
    initDslashConstants(gauge, in.stride, cloverInv.even.stride);
    // faceBufferPrecise=allocateFaceBuffer(gauge.X[0]*gauge.X[1]*gauge.X[2], gauge.volume, in.stride, in.precision); 
  }    

 
  // This gathers from source spinors and starts comms
  exchangeFacesStart(gauge.faces, in, dagger);

  // This waits for comms to finish, and sprays into the 
  // pads of the SOURCE spinor
  exchangeFacesWait(gauge.faces, in, dagger);

  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);
  checkCloverSpinor(in, cloverInv);


  int Vs = gauge.X[0] * gauge.X[1] * gauge.X[2];

  // do body
  {
    int tOffset = 1;
    int tMul = 1;
    int threads = in.volume - 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      cloverDslashXpayDCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      cloverDslashXpaySCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      cloverDslashXpayHCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    }
    checkCudaError();
  }

  // do faces
  {
    int tOffset = 0;
    int tMul = gauge.X[3] - 1;
    int threads = 2*Vs;
    gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
    initTLocation(tOffset, tMul, threads);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
      cloverDslashXpayDCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
      cloverDslashXpaySCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
      cloverDslashXpayHCuda(out, gauge, cloverInv, in, parity, dagger, x, a);
    }
    checkCudaError();
  }

  dslash_quda_flops += (1320+504+48)*in.volume;
}


void cloverDslashXpayDCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv, ParitySpinor spinor, 
			   int oddBit, int daggerBit, ParitySpinor x, double a)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    }
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashDDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashSDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	} else {
	  dslashHDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a);
	}
      }
    }
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
}

void cloverDslashXpaySCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv, ParitySpinor spinor, 
			   int oddBit, int daggerBit, ParitySpinor x, double a)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes+spinor.tface_bytes;
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
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
	  dslashDSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashDSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashSSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	} else {
	  dslashHSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit, a);
	}
      }
    }
  }
}

void cloverDslashXpayHCuda(ParitySpinor res, FullGauge gauge, FullClover cloverInv, ParitySpinor spinor, 
			   int oddBit, int daggerBit, ParitySpinor x, double a)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(cloverInv.odd);
    clover_prec = cloverInv.odd.precision;
  } else {
    bindCloverTex(cloverInv.even);
    clover_prec = cloverInv.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  int spinor_norm_bytes = spinor.bytes/12 + spinor.tface_bytes/6;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_norm_bytes); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, spinor_norm_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
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
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    }
  } else {
    if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashDHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashDHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
    } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashSHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashSHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashSHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    } else {
      if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashHHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	} else {
	  dslashHHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	    ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
	}
      }
    }
  } 
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void cloverMatPCCuda(ParitySpinor out, FullGauge gauge, FullClover clover, FullClover cloverInv, ParitySpinor in,
		     double kappa, ParitySpinor tmp, MatPCType matpc_type, int dagger)
{
  double kappa2 = -kappa*kappa;

  if (((matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) || (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC))
      && (clover.even.clover == NULL)) {
    errorQuda("For asymmetric matpc_type, the uninverted clover term must be loaded");
  }

  // FIXME: For asymmetric, a "dslashCxpay" kernel would improve performance.

  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      cloverDslashCuda(tmp, gauge, cloverInv, in, 1, dagger);
      cloverCuda(out, gauge, clover, in, 0);
      dslashXpayCuda(out, gauge, tmp, 0, dagger, out, kappa2); // safe since out is not read after writing
  } else if (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      cloverDslashCuda(tmp, gauge, cloverInv, in, 0, dagger);
      cloverCuda(out, gauge, clover, in, 1);
      dslashXpayCuda(out, gauge, tmp, 1, dagger, out, kappa2);
  } else if (!dagger) { // symmetric preconditioning
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      cloverDslashCuda(tmp, gauge, cloverInv, in, 1, dagger);
      cloverDslashXpayCuda(out, gauge, cloverInv, tmp, 0, dagger, in, kappa2); 
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      cloverDslashCuda(tmp, gauge, cloverInv, in, 0, dagger);
      cloverDslashXpayCuda(out, gauge, cloverInv, tmp, 1, dagger, in, kappa2); 
    } else {
      errorQuda("Invalid matpc_type");
    }
  } else { // symmetric preconditioning, dagger
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      cloverCuda(out, gauge, cloverInv, in, 0); 
      cloverDslashCuda(tmp, gauge, cloverInv, out, 1, dagger);
      dslashXpayCuda(out, gauge, tmp, 0, dagger, in, kappa2); 
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      cloverCuda(out, gauge, cloverInv, in, 1); 
      cloverDslashCuda(tmp, gauge, cloverInv, out, 0, dagger);
      dslashXpayCuda(out, gauge, tmp, 1, dagger, in, kappa2); 
    } else {
      errorQuda("Invalid matpc_type");
    }
  }
}

void cloverMatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, FullClover clover, FullClover cloverInv, ParitySpinor in,
			     double kappa, ParitySpinor tmp, MatPCType matpc_type)
{
  ParitySpinor aux = allocateParitySpinor(out.X, out.precision, out.pad); // FIXME: eliminate aux
  cloverMatPCCuda(aux, gauge, clover, cloverInv, in, kappa, tmp, matpc_type, 0);
  cloverMatPCCuda(out, gauge, clover, cloverInv, aux, kappa, tmp, matpc_type, 1);
  freeParitySpinor(aux);
}

// Apply the full operator (FIXME: create kernel to eliminate tmp)
void cloverMatCuda(FullSpinor out, FullGauge gauge, FullClover clover,
		   FullSpinor in, double kappa, ParitySpinor tmp, int dagger)
{
  cloverCuda(tmp, gauge, clover, in.odd, 1);
  dslashXpayCuda(out.odd, gauge, in.even, 1, dagger, tmp, -kappa);
  cloverCuda(tmp, gauge, clover, in.even, 0);
  dslashXpayCuda(out.even, gauge, in.odd, 0, dagger, tmp, -kappa);
}


// ----------------------------------------------------------------------

// Apply the clover term only
void cloverCuda(ParitySpinor out, FullGauge gauge, FullClover clover,
		ParitySpinor in, int parity)
{
 // if (!initDslash) initDslashConstants(gauge, in.stride, clover.even.stride);
  if (!initDslash) {
    initDslashConstants(gauge, in.stride, clover.even.stride);
    //faceBufferPrecise=allocateFaceBuffer(gauge.X[0]*gauge.X[1]*gauge.X[2], gauge.volume, in.stride, in.precision); 
  }    

  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);
  checkCloverSpinor(in, clover);

  int tOffset = 0;
  int tMul = 1;
  int threads = in.volume;
  gridVolume = (threads + BLOCK_DIM - 1) / BLOCK_DIM;
  initTLocation(tOffset, tMul, threads);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
    cloverDCuda(out, gauge, clover, in, parity);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    cloverSCuda(out, gauge, clover, in, parity);
  } else if (in.precision == QUDA_HALF_PRECISION) {
    cloverHCuda(out, gauge, clover, in, parity);
  }
  checkCudaError();

  dslash_quda_flops += 504*in.volume;
}

void cloverDCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		 ParitySpinor spinor, int oddBit)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (clover_prec == QUDA_DOUBLE_PRECISION) {
    cloverDDKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverDSKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
  } else {
    cloverDHKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
  }
#else
  errorQuda("Double precision not supported on this GPU");
#endif
}

void cloverSCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		 ParitySpinor spinor, int oddBit)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverSDKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverSSKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
  } else {
    cloverSHKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
  }
}

void cloverHCuda(ParitySpinor res, FullGauge gauge, FullClover clover,
		 ParitySpinor spinor, int oddBit)
{
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  Precision clover_prec;

  bindGaugeTex(gauge, oddBit);

  if (oddBit) {
    bindCloverTex(clover.odd);
    clover_prec = clover.odd.precision;
  } else {
    bindCloverTex(clover.even);
    clover_prec = clover.even.precision;
  }

  int spinor_bytes = spinor.bytes + spinor.tface_bytes;
  int spinor_norm_bytes = spinor.bytes/12 + spinor.tface_bytes/6;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_norm_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverHDKernel <<<gridDim, blockDim, shared_bytes>>> ((short4 *)res.spinor, (float *)res.spinorNorm, oddBit);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverHSKernel <<<gridDim, blockDim, shared_bytes>>> ((short4 *)res.spinor, (float *)res.spinorNorm, oddBit);
  } else {
    cloverHHKernel <<<gridDim, blockDim, shared_bytes>>> ((short4 *)res.spinor, (float *)res.spinorNorm, oddBit);
  }
}

#ifdef BUILD_3D_DSLASH

void dslash3DDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

#if (__CUDA_ARCH__ == 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      } else {
	dslash3DHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit);
      }
    }
  }
#endif
  
}

void dslash3DSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
  
  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DSS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      } else {
	dslash3DHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float4 *)res.spinor, oddBit);
      }
    }
  }
  
}


void dslash3DHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {

  dim3 gridDim(gridVolume, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = res.length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslash3DHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslash3DHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslash3DHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  }
  
}

#endif // BUILD_3D_DSLASH
