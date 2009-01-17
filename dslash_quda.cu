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

texture<float4, 1, cudaReadModeElementType> spinorTex;
texture<float4, 1, cudaReadModeElementType> accumTex;

QudaGaugeParam *gauge_param;

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

  cudaBindTexture(0, spinorTex, spinor, SPINOR_BYTES); 
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSingle12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      } else {
	dslashSingle12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSingle8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      } else {
	dslashSingle8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      }
    }
  } else {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHalf12Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      } else {
	dslashHalf12DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHalf8Kernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
      } else {
	dslashHalf8DaggerKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit);
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

  cudaBindTexture(0, spinorTex, spinor, SPINOR_BYTES); 
  cudaBindTexture(0, accumTex, x, SPINOR_BYTES); 
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  if (gauge_param->cuda_prec == QUDA_SINGLE_PRECISION) {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSingle12XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      } else {
	dslashSingle12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
    } else if (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSingle8XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
      else {
	dslashSingle8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
    }
  } else {
    if (gauge_param->reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHalf12XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
    } else {
	dslashHalf12DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
    }
    } else if (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHalf8XpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
	  ((float4 *)res, oddBit, a);
      }
      else {
	dslashHalf8DaggerXpayKernel <<<gridDim, blockDim, SHARED_BYTES>>> 
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
cuComplex MatPCcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
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
cuComplex MatPCDagcDotWXCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, float kappa, 
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
