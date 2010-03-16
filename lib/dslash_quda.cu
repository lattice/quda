#include <stdlib.h>
#include <stdio.h>

#include <quda_internal.h>
#include <dslash_quda.h>

#define BLOCK_DIM 64

#include <dslash_textures.h>
#include <dslash_constants.h>

// kludge to avoid '#include nested too deeply' error
#define DD_DAG 0
#include <dslash_def.h>
#undef DD_DAG
#define DD_DAG 1
#include <dslash_def.h>
#undef DD_DAG

#include <clover_def.h> // kernels for applying the clover term alone

int dslashCudaSharedBytes(QudaPrecision precision) {
  return BLOCK_DIM*SHARED_FLOATS_PER_THREAD*precision;
}

/*template <QudaReconstructType reconType, typename sFloat, typename gFloat>
__global__ dslashKernel(sFloat *out, gFloat *gauge, sFloat *in, int parity, float *outNorm, float *inNorm);

template <QudaReconstructType reconType, typename sFloat, typename gFloat>
__global__ dslashDaggerKernel(sFloat *out, gFloat *gauge, sFloat *in, int parity, float *outNorm, float *inNorm);*/

template <int spinorN, typename spinorFloat, typename gaugeFloat>
void dslashCuda(spinorFloat *out, gaugeFloat *gauge0, gaugeFloat *gauge1, 
		QudaReconstructType reconstruct, spinorFloat *in, 
		int parity, int dagger, int volume, int length, 
		float *outNorm, float *inNorm) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<spinorN>(length, in, inNorm);

  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (!dagger) {
      dslash12Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, gauge0, gauge1, in, inNorm, parity);
    } else {
      dslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, gauge0, gauge1, in, inNorm, parity);
    }
  } else {
    if (!dagger) {
      dslash8Kernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, gauge0, gauge1, in, inNorm, parity);
    } else {
      dslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> (out, outNorm, gauge0, gauge1, in, inNorm, parity);
    }
  }
  
}

template <int spinorN, typename spinorFloat>
void dslashCuda(spinorFloat *out, FullGauge gauge, spinorFloat *in, int parity, int dagger,
		int volume, int length, float *outNorm, float *inNorm) {

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    dslashCuda<spinorN>(out, (double2*)gauge0, (double2*)gauge1, gauge.reconstruct, in, parity, 
			dagger, volume, length, outNorm, inNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    dslashCuda<spinorN>(out, (float4*)gauge0, (float4*)gauge1, gauge.reconstruct, in, parity, 
			dagger, volume, length, outNorm, inNorm);
  } else {
    dslashCuda<spinorN>(out, (short4*)gauge0, (short4*)gauge1, gauge.reconstruct, in, parity, 
			dagger, volume, length, outNorm, inNorm);
  }

}

void dslashCuda(void *out, FullGauge gauge, void *in, int parity, int dagger,
		int volume, int length, void *outNorm, void *inNorm, const QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    dslashCuda<2>((double2*)out, gauge, (double2*)in, parity, dagger, volume, length,
		  (float*)0, (float*)0);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    dslashCuda<4>((float4*)out, gauge, (float4*)in, parity, dagger, volume, length,
		  (float*)0, (float*)0);
  } else if (precision == QUDA_HALF_PRECISION) {
    dslashCuda<4>((short4*)out, gauge, (short4*)in, parity, dagger, volume, length,
		  (float*)outNorm, (float*)inNorm);
  }

}

template <int spinorN, typename spinorFloat, typename gaugeFloat>
void dslashXpayCuda(spinorFloat *out, gaugeFloat *gauge0, gaugeFloat *gauge1, 
		QudaReconstructType reconstruct, spinorFloat *in, 
		int parity, int dagger, spinorFloat *x, double a, 
		int volume, int length,  float *outNorm, 
		float *inNorm, float *xNorm) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<spinorN>(length, in, inNorm, x, xNorm);

  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (!dagger) {
      dslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, gauge0, gauge1, in, inNorm, parity, a);
    } else {
      dslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, gauge0, gauge1, in, inNorm, parity, a);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    if (!dagger) {
      dslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, gauge0, gauge1, in, inNorm, parity, a);
    } else {
      dslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	(out, outNorm, gauge0, gauge1, in, inNorm, parity, a);
    }
  }
}


template <int spinorN, typename spinorFloat>
void dslashXpayCuda(spinorFloat *out, FullGauge gauge, spinorFloat *in, int parity, 
		    int dagger, spinorFloat *x, double a, int volume, int length, 
		    float *outNorm, float *inNorm, float *xNorm) {

  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    dslashXpayCuda<spinorN>(out, (double2*)gauge0, (double2*)gauge1, gauge.reconstruct, in, parity, 
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    dslashXpayCuda<spinorN>(out, (float4*)gauge0, (float4*)gauge1, gauge.reconstruct, in, parity, 
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  } else {
    dslashXpayCuda<spinorN>(out, (short4*)gauge0, (short4*)gauge1, gauge.reconstruct, in, parity, 
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  }

}

void dslashXpayCuda(void *out, FullGauge gauge, void *in, int parity, int dagger,
		    void *x, double k, int volume, int length, void *outNorm, 
		    void *inNorm, void *xNorm, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    dslashXpayCuda<2>((double2*)out, gauge, (double2*)in, parity, dagger, (double2*)x, k,
		      volume, length, (float*)0, (float*)0, (float*)0);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    dslashXpayCuda<4>((float4*)out, gauge, (float4*)in, parity, dagger, (float4*)x, k,
		      volume, length, (float*)0, (float*)0, (float*)0);
  } else if (precision == QUDA_HALF_PRECISION) {
    dslashXpayCuda<4>((short4*)out, gauge, (short4*)in, parity, dagger, (short4*)x, k,
		      volume, length, (float*)outNorm, (float*)inNorm, (float*)xNorm);
  }
  checkCudaError();

}


template <int N, typename spinorFloat, typename cloverFloat, typename gaugeFloat>
void cloverDslashCuda(spinorFloat *out, gaugeFloat gauge0, gaugeFloat gauge1, 
		      QudaReconstructType reconstruct, cloverFloat *clover, float *cloverNorm, 
		      spinorFloat *in, int parity, int dagger, int volume, int length, 
		      float *outNorm, float *inNorm)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm);

  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (!dagger) {
      dslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
    } else {
      dslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
 	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
    }
    if (!dagger) {
      dslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 	
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
    } else {
      dslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
    }
  }

}

template <int N, typename spinorFloat, typename cloverFloat>
void cloverDslashCuda(spinorFloat *out, FullGauge gauge, cloverFloat *clover, float *cloverNorm, 
		      spinorFloat *in, int parity, int dagger, int volume, int length, 
		      float *outNorm, float *inNorm)
{
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashCuda<N>(out, (double2*)gauge0, (double2*)gauge1, gauge.reconstruct, 
			clover, cloverNorm, in, parity, dagger, volume, length,
			outNorm, inNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    cloverDslashCuda<N>(out, (float4*)gauge0, (float4*)gauge1, gauge.reconstruct, 
			clover, cloverNorm, in, parity, dagger, volume, length,
			outNorm, inNorm);
  } else {
    cloverDslashCuda<N>(out, (short4*)gauge0, (short4*)gauge1, gauge.reconstruct, 
			clover, cloverNorm, in, parity, dagger, volume, length, 
			outNorm, inNorm);
  }
}

template <int N, typename spinorFloat>
void cloverDslashCuda(spinorFloat *out, FullGauge gauge, FullClover cloverInv, 
		      spinorFloat *in, int parity, int dagger, int volume, int length, 
		      float *outNorm, float *inNorm) {

  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashCuda<N>(out, gauge, (double2*)cloverP, (float*)cloverNormP, in, parity, 
			dagger, volume, length, outNorm, inNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverDslashCuda<N>(out, gauge, (float4*)cloverP, (float*)cloverNormP, in, parity, 
			dagger, volume, length, outNorm, inNorm);
  } else {
    cloverDslashCuda<N>(out, gauge, (short4*)cloverP, (float*)cloverNormP, in, parity, 
			dagger, volume, length, outNorm, inNorm);
  }

}

void cloverDslashCuda(void *out, FullGauge gauge, FullClover cloverInv, void *in, 
		      int parity, int dagger, int volume, int length, 
		      void *outNorm, void *inNorm, const QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashCuda<2>((double2*)out, gauge, cloverInv, (double2*)in, parity, dagger,
			volume, length, (float*)0, (float*)0);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    cloverDslashCuda<4>((float4*)out, gauge, cloverInv, (float4*)in, parity, dagger,
			volume, length, (float*)0, (float*)0);
  } else if (precision == QUDA_HALF_PRECISION) {
    cloverDslashCuda<4>((short4*)out, gauge, cloverInv, (short4*)in, parity, dagger,
			volume, length, (float*)outNorm, (float*)inNorm);
  }
  checkCudaError();

}

template <int N, typename spinorFloat, typename cloverFloat, typename gaugeFloat>
void cloverDslashXpayCuda(spinorFloat *out, gaugeFloat gauge0, gaugeFloat gauge1, 
			  QudaReconstructType reconstruct, cloverFloat *clover, float *cloverNorm, 
			  spinorFloat *in, int parity, int dagger, spinorFloat *x, double a, 
			  int volume, int length, float *outNorm, float *inNorm, float *xNorm)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm);

  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (!dagger) {
      dslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, a);
    } else {
      dslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
 	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, a);
    }
    if (!dagger) {
      dslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 	
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, a);
    } else {
      dslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	(out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, a);
    }
  }

}

template <int N, typename spinorFloat, typename cloverFloat>
void cloverDslashXpayCuda(spinorFloat *out, FullGauge gauge, cloverFloat *clover, float *cloverNorm, 
			  spinorFloat *in, int parity, int dagger, spinorFloat *x, double a, 
			  int volume, int length, float *outNorm, float *inNorm, float *xNorm)
{
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashXpayCuda<N>(out, (double2*)gauge0, (double2*)gauge1, gauge.reconstruct, clover, cloverNorm, in, 
			    parity, dagger, x, a, volume, length, outNorm, inNorm, xNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    cloverDslashXpayCuda<N>(out, (float4*)gauge0, (float4*)gauge1, gauge.reconstruct, clover, cloverNorm, in, 
			    parity, dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  } else {
    cloverDslashXpayCuda<N>(out, (short4*)gauge0, (short4*)gauge1, gauge.reconstruct, clover, cloverNorm, in, 
			    parity, dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  }
}

template <int N, typename spinorFloat>
void cloverDslashXpayCuda(spinorFloat *out, FullGauge gauge, FullClover cloverInv, spinorFloat *in, 
			  int parity, int dagger, spinorFloat *x, double a, int volume, int length, 
			  float *outNorm, float *inNorm, float *xNorm) {

  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashXpayCuda<N>(out, gauge, (double2*)cloverP, (float*)cloverNormP, in, parity, 
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverDslashXpayCuda<N>(out, gauge, (float4*)cloverP, (float*)cloverNormP, in, parity, 
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  } else {
    cloverDslashXpayCuda<N>(out, gauge, (short4*)cloverP, (float*)cloverNormP, in, parity,
			    dagger, x, a, volume, length, outNorm, inNorm, xNorm);
  }

}

void cloverDslashXpayCuda(void *out, FullGauge gauge, FullClover cloverInv, void *in, 
			  int parity, int dagger, void *x, double a, int volume, 
			  int length, void *outNorm, void *inNorm, void *xNorm,
			  const QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverDslashXpayCuda<2>((double2*)out, gauge, cloverInv, (double2*)in, parity, 
			    dagger, (double2*)x, a, volume, length, (float*)0, (float*)0, (float*)0);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    cloverDslashXpayCuda<4>((float4*)out, gauge, cloverInv, (float4*)in, parity, 
			    dagger, (float4*)x, a, volume, length, (float*)0, (float*)0, (float*)0);
  } else if (precision == QUDA_HALF_PRECISION) {
    cloverDslashXpayCuda<4>((short4*)out, gauge, cloverInv, (short4*)in, parity, dagger, 
			    (short4*)x, a, volume, length, 
			    (float*)outNorm, (float*)inNorm, (float*)xNorm);
  }
  checkCudaError();

}


template <int N, typename FloatN>
void cloverCuda(FloatN *out, float *outNorm, FullGauge gauge, FullClover clover, 
		FloatN *in, float *inNorm, int parity, int volume, int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm);

  if (clover_prec == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverKernel <<<gridDim, blockDim, shared_bytes>>> 
      (out, outNorm, (double2*)cloverP, (float*)cloverNormP, in, inNorm, parity);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (clover_prec == QUDA_SINGLE_PRECISION) {
    cloverKernel <<<gridDim, blockDim, shared_bytes>>> 
      (out, outNorm, (float4*)cloverP, (float*)cloverNormP, in, inNorm, parity);
  } else {
    cloverKernel <<<gridDim, blockDim, shared_bytes>>> 
      (out, outNorm, (short4*)cloverP, (float*)cloverNormP, in, inNorm, parity);
  }
}

void cloverCuda(void *out, FullGauge gauge, FullClover clover, 
		void *in, int parity, int volume, int length, 
		void *outNorm, void *inNorm, const QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ == 130)
    cloverCuda<2>((double2*)out, (float*)outNorm, gauge, clover, (double2*)in, 
		  (float*)inNorm, parity, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    cloverCuda<4>((float4*)out, (float*)outNorm, gauge, clover, (float4*)in, 
		  (float*)inNorm, parity, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    cloverCuda<4>((short4*)out, (float*)outNorm, gauge, clover, (short4*)in,
		  (float*)inNorm, parity, volume, length);
  }
  checkCudaError();

}
