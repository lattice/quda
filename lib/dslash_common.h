static void checkSpinor(ParitySpinor out, ParitySpinor in) {
  if (in.precision != out.precision) {
    errorQuda("Input and output spinor precisions don't match in dslash_quda");
  }

  if (in.stride != out.stride) {
    errorQuda("Input and output spinor strides don't match in dslash_quda");
  }

#if (__CUDA_ARCH__ != 130)
  if (in.precision == QUDA_DOUBLE_PRECISION) {
    errorQuda("Double precision not supported on this GPU");
  }
#endif
}

static void checkGaugeSpinor(ParitySpinor spinor, FullGauge gauge) {
  if (spinor.volume != gauge.volume) {
    errorQuda("Spinor volume %d doesn't match gauge volume %d", spinor.volume, gauge.volume);
  }

#if (__CUDA_ARCH__ != 130)
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    errorQuda("Double precision not supported on this GPU");
  }
#endif
}

static void checkCloverSpinor(ParitySpinor spinor, FullClover clover) {
  if (spinor.volume != clover.even.volume) {
    errorQuda("Spinor volume %d doesn't match even clover volume %d",
	      spinor.volume, clover.even.volume);
  }
  if (spinor.volume != clover.odd.volume) {
    errorQuda("Spinor volume %d doesn't match odd clover volume %d",
	      spinor.volume, clover.odd.volume);
  }

#if (__CUDA_ARCH__ != 130)
  if ((clover.even.precision == QUDA_DOUBLE_PRECISION) ||
      (clover.odd.precision == QUDA_DOUBLE_PRECISION)) {
    errorQuda("Double precision not supported on this GPU");
  }
#endif
}
