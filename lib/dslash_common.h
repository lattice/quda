static void checkSpinor(ParitySpinor out, ParitySpinor in) {
  if (in.precision != out.precision) {
    printf("Error in dslash quda: input and out spinor precisions don't match\n");
    exit(-1);
  }

  if (in.stride != out.stride) {
    printf("Error in dslash quda: input and out spinor strides don't match\n");
    exit(-1);
  }

#if (__CUDA_ARCH__ != 130)
  if (in.precision == QUDA_DOUBLE_PRECISION) {
    printf("Double precision not supported on this GPU\n");
    exit(-1);    
  }
#endif
}

static void checkGaugeSpinor(ParitySpinor spinor, FullGauge gauge) {
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

static void checkCloverSpinor(ParitySpinor spinor, FullClover clover) {
  if (spinor.volume != clover.even.volume) {
    printf("Error, spinor volume %d doesn't match even clover volume %d\n",
	   spinor.volume, clover.even.volume);
    exit(-1);
  }
  if (spinor.volume != clover.odd.volume) {
    printf("Error, spinor volume %d doesn't match odd clover volume %d\n",
	   spinor.volume, clover.odd.volume);
    exit(-1);
  }

#if (__CUDA_ARCH__ != 130)
  if ((clover.even.precision == QUDA_DOUBLE_PRECISION) ||
      (clover.odd.precision == QUDA_DOUBLE_PRECISION)) {
    printf("Double precision not supported on this GPU\n");
    exit(-1);    
  }
#endif
}


