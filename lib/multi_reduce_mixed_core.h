namespace mixed {

  /**
     Driver for multi-reduce with up to five vectors
  */
  template<int NXZ, typename doubleN, typename ReduceType,
    template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write, bool siteUnroll, typename T>
    void multiReduceCuda(doubleN result[], const reduce::coeff_array<T> &a, const reduce::coeff_array<T> &b, const reduce::coeff_array<T> &c,
			 CompositeColorSpinorField& x, CompositeColorSpinorField& y,
			 CompositeColorSpinorField& z, CompositeColorSpinorField& w){
    const int NYW = y.size();

    checkPrecision(*x[0],*z[0]);
    checkPrecision(*y[0],*w[0]);

    assert(siteUnroll==true);
    int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

    if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_SINGLE_PRECISION) {

      if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 12; // determines how much work per thread to do
	multiReduceCuda<doubleN,ReduceType,double2,float4,double2,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3; // determines how much work per thread to do
	multiReduceCuda<doubleN,ReduceType,double2,float2,double2,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }

    } else if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

      if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6; // determines how much work per thread to do
	multiReduceCuda<doubleN,ReduceType,double2,short4,double2,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, reduce_length/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if(x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC)
	const int M = 3;
	multiReduceCuda<doubleN,ReduceType,double2,short2,double2,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }

    } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

      if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6;
	multiReduceCuda<doubleN,ReduceType,float4,short4,float4,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, x[0]->Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if(x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3;
	multiReduceCuda<doubleN,ReduceType,float2,short2,float2,M,NXZ,Reducer,write>
	  (result, a, b, c, x, y, z, w, x[0]->Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }

    } else {
      errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
    }

    return;
  }

} // namespace mixed
