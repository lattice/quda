/*
  Wilson
  double double2 M = 1/12
  single float4  M = 1/6
  half   short4  M = 6/6

  Staggered 
  double double2 M = 1/3
  single float2  M = 1/3
  half   short2  M = 3/3
 */

/**
   Driver for generic reduction routine with five loads.
   @param ReduceType 
   @param siteUnroll - if this is true, then one site corresponds to exactly one thread
 */
template <typename doubleN, typename ReduceType,
	  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
doubleN reduceCuda(const double2 &a, const double2 &b, ColorSpinorField &x, 
		   ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w,
		   ColorSpinorField &v) {

  doubleN value;
  if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

    // cannot do site unrolling for arbitrary color (needs JIT)
    if (siteUnroll && x.Ncolor()!=3) errorQuda("Not supported");
    
    int reduce_length = siteUnroll ? x.RealLength() : x.Length();

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      if (x.Nspin() == 4 || x.Nspin() == 2) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	value = reduceCuda<doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) { //staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	value = reduceCuda<doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
	value = reduceCuda<doubleN,ReduceType,float4,float4,float4,M,Reducer,
	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, reduce_length/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1 || x.Nspin() == 2) {
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	value = reduceCuda<doubleN,ReduceType,float2,float2,float2,M,Reducer,
	  	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_HALF_PRECISION) { // half precision
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6; // determines how much work per thread to do
	value = reduceCuda<doubleN,ReduceType,float4,short4,short4,M,Reducer,
	  	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, y.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3; // determines how much work per thread to do
	value = reduceCuda<doubleN,ReduceType,float2,short2,short2,M,Reducer,
	  	  writeX,writeY,writeZ,writeW,writeV>
	  (a, b, x, y, z, w, v, y.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x.Nspin()); }
    } else {
      errorQuda("precision=%d is not supported\n", x.Precision());
    }
  } else { // fields are on the CPU
    // we don't have quad precision support on the GPU so use doubleN instead of ReduceType
    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      Reducer<doubleN, double2, double2> r(a, b);
      value = genericReduce<doubleN,doubleN,double,double,writeX,writeY,writeZ,writeW,writeV,Reducer<doubleN,double2,double2> >(x,y,z,w,v,r);
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      Reducer<doubleN, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      value = genericReduce<doubleN,doubleN,float,float,writeX,writeY,writeZ,writeW,writeV,Reducer<doubleN,float2,float2> >(x,y,z,w,v,r);
    } else {
      errorQuda("Precision %d not implemented", x.Precision());
    }
  }

  const int Nreduce = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray((double*)&value, Nreduce);

  return value;
}

template <typename doubleN, typename ReduceType,
	  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeS,int writeP,int writeZ,int writeR, int writeX, int writeW,int writeQ, bool siteUnroll>
doubleN mergedReduceCuda(const double2 &a, const double2 &b, ColorSpinorField &s, 
		   ColorSpinorField &p, ColorSpinorField &z, ColorSpinorField &r,
		   ColorSpinorField &x, ColorSpinorField &w, ColorSpinorField &q) { 

  doubleN value;
  if (checkLocation(x, s, r, p, z) == QUDA_CUDA_FIELD_LOCATION) {//must be 10 args.

    // cannot do site unrolling for arbitrary color (needs JIT)
    if (siteUnroll && x.Ncolor()!=3) errorQuda("Not supported");
    
    int reduce_length = siteUnroll ? x.RealLength() : x.Length();

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      if (x.Nspin() == 4 || x.Nspin() == 2) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	value = reduceCudaExp<doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeS,writeP,writeZ,writeR,writeX,writeW,writeQ>
	  (a, b, s, p, z, r, x, w, q, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) { //staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	value = reduceCudaExp<doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeS,writeP,writeZ,writeR,writeX,writeW,writeQ>
	  (a, b, s, p, z, r, x, w, q, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
	value = reduceCudaExp<doubleN,ReduceType,float4,float4,float4,M,Reducer,
	  writeS,writeP,writeZ,writeR,writeX,writeW,writeQ>
	  (a, b, s, p, z, r, x, w, q,reduce_length/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1 || x.Nspin() == 2) {
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	value = reduceCudaExp<doubleN,ReduceType,float2,float2,float2,M,Reducer,
	  	  writeS,writeP,writeZ,writeR,writeX,writeW,writeQ>
	  (a, b, s, p, z, r, x, w, q,reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
#if 0 //BEGIN of HALF precision
    } else if (x.Precision() == QUDA_HALF_PRECISION) { // half precision
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6; // determines how much work per thread to do
	value = reduceCudaExp<doubleN,ReduceType,float4,short4,short4,M,Reducer,
	  	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (a, b, x, p, u, r, s, m, q, w, n, z, y.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3; // determines how much work per thread to do
	value = reduceCudaExp<doubleN,ReduceType,float2,short2,short2,M,Reducer,
	  	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (a, b, x, p, u, r, s, m, q, w, n, z, y.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x.Nspin()); }
#endif //END of HALF precision
    } else {
      errorQuda("precision=%d is not supported\n", x.Precision());
    }
  } else { // fields are on the CPU
    errorQuda("\nCPU version is not implemented..\n"); 
  }

  const int Nreduce = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray((double*)&value, Nreduce);

  return value;
}






