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
template <int Nreduce, typename doubleN, typename ReduceType,
	  template <int Nreduce_, typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX,int writeP,int writeU,int writeR, int writeS, int writeM,int writeQ,int writeW,int writeN, int writeZ, bool siteUnroll>
void reduceCudaExp(doubleN *reduce_buffer, const double2 &a, const double2 &b, ColorSpinorField &x, 
		   ColorSpinorField &p, ColorSpinorField &u, ColorSpinorField &r,
		   ColorSpinorField &s, ColorSpinorField &m, ColorSpinorField &q, 
                   ColorSpinorField &w, ColorSpinorField &n, ColorSpinorField &z) {

  if (checkLocation(x, u, r, p, z) == QUDA_CUDA_FIELD_LOCATION) {//must be 10 args.

    // cannot do site unrolling for arbitrary color (needs JIT)
    if (siteUnroll && x.Ncolor()!=3) errorQuda("Not supported");
    
    int reduce_length = siteUnroll ? x.RealLength() : x.Length();

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      if (x.Nspin() == 4 || x.Nspin() == 2) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	reduceCudaExp<Nreduce,doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ >
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) { //staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	reduceCudaExp<Nreduce,doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
	reduceCudaExp<Nreduce,doubleN,ReduceType,float4,float4,float4,M,Reducer,
	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, reduce_length/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1 || x.Nspin() == 2) {
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	reduceCudaExp<Nreduce,doubleN,ReduceType,float2,float2,float2,M,Reducer,
	  	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_HALF_PRECISION) { // half precision
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6; // determines how much work per thread to do
	reduceCudaExp<Nreduce,doubleN,ReduceType,float4,short4,short4,M,Reducer,
	  	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, x.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3; // determines how much work per thread to do
	reduceCudaExp<Nreduce,doubleN,ReduceType,float2,short2,short2,M,Reducer,
	  	  writeX,writeP,writeU,writeR,writeS,writeM,writeQ,writeW,writeN,writeZ>
	  (reduce_buffer, a, b, x, p, u, r, s, m, q, w, n, z, x.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x.Nspin()); }
    } else {
      errorQuda("precision=%d is not supported\n", x.Precision());
    }
  } else { // fields are on the CPU
    errorQuda("\nCPU version is not implemented..\n"); 
  }

  const int Nreglen = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray(reinterpret_cast<double*>(reduce_buffer), Nreduce*Nreglen);

  return;
}

//Component-wise reduction 

template <int Nreduce, typename doubleN, typename ReduceType,
	  template <int Nreduce_, typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX,int writeR,int writeW,int writeQ, int writeD, int writeH,int writeZ,int writeP,int writeU, int writeG, bool siteUnroll>
void reduceComponentwiseCudaExp(doubleN *reduce_buffer, const double2 &a, const double2 &b, const double2 &c, 
                   const double2 &a2, const double2 &b2, const double2 &c2,  ColorSpinorField &x1, ColorSpinorField &r1,
		   ColorSpinorField &w1, ColorSpinorField &q1, ColorSpinorField &d1,
		   ColorSpinorField &h1, ColorSpinorField &z1, ColorSpinorField &p1, 
                   ColorSpinorField &u1, ColorSpinorField &g1,
                   ColorSpinorField &x2, ColorSpinorField &r2,
		   ColorSpinorField &w2, ColorSpinorField &q2, ColorSpinorField &d2,
		   ColorSpinorField &h2, ColorSpinorField &z2, ColorSpinorField &p2, 
                   ColorSpinorField &u2, ColorSpinorField &g2) {

  if ((checkLocation(x1, u1, r1, p1, z1) == QUDA_CUDA_FIELD_LOCATION) and (checkLocation(x2, u2, r2, p2, z2) == QUDA_CUDA_FIELD_LOCATION)) {//must be 10 args.

    // cannot do site unrolling for arbitrary color (needs JIT)
    if (siteUnroll && x1.Ncolor()!=3) errorQuda("Not supported");
    
    int reduce_length = siteUnroll ? x1.RealLength() : x1.Length();

    if (x1.Precision() == QUDA_DOUBLE_PRECISION) {
      if (x1.Nspin() == 4 || x1.Nspin() == 2) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x1.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,double2,double2,double2,M,Reducer,
	 writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG >
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else if (x1.Nspin() == 1) { //staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,double2,double2,double2,M,Reducer,
	  writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG>
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x1.Nspin()); }
    } else if (x1.Precision() == QUDA_SINGLE_PRECISION) {
      if (x1.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,float4,float4,float4,M,Reducer,
	 writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG>
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, reduce_length/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else if (x1.Nspin() == 1 || x1.Nspin() == 2) {
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	if (x1.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,float2,float2,float2,M,Reducer,
	  	  writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG>
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, reduce_length/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x1.Nspin()); }
    } else if (x1.Precision() == QUDA_HALF_PRECISION) { // half precision
      if (x1.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6; // determines how much work per thread to do
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,float4,short4,short4,M,Reducer,
	  	  writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG>
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, x1.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else if (x1.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3; // determines how much work per thread to do
	reduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,float2,short2,short2,M,Reducer,
	  	  writeX,writeR,writeW,writeQ,writeD,writeH,writeZ,writeP,writeU,writeG>
	  (reduce_buffer, a, b, c, a2, b2, c2, x1, r1, w1, q1, d1, h1, z1, p1, u1, g1, x2, r2, w2, q2, d2, h2, z2, p2, u2, g2, x1.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x1.Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x1.Nspin()); }
    } else {
      errorQuda("precision=%d is not supported\n", x1.Precision());
    }
  } else { // fields are on the CPU
    errorQuda("\nCPU version is not implemented..\n"); 
  }

  const int Nreglen = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray(reinterpret_cast<double*>(reduce_buffer), Nreduce*Nreglen);

  return;
}
