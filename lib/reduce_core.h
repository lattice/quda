#include <reduce_core.cuh>

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
template <typename doubleN, typename ReduceType, typename ReduceSimpleType,
	  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
doubleN reduceCuda(const double2 &a, const double2 &b, ColorSpinorField &x, 
		   ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w,
		   ColorSpinorField &v) {
  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);
  checkSpinor(x, v);

  doubleN value;
  if (Location(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

    if (!x.isNative()) {
      warningQuda("Device reductions on non-native fields is not supported\n");
      doubleN value;
      ::quda::zero(value);
      return value;
    }

    blasStrings.vol_str = x.VolString();
    blasStrings.aux_str = x.AuxString();

    // FIXME this condition should be outside of the Location test but
    // Even and Odd must be implemented for cpu fields first
    if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      doubleN even =
	reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,
	writeY,writeZ,writeW,writeV,siteUnroll>
	(a, b, x.Even(), y.Even(), z.Even(), w.Even(), v.Even());
      doubleN odd = 
	reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,
	writeY,writeZ,writeW,writeV,siteUnroll>
	(a, b, x.Odd(), y.Odd(), z.Odd(), w.Odd(), v.Odd());
      return even + odd;
    }

    int reduce_length = siteUnroll ? x.RealLength() : x.Length();

    // cannot do site unrolling for arbitrary color (needs JIT)
    if (siteUnroll && x.Ncolor()!=3) errorQuda("Not supported");
    
    // FIXME: use traits to encapsulate register type for shorts -
    // will reduce template type parameters from 3 to 2
    
    size_t bytes[] = {x.Bytes(), y.Bytes(), z.Bytes(), w.Bytes(), v.Bytes()};
    size_t norm_bytes[] = {x.NormBytes(), y.NormBytes(), z.NormBytes(), w.NormBytes(), v.NormBytes()};

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      if (x.Nspin() == 4 || x.Nspin() == 2){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	Spinor<double2,double2,double2,M,writeX> X(x);
	Spinor<double2,double2,double2,M,writeY> Y(y);
	Spinor<double2,double2,double2,M,writeZ> Z(z);
	Spinor<double2,double2,double2,M,writeW> W(w);
	Spinor<double2,double2,double2,M,writeV> V(v);
	Reducer<ReduceType, double2, double2> r(a,b);
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,double2,M,
	  Spinor<double2,double2,double2,M,writeX>, Spinor<double2,double2,double2,M,writeY>,
	  Spinor<double2,double2,double2,M,writeZ>, Spinor<double2,double2,double2,M,writeW>,
	  Spinor<double2,double2,double2,M,writeV>, Reducer<ReduceType, double2, double2> >
	  reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M), bytes, norm_bytes);
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1){ //staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	Spinor<double2,double2,double2,M,writeX> X(x);
	Spinor<double2,double2,double2,M,writeY> Y(y);
	Spinor<double2,double2,double2,M,writeZ> Z(z);
	Spinor<double2,double2,double2,M,writeW> W(w);
	Spinor<double2,double2,double2,M,writeV> V(v);
	Reducer<ReduceType, double2, double2> r(a,b);
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,double2,M,
	  Spinor<double2,double2,double2,M,writeX>, Spinor<double2,double2,double2,M,writeY>,
	  Spinor<double2,double2,double2,M,writeZ>, Spinor<double2,double2,double2,M,writeW>,
	  Spinor<double2,double2,double2,M,writeV>, Reducer<ReduceType, double2, double2> >
	  reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M), bytes, norm_bytes);
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
	Spinor<float4,float4,float4,M,writeX,0> X(x);
	Spinor<float4,float4,float4,M,writeY,1> Y(y);
	Spinor<float4,float4,float4,M,writeZ,2> Z(z);
	Spinor<float4,float4,float4,M,writeW,3> W(w);
	Spinor<float4,float4,float4,M,writeV,4> V(v);
	Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float4,M,
	  Spinor<float4,float4,float4,M,writeX,0>, Spinor<float4,float4,float4,M,writeY,1>,
	  Spinor<float4,float4,float4,M,writeZ,2>, Spinor<float4,float4,float4,M,writeW,3>,
	  Spinor<float4,float4,float4,M,writeV,4>, Reducer<ReduceType, float2, float4> >
	  reduce(value, X, Y, Z, W, V, r, reduce_length/(4*M), bytes, norm_bytes);
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1 || x.Nspin() == 2) {
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
	if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
	Spinor<float2,float2,float2,M,writeX,0> X(x);
	Spinor<float2,float2,float2,M,writeY,1> Y(y);
	Spinor<float2,float2,float2,M,writeZ,2> Z(z);
	Spinor<float2,float2,float2,M,writeW,3> W(w);
	Spinor<float2,float2,float2,M,writeV,4> V(v);
	Reducer<ReduceType, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float2,M,
	  Spinor<float2,float2,float2,M,writeX,0>, Spinor<float2,float2,float2,M,writeY,1>,
	  Spinor<float2,float2,float2,M,writeZ,2>, Spinor<float2,float2,float2,M,writeW,3>,
	  Spinor<float2,float2,float2,M,writeV,4>, Reducer<ReduceType, float2, float2> >
	  reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M), bytes, norm_bytes);
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    } else { // half precision
      if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	Spinor<float4,float4,short4,6,writeX,0> X(x);
	Spinor<float4,float4,short4,6,writeY,1> Y(y);
	Spinor<float4,float4,short4,6,writeZ,2> Z(z);
	Spinor<float4,float4,short4,6,writeW,3> W(w);
	Spinor<float4,float4,short4,6,writeV,4> V(v);
	Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float4,6,
	  Spinor<float4,float4,short4,6,writeX,0>, Spinor<float4,float4,short4,6,writeY,1>,
	  Spinor<float4,float4,short4,6,writeZ,2>, Spinor<float4,float4,short4,6,writeW,3>,
	  Spinor<float4,float4,short4,6,writeV,4>, Reducer<ReduceType, float2, float4> >
	  reduce(value, X, Y, Z, W, V, r, y.Volume(), bytes, norm_bytes);
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	Spinor<float2,float2,short2,3,writeX,0> X(x);
	Spinor<float2,float2,short2,3,writeY,1> Y(y);
	Spinor<float2,float2,short2,3,writeZ,2> Z(z);
	Spinor<float2,float2,short2,3,writeW,3> W(w);
	Spinor<float2,float2,short2,3,writeV,4> V(v);
	Reducer<ReduceType, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
	ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float2,3,
	  Spinor<float2,float2,short2,3,writeX,0>, Spinor<float2,float2,short2,3,writeY,1>,
	  Spinor<float2,float2,short2,3,writeZ,2>, Spinor<float2,float2,short2,3,writeW,3>,
	  Spinor<float2,float2,short2,3,writeV,4>, Reducer<ReduceType, float2, float2> >
	  reduce(value, X, Y, Z, W, V, r, y.Volume(), bytes, norm_bytes );
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
      blas::bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x.Volume()*sizeof(float);
    }
  } else { // fields are on the CPU
    // we don't have quad precision support on the GPU so use doubleN instead of ReduceType
    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      Reducer<doubleN, double2, double2> r(a, b);
      value = genericReduce<doubleN,doubleN,double,writeX,writeY,writeZ,writeW,writeV,Reducer<doubleN,double2,double2> >(x,y,z,w,v,r);
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      Reducer<doubleN, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      value = genericReduce<doubleN,doubleN,float,writeX,writeY,writeZ,writeW,writeV,Reducer<doubleN,float2,float2> >(x,y,z,w,v,r);
    } else {
      errorQuda("Precision %d not implemented", x.Precision());
    }
  }

  const int Nreduce = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray((double*)&value, Nreduce);

  blas::bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  blas::flops += Reducer<ReduceType,double2,double2>::flops()*(unsigned long long)x.RealLength();
    
  checkCudaError();

  return value;
}

#include "multi_reduce_core.h"
