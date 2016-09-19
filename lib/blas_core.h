/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
  int writeX, int writeY, int writeZ, int writeW>
  void blasCuda(const double2 &a, const double2 &b, const double2 &c,
		ColorSpinorField &x, ColorSpinorField &y, 
		ColorSpinorField &z, ColorSpinorField &w) {

  if (Location(x, y, z, w) == QUDA_CUDA_FIELD_LOCATION) {

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
      const int M = 1;
      blasCuda<double2,double2,double2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Length()/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      if (x.Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 1;
	blasCuda<float4,float4,float4,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Length()/(4*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin()==2 || x.Nspin()==1) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
	const int M = 1;
	blasCuda<float2,float2,float2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Length()/(2*M));
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else { errorQuda("nSpin=%d is not supported\n", x.Nspin()); }
    } else {
      if (x.Ncolor() != 3) { errorQuda("nColor = %d is not supported", x.Ncolor()); }
      if (x.Nspin() == 4) { //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	const int M = 6;
	blasCuda<float4,short4,short4,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	const int M = 3;
	blasCuda<float2,short2,short2,M,Functor,writeX,writeY,writeZ,writeW>(a,b,c,x,y,z,w,x.Volume());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else {
	errorQuda("nSpin=%d is not supported\n", x.Nspin());
      }
    }
  } else { // fields on the cpu
    using namespace quda::colorspinor;
    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      Functor<double2, double2> f(a, b, c);
      genericBlas<double, double, writeX, writeY, writeZ, writeW>(x, y, z, w, f);
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      Functor<float2, float2> f(make_float2(a.x,a.y), make_float2(b.x,b.y), make_float2(c.x,c.y) );
      genericBlas<float, float, writeX, writeY, writeZ, writeW>(x, y, z, w, f);
    } else {
      errorQuda("Not implemented");
    }
  }

}
  
