#include <multi_reduce_mixed_core.h>

/**
   Driver for multi-reduce with up to five vectors
*/
template<int NXZ, typename doubleN, typename ReduceType,
  template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write, bool siteUnroll, typename T>
  void multiReduceCuda(doubleN result[], const reduce::coeff_array<T> &a, const reduce::coeff_array<T> &b, const reduce::coeff_array<T> &c,
		       CompositeColorSpinorField& x, CompositeColorSpinorField& y,
		       CompositeColorSpinorField& z, CompositeColorSpinorField& w){
  const int NYW = y.size();

  int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

  if (x[0]->Precision() == QUDA_DOUBLE_PRECISION) {
    if (x[0]->Nspin() == 4 || x[0]->Nspin() == 2) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
      const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
      if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
      multiReduceCuda<doubleN,ReduceType,double2,double2,double2,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
      const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
      multiReduceCuda<doubleN,ReduceType,double2,double2,double2,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  } else if (x[0]->Precision() == QUDA_SINGLE_PRECISION) {
    if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
      multiReduceCuda<doubleN,ReduceType,float4,float4,float4,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, reduce_length/(4*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if(x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
      const int M = siteUnroll ? 3 : 1;
      if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
      multiReduceCuda<doubleN,ReduceType,float2,float2,float2,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  } else { // half precision
    if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = 6;
      multiReduceCuda<doubleN,ReduceType,float4,short4,short4,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if(x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
      const int M = 3;
      multiReduceCuda<doubleN,ReduceType,float2,short2,short2,M,NXZ,Reducer,write>
	(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  }

  return;
}

template<int NXZ, typename doubleN, typename ReduceType,
  template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal, typename writeDiagonal,
  template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal, typename writeOffDiagonal,
  bool siteUnroll, typename T>
  void multiReduceCuda(doubleN result[], const reduce::coeff_array<T> &a, const reduce::coeff_array<T> &b, const reduce::coeff_array<T> &c,
		       CompositeColorSpinorField& x, CompositeColorSpinorField& y,
		       CompositeColorSpinorField& z, CompositeColorSpinorField& w, int i, int j){

  if (x[0]->Precision()==y[0]->Precision()) {
    if (i==j) { // we are on the diagonal so invoke the diagonal reducer
      multiReduceCuda<NXZ,doubleN,ReduceType,ReducerDiagonal,writeDiagonal,siteUnroll,T>(result, a, b, c, x, y, z, w);
    } else { // we are on the diagonal so invoke the off-diagonal reducer
      multiReduceCuda<NXZ,doubleN,ReduceType,ReducerOffDiagonal,writeOffDiagonal,siteUnroll,T>(result, a, b, c, x, y, z, w);
    }
  } else {
    if (i==j) { // we are on the diagonal so invoke the diagonal reducer
      mixed::multiReduceCuda<NXZ,doubleN,ReduceType,ReducerDiagonal,writeDiagonal,true,T>(result, a, b, c, x, y, z, w);
    } else { // we are on the diagonal so invoke the off-diagonal reducer
      mixed::multiReduceCuda<NXZ,doubleN,ReduceType,ReducerOffDiagonal,writeOffDiagonal,true,T>(result, a, b, c, x, y, z, w);
    }
  }

}
