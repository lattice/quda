/**
   Generic blas kernel with four loads and up to four stores.
 */
template <typename FloatN, int M, int writeX, int writeY, int writeZ, int writeW, 
  typename InputX, typename InputY, typename InputZ, typename InputW, 
  typename OutputX, typename OutputY, typename OutputZ, typename OutputW, typename Functor>
__global__ void blasKernel(InputX X, InputY Y, InputZ Z, InputW W, Functor f, 
			   OutputX XX, OutputY YY, OutputZ ZZ, OutputW WW, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    FloatN x[M], y[M], z[M], w[M];
    X.load(x, i);
    Y.load(y, i);
    Z.load(z, i);
    W.load(w, i);

#pragma unroll
    for (int j=0; j<M; j++) f(x[j], y[j], z[j], w[j]);

    if (writeX) XX.save(x, i);
    if (writeY) YY.save(y, i);
    if (writeZ) ZZ.save(z, i);
    if (writeW) WW.save(w, i);
    i += gridSize;
  }
}

/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void blasCuda(const int kernel, const double2 &a, const double2 &b, const double2 &c,
	      cudaColorSpinorField &x, cudaColorSpinorField &y, 
	      cudaColorSpinorField &z, cudaColorSpinorField &w) {
  int block_length = (x.Precision() == QUDA_HALF_PRECISION) ? x.Stride() : x.Length();
  setBlock(kernel, block_length, x.Precision());
  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);

  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (kernel, a, b, c, x.Even(), y.Even(), z.Even(), w.Even());
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (kernel, a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd());
    return;
  }

  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    const int M = 1;
    SpinorTexture<double2,double2,double2,M,0> xTex(x);
    SpinorTexture<double2,double2,double2,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<double2,double2,double2,M,1>(y);    
    SpinorTexture<double2,double2,double2,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<double2,double2,double2,M,2>(z);    
    SpinorTexture<double2,double2,double2,M,3> wTex;
    if (x.V() != w.V()) wTex = SpinorTexture<double2,double2,double2,M,3>(w);    
    Spinor<double2,double2,double2,M> X(x);
    Spinor<double2,double2,double2,M> Y(y);
    Spinor<double2,double2,double2,M> Z(z);
    Spinor<double2,double2,double2,M> W(w);
    Functor<double2, double2> f(a,b,c);
    blasKernel<double2,M,writeX,writeY,writeZ,writeW><<<blasGrid, blasBlock, 0, *blasStream>>>
      (xTex, yTex, zTex, wTex, f, X, Y, Z, W, x.Length()/(2*M));
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    const int M = 1;
    SpinorTexture<float4,float4,float4,M,0> xTex(x);
    SpinorTexture<float4,float4,float4,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,float4,M,1>(y);
    SpinorTexture<float4,float4,float4,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,float4,M,2>(z);
    SpinorTexture<float4,float4,float4,M,3> wTex;
    if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,float4,M,3>(w);
    Spinor<float4,float4,float4,M> X(x);
    Spinor<float4,float4,float4,M> Y(y);
    Spinor<float4,float4,float4,M> Z(z);
    Spinor<float4,float4,float4,M> W(w);
    Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
    blasKernel<float4,M,writeX,writeY,writeZ,writeW><<<blasGrid, blasBlock, 0, *blasStream>>>
      (xTex, yTex, zTex, wTex, f, X, Y, Z, W, x.Length()/(4*M));
  } else {
    if (x.Nspin() == 4){ //wilson
      SpinorTexture<float4,float4,short4,6,0> xTex(x);
      SpinorTexture<float4,float4,short4,6,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,short4,6,1>(y);
      SpinorTexture<float4,float4,short4,6,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,short4,6,2>(z);
      SpinorTexture<float4,float4,short4,6,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,short4,6,3>(w);
      Spinor<float4,float4,short4,6> xStore(x);
      Spinor<float4,float4,short4,6> yStore(y);
      Spinor<float4,float4,short4,6> zStore(z);
      Spinor<float4,float4,short4,6> wStore(w);
      Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      blasKernel<float4, 6, writeX, writeY, writeZ, writeW> <<<blasGrid, blasBlock, 0, *blasStream>>> 
	(xTex, yTex, zTex, wTex, f, xStore, yStore, zStore, wStore, y.Volume());
    } else if (x.Nspin() == 1) {//staggered
      SpinorTexture<float2,float2,short2,3,0> xTex(x);
      SpinorTexture<float2,float2,short2,3,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float2,float2,short2,3,1>(y);
      SpinorTexture<float2,float2,short2,3,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float2,float2,short2,3,2>(z);
      SpinorTexture<float2,float2,short2,3,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float2,float2,short2,3,3>(w);
      Spinor<float2,float2,short2,3> xStore(x);
      Spinor<float2,float2,short2,3> yStore(y);
      Spinor<float2,float2,short2,3> zStore(z);
      Spinor<float2,float2,short2,3> wStore(w);
      Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      blasKernel<float2, 3,writeX,writeY,writeZ,writeW> <<<blasGrid, blasBlock, 0, *blasStream>>>
	(xTex, yTex, zTex, wTex, f, xStore, yStore, zStore, wStore, y.Volume());
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    quda::blas_bytes += Functor<double2,double2>::streams()*x.Volume()*sizeof(float);
  }
  quda::blas_bytes += Functor<double2,double2>::streams()*x.RealLength()*x.Precision();
  quda::blas_flops += Functor<double2,double2>::flops()*x.RealLength();

  if (!blasTuning) checkCudaError();
}

