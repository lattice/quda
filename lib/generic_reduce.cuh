/**
   Generic reduce kernel with four loads and up to four stores.
  */
template <typename ReduceType, typename Float, int writeX, int writeY, int writeZ, int writeW, int writeV,
    typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
ReduceType genericReduce(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Reducer r)
{

  ReduceType sum;
  ::quda::zero(sum);

  for (int parity = 0; parity < X.Nparity(); parity++) {
    for (int x = 0; x < X.VolumeCB(); x++) {
      r.pre();
      for (int s = 0; s < X.Nspin(); s++) {
        for (int c = 0; c < X.Ncolor(); c++) {
          complex<Float> X_ = X(parity, x, s, c);
          complex<Float> Y_ = Y(parity, x, s, c);
          complex<Float> Z_ = Z(parity, x, s, c);
          complex<Float> W_ = W(parity, x, s, c);
          complex<Float> V_ = V(parity, x, s, c);
          r(sum, X_, Y_, Z_, W_, V_);
          if (writeX) X(parity, x, s, c) = X_;
          if (writeY) Y(parity, x, s, c) = Y_;
          if (writeZ) Z(parity, x, s, c) = Z_;
          if (writeW) W(parity, x, s, c) = W_;
          if (writeV) V(parity, x, s, c) = V_;
        }
      }
      r.post(sum);
    }
  }

  return sum;
}

template <typename ReduceType, typename Float, typename zFloat, int nSpin, int nColor, QudaFieldOrder order, int writeX,
    int writeY, int writeZ, int writeW, int writeV, typename R>
ReduceType genericReduce(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  colorspinor::FieldOrderCB<Float, nSpin, nColor, 1, order> X(x), Y(y), W(w), V(v);
  colorspinor::FieldOrderCB<zFloat, nSpin, nColor, 1, order> Z(z);
  return genericReduce<ReduceType, zFloat, writeX, writeY, writeZ, writeW, writeV>(X, Y, Z, W, V, r);
}

template <typename ReduceType, typename Float, typename zFloat, int nSpin, QudaFieldOrder order, int writeX, int writeY,
    int writeZ, int writeW, int writeV, typename R>
ReduceType genericReduce(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  ReduceType value;
  if (x.Ncolor() == 3) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 3, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
#ifdef GPU_MULTIGRID
  } else if (x.Ncolor() == 4) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 4, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 6) { // free field Wilson
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 6, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 8) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 8, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 12) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 12, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 16) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 16, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 20) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 20, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 24) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 24, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 32) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 32, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 72) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 72, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
  } else if (x.Ncolor() == 576) {
    value = genericReduce<ReduceType, Float, zFloat, nSpin, 576, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
#endif
  } else {
    ::quda::zero(value);
    errorQuda("nColor = %d not implemented", x.Ncolor());
  }
  return value;
}

template <typename ReduceType, typename Float, typename zFloat, QudaFieldOrder order, int writeX, int writeY,
    int writeZ, int writeW, int writeV, typename R>
ReduceType genericReduce(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  ReduceType value;
  ::quda::zero(value);
  if (x.Nspin() == 4) {
#ifdef NSPIN4
    value = genericReduce<ReduceType, Float, zFloat, 4, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 2) {
#ifdef NSPIN2
    value = genericReduce<ReduceType, Float, zFloat, 2, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 1) {
#ifdef NSPIN1
    value = genericReduce<ReduceType, Float, zFloat, 1, order, writeX, writeY, writeZ, writeW, writeV, R>(
        x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else {
    errorQuda("nSpin = %d not implemented", x.Nspin());
  }
  return value;
}

template <typename doubleN, typename ReduceType, typename Float, typename zFloat, int writeX, int writeY, int writeZ,
    int writeW, int writeV, typename R>
doubleN genericReduce(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  ReduceType value;
  ::quda::zero(value);
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    value = genericReduce<ReduceType, Float, zFloat, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, writeX, writeY, writeZ, writeW,
        writeV, R>(x, y, z, w, v, r);
  } else {
    warningQuda("CPU reductions not implemented for %d field order", x.FieldOrder());
  }
  return set(value);
}
