/**
   Generic reduce kernel with four loads and up to four stores.
  */
template <typename reduce_t, typename Float, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
auto genericReduce(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Reducer r)
{
  using vec = vector_type<complex<Float>, 1>;
  reduce_t sum;
  ::quda::zero(sum);

  vec X_, Y_, Z_, W_, V_;
  for (int parity = 0; parity < X.Nparity(); parity++) {
    for (int x = 0; x < X.VolumeCB(); x++) {
      r.pre();
      for (int s = 0; s < X.Nspin(); s++) {
        for (int c = 0; c < X.Ncolor(); c++) {
          X_[0] = X(parity, x, s, c);
          Y_[0] = Y(parity, x, s, c);
          Z_[0] = Z(parity, x, s, c);
          W_[0] = W(parity, x, s, c);
          V_[0] = V(parity, x, s, c);
          r(sum, X_, Y_, Z_, W_, V_);
          if (r.write.X) X(parity, x, s, c) = X_[0];
          if (r.write.Y) Y(parity, x, s, c) = Y_[0];
          if (r.write.Z) Z(parity, x, s, c) = Z_[0];
          if (r.write.W) W(parity, x, s, c) = W_[0];
          if (r.write.V) V(parity, x, s, c) = V_[0];
        }
      }
      r.post(sum);
    }
  }

  return sum;
}

template <typename reduce_t, typename Float, typename zFloat, int nSpin, int nColor, QudaFieldOrder order, typename R>
auto genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  colorspinor::FieldOrderCB<Float, nSpin, nColor, 1, order> X(x), Y(y), W(w), V(v);
  colorspinor::FieldOrderCB<zFloat, nSpin, nColor, 1, order> Z(z);
  return genericReduce<reduce_t, zFloat>(X, Y, Z, W, V, r);
}

template <typename reduce_t, typename Float, typename zFloat, int nSpin, QudaFieldOrder order, typename R>
auto genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  if (x.Ncolor() != N_COLORS && x.Nspin() != 2) errorQuda("Unsupported nSpin = %d and nColor = %d combination", x.Ncolor(), x.Nspin());
  reduce_t value;
  if (x.Ncolor() == N_COLORS) {
    value = genericReduce<reduce_t, Float, zFloat, nSpin, N_COLORS, order, R>(x, y, z, w, v, r);
#ifdef GPU_MULTIGRID
#ifdef NSPIN4
  } else if (x.Ncolor() == 6) { // free field Wilson
    value = genericReduce<reduce_t, Float, zFloat, 2, 6, order, R>(x, y, z, w, v, r);
#endif
  } else if (x.Ncolor() == 24) {
    value = genericReduce<reduce_t, Float, zFloat, 2, 24, order, R>(x, y, z, w, v, r);
#ifdef NSPIN4
  } else if (x.Ncolor() == 32) {
    value = genericReduce<reduce_t, Float, zFloat, 2, 32, order, R>(x, y, z, w, v, r);
#endif // NSPIN4
#ifdef NSPIN1
  } else if (x.Ncolor() == 64) {
    value = genericReduce<reduce_t, Float, zFloat, 2, 64, order, R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 96) {
    value = genericReduce<reduce_t, Float, zFloat, 2, 96, order, R>(x, y, z, w, v, r);
#endif // NSPIN1
#endif
  } else {
    ::quda::zero(value);
    errorQuda("nColor = %d not implemented", x.Ncolor());
  }
  return value;
}

template <typename reduce_t,typename Float, typename zFloat, QudaFieldOrder order, typename R>
auto genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  reduce_t value;
  ::quda::zero(value);
  if (x.Nspin() == 4) {
#ifdef NSPIN4
    value = genericReduce<reduce_t, Float, zFloat, 4, order, R>(x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 2) {
#ifdef NSPIN2
    value = genericReduce<reduce_t, Float, zFloat, 2, order, R>(x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 1) {
#ifdef NSPIN1
    value = genericReduce<reduce_t, Float, zFloat, 1, order, R>(x, y, z, w, v, r);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else {
    errorQuda("nSpin = %d not implemented", x.Nspin());
  }
  return value;
}

template <typename reduce_t, typename Float, typename zFloat, typename R>
auto genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r)
{
  reduce_t value;
  ::quda::zero(value);
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    value = genericReduce<reduce_t, Float, zFloat, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, R>(x, y, z, w, v, r);
  } else {
    warningQuda("CPU reductions not implemented for %d field order", x.FieldOrder());
  }
  return value;
}
