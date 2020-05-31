/**
   Generic blas kernel with four loads and up to four stores.
  */
template <typename Float, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Functor>
void genericBlas(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Functor f)
{
  for (int parity = 0; parity < X.Nparity(); parity++) {
    for (int x = 0; x < X.VolumeCB(); x++) {
      for (int s = 0; s < X.Nspin(); s++) {
        for (int c = 0; c < X.Ncolor(); c++) {
          complex<Float> X_(X(parity, x, s, c));
          complex<Float> Y_ = Y(parity, x, s, c);
          complex<Float> Z_ = Z(parity, x, s, c);
          complex<Float> W_ = W(parity, x, s, c);
          complex<Float> V_ = V(parity, x, s, c);
          f(X_, Y_, Z_, W_, V_);
          if (f.write.X) X(parity, x, s, c) = X_;
          if (f.write.Y) Y(parity, x, s, c) = Y_;
          if (f.write.Z) Z(parity, x, s, c) = Z_;
          if (f.write.W) W(parity, x, s, c) = W_;
          if (f.write.V) V(parity, x, s, c) = V_;
        }
      }
    }
  }
}

template <typename Float, typename yFloat, int nSpin, int nColor, QudaFieldOrder order, typename Functor>
void genericBlas(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, Functor f)
{
  colorspinor::FieldOrderCB<Float, nSpin, nColor, 1, order> X(x), Z(z), W(w);
  colorspinor::FieldOrderCB<yFloat, nSpin, nColor, 1, order> Y(y), V(v);
  genericBlas<yFloat>(X, Y, Z, W, V, f);
}

template <typename Float, typename yFloat, int nSpin, QudaFieldOrder order, typename Functor>
void genericBlas(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, Functor f)
{
  if (x.Ncolor() != 3 && x.Nspin() != 2) errorQuda("Unsupported nSpin = %d and nColor = %d combination", x.Ncolor(), x.Nspin());
  if (x.Ncolor() == 3) {
    genericBlas<Float, yFloat, nSpin, 3, order, Functor>(x, y, z, w, v, f);
#ifdef GPU_MULTIGRID
#ifdef NSPIN4
  } else if (x.Ncolor() == 6) { // free field Wilson
    genericBlas<Float, yFloat, 2, 6, order, Functor>(x, y, z, w, v, f);
#endif
  } else if (x.Ncolor() == 24) {
    genericBlas<Float, yFloat, 2, 24, order, Functor>(x, y, z, w, v, f);
#ifdef NSPIN4
  } else if (x.Ncolor() == 32) {
    genericBlas<Float, yFloat, 2, 32, order, Functor>(x, y, z, w, v, f);
#endif // NSPIN4
#ifdef NSPIN1
  } else if (x.Ncolor() == 64) {
    genericBlas<Float, yFloat, 2, 64, order, Functor>(x, y, z, w, v, f);
  } else if (x.Ncolor() == 96) {
    genericBlas<Float, yFloat, 2, 96, order, Functor>(x, y, z, w, v, f);
#endif // NSPIN1
#endif
  } else {
    errorQuda("nColor = %d not implemented", x.Ncolor());
  }
}

template <typename Float, typename yFloat, QudaFieldOrder order, typename Functor>
void genericBlas(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, Functor f)
{
  if (x.Nspin() == 4) {
#ifdef NSPIN4
    genericBlas<Float, yFloat, 4, order, Functor>(x, y, z, w, v, f);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 2) {
#ifdef NSPIN2
    genericBlas<Float, yFloat, 2, order, Functor>(x, y, z, w, v, f);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  } else if (x.Nspin() == 1) {
#ifdef NSPIN1
    genericBlas<Float, yFloat, 1, order, Functor>(x, y, z, w, v, f);
#else
    errorQuda("nSpin = %d not enabled", x.Nspin());
#endif
  }
}

template <typename Float, typename yFloat, typename Functor>
void genericBlas(
    ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, Functor f)
{
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    genericBlas<Float, yFloat, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, Functor>(x, y, z, w, v, f);
  } else {
    errorQuda("Not implemented");
  }
}
