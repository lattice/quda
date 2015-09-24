#include <multigrid.h>
#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>

namespace quda {

  //Apply the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float, int nDim, typename F, typename Gauge>
  void coarseDslash(F &out, F &in, Gauge &Y, Gauge &X, Float kappa) {
    const int Nc = in.Ncolor();
    const int Ns = in.Nspin();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) x_size[d] = in.X(d);

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      for(int x_cb = 0; x_cb < in.Volume()/2; x_cb++) { //Volume
	int coord[QUDA_MAX_DIM];
	in.LatticeIndex(coord,parity*in.Volume()/2+x_cb);

	for(int s = 0; s < Ns; s++) for(int c = 0; c < Nc; c++) out(parity, x_cb, s, c) = (Float)0.0; 

	for(int d = 0; d < nDim; d++) { //Ndim
	  //Forward link - compute fwd offset for spinor fetch
	  int coordTmp = coord[d];
	  coord[d] = (coord[d] + 1)%x_size[d];
	  int fwd_idx = 0;
	  for(int dim = nDim-1; dim >= 0; dim--) fwd_idx = x_size[dim] * fwd_idx + coord[dim];
	  coord[d] = coordTmp;

	  for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	    for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	      for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
		Float sign = (s_row == s_col) ? 1.0 : -1.0;    
		for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		  out(parity, x_cb, s_row, c_row) += sign*Y(d, parity, x_cb, s_row, s_col, c_row, c_col)
		    * in((parity+1)&1, fwd_idx/2, s_col, c_col);
		} //Color column
	      } //Spin column
	    } //Color row
	  } //Spin row 

	  //Backward link - compute back offset for spinor and gauge fetch
	  int back_idx = 0;
	  coord[d] = (coordTmp - 1 + x_size[d])%x_size[d];
	  for(int dim = nDim-1; dim >= 0; dim--) back_idx = x_size[dim] * back_idx + coord[dim];
	  coord[d] = coordTmp;

	  for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	    for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	      for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
		for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		  out(parity, x_cb, s_row, c_row) += conj(Y(d,(parity+1)&1, back_idx/2, s_col, s_row, c_col, c_row))
		    * in((parity+1)&1, back_idx/2, s_col, c_col);
		} //Color column
	      } //Spin column
	    } //Color row
	  } //Spin row 
	} //nDim

	// apply kappa
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out(parity, x_cb, s, c) *= -(Float)2.0*kappa;

	// apply clover term
	for(int s = 0; s < Ns; s++) { //Spin out
	  for(int c = 0; c < Nc; c++) { //Color out
	    //This term is now incorporated into the matrix X.
	    //out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	        //Factor of 2*kappa now incorporated in X
		//out(parity,x_cb,s,c) -= 2*kappa*X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
                out(parity,x_cb,s,c) += X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
	      } //Color in
	    } //Spin in
	  } //Color out
	} //Spin out

      }//VolumeCB
    } // parity
    
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;
    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessor(const_cast<ColorSpinorField&>(in));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    coarseDslash<Float,4,F,G>(outAccessor, inAccessor, yAccessor, xAccessor, (Float)kappa);
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (in.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",in.Nspin());

    if (in.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, in, Y, X, kappa);
    } else if (in.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder fOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_QDP_GAUGE_ORDER>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", Y.FieldOrder());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", in.FieldOrder());
    }
  }


  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (Y.Precision() != in.Precision() || X.Precision() != Y.Precision() || Y.Precision() != out.Precision())
      errorQuda("Unsupported precision mix");

    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (out.Precision() != in.Precision() ||
	Y.Precision() != in.Precision() ||
	X.Precision() != in.Precision()) 
      errorQuda("Precision mismatch out=%d in=%d Y=%d X=%d", 
		out.Precision(), in.Precision(), Y.Precision(), X.Precision());

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, in, Y, X, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
  }//ApplyCoarse

}
