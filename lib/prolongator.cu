#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  template <typename Float>
  ColorSpinorFieldOrder<Float>* createOrder(const cpuColorSpinorField &a) {
    ColorSpinorFieldOrder<Float>* ptr=0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      ptr = new SpaceSpinColorOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      ptr = new SpaceColorSpinOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      ptr = new QOPDomainWallOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", a.FieldOrder());
    return ptr;
  }

  // Applies the grid prolongation operator (coarse to fine)
  template <class FineSpinor, class CoarseSpinor>
  void prolongate(FineSpinor &out, const CoarseSpinor &in, const int *geo_map, const int *spin_map) {

    for (int x=0; x<out.Volume(); x++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int c=0; c<out.Ncolor(); c++) {
	  out(x, s, c) = in(geo_map[x], spin_map[s], c);
	}
      }
    }

  }
  
  /*
    Rotates from the coarse-color basis into the fine-color basis.  This
    is the second step of applying the prolongator.
  */
  template <class FineColor, class CoarseColor, class Rotator>
  void rotateFineColor(FineColor &out, const CoarseColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<in.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) { 
	    // V is a ColorMatrixField with dimension
	    // [out.Nc][in.Ns*in.Nc] (the rotation has spin dependence)
	    out(x, s, i) += V(x, i, s*in.Ncolor() + j) * in(x, s, j);
	  }
	}
      }
      
    }

  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, const int *geo_map, const int *spin_map) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *outOrder = createOrder<double>(out);
      ColorSpinorFieldOrder<double> *inOrder = createOrder<double>(in);
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(v);
      ColorSpinorFieldOrder<double> *tmpOrder = createOrder<double>(tmp);
      prolongate(*tmpOrder, *inOrder, geo_map, spin_map);
      rotateFineColor(*outOrder, *tmpOrder, *vOrder);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
      ColorSpinorFieldOrder<float> *outOrder = createOrder<float>(out);
      ColorSpinorFieldOrder<float> *inOrder = createOrder<float>(in);
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(v);
      ColorSpinorFieldOrder<float> *tmpOrder = createOrder<float>(tmp);
      prolongate(*tmpOrder, *inOrder, geo_map, spin_map);
      rotateFineColor(*outOrder, *tmpOrder, *vOrder);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }
  }

} // end namespace quda
