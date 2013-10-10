#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;

  // Applies the grid restriction operator (fine to coarse)
  template <class CoarseSpinor, class FineSpinor>
  void restrict(CoarseSpinor &out, const FineSpinor &in, const int* geo_map, const int* spin_map) {

    // We need to zero all elements first, since this is a reduction operation
    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) = 0.0;
	}
      }
    }

    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) += in(x, s, c);
	}
      }
    }

  }

  /*
    Rotates from the fine-color basis into the coarse-color basis.
  */
  template <class CoarseColor, class FineColor, class Rotator>
  void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<out.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) {
	    out(x, s, i) += quda::conj(V(x, s, j, i)) * in(x, s, j);
	  }
	}
      }
    }

  }

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *geo_map, const int *spin_map) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *outOrder = createOrder<double>(out);
      FieldOrder<double> *inOrder = createOrder<double>(in);
      FieldOrder<double> *vOrder = createOrder<double>(v, Nvec);
      FieldOrder<double> *tmpOrder = createOrder<double>(tmp);
      rotateCoarseColor(*tmpOrder, *inOrder, *vOrder);
      restrict(*outOrder, *tmpOrder, geo_map, spin_map);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
      FieldOrder<float> *outOrder = createOrder<float>(out);
      FieldOrder<float> *inOrder = createOrder<float>(in);
      FieldOrder<float> *vOrder = createOrder<float>(v, Nvec);
      FieldOrder<float> *tmpOrder = createOrder<float>(tmp);
      rotateCoarseColor(*tmpOrder, *inOrder, *vOrder);
      restrict(*outOrder, *tmpOrder, geo_map, spin_map);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }
  }

} // namespace quda
