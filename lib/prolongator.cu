#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;

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
  // Applies the grid prolongation operator (coarse to fine)
  template <class FineSpinor, class CoarseSpinor>
  __global__ void prolongate(FineSpinor &out, const CoarseSpinor &in, const int *geo_map, const int *spin_map) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    for (int s=0; s<out.Nspin(); s++) {
      for (int c=0; c<out.Ncolor(); c++) {
	out(x, s, c) = in(geo_map[x], spin_map[s], c);
      }
    }

    }*/
  
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
	    // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	    out(x, s, i) += V(x, s, i, j) * in(x, s, j);
	  }
	}
      }
      
    }

  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, int Nvec, const int *geo_map, const int *spin_map) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *outOrder = createOrder<double>(out);
      FieldOrder<double> *inOrder = createOrder<double>(in);
      FieldOrder<double> *vOrder = createOrder<double>(v, Nvec);
      FieldOrder<double> *tmpOrder = createOrder<double>(tmp);
      prolongate(*tmpOrder, *inOrder, geo_map, spin_map);
      rotateFineColor(*outOrder, *tmpOrder, *vOrder);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
      FieldOrder<float> *outOrder = createOrder<float>(out);
      FieldOrder<float> *inOrder = createOrder<float>(in);
      FieldOrder<float> *vOrder = createOrder<float>(v, Nvec);
      FieldOrder<float> *tmpOrder = createOrder<float>(tmp);
      prolongate(*tmpOrder, *inOrder, geo_map, spin_map);
      rotateFineColor(*outOrder, *tmpOrder, *vOrder);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }

  }

} // end namespace quda
