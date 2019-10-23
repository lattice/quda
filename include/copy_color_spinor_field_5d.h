#include <color_spinor_field.h>
#include <complex_quda.h>

namespace quda {

void transfer_5d_hh(ColorSpinorField& out, const ColorSpinorField& in, const complex<float>* transfer_matrix, bool dagger);

void tensor_5d_hh(ColorSpinorField& out, const ColorSpinorField& in, complex<float>* tensor_matrix);

}
