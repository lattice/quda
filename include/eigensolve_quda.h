#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

#include <quda_arpack_interface.h>

namespace quda {

  void irlmSolve(std::vector<ColorSpinorField*> kSpace,
		 std::vector<Complex> &evals, const Dirac &mat,
		 QudaEigParam *eig_param);

  void iramSolve(std::vector<ColorSpinorField*> kSpace,
		 std::vector<Complex> &evals, const Dirac &mat,
		 QudaEigParam *eig_param);

}
