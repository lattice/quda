#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <contract_quda.h>

using namespace quda;

// Forward declarations for profiling and parameter checking
// The helper functions are defined in interface_quda.cpp
void checkBLASParam(QudaBLASParam &param);

void createLAPHsource(void *source, void **evecs, int source_t, int source_s, int eig_n)
{ 
  
}
