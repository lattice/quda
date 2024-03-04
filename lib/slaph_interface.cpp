#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <contract_quda.h>

using namespace quda;

// Delclaration of function in interface_quda.cpp
TimeProfile &getProfileSinkProject();

void laphSinkProject(double _Complex *host_sinks, void **host_quark, unsigned int nQuark, void **host_evec,
                     unsigned int nEv, QudaInvertParam *inv_param, const int X[4])
{
  auto profile = pushProfile(getProfileSinkProject());

  // Parameter object describing the sources and smeared quarks
  lat_dim_t x = {X[0], X[1], X[2], X[3]};
  ColorSpinorParam cpu_quark_param(host_quark, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> quark(nQuark);
  for (auto i = 0u; i < nQuark; i++) {
    cpu_quark_param.v = host_quark[i];
    quark[i] = ColorSpinorField(cpu_quark_param);
  }

  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, *inv_param, x, false, QUDA_CPU_FIELD_LOCATION);
  // Switch to spin 1
  cpu_evec_param.nSpin = 1;
  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField> evec(nEv);
  for (auto i = 0u; i < nEv; i++) {
    cpu_evec_param.v = host_evec[i];
    evec[i] = ColorSpinorField(cpu_evec_param);
  }

  // Create device vectors
  ColorSpinorParam quda_quark_param(cpu_quark_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  std::vector<ColorSpinorField> quda_quark(1, quda_quark_param);

  // Create device vectors for evecs
  ColorSpinorParam quda_evec_param(cpu_evec_param, *inv_param, QUDA_CUDA_FIELD_LOCATION);
  std::vector<ColorSpinorField> quda_evec(1, quda_evec_param);

  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }

  auto Lt = x[3] * comm_dim(3);
  std::vector<Complex> hostSink(4 * Lt);

  for (auto i = 0u; i < nQuark; i++) { // iterate over all quarks
    // Copy quark field from host to device
    quda_quark[0] = quark[i];

    for (auto j = 0u; j < nEv; j++) { // iterate over all EV
      quda_evec[0] = evec[j];

      // We now perfrom the projection onto the eigenspace. The data
      // is placed in host_sinks in  T, spin order
      evecProjectLaplace3D(hostSink, quda_quark[0], quda_evec[0]);

      for (auto k = 0u; k < hostSink.size(); k++)
        reinterpret_cast<std::complex<double> *>(host_sinks)[(i * nEv + j) * 4 * Lt + k] = hostSink[k];
    }
  }

  comm_allreduce_sum(hostSink);
}
