#include "dslash_test_helpers.h"
#include <quda.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <quda_internal.h>

using namespace quda;

// need a better solution here but as long as they gauge field live in interface probably ok
extern cudaGaugeField *gaugePrecise;
extern cudaGaugeField *gaugeFatPrecise;
extern cudaGaugeField *gaugeLongPrecise;

void dslashQuda_4dpc(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, dslash_test_type test_type)
{
  const auto &gauge = (inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
      || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
    errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gauge.X(), true, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    blas::ax(gauge.Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  DiracDomainWall4DPC dirac(diracParam); // create the Dirac operator
  printfQuda("kappa for QUDA input : %e\n", inv_param->kappa);
  switch (test_type) {
  case dslash_test_type::Dslash: dirac.Dslash4(out, in, parity); break;
  case dslash_test_type::M5: dirac.Dslash5(out, in, parity); break;
  case dslash_test_type::M5inv: dirac.Dslash5inv(out, in, parity, inv_param->kappa); break;
  default: errorQuda("Unsupported dslash_test_type in dslashQuda_4dpc.");
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

void dslashQuda_mdwf(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, dslash_test_type test_type)
{
  const auto &gauge = (inv_param->dslash_type != QUDA_ASQTAD_DSLASH) ? *gaugePrecise : *gaugeFatPrecise;

  if ((!gaugePrecise && inv_param->dslash_type != QUDA_ASQTAD_DSLASH)
      || ((!gaugeFatPrecise || !gaugeLongPrecise) && inv_param->dslash_type == QUDA_ASQTAD_DSLASH))
    errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, *inv_param, gauge.X(), true, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    blas::ax(gauge.Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  DiracMobiusPC dirac(diracParam); // create the Dirac operator
  switch (test_type) {
  case dslash_test_type::Dslash: dirac.Dslash4(out, in, parity); break;
  case dslash_test_type::M5: dirac.Dslash5(out, in, parity); break;
  case dslash_test_type::Dslash4pre: dirac.Dslash4pre(out, in, parity); break;
  case dslash_test_type::M5inv: dirac.Dslash5inv(out, in, parity); break;
  default: errorQuda("Unsupported dslash_test_type in dslashQuda_mdwf.");
  }

  cpuParam.v = h_out;
  cpuParam.location = inv_param->output_location;
  ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
  *out_h = out;

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*out_h);
    double gpu = blas::norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

void dslashQuda_mobius_eofa(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity,
                            dslash_test_type test_type)
{
  if (inv_param->dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    setKernelPackT(true);
  } else {
    errorQuda("This type of dslashQuda operator is defined for QUDA_MOBIUS_DWF_EOFA_DSLASH ONLY");
  }

  if (gaugePrecise == nullptr) errorQuda("Gauge field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool precondition_output = test_type == dslash_test_type::Dslash ? false : true;

  ColorSpinorParam cpuParam(h_in, *inv_param, gaugePrecise->X(), precondition_output, inv_param->input_location);
  ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    double cpu = blas::norm2(*in_h);
    double gpu = blas::norm2(in);
    printfQuda("In CPU: %16.12e CUDA: %16.12e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    blas::ax(gaugePrecise->Anisotropy(), in);
  }
  constexpr bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  DiracMobiusEofaPC dirac(diracParam); // create the Dirac operator
  switch (test_type) {
  case dslash_test_type::MatPC: dirac.M(out, in); break;
  case dslash_test_type::M5: dirac.m5_eofa(out, in); break;
  case dslash_test_type::M5inv: dirac.m5inv_eofa(out, in); break;
  default: errorQuda("test_type(=%d) NOT defined for M\"obius EOFA! :( \n", static_cast<int>(test_type));
  }
}
