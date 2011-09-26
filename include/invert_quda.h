#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

class Solver {

 protected:
  QudaInvertParam &invParam;

 public:
  Solver(QudaInvertParam &invParam) : invParam(invParam) { ; }
  virtual ~Solver() { ; }

  virtual void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;
};

class CG : public Solver {

 private:
  const DiracMatrix &mat;
  const DiracMatrix &matSloppy;

 public:
  CG(const DiracMatrix &mat, const DiracMatrix &matSloppy, QudaInvertParam &invParam);
  virtual ~CG();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

class BiCGstab : public Solver {

 private:
  const DiracMatrix &mat;
  const DiracMatrix &matSloppy;
  const DiracMatrix &matPrecon;

  // pointers to fields to avoid multiple creation overhead
  cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp, *wp, *zp;
  bool init;

 public:
  BiCGstab(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
	   QudaInvertParam &invParam);
  virtual ~BiCGstab();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

class GCR : public Solver {

 private:
  const DiracMatrix &mat;
  const DiracMatrix &matSloppy;
  const DiracMatrix &matPrecon;

  Solver *K;
  QudaInvertParam Kparam; // parameters for preconditioner solve

 public:
  GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
      QudaInvertParam &invParam);
  virtual ~GCR();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

class MR : public Solver {

 private:
  const DiracMatrix &mat;
  cudaColorSpinorField *rp;
  cudaColorSpinorField *Arp;
  cudaColorSpinorField *tmpp;
  bool init;

 public:
  MR(const DiracMatrix &mat, QudaInvertParam &invParam);
  virtual ~MR();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

class MultiShiftSolver {

 protected:
  QudaInvertParam &invParam;

 public:
  MultiShiftSolver(QudaInvertParam &invParam) : invParam(invParam) { ; }
  virtual ~MultiShiftSolver() { ; }

  virtual void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in) = 0;
};

class MultiShiftCG : public MultiShiftSolver {

 protected:
  const DiracMatrix &mat;
  const DiracMatrix &matSloppy;

 public:
  MultiShiftCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, QudaInvertParam &invParam);
  virtual ~MultiShiftCG();

  void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
};

#ifdef __cplusplus
extern "C" {
#endif

  // defined in interface_quda.cpp

  extern FullGauge cudaGaugePrecise;
  extern FullGauge cudaGaugeSloppy;

  extern FullGauge cudaFatLinkPrecise;
  extern FullGauge cudaFatLinkSloppy;

  extern FullGauge cudaLongLinkPrecise;
  extern FullGauge cudaLongLinkSloppy;

  extern FullClover cudaCloverPrecise;
  extern FullClover cudaCloverSloppy;

  extern FullClover cudaCloverInvPrecise;
  extern FullClover cudaCloverInvSloppy;

  /*
  // defined in inv_cg_cuda.cpp

  void invertCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x,
		    cudaColorSpinorField &b, QudaInvertParam *param);

  // defined in inv_multi_cg_quda.cpp

  int invertMultiShiftCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField **x, 
			     cudaColorSpinorField b, QudaInvertParam *param, double *offsets, 
			     int num_offsets, double *residue_sq);

  // defined in inv_bicgstab_cuda.cpp

  void invertBiCGstabCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &pre,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, QudaInvertParam *param);

  void freeBiCGstab();

  // defined in inv_gcr_cuda.cpp

  void invertGCRCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &pre,
		     cudaColorSpinorField &x, cudaColorSpinorField &b, QudaInvertParam *param);

  // defined in inv_mr_cuda.cpp

  void invertMRCuda(const DiracMatrix &mat, cudaColorSpinorField &x, 
		    cudaColorSpinorField &b, QudaInvertParam *param);

  void freeMR();*/

#ifdef __cplusplus
}
#endif

#endif // _INVERT_QUDA_H
