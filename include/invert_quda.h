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
  CG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam);
  virtual ~CG();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

class BiCGstab : public Solver {

 private:
  DiracMatrix &mat;
  const DiracMatrix &matSloppy;
  const DiracMatrix &matPrecon;

  // pointers to fields to avoid multiple creation overhead
  cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp, *wp, *zp;
  bool init;

 public:
  BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
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
  GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
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
  bool allocate_r;

 public:
  MR(DiracMatrix &mat, QudaInvertParam &invParam);
  virtual ~MR();

  void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
};

// multigrid solver
class alphaSA : public Solver {

 protected:
  const DiracMatrix &mat;

 public:
  alphaSA(DiracMatrix &mat, QudaInvertParam &invParam);
  virtual ~alphaSA() { ; }

  void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
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
  MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam);
  virtual ~MultiShiftCG();

  void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
};

#endif // _INVERT_QUDA_H
