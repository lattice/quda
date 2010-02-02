#ifndef _QUDA_DIRAC_H
#define _QUDA_DIRAC_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>

// Params for Dirac operator
class DiracParam {

 public:
  QudaDiracType type;
  double kappa;
  MatPCType matpcType;
  FullGauge *gauge;
  FullClover *clover;
  FullClover *cloverInv;
  cudaColorSpinorField *tmp;
  QudaVerbosity verbose;

 DiracParam() 
   : type(QUDA_INVALID_DIRAC), kappa(0.0), matpcType(QUDA_MATPC_INVALID),
    gauge(0), clover(0), cloverInv(0), tmp(0), verbose(QUDA_SILENT)
  {

  }

};

void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param);
void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param);

// Abstract base class
class Dirac {

 protected:
  FullGauge &gauge;
  double kappa;
  MatPCType matpcType;
  unsigned long long flops;

 public:
  Dirac(const DiracParam &param);
  Dirac(const Dirac &dirac);
  virtual ~Dirac();
  Dirac& operator=(const Dirac &dirac);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &);
  virtual void checkFullSpinor(const cudaColorSpinorField &, const cudaColorSpinorField &);

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const int parity, const QudaDagType) = 0;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const int parity, const QudaDagType,
			  const cudaColorSpinorField &tmp, const double &k) = 0;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		 const QudaDagType = QUDA_DAG_NO) = 0;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) = 0;

  // required methods to use e-o preconditioning for solving full system
  virtual void Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
		       const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO) = 0;
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO) = 0;

  // Dirac operator factory
  static Dirac* create(const DiracParam &param);
};

// Full Wilson
class DiracWilson : public Dirac {

 protected:

 public:
  DiracWilson(const DiracParam &param);
  DiracWilson(const DiracWilson &dirac);
  virtual ~DiracWilson();
  DiracWilson& operator=(const DiracWilson &dirac);

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const int parity, const QudaDagType);
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const int parity, const QudaDagType,
			  const cudaColorSpinorField &tmp, const double &k);
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  virtual void Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
		       const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Even-Odd preconditioned Wilson
class DiracWilsonPC : public DiracWilson {

 private:
  cudaColorSpinorField &tmp; // used when applying the operator

 public:
  DiracWilsonPC(const DiracParam &param);
  DiracWilsonPC(const DiracWilsonPC &dirac);
  virtual ~DiracWilsonPC();
  DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  void Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
	       const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
	       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Full clover
class DiracClover : public DiracWilson {

 protected:
  FullClover &clover;
  void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &, 
			 const FullClover &);
  void cloverApply(cudaColorSpinorField &out, const FullClover &clover, const cudaColorSpinorField &in, 
		   const int parity);

 public:
  DiracClover(const DiracParam &param);
  DiracClover(const DiracClover &dirac);
  virtual ~DiracClover();
  DiracClover& operator=(const DiracClover &dirac);

  void Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int parity);
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  virtual void Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
		       const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Even-Odd preconditioned clover
class DiracCloverPC : public DiracClover {

 private:
  FullClover &cloverInv;
  cudaColorSpinorField &tmp;

 public:
  DiracCloverPC(const DiracParam &param);
  DiracCloverPC(const DiracCloverPC &dirac);
  virtual ~DiracCloverPC();
  DiracCloverPC& operator=(const DiracCloverPC &dirac);

  void CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int parity);  
  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
	      const int parity, const QudaDagType);
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  const int parity, const QudaDagType,
		  const cudaColorSpinorField &tmp, const double &k);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  void Prepare(cudaColorSpinorField &src, cudaColorSpinorField &sol,
	       const cudaColorSpinorField &x, const cudaColorSpinorField &b, 
	       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

#endif // _QUDA_DIRAC_H
