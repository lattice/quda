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
  double mass;
  MatPCType matpcType;
  FullGauge *gauge;
  FullClover *clover;
  FullClover *cloverInv;
  cudaColorSpinorField *tmp1;
  cudaColorSpinorField *tmp2; // used only by Clover operators
  
  FullGauge* fatGauge; //used by staggered only
  FullGauge* longGauge;//used by staggered only
  
  QudaVerbosity verbose;

 DiracParam() 
   : type(QUDA_INVALID_DIRAC), kappa(0.0), matpcType(QUDA_MATPC_INVALID),
   gauge(0), clover(0), cloverInv(0), tmp1(0), tmp2(0), verbose(QUDA_SILENT)
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
  double mass;
  MatPCType matpcType;
  unsigned long long flops;
  cudaColorSpinorField *tmp1;
  cudaColorSpinorField *tmp2;

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
			  const cudaColorSpinorField &x, const double &k) = 0;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		 const QudaDagType = QUDA_DAG_NO) = 0;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) = 0;

  // required methods to use e-o preconditioning for solving full system
  virtual void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO) = 0;
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO) = 0;

  // Dirac operator factory
  static Dirac* create(const DiracParam &param);

  unsigned long long Flops() { unsigned long long rtn = flops; flops = 0; return rtn; }
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
			  const cudaColorSpinorField &x, const double &k);
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  virtual void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Even-Odd preconditioned Wilson
class DiracWilsonPC : public DiracWilson {

 private:

 public:
  DiracWilsonPC(const DiracParam &param);
  DiracWilsonPC(const DiracWilsonPC &dirac);
  virtual ~DiracWilsonPC();
  DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
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

  virtual void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Even-Odd preconditioned clover
class DiracCloverPC : public DiracClover {

 private:
  FullClover &cloverInv;

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
		  const cudaColorSpinorField &x, const double &k);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};

// Full Wilson
class DiracStaggered : public Dirac {

 protected:
    FullGauge* fatGauge;
    FullGauge* longGauge;

 public:
  DiracStaggered(const DiracParam &param);
  DiracStaggered(const DiracStaggered &dirac);
  virtual ~DiracStaggered();
  DiracStaggered& operator=(const DiracStaggered &dirac);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &);
  
  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const int parity, const QudaDagType);
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const int parity, const QudaDagType,
			  const cudaColorSpinorField &x, const double &k);
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaDagType = QUDA_DAG_NO);
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in);

  virtual void Prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
  virtual void Reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType, const QudaDagType dagger = QUDA_DAG_NO);
};


#endif // _QUDA_DIRAC_H
