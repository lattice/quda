#ifndef _DIRAC_QUDA_H
#define _DIRAC_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash_quda.h>

#include <face_quda.h>

// Params for Dirac operator
class DiracParam {

 public:
  QudaDiracType type;
  double kappa;
  double mass;
  double m5; // used by domain wall only
  MatPCType matpcType;
  DagType dagger;
  FullGauge *gauge;
  FullGauge *fatGauge;  // used by staggered only
  FullGauge *longGauge; // used by staggered only
  cudaCloverField *clover;
  
  double mu; // used by twisted mass only

  cudaColorSpinorField *tmp1;
  cudaColorSpinorField *tmp2; // used only by Clover and TM

  QudaVerbosity verbose;

  int commDim[QUDA_MAX_DIM]; // whether to do comms or not

  DiracParam() 
    : type(QUDA_INVALID_DIRAC), kappa(0.0), m5(0.0), matpcType(QUDA_MATPC_INVALID),
    dagger(QUDA_DAG_INVALID), gauge(0), clover(0), mu(0.0), 
    tmp1(0), tmp2(0), verbose(QUDA_SILENT)
  {

  }

};

void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);
void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);

// forward declarations
class DiracMatrix;
class DiracM;
class DiracMdagM;
class DiracMdag;

// Abstract base class
class Dirac {

  friend class DiracMatrix;
  friend class DiracM;
  friend class DiracMdagM;
  friend class DiracMdag;

  protected:
  FullGauge &gauge;
  double kappa;
  double mass;
  MatPCType matpcType;
  mutable DagType dagger; // mutable to simplify implementation of Mdag
  mutable unsigned long long flops;
  mutable cudaColorSpinorField *tmp1; // temporary hack
  mutable cudaColorSpinorField *tmp2; // temporary hack

  bool newTmp(cudaColorSpinorField **, const cudaColorSpinorField &) const;
  void deleteTmp(cudaColorSpinorField **, const bool &reset) const;

  QudaTune tune;
  QudaVerbosity verbose;  

  int commDim[QUDA_MAX_DIM]; // whether do comms or not

 public:
  Dirac(const DiracParam &param);
  Dirac(const Dirac &dirac);
  virtual ~Dirac();
  Dirac& operator=(const Dirac &dirac);

  // Autotunes the block sizes for optimum performance
  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &) = 0;

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  virtual void checkFullSpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  void checkSpinorAlias(const cudaColorSpinorField &, const cudaColorSpinorField &) const;

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const = 0;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x,
			  const double &k) const = 0;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  void Mdag(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  // required methods to use e-o preconditioning for solving full system
  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const = 0;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const = 0;
  void setMass(double mass){ this->mass = mass;}
  // Dirac operator factory
  static Dirac* create(const DiracParam &param);

  unsigned long long Flops() const { unsigned long long rtn = flops; flops = 0; return rtn; }
  QudaVerbosity Verbose() const { return verbose; }
};

// Full Wilson
class DiracWilson : public Dirac {

 private:
  dim3 blockDslash[5]; // thread block size for Dslash (full volume or just body for overlapping comms)
  dim3 blockDslashXpay[5]; // thread block size for DslashXpay (full volume or just body for overlapping comms)

 protected:
  FaceBuffer face; // multi-gpu communication buffers

 public:
  DiracWilson(const DiracParam &param);
  DiracWilson(const DiracWilson &dirac);
  virtual ~DiracWilson();
  DiracWilson& operator=(const DiracWilson &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// Even-odd preconditioned Wilson
class DiracWilsonPC : public DiracWilson {

 private:

 public:
  DiracWilsonPC(const DiracParam &param);
  DiracWilsonPC(const DiracWilsonPC &dirac);
  virtual ~DiracWilsonPC();
  DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
};

// Full clover
class DiracClover : public DiracWilson {

 protected:
  cudaCloverField &clover;
  dim3 blockClover; // thread block size for applying clover (or inverse) term
  void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &, 
			 const cudaCloverField &) const;

 public:
  DiracClover(const DiracParam &param);
  DiracClover(const DiracClover &dirac);
  virtual ~DiracClover();
  DiracClover& operator=(const DiracClover &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  void Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// Even-odd preconditioned clover
class DiracCloverPC : public DiracClover {

 private:
  dim3 blockDslash[5]; // thread block size for Dslash (full volume or just body for overlapping comms)
  dim3 blockDslashXpay[5]; // thread block size for DslashXpay (full volume or just body for overlapping comms)

 public:
  DiracCloverPC(const DiracParam &param);
  DiracCloverPC(const DiracCloverPC &dirac);
  virtual ~DiracCloverPC();
  DiracCloverPC& operator=(const DiracCloverPC &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  void CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
	      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
};



// Full domain wall 
class DiracDomainWall : public DiracWilson {

 private:
  dim3 blockDslash[5]; // thread block size for Dslash (full volume or just body for overlapping comms)
  dim3 blockDslashXpay[5]; // thread block size for DslashXpay (full volume or just body for overlapping comms)

 protected:
  double m5;
  double kappa5;

 public:
  DiracDomainWall(const DiracParam &param);
  DiracDomainWall(const DiracDomainWall &dirac);
  virtual ~DiracDomainWall();
  DiracDomainWall& operator=(const DiracDomainWall &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
	      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// 5d Even-odd preconditioned domain wall
class DiracDomainWallPC : public DiracDomainWall {

 private:

 public:
  DiracDomainWallPC(const DiracParam &param);
  DiracDomainWallPC(const DiracDomainWallPC &dirac);
  virtual ~DiracDomainWallPC();
  DiracDomainWallPC& operator=(const DiracDomainWallPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
};

// Full twisted mass
class DiracTwistedMass : public DiracWilson {

 private:
  dim3 blockTwist; // thread block size for applying the twist kernel

 protected:
  double mu;
  void twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaTwistGamma5Type twistType) const;

 public:
  DiracTwistedMass(const DiracParam &param);
  DiracTwistedMass(const DiracTwistedMass &dirac);
  virtual ~DiracTwistedMass();
  DiracTwistedMass& operator=(const DiracTwistedMass &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  void Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// Even-odd preconditioned twisted mass
class DiracTwistedMassPC : public DiracTwistedMass {

 private:
  dim3 blockDslash[5]; // thread block size for Dslash (full volume or just body for overlapping comms)
  dim3 blockDslashXpay[5]; // thread block size for DslashXpay (full volume or just body for overlapping comms)

 public:
  DiracTwistedMassPC(const DiracParam &param);
  DiracTwistedMassPC(const DiracTwistedMassPC &dirac);
  virtual ~DiracTwistedMassPC();
  DiracTwistedMassPC& operator=(const DiracTwistedMassPC &dirac);

  virtual void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  void TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
};

// Full staggered
class DiracStaggered : public Dirac {

 private:
  dim3 blockDslash[5]; // thread block size for Dslash (body + face kernels)
  dim3 blockDslashXpay[5]; // thread block size for DslashXpay (body + face kernels)

 protected:
  FullGauge *fatGauge;
  FullGauge *longGauge;
  FaceBuffer face; // multi-gpu communication buffers

 public:
  DiracStaggered(const DiracParam &param);
  DiracStaggered(const DiracStaggered &dirac);
  virtual ~DiracStaggered();
  DiracStaggered& operator=(const DiracStaggered &dirac);

  void Tune(cudaColorSpinorField &, const cudaColorSpinorField &, const cudaColorSpinorField &);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  
  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// Even-odd preconditioned staggered
class DiracStaggeredPC : public DiracStaggered {

 protected:

 public:
  DiracStaggeredPC(const DiracParam &param);
  DiracStaggeredPC(const DiracStaggeredPC &dirac);
  virtual ~DiracStaggeredPC();
  DiracStaggeredPC& operator=(const DiracStaggeredPC &dirac);

  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
};

// Functor base class for applying a given Dirac matrix (M, MdagM, etc.)
class DiracMatrix {

 protected:
  const Dirac *dirac;

 public:
  DiracMatrix(const Dirac &d) : dirac(&d) { }
  DiracMatrix(const Dirac *d) : dirac(d) { }
  virtual ~DiracMatrix() = 0;

  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			  cudaColorSpinorField &tmp) const = 0;
  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const = 0;

  unsigned long long flops() const { return dirac->Flops(); }
};

inline DiracMatrix::~DiracMatrix()
{

}

class DiracM : public DiracMatrix {

 public:
 DiracM(const Dirac &d) : DiracMatrix(d) { }
 DiracM(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->M(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->M(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->M(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
};

class DiracMdagM : public DiracMatrix {

 public:
  DiracMdagM(const Dirac &d) : DiracMatrix(d) { }
  DiracMdagM(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->MdagM(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->MdagM(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->MdagM(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
};

class DiracMdag : public DiracMatrix {

 public:
  DiracMdag(const Dirac &d) : DiracMatrix(d) { }
  DiracMdag(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->Mdag(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->Mdag(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->Mdag(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
};

#endif // _DIRAC_QUDA_H
