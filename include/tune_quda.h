#ifndef _TUNE_QUDA_H
#define _TUNE_QUDA_H

#include <quda_internal.h>
#include <dirac_quda.h>

// Curried wrappers to Cuda functions used for auto-tuning
class TuneBase {

 protected:
  const char *name;
  QudaVerbosity verbose;

 public:
  TuneBase(const char *name, QudaVerbosity verbose) : 
    name(name), verbose(verbose) { ; }
   
  virtual ~TuneBase() { ; }
  virtual void Apply() const = 0;
  virtual unsigned long long Flops() const = 0;
 
  const char* Name() const { return name; }

  // Varies the block size of the given function and finds the performance maxiumum
  void Benchmark(TuneParam &tune); 
};

class TuneDiracWilsonDslash : public TuneBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracWilsonDslash(const DiracWilson &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneBase("DiracWilsonDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracWilsonDslash() { ; }

  void Apply() const { dirac.DiracWilson::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracWilsonDslashXpay : public TuneBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracWilsonDslashXpay(const DiracWilson &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracWilsonDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracWilsonDslashXpay() { ; }

  void Apply() const { dirac.DiracWilson::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracClover : public TuneBase {

 private:
  const DiracClover &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracClover(const DiracClover &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneBase("DiracClover", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracClover() { ; }

  void Apply() const { dirac.Clover(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslash : public TuneBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracCloverDslash(const DiracCloverPC &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneBase("DiracCloverDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracCloverDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslashXpay : public TuneBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracCloverDslashXpay(const DiracCloverPC &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracCloverDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracCloverDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMass : public TuneBase {

 private:
  const DiracTwistedMass &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMass(const DiracTwistedMass &d, cudaColorSpinorField &a, 
		       const cudaColorSpinorField &b) : 
  TuneBase("DiracTwistedMass", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMass() { ; }

  void Apply() const { dirac.Twist(a, b); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslash : public TuneBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMassDslash(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
			     const cudaColorSpinorField &b) : 
  TuneBase("DiracTwistedMassDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMassDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslashXpay : public TuneBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracTwistedMassDslashXpay(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
				 const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracTwistedMassDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracTwistedMassDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslash : public TuneBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracDomainWallDslash(const DiracDomainWall &d, cudaColorSpinorField &a, 
			    const cudaColorSpinorField &b) : 
  TuneBase("DiracDomainWallDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracDomainWallDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslashXpay : public TuneBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracDomainWallDslashXpay(const DiracDomainWall &d, cudaColorSpinorField &a, 
				const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracDomainWallDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracDomainWallDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslash : public TuneBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracStaggeredDslash(const DiracStaggered &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneBase("DiracStaggeredDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracStaggeredDslash() { ; }

  void Apply() const { dirac.DiracStaggered::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslashXpay : public TuneBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracStaggeredDslashXpay(const DiracStaggered &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneBase("DiracStaggeredDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracStaggeredDslashXpay() { ; }

  void Apply() const { dirac.DiracStaggered::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

#endif // _TUNE_QUDA_H
