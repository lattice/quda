#ifndef _TUNE_QUDA_H
#define _TUNE_QUDA_H

#include <quda_internal.h>
#include <dirac_quda.h>

#include <string>
#include <iostream>
#include <iomanip>

class TuneKey {

 public:
  std::string volume;
  std::string name;
  std::string aux;

  TuneKey(std::string v, std::string n, std::string a=std::string())
    : volume(v), name(n), aux(a) { }  
  TuneKey(const TuneKey &key)
    : volume(key.volume), name(key.name), aux(key.aux) { }

  TuneKey& operator=(const TuneKey &key) {
    if (&key != this) {
      volume = key.volume;
      name = key.name;
      aux = key.aux;
    }
    return *this;
  }

  bool operator<(const TuneKey &other) const {
    return (volume < other.volume) ||
      ((volume == other.volume) && (name < other.name)) ||
      ((volume == other.volume) && (name == other.name) && (aux < other.aux));
  }

};


class TuneParam {

 public:
  dim3 block;
  dim3 grid;
  int shared_bytes;

  TuneParam() : block(32, 1, 1), grid(1, 1, 1), shared_bytes(0) { }
  TuneParam(const TuneParam &param)
    : block(param.block), grid(param.grid), shared_bytes(param.shared_bytes) { }
  TuneParam& operator=(const TuneParam &param) {
    if (&param != this) {
      block = param.block;
      grid = param.grid;
      shared_bytes = param.shared_bytes;
    }
    return *this;
  }

};


class Tunable {

 protected:
  virtual long long flops() const { return 1e9; } // FIXME: make pure virtual
  virtual long long bytes() const { return 1e9; } // FIXME
  virtual int sharedBytesPerThread() const = 0;
  virtual int sharedBytesPerBlock() const = 0;

  virtual bool advanceGridDim(TuneParam &param) const
  {
    return false; // FIXME: generalize for blas
  }

  virtual bool advanceBlockDim(TuneParam &param) const
  {
    const unsigned int max_threads = 512; // FIXME: use deviceProp.maxThreadsDim[0];
    const int step = 32; // FIXME: use deviceProp.warpSize;
    param.block.x += step;
    if (param.block.x > max_threads) {
      param.block.x = step;
      return false;
    } else {
      return true;
    }
  }

  /**
   * The goal here is to throttle the number of thread blocks per SM by over-allocating shared memory (in order to improve
   * L2 utilization, etc.).  Note that:
   * - On Fermi, requesting greater than 16 KB will switch the cache config, so we restrict ourselves to 16 KB for now.
   * - On GT200 and older, kernel arguments are passed via shared memory, so available space may be smaller than 16 KB.
   *   We thus request the smallest amount of dynamic shared memory that guarantees throttling to a given number of blocks,
   *   in order to allow some extra leeway.
   */
  virtual bool advanceSharedBytes(TuneParam &param) const
  {
    const int max_shared = 16384; // FIXME: use deviceProp.sharedMemPerBlock;
    int blocks_per_sm = max_shared / (param.shared_bytes ? param.shared_bytes : 1);
    param.shared_bytes = max_shared / blocks_per_sm + 1;
    if (param.shared_bytes > max_shared) {
      TuneParam next(param);
      advanceBlockDim(next); // to get next blockDim
      int nthreads = next.block.x * next.block.y * next.block.z;
      param.shared_bytes = sharedBytesPerThread()*nthreads + sharedBytesPerBlock();
      return false;
    } else {
      return true;
    }
  }

 public:
  Tunable() { }
  virtual ~Tunable() { }
  virtual TuneKey tuneKey() const = 0;
  virtual void apply(const cudaStream_t &stream) = 0;
  virtual void preTune() { }
  virtual void postTune() { }
  virtual int tuningIter() const { return 1; }

  virtual std::string paramString(const TuneParam &param) const
  {
    std::stringstream ss;
    ss << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ss << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
    ss << "shared=" << param.shared_bytes;
    return ss.str();
  }

  virtual std::string perfString(float time) const
  {
    float gflops = flops() / (1e9 * time);
    float gbytes = bytes() / (1e9 * time);
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << gflops << " Gflop/s, ";
    ss << gbytes << " GB/s";
    return ss.str();
  }

  virtual void initTuneParam(TuneParam &param) const
  {
    param.block = dim3(32,1,1);
    param.grid = dim3(1,1,1);
    param.shared_bytes = sharedBytesPerThread()*32 + sharedBytesPerBlock();
  }

  virtual bool advanceTuneParam(TuneParam &param) const
  {
    return advanceSharedBytes(param) || advanceBlockDim(param) || advanceGridDim(param);
  }

};


TuneParam tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity);


#if 0
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
  virtual bool checkLaunch() const = 0;

  const char* Name() const { return name; }

  // Varies the block size of the given function and finds the performance maxiumum
  void Benchmark(TuneParam &tune); 
};

class TuneDiracBase : public TuneBase {

 protected:
  bool checkLaunch() const { return getDslashLaunch(); }

 public:
  TuneDiracBase(const char *name, QudaVerbosity verbose) 
    : TuneBase(name, verbose) { ; }
    virtual ~TuneDiracBase() { ; }

};

class TuneDiracWilsonDslash : public TuneDiracBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracWilsonDslash(const DiracWilson &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracWilsonDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracWilsonDslash() { ; }

  void Apply() const { dirac.DiracWilson::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracWilsonDslashXpay : public TuneDiracBase {

 private:
  const DiracWilson &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracWilsonDslashXpay(const DiracWilson &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneDiracBase("DiracWilsonDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracWilsonDslashXpay() { ; }

  void Apply() const { dirac.DiracWilson::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracClover : public TuneDiracBase {

 private:
  const DiracClover &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracClover(const DiracClover &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracClover", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracClover() { ; }

  void Apply() const { dirac.Clover(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslash : public TuneDiracBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracCloverDslash(const DiracCloverPC &d, cudaColorSpinorField &a, 
		  const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracCloverDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracCloverDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracCloverDslashXpay : public TuneDiracBase {

 private:
  const DiracCloverPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracCloverDslashXpay(const DiracCloverPC &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneDiracBase("DiracCloverDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracCloverDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMass : public TuneDiracBase {

 private:
  const DiracTwistedMass &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMass(const DiracTwistedMass &d, cudaColorSpinorField &a, 
		       const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracTwistedMass", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMass() { ; }

  void Apply() const { dirac.Twist(a, b); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslash : public TuneDiracBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracTwistedMassDslash(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
			     const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracTwistedMassDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracTwistedMassDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracTwistedMassDslashXpay : public TuneDiracBase {

 private:
  const DiracTwistedMassPC &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracTwistedMassDslashXpay(const DiracTwistedMassPC &d, cudaColorSpinorField &a, 
				 const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneDiracBase("DiracTwistedMassDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracTwistedMassDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslash : public TuneDiracBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracDomainWallDslash(const DiracDomainWall &d, cudaColorSpinorField &a, 
			    const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracDomainWallDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracDomainWallDslash() { ; }

  void Apply() const { dirac.Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracDomainWallDslashXpay : public TuneDiracBase {

 private:
  const DiracDomainWall &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracDomainWallDslashXpay(const DiracDomainWall &d, cudaColorSpinorField &a, 
				const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneDiracBase("DiracDomainWallDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracDomainWallDslashXpay() { ; }

  void Apply() const { dirac.DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslash : public TuneDiracBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;

 public:
  TuneDiracStaggeredDslash(const DiracStaggered &d, cudaColorSpinorField &a, 
			const cudaColorSpinorField &b) : 
  TuneDiracBase("DiracStaggeredDslash", d.Verbose()), dirac(d), a(a), b(b) { ; }
  virtual ~TuneDiracStaggeredDslash() { ; }

  void Apply() const { dirac.DiracStaggered::Dslash(a, b, QUDA_EVEN_PARITY); }
  unsigned long long Flops() const { return dirac.Flops(); }
};

class TuneDiracStaggeredDslashXpay : public TuneDiracBase {

 private:
  const DiracStaggered &dirac;
  cudaColorSpinorField &a;
  const cudaColorSpinorField &b;
  const cudaColorSpinorField &c;

 public:
  TuneDiracStaggeredDslashXpay(const DiracStaggered &d, cudaColorSpinorField &a, 
		      const cudaColorSpinorField &b, const cudaColorSpinorField &c) : 
  TuneDiracBase("DiracStaggeredDslashXpay", d.Verbose()), dirac(d), a(a), b(b), c(c) { ; }
  virtual ~TuneDiracStaggeredDslashXpay() { ; }

  void Apply() const { dirac.DiracStaggered::DslashXpay(a, b, QUDA_EVEN_PARITY, c, 1.0); }
  unsigned long long Flops() const { return dirac.Flops(); }
};
#endif // 0

#endif // _TUNE_QUDA_H
