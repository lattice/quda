#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>
#include <lattice_field.h>

struct CloverFieldParam : public LatticeFieldParam {
  
};

class CloverField : public LatticeField {

 protected:
  size_t bytes; // bytes allocated per clover full field 
  size_t norm_bytes; // sizeof each norm full field
  int length;
  int real_length;
  int nColor;
  int nSpin;

 public:
  CloverField(const CloverFieldParam &param);
  virtual ~CloverField();
};

class cudaCloverField : public CloverField {

 private:
  void *clover, *even, *odd;
  void *norm, *evenNorm, *oddNorm;

  void *cloverInv, *evenInv, *oddInv;
  void *invNorm, *evenInvNorm, *oddInvNorm;

  void loadCPUField(void *d_clover, void *d_norm, const void *h_clover, 
		    const QudaPrecision cpu_prec, const CloverFieldOrder order);
  void loadParityField(void *d_clover, void *d_norm, const void *h_clover, 
		       const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order);
  void loadFullField(void *d_even, void *d_even_norm, void *d_odd, void *d_odd_norm, 
		     const void *h_clover, const QudaPrecision cpu_prec, const CloverFieldOrder cpu_order);

  // computes the clover field given the input gauge field
  void compute(const cudaGaugeField &gauge);

 public:
  // create a cudaCloverField from a cpu pointer
  cudaCloverField(const void *h_clov, const void *h_clov_inv, 
		  const QudaPrecision cpu_prec, 
		  const QudaCloverFieldOrder cpu_order,
		  const CloverFieldParam &param);

  // create a cudaCloverField from a cudaGaugeField
  cudaCloverField(const cudaGaugeField &gauge, const CloverFieldParam &param);
  virtual ~cudaCloverField();

  friend class DiracClover;
  friend class DiracCloverPC;
};

// this is a place holder for a future host-side clover object
class cpuCloverField {

 private:

 public:
  cpuCloverField();
  virtual ~cpuCloverField() = 0;
};

// lightweight struct used to send pointers to cuda driver code
struct FullClover {
  void *even;
  void *odd;
  void *evenNorm;
  void *oddNorm;
  QudaPrecision precision;
  size_t bytes; // sizeof each clover field (per parity)
  size_t norm_bytes; // sizeof each norm field (per parity)
};

// driver for computing the clover field from the gauge field
void computeCloverCuda(cudaCloverField &clover, const cudaGaugeField &gauge);

#endif // _CLOVER_QUDA_H
