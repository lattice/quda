#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>

class CloverField {

 protected:
  size_t bytes; // bytes allocated per clover full field 
  size_t norm_bytes; // sizeof each norm full field
  size_t total_bytes; // total bytes allocated
  QudaPrecision precision;
  int length;
  int real_length;
  int volume;
  int volumeCB;
  int X[QUDA_MAX_DIM];
  int Nc;
  int Ns;
  int pad;
  int stride;

 public:
  CloverField(const int *, const int pad, const QudaPrecision);
  virtual ~CloverField();

  int Volume() const { return volume; }
  int VolumeCB() const { return volumeCB; }
  size_t GBytes() const { return total_bytes / (1<<30); } // returns total storage allocated in the clover object
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

 public:
  cudaCloverField(const void *, const void *, const int *X, const int pad, 
		  const QudaPrecision precision, const QudaPrecision cpu_prec,
		  const QudaCloverFieldOrder cpu_order);
  virtual ~cudaCloverField();


  // TODO - improve memory efficiency for asymmetric clover?

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

#endif // _CLOVER_QUDA_H
