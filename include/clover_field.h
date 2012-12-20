#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>
#include <lattice_field.h>

namespace quda {

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

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t evenTex;
    cudaTextureObject_t evenNormTex;
    cudaTextureObject_t oddTex;
    cudaTextureObject_t oddNormTex;
    cudaTextureObject_t evenInvTex;
    cudaTextureObject_t evenInvNormTex;
    cudaTextureObject_t oddInvTex;
    cudaTextureObject_t oddInvNormTex;
    void createTexObject(cudaTextureObject_t &tex, cudaTextureObject_t &texNorm, void *field, void *norm);
    void destroyTexObject();
#endif

  public:
    // create a cudaCloverField from a cpu pointer
    cudaCloverField(const void *h_clov, const void *h_clov_inv, 
		    const QudaPrecision cpu_prec, 
		    const QudaCloverFieldOrder cpu_order,
		    const CloverFieldParam &param);

    // create a cudaCloverField from a cudaGaugeField
    cudaCloverField(const cudaGaugeField &gauge, const CloverFieldParam &param);
    virtual ~cudaCloverField();

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& EvenNormTex() const { return evenNormTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
    const cudaTextureObject_t& OddNormTex() const { return oddNormTex; }
    const cudaTextureObject_t& EvenInvTex() const { return evenInvTex; }
    const cudaTextureObject_t& EvenInvNormTex() const { return evenInvNormTex; }
    const cudaTextureObject_t& OddInvTex() const { return oddInvTex; }
    const cudaTextureObject_t& OddInvNormTex() const { return oddInvNormTex; }
#endif

    friend class DiracClover;
    friend class DiracCloverPC;
    friend struct FullClover;
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

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t &evenTex;
    const cudaTextureObject_t &evenNormTex;
    const cudaTextureObject_t &oddTex;
    const cudaTextureObject_t &oddNormTex;
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& EvenNormTex() const { return evenNormTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
    const cudaTextureObject_t& OddNormTex() const { return oddNormTex; }    
#endif

    FullClover(const cudaCloverField &clover, bool inverse=false) :
      precision(clover.precision), bytes(clover.bytes), norm_bytes(clover.norm_bytes)
#ifdef USE_TEXTURE_OBJECTS
	, evenTex(inverse ? clover.evenInvTex : clover.evenTex)
	, evenNormTex(inverse ? clover.evenInvNormTex : clover.evenNormTex)
	, oddTex(inverse ? clover.oddInvTex : clover.oddTex)
	, oddNormTex(inverse ? clover.oddInvNormTex : clover.oddNormTex)
#endif
      { 
	if (inverse) {
	  even = clover.evenInv;
	  evenNorm = clover.evenInvNorm;
	  odd = clover.oddInv;	
	  oddNorm = clover.oddInvNorm;
	} else {
	  even = clover.even;
	  evenNorm = clover.evenNorm;
	  odd = clover.odd;	
	  oddNorm = clover.oddNorm;
	}
    }
  };

  // driver for computing the clover field from the gauge field
  void computeCloverCuda(cudaCloverField &clover, const cudaGaugeField &gauge);

} // namespace quda

#endif // _CLOVER_QUDA_H
