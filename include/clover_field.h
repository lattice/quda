#ifndef _CLOVER_QUDA_H
#define _CLOVER_QUDA_H

#include <quda_internal.h>
#include <lattice_field.h>

namespace quda {

  struct CloverFieldParam : public LatticeFieldParam {
    bool direct; // whether to create the direct clover 
    bool inverse; // whether to create the inverse clover
    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;
    CloverFieldOrder order;
    QudaFieldCreate create;
    void setPrecision(QudaPrecision precision) {
      this->precision = precision;
      order = (precision == QUDA_DOUBLE_PRECISION) ? 
	QUDA_FLOAT2_CLOVER_ORDER : QUDA_FLOAT4_CLOVER_ORDER;
    }
  };

  class CloverField : public LatticeField {

  protected:
    size_t bytes; // bytes allocated per clover full field 
    size_t norm_bytes; // sizeof each norm full field
    int length;
    int real_length;
    int nColor;
    int nSpin;

    void *clover;
    void *norm;
    void *cloverInv;
    void *invNorm;

    CloverFieldOrder order;
    QudaFieldCreate create;

  public:
    CloverField(const CloverFieldParam &param);
    virtual ~CloverField();

    void* V(bool inverse=false) { return inverse ? cloverInv : clover; }
    void* Norm(bool inverse=false) { return inverse ? invNorm : norm; }
    const void* V(bool inverse=false) const { return inverse ? cloverInv : clover; }
    const void* Norm(bool inverse=false) const { return inverse ? invNorm : norm; }
    
    CloverFieldOrder Order() const { return order; }
    size_t Bytes() const { return bytes; }
    size_t NormBytes() const { return norm_bytes; }
  };

  class cudaCloverField : public CloverField {

  private:
    void *even, *odd;
    void *evenNorm, *oddNorm;

    void *evenInv, *oddInv;
    void *evenInvNorm, *oddInvNorm;

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
    // create a cudaCloverField from a CloverFieldParam
    cudaCloverField(const CloverFieldParam &param);

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

    /**
       Copy into this CloverField from the generic CloverField src
       @param src The clover field from which we want to copy
     */
    void copy(const CloverField &src);

    /**
       Copy into this CloverField from the cpuCloverField cpu
       @param cpu The cpu clover field from which we want to copy
     */
    void loadCPUField(const cpuCloverField &cpu);

    friend class DiracClover;
    friend class DiracCloverPC;
    friend struct FullClover;
  };

  // this is a place holder for a future host-side clover object
  class cpuCloverField : public CloverField {

  private:

  public:
    cpuCloverField(const CloverFieldParam &param);
    virtual ~cpuCloverField();
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

  // driver for generic clover field copying
  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in copy_gauge.cu
     @param out The output field to which we are copying
     @param in The input field from which we are copying
     @param inverse Whether we are copying the inverse term or not
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
     @param outNorm The output norm buffer (optional)
     @param inNorm The input norm buffer (optional)
  */
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
			 void *Out=0, void *In=0, void *outNorm=0, void *inNorm=0);

} // namespace quda

#endif // _CLOVER_QUDA_H
