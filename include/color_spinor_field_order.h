/**
 * @file color_spinor_field_order.h
 *
 * @section DESCRIPTION 
 *
 * Define functors to allow for generic accessors regardless of field
 * ordering.  Currently this is used for cpu fields only with limited
 * ordering support, but this will be expanded for device ordering
 *  also.
 */

namespace quda {

  template <typename Float>
    class ColorSpinorFieldOrder {

  protected:
    /** An internal reference to the actual field we are accessing */
    cpuColorSpinorField &field;
    
  public:
  
    /** 
     * Constructor for the ColorSpinorFieldOrder class
     * @param field The field that we are accessing
     */
  ColorSpinorFieldOrder(cpuColorSpinorField &field) : field(field) { ; }

    /**
     * Destructor for the ColorSpinorFieldOrder class
     */
    virtual ~ColorSpinorFieldOrder() { ; }

    /**
     * Read-only real-member accessor function
     * @param x 1-d site index
     * @param s spin index
     * @param c color index
     * @param z complexity index
     */
    virtual const Float& operator()(const int &x, const int &s, const int &c, const int &z) const = 0;

    /**
     * Writable real-member accessor function
     * @param x 1-d site index
     * @param s spin index
     * @param c color index
     * @param z complexity index
     */
    virtual Float& operator()(const int &x, const int &s, const int &c, const int &z) = 0;

    /**
     * Read-only complex-member accessor function
     * @param x 1-d site index
     * @param s spin index
     * @param c color index
     */
    virtual const std::complex<Float>& operator()(const int &x, const int &s, const int &c) const = 0;

    /**
     * Writable complex-member accessor function
     * @param x 1-d site index
     * @param s spin index
     * @param c color index
     */
    virtual std::complex<Float>& operator()(const int &x, const int &s, const int &c) = 0;

    /** Returns the number of field colors */
    int Ncolor() const { return field.Ncolor(); }

    /** Returns the number of field spins */
    int Nspin() const { return field.Nspin(); }

    /** Returns the field volume */
    int Volume() const { return field.Volume(); }

    /** Returns the field geometric dimension */
    int Ndim() const { return field.Ndim(); }
  };

  template <typename Float>
    class SpaceSpinColorOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field; // convenient to have a "local" reference for code brevity

  public:
  SpaceSpinColorOrder(cpuColorSpinorField &field): ColorSpinorFieldOrder<Float>(field), field(field) 
    { ; }
    virtual ~SpaceSpinColorOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
      return *((Float*)(field.v) + index);
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
      return *((Float*)(field.v) + index);
    }

    const std::complex<Float>& operator()(const int &x, const int &s, const int &c) const {
      unsigned long index = (x*field.nSpin+s)*field.nColor+c;
      return *(static_cast<std::complex<Float>*>(field.v) + index);
    }

    std::complex<Float>& operator()(const int &x, const int &s, const int &c) {
      unsigned long index = (x*field.nSpin+s)*field.nColor+c;
      return *(static_cast<std::complex<Float>*>(field.v) + index);
    }

  };

  template <typename Float>
    class SpaceColorSpinOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field;  // convenient to have a "local" reference for code brevity

  public:
  SpaceColorSpinOrder(cpuColorSpinorField &field) : ColorSpinorFieldOrder<Float>(field), field(field)
    { ; }
    virtual ~SpaceColorSpinOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;
      return *((Float*)(field.v) + index);
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;    
      return *((Float*)(field.v) + index);
    }

    const std::complex<Float>& operator()(const int &x, const int &s, const int &c) const {
      unsigned long index = (x*field.nColor+c)*field.nSpin+s;
      return *(static_cast<std::complex<Float>*>(field.v) + index);
    }

    std::complex<Float>& operator()(const int &x, const int &s, const int &c) {
      unsigned long index = (x*field.nColor+c)*field.nSpin+s;    
      return *(static_cast<std::complex<Float>*>(field.v) + index);
    }
  };

  template <typename Float>
    class QOPDomainWallOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field;  // convenient to have a "local" reference for code brevity
    int volume_4d;
    int Ls;

  public:
  QOPDomainWallOrder(cpuColorSpinorField &field) : ColorSpinorFieldOrder<Float>(field), 
      field(field), volume_4d(1), Ls(0)
      { 
	if (field.Ndim() != 5) errorQuda("Error, wrong number of dimensions for this ColorSpinorFieldOrder");
	for (int i=0; i<4; i++) volume_4d *= field.x[i];
	Ls = field.x[4];
      }
    virtual ~QOPDomainWallOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = ((x_4d*field.nColor+c)*field.nSpin+s)*2+z;
      return ((Float**)(field.v))[ls][index_4d];
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = ((x_4d*field.nColor+c)*field.nSpin+s)*2+z;
      return ((Float**)(field.v))[ls][index_4d];
    }

    const std::complex<Float>& operator()(const int &x, const int &s, const int &c) const {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = (x_4d*field.nColor+c)*field.nSpin+s;
      return (static_cast<std::complex<Float>**>(field.v))[ls][index_4d];
    }

    std::complex<Float>& operator()(const int &x, const int &s, const int &c) {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = (x_4d*field.nColor+c)*field.nSpin+s;
      return (static_cast<std::complex<Float>**>(field.v))[ls][index_4d];
    }
  };

} // namespace quda
