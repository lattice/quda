#ifndef _EIG_VARIABLES_H
#define _EIG_VARIABLES_H

#include <quda_internal.h>
#include <quda.h>

#include <iostream>
#include <complex>

#include <lattice_field.h>

namespace quda {
  typedef std::complex<double> Complex;

  class EigParam {
  public:
    int nk; // Number of wanted eigen values
    int np; // Number of unwanted eigen values, but it is used for accelerating iteration sequence
    int f_size; // number of eigen vector elements
    QudaPrecision precision; // Precision of the field
    ColorSpinorParam *CSPparam;
    QudaFieldCreate create; //

    void *v; // pointer to eigen-value set

    EigParam(const EigValueSet &a);
    EigParam(const EigVecSet &a);

    EigParam()
      : nk(0), np(0), f_size(0), precision(QUDA_INVALID_PRECISION), CSPparam(0), 
        create(QUDA_INVALID_FIELD_CREATE), v(0)
    {}


    // used to create cpu params
    EigParam(QudaEigParam &eig_param, QudaFieldLocation location, void *h_v, ColorSpinorParam *CSP_param)
      : nk(eig_param.nk), np(eig_param.np), f_size(eig_param.f_size), 
        create(QUDA_REFERENCE_FIELD_CREATE), CSPparam(CSP_param), v(h_v)
    {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        precision = eig_param.invert_param->cpu_prec;
      } else {
        precision = eig_param.invert_param->cuda_prec;
      }
      if( precision != QUDA_DOUBLE_PRECISION)
      {
        printfQuda("ERROR!!! current lanczos program supports double precision only\n");
        exit(0);
      }
    }

    // used to create cuda param from a cpu param
    EigParam(EigParam &cpuParam, QudaEigParam &eig_param) 
      : nk(cpuParam.nk),np(cpuParam.np),f_size(cpuParam.f_size), 
        precision(eig_param.invert_param->cuda_prec),
        create(QUDA_COPY_FIELD_CREATE), CSPparam(cpuParam.CSPparam), v(0)
    {
      // Currently lanczos program supports double precision only
      if( precision != QUDA_DOUBLE_PRECISION)
      {
        printfQuda("ERROR!!! current lanczos program supports double precision only\n");
        exit(0);
      }
    }


    void setPrecision(QudaPrecision precision) {
      this->precision = precision;
      // Currently lanczos program supports double precision only
      if( precision != QUDA_DOUBLE_PRECISION)
      {
        printfQuda("ERROR!!! current lanczos program supports double precision only\n");
        exit(0);
      }
    }

    void print() {
      printfQuda("nk = %d\n", nk);
      printfQuda("np = %d\n", np);
      printfQuda("eigen vector elements = %d\n",f_size);
      printfQuda("precision = %d\n", precision);
      printfQuda("Memory Addr = %lx\n", (unsigned long)v);
    }

    virtual ~EigParam() {
    }
  };

  class cpuEigValueSet;
  class cudaEigValueSet;

  class EigValueSet {
    private:
      void create(int nk, int np, QudaPrecision precision); 
      void destroy();  
      QudaVerbosity verbose;
    protected:
      bool init;
      void *v;
      int bytes;
      void fill(EigParam &) const;
    public:
      EigValueSet(const EigParam &);
      virtual ~EigValueSet();

      EigParam self_param;

      virtual EigValueSet& operator=(const EigValueSet&);

      int NK() const { return self_param.nk; } 
      int NP() const { return self_param.np; } 
      void* V() {return v;}
      const void* V() const {return v;}
      QudaPrecision Precision() const { return self_param.precision; }

      friend std::ostream& operator<<(std::ostream &out, const EigValueSet &);
      friend class EigParam;
  };

  // CUDA implementation
  class cudaEigValueSet : public EigValueSet {
    friend class cpuEigValueSet;
  private:
    void create(const QudaFieldCreate);
    void destroy();
    bool reference; // whether the field is a reference or not

  public:
    cudaEigValueSet(const EigValueSet &src, const EigParam&);
    virtual ~cudaEigValueSet();
    cudaEigValueSet& operator=(const cudaEigValueSet&);
    cudaEigValueSet& operator=(const cpuEigValueSet&);
    void copy(const cpuEigValueSet &src);
    void save(const cudaEigValueSet &src);
    void zero();

    friend std::ostream& operator<<(std::ostream &out, const cudaEigValueSet&);
  };
  
  // CPU implementation
  class cpuEigValueSet : public EigValueSet{
    friend class cudaEigValueSet;

  private:
    void create(const QudaFieldCreate);
    void destroy();
    bool reference; // whether the field is a reference or not
  public:
    cpuEigValueSet(const EigParam&);
    virtual ~cpuEigValueSet();
    cpuEigValueSet& operator=(const cpuEigValueSet&);
    cpuEigValueSet& operator=(const cudaEigValueSet&);

    void copy(const cpuEigValueSet &src);
    void PrintEigVal(unsigned int n);
    void zero();
    QudaFieldLocation Location() const;
  };

  class cpuEigVecSet;
  class cudaEigVecSet;

  class EigVecSet {
    private:
      void create(int nk, int np, int f_size, QudaPrecision precision); 
      void destroy();  
      QudaVerbosity verbose;
    protected:
      bool init;
      void *v;
      int bytes;
      void fill(EigParam &) const;
    public:
      EigVecSet(const EigParam &);
      virtual ~EigVecSet();

      EigParam self_param;
      
      virtual EigVecSet& operator=(const EigVecSet&);

      int NK() const { return self_param.nk; } 
      int NP() const { return self_param.np; } 
      int Vec_size() const { return self_param.f_size; } 
      void* V() {return v;}
      const void* V() const {return v;}
      QudaPrecision Precision() const { return self_param.precision; }

      friend std::ostream& operator<<(std::ostream &out, const EigVecSet &);
      friend class EigParam;
  };

  // CUDA implementation
  class cudaEigVecSet : public EigVecSet {
    friend class cpuEigVecSet;
  private:
    void create(const QudaFieldCreate);
    void destroy();
    bool reference; // whether the field is a reference or not
    __constant__ void *cuda_eigvec;
  public:
    cudaEigVecSet(const EigVecSet &, const EigParam&);
    virtual ~cudaEigVecSet();
    cudaEigVecSet& operator=(const cudaEigVecSet&);
    cudaEigVecSet& operator=(const cpuEigVecSet&);

    friend std::ostream& operator<<(std::ostream &out, const cudaEigVecSet&);
  };
  
  // CPU implementation
  class cpuEigVecSet : public EigVecSet{
    friend class cudaEigVecSet;

  private:
    void create(const QudaFieldCreate);
    void destroy();
    bool reference; // whether the field is a reference or not
  public:
    cpuEigVecSet(const EigParam&);
    virtual ~cpuEigVecSet();
    cpuEigVecSet& operator=(const cpuEigVecSet&);
    cpuEigVecSet& operator=(const cudaEigVecSet&);
    void PrintEigVec(unsigned int n);
    QudaFieldLocation Location() const;
  };

} // namespace quda

#endif // _EIG_VARIABLES_H
