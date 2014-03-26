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


    EigParam()
      : nk(0), np(0), f_size(0), precision(QUDA_INVALID_PRECISION), CSPparam(0), 
        create(QUDA_INVALID_FIELD_CREATE), v(0)
    {}


    // used to create cpu params
    EigParam(QudaEigParam &eig_param, QudaFieldLocation location, void *h_v, ColorSpinorParam *CSP_param)
      : nk(eig_param.nk), np(eig_param.np), f_size(eig_param.f_size), 
        CSPparam(CSP_param), create(QUDA_REFERENCE_FIELD_CREATE), v(h_v)
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
      : nk(cpuParam.nk), np(cpuParam.np), f_size(cpuParam.f_size), 
        precision(eig_param.invert_param->cuda_prec),
        CSPparam(cpuParam.CSPparam), create(QUDA_COPY_FIELD_CREATE), v(0)
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

} // namespace quda

#endif // _EIG_VARIABLES_H
