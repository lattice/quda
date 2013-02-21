#ifndef __SPINOR_TYPES_H__
#define __SPINOR_TYPES_H__



#include <resize_texture.h>
  
  class NullType {};

  template <int Nspin, QudaPrecision outPrec, QudaPrecision inPrec>
    struct SpinorType {
      typedef NullType InType;
      typedef NullType OutType;
      typedef NullType TexType;
    };

  // double-precision source
  // double-precision destination
  template<>
    struct SpinorType<1, QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION>
    {
      typedef Spinor<double2, double2, double2, 3, 0> InType;
      typedef Spinor<double2, double2, double2, 3, 1> OutType;
      typedef Spinor<double2, double2, double2, 3, 0, 1> TexType;
    };

  // single-precision source
  // single-precision destination
  template<>
    struct SpinorType<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>
    {
      typedef Spinor<float2, float2, float2, 3, 0> InType;
      typedef Spinor<float2, float2, float2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, float2, 3, 1> OutType;
    };

  // single-precision source
  // double-precision destination
  template<>
    struct SpinorType<1, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION>
    {
      // Spinor<RegType, InterType, StoreType ...>
      typedef Spinor<float2, float2, float2, 3, 0> InType;
      typedef Spinor<float2, float2, float2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, double2, 3, 1> OutType;
    };

  // double-precision source
  // single-precision destination
  template<>
    struct SpinorType<1, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION>
    {
      typedef Spinor<float2, float2, double2, 3, 0> InType;
      typedef Spinor<float2, float2, double2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, float2, 3, 1> OutType;
    };

  // half-precision source
  // half-precision destination 
  // need to convert to single precision as an intermediate step
  template<>
    struct SpinorType<1, QUDA_HALF_PRECISION, QUDA_HALF_PRECISION>
    {
      typedef Spinor<float2, float2, short2, 3, 0> InType;
      typedef Spinor<float2, float2, short2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, short2, 3, 1> OutType;
    };

  // half-precision source 
  // single-precision destination
  template<>
    struct SpinorType<1, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION>
    {
      typedef Spinor<float2, float2, short2, 3, 0> InType;
      typedef Spinor<float2, float2, short2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, float2, 3, 1> OutType;
    };

  // half-precision source
  // double-precision destination
  template<>
    struct SpinorType<1, QUDA_DOUBLE_PRECISION, QUDA_HALF_PRECISION>
    {
      typedef Spinor<double2, float2, short2, 3, 0> InType;
      typedef Spinor<double2, float2, short2, 3, 0, 1> TexType;
      typedef Spinor<double2, double2, double2, 3, 1> OutType;
    };

  // single-precision source 
  // half-precision destination
  template<>
    struct SpinorType<1, QUDA_HALF_PRECISION, QUDA_SINGLE_PRECISION>
    {
      typedef Spinor<float2, float2, float2, 3, 0> InType;
      typedef Spinor<float2, float2, float2, 3, 0, 1> TexType;
      typedef Spinor<float2, float2, short2, 3, 1> OutType;
    };

  // double-precision source
  // half-precision destination
  template<>
    struct SpinorType<1, QUDA_HALF_PRECISION, QUDA_DOUBLE_PRECISION>
    {
      typedef Spinor<double2, double2, double2, 3, 0> InType;
      typedef Spinor<double2, double2, double2, 3, 0, 1> TexType;
      typedef Spinor<double2, double2, short2, 3, 1> OutType;
    };

  /*
  // I could be much more sophisticated and use templates
  template<>
  struct SpinorType<1, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION>
  {
  typedef Spinor<
  typename ConversionTraits<1, InterMediatePrecision<, QUDA_SINGLE_PRECISION>::DstType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>::InterType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>::SRCType,
  3, 0 > InType;

  typedef Spinor<
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>::DstType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>::InterType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>::SRCType,
  3, 0, 1> TexType;

  typedef Spinor< 
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION>::DstType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION>::InterType,
  typename ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION>::SrcType,
  3, 1> TexType; 
  };



  template<QudaPrecision DstPrec, QudaPrecision SrcPrec>
  IntermediatePrecision
  {
  enum val { QUDA_SINGLE_PRECISION };
  };

  template<>
  struct InterMediatePrecision<QUDA_DOUBLE_PRECISON, QUDA_DOUBLE_PRECISION>
  {
  enum val { QUDA_DOUBLE_PRECISION };
  };




  template<int Nspin, QudaPrecision DstPrec, QudaPrecision SrcPrec>
  struct ConversionTraits
  {
  typedef NullType SrcType
  typedef NullType InterType;
  typedef NullType DstType;
  };


  template<>
  struct ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>
  {
  typedef float2 SrcType;
  typedef float2 InterType;
  typedef float2 DstType;
  };

  template<>
  struct ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION>
  {
  typedef double2 SrcType;
  typedef double2 InterType;
  typedef double2 DstType;
  };


  template<>
  struct ConversionTraits<1, QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION>
  {
  typedef float2 SrcType;
  typedef float2 InterType;
  typedef double2 DstType;
};



template<>
struct ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION>
{
  typedef double2 SrcType;
  typedef float2 InterType;
  typedef float2 DstType;
};


template<>
struct ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION>
{
  typedef short2 SrcType;
  typedef float2 InterType;
  typedef float2 DstType;
};

template<>
struct ConversionTraits<1, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION>
{
  typedef float2 SrcType;
  typedef float2 InterType; 
  typedef short2 DstType;
};
*/

#endif
