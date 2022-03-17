/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file complex.h
 *  \brief Complex number class, designed to provide std::complex functionality to quda.
 */

#pragma once

#include <cmath>
#include <complex>
#include <sstream>
#include <cstdint>
#include <type_traits>
#include <quda_arch.h> // for double2 / float2

namespace quda {
  namespace gauge {
    template<typename Float, typename storeFloat> struct fieldorder_wrapper;
  }

  namespace colorspinor {
    template <typename Float, typename storeFloat, bool block_float, typename norm_t> struct fieldorder_wrapper;
  }
}

// We need this to make sure code inside quda:: that calls sqrt() using real numbers
// doesn't try to call the complex sqrt, but the standard sqrt
namespace quda
{
  template <typename ValueType>
    __host__ __device__
    inline ValueType cos(ValueType x){
    return std::cos(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType sin(ValueType x){
    return std::sin(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType tan(ValueType x){
    return std::tan(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType acos(ValueType x){
    return std::acos(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType asin(ValueType x){
    return std::asin(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType atan(ValueType x){
    return std::atan(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType atan2(ValueType x,ValueType y){
    return std::atan2(x,y);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType cosh(ValueType x){
    return std::cosh(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType sinh(ValueType x){
    return std::sinh(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType tanh(ValueType x){
    return std::tanh(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType exp(ValueType x){
    return std::exp(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType log(ValueType x){
    return std::log(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType log10(ValueType x){
    return std::log10(x);
  }
  template <typename ValueType, typename ExponentType>
    __host__ __device__
    inline ValueType pow(ValueType x, ExponentType e){
    return std::pow(x,static_cast<ValueType>(e));
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType sqrt(ValueType x){
    return std::sqrt(x);
  }
  template <typename ValueType>
    __host__ __device__
    inline ValueType abs(ValueType x){
    return std::abs(x);
  }

  __host__ __device__ inline float conj(float x) { return x; }
  __host__ __device__ inline double conj(double x) { return x; }

  template <typename ValueType> struct complex;
  //template <> struct complex<float>;
  //template <> struct complex<double>;


  /// Returns the magnitude of z.
  template<typename ValueType>
  __host__ __device__
  ValueType abs(const complex<ValueType>& z);
  /// Returns the phase angle of z.
  template<typename ValueType>
  __host__ __device__
  ValueType arg(const complex<ValueType>& z);
  /// Returns the magnitude of z squared.
  template<typename ValueType>
  __host__ __device__
  ValueType norm(const complex<ValueType>& z);

  /// Returns the complex conjugate of z.
  template<typename ValueType>
  __host__ __device__
  complex<ValueType> conj(const complex<ValueType>& z);
  /// Returns the complex with magnitude m and angle theta in radians.

  template<typename ValueType>
  __host__ __device__
  complex<ValueType> polar(const ValueType& m, const ValueType& theta = 0);

  // Arithmetic operators:
  // Multiplication
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs);
  // Division
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <>
    __host__ __device__
    inline complex<float> operator/(const complex<float>& lhs, const complex<float>& rhs);
  template <>
    __host__ __device__
    inline complex<double> operator/(const complex<double>& lhs, const complex<double>& rhs);

  // Addition
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs);
  // Subtraction
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs);

  // Unary plus and minus
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &rhs);

  // Transcendentals:
  // Returns the complex cosine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> cos(const complex<ValueType>& z);
  // Returns the complex hyperbolic cosine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> cosh(const complex<ValueType>& z);
  // Returns the complex base e exponential of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> exp(const complex<ValueType>& z);
  // Returns the complex natural logarithm of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> log(const complex<ValueType>& z);
  // Returns the complex base 10 logarithm of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> log10(const complex<ValueType>& z);
  // Returns z to the n'th power.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> pow(const complex<ValueType>& z, const int& n);
  // Returns z to the x'th power.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> pow(const complex<ValueType>&z, const ValueType&x);
  // Returns z to the z2'th power.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> pow(const complex<ValueType>&z, const complex<ValueType>&z2);
  // Returns x to the z'th power.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> pow(const ValueType& x, const complex<ValueType>& z);
  // Returns the complex sine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> sin(const complex<ValueType>&z);
  // Returns the complex hyperbolic sine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> sinh(const complex<ValueType>&z);
  // Returns the complex square root of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> sqrt(const complex<ValueType>&z);
  // Returns the complex tangent of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> tan(const complex<ValueType>&z);
  // Returns the complex hyperbolic tangent of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> tanh(const complex<ValueType>&z);


  // Inverse Trigonometric:
  // Returns the complex arc cosine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> acos(const complex<ValueType>& z);
  // Returns the complex arc sine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> asin(const complex<ValueType>& z);
  // Returns the complex arc tangent of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> atan(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc cosine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> acosh(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc sine of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> asinh(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc tangent of z.
  template <typename ValueType>
    __host__ __device__
  complex<ValueType> atanh(const complex<ValueType>& z);



  // Stream operators:
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z);
  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z);

  // Stream operators
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z)
    {
      os << '(' << z.real() << ',' << z.imag() << ')';
      return os;
    }

  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z)
    {
      ValueType re, im;

      charT ch;
      is >> ch;

      if(ch == '(')
      {
        is >> re >> ch;
        if (ch == ',')
        {
          is >> im >> ch;
          if (ch == ')')
          {
            z = complex<ValueType>(re, im);
          }
          else
          {
            is.setstate(std::ios_base::failbit);
          }
        }
        else if (ch == ')')
        {
          z = re;
        }
        else
        {
          is.setstate(std::ios_base::failbit);
        }
      }
      else
      {
        is.putback(ch);
        is >> re;
        z = re;
      }
      return is;
    }

template <typename T>
  struct norm_type {
    typedef T type;
  };
 template <typename T>
   struct norm_type< complex<T> > {
   typedef T type;
 };

template <typename ValueType>
struct complex
{
public:
  typedef ValueType value_type;

  // Constructors
  __host__ __device__ inline complex<ValueType>(const ValueType &re = ValueType(), const ValueType &im = ValueType())
  {
    real(re);
    imag(im);
    }

  template <class X>
    __host__ __device__
    inline complex<ValueType>(const complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }

  template <class X>
    __host__ __device__
    inline complex<ValueType>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }

    template <typename T> __host__ __device__ inline complex<ValueType> &operator=(const complex<T> &z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator+=(const complex<ValueType> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator-=(const complex<ValueType> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator*=(const complex<ValueType> z)
    {
      *this = *this * z;
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator/=(const complex<ValueType> z)
    {
      *this = *this / z;
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator*=(const ValueType z)
    {
      this->x *= z;
      this->y *= z;
      return *this;
    }

  __host__ __device__ inline ValueType real() const;
  __host__ __device__ inline ValueType imag() const;
  __host__ __device__ inline void real(ValueType);
  __host__ __device__ inline void imag(ValueType);
};

template <> struct complex<float> : public float2 {
public:
  typedef float value_type;
  complex<float>() = default;
  constexpr complex<float>(const float &re, const float &im = float()) : float2 {re, im} { }

  template <typename X>
  constexpr complex<float>(const std::complex<X> &z) :
    float2 {static_cast<float>(z.real()), static_cast<float>(z.imag())}
  {
  }

  template <typename T> __host__ __device__ inline complex<float> &operator=(const complex<T> &z)
  {
    real(z.real());
    imag(z.imag());
    return *this;
    }

    __host__ __device__ inline complex<float> &operator+=(const complex<float> &z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

    __host__ __device__ inline complex<float> &operator-=(const complex<float> &z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

    __host__ __device__ inline complex<float> &operator*=(const complex<float> &z)
    {
      *this = *this * z;
      return *this;
    }

    __host__ __device__ inline complex<float> &operator/=(const complex<float> &z)
    {
      *this = *this / z;
      return *this;
    }

    __host__ __device__ inline complex<float> &operator*=(const float &z)
    {
      this->x *= z;
      this->y *= z;
      return *this;
    }

    constexpr float real() const { return x; }
    constexpr float imag() const { return y; }
    __host__ __device__ inline void real(float re) { x = re; }
    __host__ __device__ inline void imag(float im) { y = im; }

    // cast operators
    inline operator std::complex<float>() const { return std::complex<float>(real(), imag()); }
    template <typename T> inline __host__ __device__ operator complex<T>() const
    {
      return complex<T>(static_cast<T>(real()), static_cast<T>(imag())); }
};

template <> struct complex<double> : public double2 {
public:
  typedef double value_type;
  complex<double>() = default;
  constexpr complex<double>(const double &re, const double &im = double()) : double2 {re, im} { }

  template <typename X>
  constexpr complex<double>(const std::complex<X> &z) :
    double2 {static_cast<double>(z.real()), static_cast<double>(z.imag())}
  {
  }

  template <typename T> __host__ __device__ inline complex<double> &operator=(const complex<T> &z)
  {
    real(z.real());
    imag(z.imag());
    return *this;
    }

    __host__ __device__ inline complex<double> &operator+=(const complex<double> &z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

    __host__ __device__ inline complex<double> &operator+=(const complex<float> &z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

    __host__ __device__ inline complex<double> &operator-=(const complex<double> &z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

    __host__ __device__ inline complex<double> &operator*=(const complex<double> &z)
    {
      *this = *this * z;
      return *this;
    }

    __host__ __device__ inline complex<double> &operator/=(const complex<double> &z)
    {
      *this = *this / z;
      return *this;
    }

    __host__ __device__ inline complex<double> &operator*=(const double &z)
    {
      this->x *= z;
      this->y *= z;
      return *this;
    }

    constexpr double real() const { return x; }
    constexpr double imag() const { return y; }
    __host__ __device__ inline void real(double re) { x = re; }
    __host__ __device__ inline void imag(double im) { y = im; }

    // cast operators
    inline operator std::complex<double>() const { return std::complex<double>(real(), imag()); }
    template <typename T> inline __host__ __device__ operator complex<T>() const
    {
      return complex<T>(static_cast<T>(real()), static_cast<T>(imag())); }
};

template <> struct complex<int8_t> : public char2 {
public:
  typedef int8_t value_type;

  complex<int8_t>() = default;

  constexpr complex<int8_t>(const int8_t &re, const int8_t &im = int8_t()) : char2 {re, im} { }

  __host__ __device__ inline complex<int8_t> &operator+=(const complex<int8_t> &z)
  {
    real(real() + z.real());
    imag(imag() + z.imag());
    return *this;
  }

  __host__ __device__ inline complex<int8_t> &operator-=(const complex<int8_t> &z)
  {
    real(real() - z.real());
    imag(imag() - z.imag());
    return *this;
  }

  constexpr int8_t real() const { return x; }
  constexpr int8_t imag() const { return y; }
  __host__ __device__ inline void real(int8_t re) { x = re; }
  __host__ __device__ inline void imag(int8_t im) { y = im; }

  // cast operators
  template <typename T> inline __host__ __device__ operator complex<T>() const
  {
    return complex<T>(static_cast<T>(real()), static_cast<T>(imag()));
  }
};

template<>
struct complex <short> : public short2
{
public:
  typedef short value_type;

  complex<short>() = default;

  constexpr complex<short>(const short &re, const short &im = short()) : short2 {re, im} { }

  __host__ __device__ inline complex<short> &operator+=(const complex<short> &z)
  {
    real(real() + z.real());
    imag(imag() + z.imag());
    return *this;
    }

    __host__ __device__ inline complex<short> &operator-=(const complex<short> &z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

    constexpr short real() const { return x; }
    constexpr short imag() const { return y; }
    __host__ __device__ inline void real(short re) { x = re; }
    __host__ __device__ inline void imag(short im) { y = im; }

    // cast operators
    template <typename T> inline __host__ __device__ operator complex<T>() const
    {
      return complex<T>(static_cast<T>(real()), static_cast<T>(imag())); }

};

template<>
struct complex <int> : public int2
{
public:
  typedef int value_type;

  complex<int>() = default;

  constexpr complex<int>(const int &re, const int &im = int()) : int2 {re, im} { }

  __host__ __device__ inline complex<int> &operator+=(const complex<int> &z)
  {
    real(real() + z.real());
    imag(imag() + z.imag());
    return *this;
    }

    __host__ __device__ inline complex<int> &operator-=(const complex<int> &z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

    constexpr int real() const { return x; }
    constexpr int imag() const { return y; }
    __host__ __device__ inline void real(int re) { x = re; }
    __host__ __device__ inline void imag(int im) { y = im; }

    // cast operators
    template <typename T> inline __host__ __device__ operator complex<T>() const
    {
      return complex<T>(static_cast<T>(real()), static_cast<T>(imag())); }

};

  // Binary arithmetic operations

  template<typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs,
const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()+rhs.real(),lhs.imag()+rhs.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()+rhs,lhs.imag());
  }
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()+lhs,rhs.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()-rhs.real(),lhs.imag()-rhs.imag());
  }
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()-rhs,lhs.imag());
  }
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs-rhs.real(),-rhs.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs,
const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),
lhs.real()*rhs.imag()+lhs.imag()*rhs.real());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()*rhs,lhs.imag()*rhs);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()*lhs,rhs.imag()*lhs);
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    const ValueType cross_norm = lhs.real() * rhs.real() + lhs.imag() * rhs.imag();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm/rhs_norm,
(lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / rhs_norm);
  }

  template <>
    __host__ __device__
    inline complex<float> operator/(const complex<float>& lhs, const complex<float>& rhs){

    float s = fabsf(rhs.real()) + fabsf(rhs.imag());
    float oos = 1.0f / s;
    float ars = lhs.real() * oos;
    float ais = lhs.imag() * oos;
    float brs = rhs.real() * oos;
    float bis = rhs.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return complex<float>(((ars * brs) + (ais * bis)) * oos,
			  ((ais * brs) - (ars * bis)) * oos);
  }

  template <>
    __host__ __device__
    inline complex<double> operator/(const complex<double>& lhs, const complex<double>& rhs){

    double s = fabs(rhs.real()) + fabs(rhs.imag());
    double oos = 1.0 / s;
    double ars = lhs.real() * oos;
    double ais = lhs.imag() * oos;
    double brs = rhs.real() * oos;
    double bis = rhs.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return complex<double>(((ars * brs) + (ais * bis)) * oos,
			   ((ais * brs) - (ars * bis)) * oos);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()/rhs,lhs.imag()/rhs);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const ValueType& lhs, const complex<ValueType>& rhs){
    const ValueType cross_norm = lhs * rhs.real();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm/rhs_norm,(-lhs.real() * rhs.imag()) / rhs_norm);
  }

  template <>
    __host__ __device__
    inline complex<float> operator/(const float& lhs, const complex<float>& rhs){
    return complex<float>(lhs) / rhs;
  }
  template <>
    __host__ __device__
    inline complex<double> operator/(const double& lhs, const complex<double>& rhs){
    return complex<double>(lhs) / rhs;
  }

  // Unary arithmetic operations
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& rhs){
    return rhs;
  }
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& rhs){
    return rhs*-ValueType(1);
  }

  // Equality operators
  template <typename ValueType>
    __host__ __device__
    inline bool operator==(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    if(lhs.real() == rhs.real() && lhs.imag() == rhs.imag()){
      return true;
    }
    return false;
  }

  template <typename ValueType>
    __host__ __device__
    inline bool operator==(const ValueType & lhs, const complex<ValueType>& rhs){
    if(lhs == rhs.real() && rhs.imag() == 0){
      return true;
    }
    return false;
  }
  template <typename ValueType>
    __host__ __device__
    inline bool operator==(const complex<ValueType> & lhs, const ValueType& rhs){
    if(lhs.real() == rhs && lhs.imag() == 0){
      return true;
    }
    return false;
  }


  template <typename ValueType>
    __host__ __device__
    inline bool operator!=(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType>
    __host__ __device__
    inline bool operator!=(const ValueType & lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType>
    __host__ __device__
    inline bool operator!=(const complex<ValueType> & lhs, const ValueType& rhs){
    return !(lhs == rhs);
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> conj(const complex<ValueType>& z){
    return complex<ValueType>(z.real(),-z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType abs(const complex<ValueType>& z){
    return ::hypot(z.real(),z.imag());
  }
  template <>
    __host__ __device__
    inline float abs(const complex<float>& z){
    return ::hypotf(z.real(),z.imag());
  }
  template<>
    __host__ __device__
    inline double abs(const complex<double>& z){
    return ::hypot(z.real(),z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType arg(const complex<ValueType>& z){
    return atan2(z.imag(),z.real());
  }
  template<>
    __host__ __device__
    inline float arg(const complex<float>& z){
    return atan2f(z.imag(),z.real());
  }
  template<>
    __host__ __device__
    inline double arg(const complex<double>& z){
    return atan2(z.imag(),z.real());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType norm(const complex<ValueType>& z){
    return z.real()*z.real() + z.imag()*z.imag();
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> polar(const ValueType & m, const ValueType & theta){
    return complex<ValueType>(m * ::cos(theta),m * ::sin(theta));
  }

  template <>
    __host__ __device__
    inline complex<float> polar(const float & magnitude, const float & angle){
    return complex<float>(magnitude * ::cosf(angle),magnitude * ::sinf(angle));
  }

  template <>
    __host__ __device__
    inline complex<double> polar(const double & magnitude, const double & angle){
    return complex<double>(magnitude * ::cos(angle),magnitude * ::sin(angle));
  }

  // Transcendental functions implementation
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cos(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::cos(re) * ::cosh(im), -::sin(re) * ::sinh(im));
  }

  template <>
    __host__ __device__
    inline complex<float> cos(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(cosf(re) * coshf(im), -sinf(re) * sinhf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cosh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::cosh(re) * ::cos(im), ::sinh(re) * ::sin(im));
  }

  template <>
    __host__ __device__
    inline complex<float> cosh(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::coshf(re) * ::cosf(im), ::sinhf(re) * ::sinf(im));
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> exp(const complex<ValueType>& z){
    return polar(::exp(z.real()),z.imag());
  }

  template <>
    __host__ __device__
    inline complex<float> exp(const complex<float>& z){
    return polar(::expf(z.real()),z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> log(const complex<ValueType>& z){
    return complex<ValueType>(::log(abs(z)),arg(z));
  }

  template <>
    __host__ __device__
    inline complex<float> log(const complex<float>& z){
    return complex<float>(::logf(abs(z)),arg(z));
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> log10(const complex<ValueType>& z){
    // Using the explicit literal prevents compile time warnings in
    // devices that don't support doubles
    return log(z)/ValueType(2.30258509299404568402);
    // return log(z)/ValueType(::log(10.0));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const ValueType & exponent){
    return exp(log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const complex<ValueType> & exponent){
    return exp(log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const ValueType & x, const complex<ValueType> & exponent){
    return exp(::log(x)*exponent);
  }

  template <>
    __host__ __device__
    inline complex<float> pow(const float & x, const complex<float> & exponent){
    return exp(::logf(x)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z,const int & exponent){
    return exp(log(z)*ValueType(exponent));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sin(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::sin(re) * ::cosh(im), ::cos(re) * ::sinh(im));
  }

  template <>
    __host__ __device__
    inline complex<float> sin(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::sinf(re) * ::coshf(im), ::cosf(re) * ::sinhf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sinh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::sinh(re) * ::cos(im), ::cosh(re) * ::sin(im));
  }

  template <>
    __host__ __device__
    inline complex<float> sinh(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::sinhf(re) * ::cosf(im), ::coshf(re) * ::sinf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sqrt(const complex<ValueType>& z){
    return polar(::sqrt(abs(z)),arg(z)/ValueType(2));
  }

  template <>
    __host__ __device__
    inline complex<float> sqrt(const complex<float>& z){
    return polar(::sqrtf(abs(z)),arg(z)/float(2));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tan(const complex<ValueType>& z){
    return sin(z)/cos(z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tanh(const complex<ValueType>& z){
    // This implementation seems better than the simple sin/cos
    return (exp(ValueType(2)*z)-ValueType(1))/(exp(ValueType(2)*z)+ValueType(1));
    // return sinh(z)/cosh(z);
  }

  // Inverse trigonometric functions implementation

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acos(const complex<ValueType>& z){
    const complex<ValueType> ret = asin(z);
    return complex<ValueType>(ValueType(M_PI/2.0) - ret.real(),-ret.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asin(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*asinh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atan(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*atanh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acosh(const complex<ValueType>& z){
    quda::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				 ValueType(2.0) * z.real() * z.imag());
    ret = sqrt(ret);
    if (z.real() < ValueType(0.0)){
      ret = -ret;
    }
    ret += z;
    ret = log(ret);
    if (ret.real() < ValueType(0.0)){
      ret = -ret;
    }
    return ret;

    /*
      quda::complex<ValueType> ret = log(sqrt(z*z-ValueType(1))+z);
      if(ret.real() < 0){
      ret.real(-ret.real());
      }
      return ret;
    */
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asinh(const complex<ValueType>& z){
    return log(sqrt(z*z+ValueType(1))+z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atanh(const complex<ValueType>& z){
    ValueType imag2 = z.imag() * z.imag();
    ValueType n = ValueType(1.0) + z.real();
    n = imag2 + n * n;

    ValueType d = ValueType(1.0) - z.real();
    d = imag2 + d * d;
    complex<ValueType> ret(ValueType(0.25) * (::log(n) - ::log(d)),0);

    d = ValueType(1.0) - z.real() * z.real() - imag2;

    ret.imag(ValueType(0.5) * ::atan2(ValueType(2.0) * z.imag(), d));
    return ret;
    //return (log(ValueType(1)+z)-log(ValueType(1)-z))/ValueType(2);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<float> atanh(const complex<float>& z){
    float imag2 = z.imag() * z.imag();
    float n = float(1.0) + z.real();
    n = imag2 + n * n;

    float d = float(1.0) - z.real();
    d = imag2 + d * d;
    complex<float> ret(float(0.25) * (::logf(n) - ::logf(d)),0);

    d = float(1.0) - z.real() * z.real() - imag2;

    ret.imag(float(0.5) * ::atan2f(float(2.0) * z.imag(), d));
    return ret;
    //return (log(ValueType(1)+z)-log(ValueType(1)-z))/ValueType(2);

  }

  template <typename real> __host__ __device__ inline complex<real> cmul(const complex<real> &x, const complex<real> &y)
  {
    complex<real> w;
    w.x = x.real() * y.real();
    w.x -= x.imag() * y.imag();
    w.y = x.imag() * y.real();
    w.y += x.real() * y.imag();
    return w;
  }

  template <typename real>
  __host__ __device__ inline complex<real> cmac(const complex<real> &x, const complex<real> &y, const complex<real> &z)
  {
    complex<real> w = z;
    w.x += x.real() * y.real();
    w.x -= x.imag() * y.imag();
    w.y += x.imag() * y.real();
    w.y += x.real() * y.imag();
    return w;
  }

  template <typename T1, typename T2, typename T3>
  __host__ __device__ inline auto cmac(const T1 &x, const T2 &y, const T3 &z)
  {
    static_assert(std::is_same<typename T1::value_type, typename T2::value_type>::value
                    && std::is_same<typename T1::value_type, typename T3::value_type>::value,
                  "precisions do not match");

    using real = typename T1::value_type;
    complex<real> X = x;
    complex<real> Y = y;
    complex<real> Z = z;
    Z.real(Z.real() + X.real() * Y.real());
    Z.real(Z.real() - X.imag() * Y.imag());
    Z.imag(Z.imag() + X.imag() * Y.real());
    Z.imag(Z.imag() + X.real() * Y.imag());
    return Z;
  }

  template <typename real> __host__ __device__ inline complex<real> i_(const complex<real> &a)
  {
    // FIXME compiler generates worse code with "optimal" code
#if 1
    return complex<real>(0.0, 1.0) * a;
#else
    return complex<real>(-a.imag(), a.real());
#endif
  }

} // end namespace quda
