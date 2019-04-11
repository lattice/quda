#pragma once

#include <complex_quda.h>
#include <quda_matrix.h>

/**
 * @file    color_spinor.h
 *
 * @section Description
 *
 * The header file defines some helper structs for dealing with
 * ColorSpinors (e.g., a vector with both color and spin degrees of
 * freedom).
 */
namespace quda {

  template<typename Float, typename T> struct colorspinor_wrapper;
  template<typename Float, typename T> struct colorspinor_ghost_wrapper;

  /**
     This is the generic declaration of ColorSpinor.
   */
  template <typename Float, int Nc, int Ns>
  struct ColorSpinorSextet {

    static constexpr int color_size = Nc * (Nc + 1) / 2;
    static constexpr int size = color_size * Ns;
    complex<Float> data[size];

    __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>()
    {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
      }

      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>(const ColorSpinorSextet<Float, Nc, Ns> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }

      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>& operator=(const ColorSpinorSextet<Float, Nc, Ns> &a) {
	if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
        }
	return *this;
      }

      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns> operator-() const
      {
        ColorSpinorSextet<Float, Nc, Ns> a;
#pragma unroll
        for (int i = 0; i < size; i++) { a.data[i] = -data[i]; }
        return a;
      }

      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>& operator+=(const ColorSpinorSextet<Float, Nc, Ns> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] += a.data[i]; }
        return *this;
      }

      template <typename T> __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns> &operator*=(const T &a)
      {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] *= a; }
        return *this;
      }

      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns> &operator-=(const ColorSpinorSextet<Float, Nc, Ns> &a)
      {
        if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] -= a.data[i]; }
        }
        return *this;
      }

      template<typename S>
      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>(const colorspinor_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline ColorSpinorSextet<Float, Nc, Ns>(const colorspinor_ghost_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const colorspinor_ghost_wrapper<Float, S> &s);

      /**
	 @brief 2-d accessor functor
	 @param[in] s Spin index
	 @param[in] i Color row index
	 @param[in] j Color column index
	 @return Complex number at this spin and color index
      */
      __device__ __host__ inline complex<Float>& operator()(int s, int i, int j)
      {
        constexpr int n = Nc;
        int k = s*n;
	if (i==j) {
          k += i;
	} else if (j<i) {
	  k += n*(n+1)/2 - (n-j)*(n-j-1)/2 + i - j - 1;
	} else { // i>j
	  // switch coordinates to count from bottom right instead of top left of matrix
	  k += n*(n+1)/2 - (n-i)*(n-i-1)/2 + j - i - 1;
	}
        return data[k];
      }

      /**
	 @brief 2-d accessor functor
	 @param[in] s Spin index
	 @param[in] i Color row index
	 @param[in] j Color column index
	 @return Complex number at this spin and color index
      */
    __device__ __host__ inline const complex<Float>& operator()(int s, int i, int j) const {
        constexpr int n = Nc;
        int k = s*n;
	if (i==j) {
          k += i;
	} else if (j<i) {
	  k += n*(n+1)/2 - (n-j)*(n-j-1)/2 + i - j - 1;
	} else { // i>j
	  // switch coordinates to count from bottom right instead of top left of matrix
	  k += n*(n+1)/2 - (n-i)*(n-i-1)/2 + j - i - 1;
	}
        return data[k];
      }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline complex<Float>& operator()(int idx) { return data[idx]; }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline const complex<Float>& operator()(int idx) const { return data[idx]; }
//???
      __device__ __host__ void print() const
      {
        for (int s=0; s<Ns; s++) {
          for (int i=0; i<Nc; i++) {
            printf("s=%d i=%d", s, i);
            for (int j=0; j<Nc; j++) {
              printf(" (%e, %e)", (*this)(s,i,j).real(), (*this)(s,i,j).imag());
            }
            printf("\n");
          }
        }
      }
    };

  /**
     @brief ColorSpinorSextet addition operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
  ColorSpinorSextet<Float,Nc,Ns> operator+(const ColorSpinorSextet<Float,Nc,Ns> &x, const ColorSpinorSextet<Float,Nc,Ns> &y) {

    ColorSpinorSextet<Float,Nc,Ns> z;
#pragma unroll
    for (int i=0; i<x.size; i++) z.data[i] = x.data[i] + y.data[i];

    return z;
  }

  /**
     @brief ColorSpinorSextet addition operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
  ColorSpinorSextet<Float,Nc,Ns> operator-(const ColorSpinorSextet<Float,Nc,Ns> &x, const ColorSpinorSextet<Float,Nc,Ns> &y) {

    ColorSpinorSextet<Float,Nc,Ns> z;
#pragma unroll
    for (int i=0; i<x.size; i++) z.data[i] = x.data[i] - y.data[i];

    return z;
  }

  /**
     @brief Compute the scalar-vector product y = a * x
     @param[in] a Input scalar
     @param[in] x Input vector
     @return The vector a * x
  */
  template<typename Float, int Nc, int Ns, typename S> __device__ __host__ inline
    ColorSpinorSextet<Float,Nc,Ns> operator*(const S &a, const ColorSpinorSextet<Float,Nc,Ns> &x) {

    ColorSpinorSextet<Float,Nc,Ns> y;

#pragma unroll
    for (int i=0; i<x.size; i++) y.data[i] = a * x.data[i];

    return y;
  }

  /**
     @brief Compute the matrix-vector product y = A * x
     @param[in] A Input matrix
     @param[in] x Input vector
     @return The vector A * x
  */
  template<typename Float, int Nc, int Ns> __device__ __host__ inline
  ColorSpinorSextet<Float,Nc,Ns> operator*(const Matrix<complex<Float>,Nc> &A, const ColorSpinorSextet<Float,Nc,Ns> &x)
  {
    ColorSpinorSextet<Float,Nc,Ns> y;

#pragma unroll
    for (int s=0; s<Ns; s++) {
      complex<Float> a[Nc*Nc];

#pragma unroll
      for (int l=0; l<Nc; l++) {
#pragma unroll
        for (int k=0; k<Nc; k++) {

#pragma unroll
          for (int i=0;i<Nc*Nc;i++) a[i] = 0;

#pragma unroll
          for (int i=0; i<Nc; i++) {
#pragma unroll
            for (int j=0; j<Nc; j++) {
              a[i*Nc+l] += x(s,i,j)*A(j,l);
            }
            y(s,k,l) = A(k,i) * a[i*Nc+l];
          }
        }
      }
    }

    return y;
  }

} // namespace quda
