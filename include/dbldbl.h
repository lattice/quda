/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Release 1.2
 *
 * (1) Deployed new implementation of div_dbldbl() and sqrt_dbldbl() based on
 *     Newton-Raphson iteration, providing significant speedup.
 * (2) Added new function rsqrt_dbldbl() which provides reciprocal square root.
 *
 * Release 1.1
 *
 * (1) Fixed a bug affecting add_dbldbl() and sub_dbldbl() that in very rare
 *     cases returned results with reduced accuracy.
 * (2) Replaced the somewhat inaccurate error bounds with the experimentally
 *     observed maximum relative error.
 */

#pragma once

#include <math_helper.cuh>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#include <math.h>       /* import sqrt() */

/* The head of a double-double number is stored in the most significant part
   of a double2 (the y-component). The tail is stored in the least significant
   part of the double2 (the x-component). All double-double operands must be
   normalized on both input to and return from all basic operations, i.e. the
   magnitude of the tail shall be <= 0.5 ulp of the head.
*/
typedef double2 dbldbl;

/* Create a double-double from two doubles. No normalization is performed,
   so the head and tail components passed in must satisfy the normalization
   requirement. To create a double-double from two arbitrary double-precision
   numbers, use add_double_to_dbldbl().
*/
__device__ __forceinline__ dbldbl make_dbldbl (double head, double tail)
{
    dbldbl z;
    z.x = tail;
    z.y = head;
    return z;
}

/* Return the head of a double-double number */
__device__ __forceinline__ double get_dbldbl_head (dbldbl a)
{
    return a.y;
}

/* Return the tail of a double-double number */
__device__ __forceinline__ double get_dbldbl_tail (dbldbl a)
{
    return a.x;
}

/* Compute error-free sum of two unordered doubles. See Knuth, TAOCP vol. 2 */
__device__ __forceinline__ dbldbl add_double_to_dbldbl (double a, double b)
{
    double t1, t2;
    dbldbl z;
    z.y = __dadd_rn (a, b);
    t1 = __dadd_rn (z.y, -a);
    t2 = __dadd_rn (z.y, -t1);
    t1 = __dadd_rn (b, -t1);
    t2 = __dadd_rn (a, -t2);
    z.x = __dadd_rn (t1, t2);
    return z;
}

/* Compute error-free product of two doubles. Take full advantage of FMA */
__device__ __forceinline__ dbldbl mul_double_to_dbldbl (double a, double b)
{
    dbldbl z;
    z.y = __dmul_rn (a, b);
    z.x = __fma_rn (a, b, -z.y);
    return z;
}

/* Negate a double-double number, by separately negating head and tail */
__device__ __forceinline__ dbldbl neg_dbldbl (dbldbl a)
{
    dbldbl z;
    z.y = -a.y;
    z.x = -a.x;
    return z;
}

/* Compute high-accuracy sum of two double-double operands. In the absence of
   underflow and overflow, the maximum relative error observed with 10 billion
   test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dbldbl add_dbldbl (dbldbl a, dbldbl b)
{
    dbldbl z;
    double t1, t2, t3, t4, t5, e;
    t1 = __dadd_rn (a.y, b.y);
    t2 = __dadd_rn (t1, -a.y);
    t3 = __dadd_rn (__dadd_rn (a.y, t2 - t1), __dadd_rn (b.y, -t2));
    t4 = __dadd_rn (a.x, b.x);
    t2 = __dadd_rn (t4, -a.x);
    t5 = __dadd_rn (__dadd_rn (a.x, t2 - t4), __dadd_rn (b.x, -t2));
    t3 = __dadd_rn (t3, t4);
    t4 = __dadd_rn (t1, t3);
    t3 = __dadd_rn (t1 - t4, t3);
    t3 = __dadd_rn (t3, t5);
    z.y = e = __dadd_rn (t4, t3);
    z.x = __dadd_rn (t4 - e, t3);
    return z;
}

/* Compute high-accuracy difference of two double-double operands. In the
   absence of underflow and overflow, the maximum relative error observed
   with 10 billion test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dbldbl sub_dbldbl (dbldbl a, dbldbl b)
{
    dbldbl z;
    double t1, t2, t3, t4, t5, e;
    t1 = __dadd_rn (a.y, -b.y);
    t2 = __dadd_rn (t1, -a.y);
    t3 = __dadd_rn (__dadd_rn (a.y, t2 - t1), - __dadd_rn (b.y, t2));
    t4 = __dadd_rn (a.x, -b.x);
    t2 = __dadd_rn (t4, -a.x);
    t5 = __dadd_rn (__dadd_rn (a.x, t2 - t4), - __dadd_rn (b.x, t2));
    t3 = __dadd_rn (t3, t4);
    t4 = __dadd_rn (t1, t3);
    t3 = __dadd_rn (t1 - t4, t3);
    t3 = __dadd_rn (t3, t5);
    z.y = e = __dadd_rn (t4, t3);
    z.x = __dadd_rn (t4 - e, t3);
    return z;
}

/* Compute high-accuracy product of two double-double operands, taking full
   advantage of FMA. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was 5.238480533564479e-32
   (~= 2**-103.9125).
*/
__device__ __forceinline__ dbldbl mul_dbldbl (dbldbl a, dbldbl b)
{
    dbldbl t, z;
    double e;
    t.y = __dmul_rn (a.y, b.y);
    t.x = __fma_rn (a.y, b.y, -t.y);
    t.x = __fma_rn (a.x, b.x, t.x);
    t.x = __fma_rn (a.y, b.x, t.x);
    t.x = __fma_rn (a.x, b.y, t.x);
    z.y = e = __dadd_rn (t.y, t.x);
    z.x = __dadd_rn (t.y - e, t.x);
    return z;
}

/* Compute high-accuracy quotient of two double-double operands, using Newton-
   Raphson iteration. Based on: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
   Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
   In Proceedings of the 8th International Conference on Computational Science,
   ICCS '08, Part I, pp. 446-455. In the absence of underflow and overflow, the
   maximum relative error observed with 10 billion test cases was
   1.0161322480099059e-31 (~= 2**-102.9566).
*/
__device__ __forceinline__ dbldbl div_dbldbl (dbldbl a, dbldbl b)
{
    dbldbl t, z;
    double e, r;
    r = 1.0 / b.y;
    t.y = __dmul_rn (a.y, r);
    e = __fma_rn (b.y, -t.y, a.y);
    t.y = __fma_rn (r, e, t.y);
    t.x = __fma_rn (b.y, -t.y, a.y);
    t.x = __dadd_rn (a.x, t.x);
    t.x = __fma_rn (b.x, -t.y, t.x);
    e = __dmul_rn (r, t.x);
    t.x = __fma_rn (b.y, -e, t.x);
    t.x = __fma_rn (r, t.x, e);
    z.y = e = __dadd_rn (t.y, t.x);
    z.x = __dadd_rn (t.y - e, t.x);
    return z;
}

/* Compute high-accuracy square root of a double-double number. Newton-Raphson
   iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
   High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
   1997, pp. 561-589. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was
   3.7564109505601846e-32 (~= 2**-104.3923).
*/
__device__ __forceinline__ dbldbl sqrt_dbldbl (dbldbl a)
{
    dbldbl t, z;
    double e, y, s, r;
    r = quda::rsqrt(a.y);
    if (a.y == 0.0) r = 0.0;
    y = __dmul_rn (a.y, r);
    s = __fma_rn (y, -y, a.y);
    r = __dmul_rn (0.5, r);
    z.y = e = __dadd_rn (s, a.x);
    z.x = __dadd_rn (s - e, a.x);
    t.y = __dmul_rn (r, z.y);
    t.x = __fma_rn (r, z.y, -t.y);
    t.x = __fma_rn (r, z.x, t.x);
    r = __dadd_rn (y, t.y);
    s = __dadd_rn (y - r, t.y);
    s = __dadd_rn (s, t.x);
    z.y = e = __dadd_rn (r, s);
    z.x = __dadd_rn (r - e, s);
    return z;
}

/* Compute high-accuracy reciprocal square root of a double-double number.
   Based on Newton-Raphson iteration. In the absence of underflow and overflow,
   the maximum relative error observed with 10 billion test cases was
   6.4937771666026349e-32 (~= 2**-103.6026)
*/
__device__ __forceinline__ dbldbl rsqrt_dbldbl (dbldbl a)
{
    dbldbl z;
    double r, s, e;
    r = quda::rsqrt(a.y);
    e = __dmul_rn (a.y, r);
    s = __fma_rn (e, -r, 1.0);
    e = __fma_rn (a.y, r, -e);
    s = __fma_rn (e, -r, s);
    e = __dmul_rn (a.x, r);
    s = __fma_rn (e, -r, s);
    e = 0.5 * r;
    z.y = __dmul_rn (e, s);
    z.x = __fma_rn (e, s, -z.y);
    s = __dadd_rn (r, z.y);
    r = __dadd_rn (r, -s);
    r = __dadd_rn (r, z.y);
    r = __dadd_rn (r, z.x);
    z.y = e = __dadd_rn (s, r);
    z.x = __dadd_rn (s - e, r);
    return z;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

/**
   This C++ wrapper over the above dbldbl functions for easy
   inclusion in C++ generic template code, e.g., CUB.
 */
struct doubledouble {

  dbldbl a;

  __device__ __host__ doubledouble() { a.x = 0.0; a.y = 0.0; }
  __device__ __host__ doubledouble(const doubledouble &a) : a(a.a) { }
  __device__ __host__ doubledouble(const dbldbl &a) : a(a) { }
  __device__ __host__ doubledouble(const double &head, const double &tail) { a.y = head; a.x = tail; }
  __device__ __host__ doubledouble(const double &head) { a.y = head; a.x = 0.0; }

  __device__ __host__ doubledouble& operator=(const double &head) {
    this->a.y = head;
    this->a.x = 0.0;
  }

  __device__ doubledouble& operator+=(const doubledouble &a) {
    this->a = add_dbldbl(this->a, a.a);
    return *this;
  }

  __device__ __host__ double head() const { return a.y; }
  __device__ __host__ double tail() const { return a.x; }

  __device__ __host__ void print() const { printf("scalar: %16.14e + %16.14e\n", head(), tail()); }

};

__device__  inline bool operator>(const doubledouble &a, const double &b) {
  return a.head() > b;
}

__device__  inline doubledouble operator+(const doubledouble &a, const doubledouble &b) {
  return doubledouble(add_dbldbl(a.a,b.a));
}

__device__  inline doubledouble operator-(const doubledouble &a, const doubledouble &b) {
  return doubledouble(sub_dbldbl(a.a,b.a));
}

__device__ inline doubledouble operator*(const doubledouble &a, const doubledouble &b) {
  return  doubledouble(mul_dbldbl(a.a,b.a));
}

__device__ inline doubledouble operator/(const doubledouble &a, const doubledouble &b) {
  return doubledouble(div_dbldbl(a.a,b.a));
}

__device__ inline doubledouble add_double_to_doubledouble(const double &a, const double &b) {
  return doubledouble(add_double_to_dbldbl(a,b));
}

__device__ inline doubledouble mul_double_to_doubledouble(const double &a, const double &b) {
  return doubledouble(mul_double_to_dbldbl(a,b));
}

struct doubledouble2 {
  doubledouble x;
  doubledouble y;

  __device__ __host__ doubledouble2() : x(), y() { }
  __device__ __host__ doubledouble2(const doubledouble2 &a) : x(a.x), y(a.y) { }
  __device__ __host__ doubledouble2(const double2 &a) : x(a.x), y(a.y) { }
  __device__ __host__ doubledouble2(const doubledouble &x, const doubledouble &y) : x(x), y(y) { }

  __device__ doubledouble2& operator+=(const doubledouble2 &a) {
    x += a.x;
    y += a.y;
    return *this;
  }

  __device__ __host__ void print() const { printf("vec2: (%16.14e + %16.14e) (%16.14e + %16.14e)\n", x.head(), x.tail(), y.head(), y.tail()); }
};

struct doubledouble3 {
  doubledouble x;
  doubledouble y;
  doubledouble z;

  __device__ __host__ doubledouble3() : x(), y() { }
  __device__ __host__ doubledouble3(const doubledouble3 &a) : x(a.x), y(a.y), z(a.z) { }
  __device__ __host__ doubledouble3(const double3 &a) : x(a.x), y(a.y), z(a.z) { }
  __device__ __host__ doubledouble3(const doubledouble &x, const doubledouble &y, const doubledouble &z) : x(x), y(y), z(z) { }

  __device__ doubledouble3& operator+=(const doubledouble3 &a) {
    x += a.x;
    y += a.y;
    z += a.z;
    return *this;
  }

  __device__ __host__ void print() const { printf("vec3: (%16.14e + %16.14e) (%16.14e + %16.14e) (%16.14e + %16.14e)\n", x.head(), x.tail(), y.head(), y.tail(), z.head(), z.tail()); }
};

__device__ doubledouble2 operator+(const doubledouble2 &a, const doubledouble2 &b)
{ return doubledouble2(a.x + b.x, a.y + b.y); }

__device__ doubledouble3 operator+(const doubledouble3 &a, const doubledouble3 &b)
{ return doubledouble3(a.x + b.x, a.y + b.y, a.z + b.z); }
