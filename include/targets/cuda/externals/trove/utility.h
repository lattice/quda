/*
Copyright (c) 2013, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <trove/array.h>

namespace trove {

template<typename T>
struct counting_array{};

template<typename T, int s>
struct counting_array<array<T, s> > {
    typedef array<T, s> Array;
    __host__ __device__
    static Array impl(T v=0, T i=1) {
        return Array(v,
                     counting_array<array<T, s-1> >::impl(v + i, i));
    }
};

template<typename T>
struct counting_array<array<T, 1> > {
    __host__ __device__
    static array<T, 1> impl(T v, T i=1) {
        return make_array(v);
    }
};

template<typename T>
struct sum_array {};

template<typename T, int s>
struct sum_array<array<T, s> > {
    typedef array<T, s> Array;
    __host__ __device__
    static T impl(const Array& a, const T& p) {
        return sum_array<typename Array::tail_type>::impl(a.tail, p + a.head);
    }
};

template<typename T>
struct sum_array<array<T, 1> > {
    typedef array<T, 1> Array;
    __host__ __device__
    static T impl(const Array& a, const T& p) {
        return p + a.head;
    }
};

template<typename T, int s>
__host__ __device__ T sum(const array<T, s>& a) {
    return sum_array<array<T, s> >::impl(a, 0);
}

template<int m>
struct static_log {
  static const int value = 1 + static_log< (m >> 1) >::value;
};

template<>
struct static_log<1> {
    static const int value = 0;
};

template<>
struct static_log<0> {
    //This functions as a static assertion
    //Don't take the log of 0!!
};

template<int m>
struct is_power_of_two {
    static const bool value = (m & (m-1)) == 0;
};

template<int m>
struct is_odd {
    static const bool value = (m & 1) == 1;
};

template<bool cond, typename T, typename Then, typename Else>
struct value_if {
    static const T value = Then::value;
};

template<typename T, typename Then, typename Else>
struct value_if<false, T, Then, Else> {
    static const T value = Else::value;
};

template<typename T, T x>
struct value_identity {
    static const T value = x;
};

template<typename T, template<T> class Fn, T x, T p=0>
struct inverse {
    static const T value =
        value_if<Fn<p>::value == x, T,
                 value_identity<T, p>, inverse<T, Fn, x, p+1> >::value;
};

struct null_type{};

template<typename T, T i, typename Tail=null_type>
struct cons_c {
    static const T head = i;
    typedef Tail tail;
};

template<int k, int l>
struct static_range {
    static const int head = k;
    typedef static_range<k+1, l> tail;
};

template<int f>
struct static_range<f, f> {
    static const int head = f;
    typedef null_type tail;
};

template<bool b, typename T=void>
struct enable_if {
    typedef T type;
};

template<typename T>
struct enable_if<false, T> {};

template<typename T, int p>
struct size_multiple_power_of_two {
    static const bool value = (sizeof(T) & ((1 << p) - 1)) == 0;
};


}
