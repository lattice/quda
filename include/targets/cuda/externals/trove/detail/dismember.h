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
#include <trove/utility.h>
#include <thrust/detail/static_assert.h>

namespace trove {
namespace detail {


template<typename T,
         bool use_int=size_multiple_power_of_two<T, 2>::value,
         bool use_int2=size_multiple_power_of_two<T, 3>::value,
         bool use_int4=size_multiple_power_of_two<T, 4>::value >
struct dismember_type {
    typedef char type;
};

template<typename T>
struct dismember_type<T, true, false, false> {
    typedef int type;
};

template<typename T>
struct dismember_type<T, true, true, false> {
    typedef int2 type;
};

template<typename T>
struct dismember_type<T, true, true, true> {
    typedef int4 type;
};


template<typename T, typename U>
struct aliased_size {
    static const int value = sizeof(T) / sizeof(U);
    //Assert sizeof(T) % sizeof(U) == 0
    THRUST_STATIC_ASSERT(sizeof(T) % sizeof(U) == 0);
};

template<typename T,
         typename U=typename dismember_type<T>::type,
         int r=aliased_size<T, U>::value>
struct dismember {
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__
    static result_type impl(const T& t) {
        U tmp;
        memcpy(&tmp, reinterpret_cast<const char*>(&t) + idx * sizeof(U), sizeof(U));
        return result_type(tmp, dismember<T, U, r-1>::impl(t));
    }
};

template<typename T, typename U>
struct dismember<T, U, 1> {
    typedef array<U, 1> result_type;
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__
    static result_type impl(const T& t) {
        U tmp;
        memcpy(&tmp, reinterpret_cast<const char*>(&t) + idx * sizeof(U), sizeof(U));
        return result_type(tmp);
    }
};


template<typename T,
         typename U=typename dismember_type<T>::type,
         int r=aliased_size<T, U>::value>
struct remember {
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__
    static void impl(const array<U, r>& d, T& t) {
        memcpy(reinterpret_cast<char*>(&t) + idx * sizeof(U), &d.head, sizeof(d.head));
        remember<T, U, r-1>::impl(d.tail, t);
    }
};

template<typename T, typename U>
struct remember<T, U, 1> {
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__
    static void impl(const array<U, 1>& d, T& t) {
        memcpy(reinterpret_cast<char*>(&t) + idx * sizeof(U), &d.head, sizeof(d.head));
    }
};


template<typename U, typename T>
__host__ __device__
array<U, detail::aliased_size<T, U>::value> lyse(const T& in) {
    return detail::dismember<T, U>::impl(in);
}

template<typename T, typename U>
__host__ __device__
T fuse(const array<U, detail::aliased_size<T, U>::value>& in) {
    T result;
    detail::remember<T, U>::impl(in, result);
    return result;
}

}
}
