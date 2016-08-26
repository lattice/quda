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
namespace detail {

template<typename Array, int i, int j=0>
struct rotate_elements;

template<typename Array, int i, int j, bool non_terminal>
struct rotate_elements_helper {
    static const int size = Array::size;
    static const int other = (i + j) % size;
    static const bool new_non_terminal = j < size-2;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        if (a & i)
            trove::get<j>(r) = trove::get<other>(t);
        rotate_elements_helper<Array, i, j+1, new_non_terminal>::impl(t, a, r);
    }
};

template<typename Array, int i, int j>
struct rotate_elements_helper<Array, i, j, false> {
    static const int size = Array::size;
    static const int other = (i + j) % size;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        if (a & i)
            trove::get<j>(r) = trove::get<other>(t);
    }
};


template<typename Array, int i, int j>
struct rotate_elements{
    static const int size = Array::size;
    static const bool non_terminal = j < size-1;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        rotate_elements_helper<Array, i, 0, non_terminal>::impl(t, a, r);
    }
};

template<typename Array, int i>
struct rotate_impl;

template<typename Array, int i, bool non_terminal>
struct rotate_impl_helper {
    static const int size = Array::size;
    static const int next_i = i * 2;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        Array rotated = t;
        rotate_elements<Array, i>::impl(t, a, rotated);
        return rotate_impl<Array, next_i>::impl(rotated, a);
    }
};

template<typename Array, int i>
struct rotate_impl_helper<Array, i, false> {
    static const int size = Array::size;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        Array rotated = t;
        rotate_elements<Array, i>::impl(t, a, rotated);
        return rotated;
    }
};
    
template<typename Array, int i>
struct rotate_impl {
    static const int size = Array::size;
    static const int next_i = i * 2;
    static const bool non_terminal = next_i < size;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        return rotate_impl_helper<Array, i, non_terminal>::impl(t, a);
    }
};

} //ends namespace detail

template<typename T, int i>
__host__ __device__
array<T, i> rotate(const array<T, i>& t, int a) {
    return detail::rotate_impl<array<T, i>, 1>::impl(t, a);
}

} //ends namespace trove
