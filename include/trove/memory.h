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
#include <trove/utility.h>
#include <trove/array.h>

namespace trove {
namespace detail {

template<typename Array>
struct warp_store_array {};

template<typename T, int s>
struct warp_store_array<array<T, s> > {
    __host__ __device__ static void impl(
        const array<T, s>& d,
        T* ptr, int offset, int stride) {
        ptr[offset] = d.head;
        warp_store_array<array<T, s-1> >::impl(
            d.tail, ptr, offset + stride, stride);
    }
};

template<typename T>
struct warp_store_array<array<T, 1> > {
    __host__ __device__ static void impl(
        const array<T, 1>& d,
        T* ptr, int offset, int stride) {
        ptr[offset] = d.head;
    }
};

template<typename Array>
struct uncoalesced_store_array{};

template<typename T, int s>
struct uncoalesced_store_array<array<T, s> > {
    __host__ __device__ static void impl(
        const array<T, s>& d,
        T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
        uncoalesced_store_array<array<T, s-1> >::impl(d.tail, ptr, offset+1,
                                                      stride);
    }
    __host__ __device__ static void impl(
        const array<T, s>& d,
        volatile T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
        uncoalesced_store_array<array<T, s-1> >::impl(d.tail, ptr, offset+1,
                                                      stride);
    }
};

template<typename T>
struct uncoalesced_store_array<array<T, 1> > {
    __host__ __device__ static void impl(
        const array<T, 1>& d,
        T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
    }
    __host__ __device__ static void impl(
        const array<T, 1>& d,
        volatile T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
    }
};

template<typename Array>
struct warp_load_array{};

template<typename T, int s>
struct warp_load_array<array<T, s> > {
    __host__ __device__ static array<T, s> impl(const T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, s>(ptr[offset],
                           warp_load_array<array<T, s-1> >::impl(ptr, offset+stride, stride));
    }
    __host__ __device__ static array<T, s> impl(const volatile T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, s>(ptr[offset],
                           warp_load_array<array<T, s-1> >::impl(ptr, offset+stride, stride));
    }
};

template<typename T>
struct warp_load_array<array<T, 1> > {
    __host__ __device__ static array<T, 1> impl(const T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, 1>(ptr[offset]);
    }
    __host__ __device__ static array<T, 1> impl(const volatile T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, 1>(ptr[offset]);
    }
};

} //end namespace detail

template<typename Array>
__host__ __device__ void warp_store(const Array& t,
                                    typename Array::head_type* ptr,
                                    int offset, int stride=32) {
    detail::warp_store_array<Array>::impl(t, ptr, offset, stride);
}

template<typename Array>
__host__ __device__ Array warp_load(const typename Array::head_type* ptr,
                                    int offset, int stride=32) {
    return detail::warp_load_array<Array>::impl(ptr, offset, stride);
}

template<typename Array>
__host__ __device__ Array warp_load(
    const volatile typename Array::head_type* ptr,
    int offset, int stride=32) {
    return detail::warp_load_array<Array>::impl(ptr, offset, stride);
}

template<typename Array>
__host__ __device__ void uncoalesced_store(const Array& t,
                                           typename Array::head_type* ptr,
                                           int stride=1) {
    detail::uncoalesced_store_array<Array>::impl(t, ptr, 0, stride);
}

template<typename Array>
__host__ __device__ void uncoalesced_store(const Array& t,
                                           volatile typename Array::head_type* ptr,
                                           int stride=1) {
    detail::uncoalesced_store_array<Array>::impl(t, ptr, 0, stride);
}

} //end namespace trove
