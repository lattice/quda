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
#include <trove/detail/dismember.h>
#include <trove/detail/fallback.h>
#include <trove/warp.h>
#include <trove/transpose.h>
#include <trove/utility.h>
#include <trove/memory.h>

namespace trove {

namespace detail {

template<typename T>
struct size_in_range {
    typedef typename dismember_type<T>::type U;
    static const int size = aliased_size<T, U>::value;
    static const bool value = (size > 1) && (size < 64);
};

template<typename T, bool s=size_multiple_power_of_two<T, 2>::value, bool r=size_in_range<T>::value>
struct use_shfl {
    static const bool value = false;
};

template<typename T>
struct use_shfl<T, true, true> {
    static const bool value = true;
};

template<typename T>
struct use_direct {
    static const bool value = !(use_shfl<T>::value);
};

}


template<typename T>
__device__ typename enable_if<detail::use_shfl<T>::value, T>::type
load_warp_contiguous(const T* src) {
    int warp_id = threadIdx.x & WARP_MASK;
    const T* warp_begin_src = src - warp_id;
    typedef typename detail::dismember_type<T>::type U;
    const U* as_int_src = (const U*)warp_begin_src;
    typedef array<U, detail::aliased_size<T, U>::value> int_store;
    int_store loaded = warp_load<int_store>(as_int_src, warp_id);
    r2c_warp_transpose(loaded);
    return detail::fuse<T>(loaded);
}

template<typename T>
__device__ typename enable_if<detail::use_direct<T>::value, T>::type
load_warp_contiguous(const T* src) {
    return detail::divergent_load(src);
}


template<typename T>
__device__ typename enable_if<detail::use_shfl<T>::value>::type
store_warp_contiguous(const T& data, T* dest) {
    int warp_id = threadIdx.x & WARP_MASK;
    T* warp_begin_dest = dest - warp_id;
    typedef typename detail::dismember_type<T>::type U;
    U* as_int_dest = (U*)warp_begin_dest;
    typedef array<U, detail::aliased_size<T, U>::value> int_store;
    int_store lysed = detail::lyse<U>(data);
    c2r_warp_transpose(lysed);
    warp_store(lysed, as_int_dest, warp_id);
}

template<typename T>
__device__ typename enable_if<detail::use_direct<T>::value>::type
store_warp_contiguous(const T& data, T* dest) {
    detail::divergent_store(data, dest);
}


namespace detail {

template<typename T>
__device__ typename detail::dismember_type<T>::type*
compute_address(T* src, int div, int mod) {
    typedef typename detail::dismember_type<T>::type U;
    T* base_ptr = __shfl(src, div);
    U* result = ((U*)(base_ptr) + mod);
    return result;
}

template<typename T>
struct address_constants {
    typedef typename detail::dismember_type<T>::type U;
    static const int m = aliased_size<T, U>::value;
    static const int mod_offset = WARP_SIZE % m;
    static const int div_offset = WARP_SIZE / m;
};

template<typename T>
__device__ void update_indices(int& div, int& mod) {
    mod += address_constants<T>::mod_offset;
    if (mod >= address_constants<T>::m) {
        mod -= address_constants<T>::m;
        div += 1;
    }
    div += address_constants<T>::div_offset;
}
        

template<int s, typename T>
struct indexed_load {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, s> impl(const T* src, int div, int mod) {
        U result;
        U* address = compute_address(src, div, mod);
        result = *address;
        update_indices<T>(div, mod);
        
        
        return array<U, s>(
            result,
            indexed_load<s-1, T>::impl(src, div, mod));
    }
};

template<typename T>
struct indexed_load<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, 1> impl(const T* src, int div, int mod) {
        U result;
        U* address = compute_address(src, div, mod);
        result = *address;
        return array<U, 1>(result);
    }
};

template<int s, typename T>
struct indexed_store {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, s>& src,
                     T* dest, int div, int mod) {
        U* address = compute_address(dest, div, mod);
        *address = src.head;
        update_indices<T>(div, mod);
        indexed_store<s-1, T>::impl(src.tail, dest, div, mod);
    }
};

template<typename T>
struct indexed_store<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, 1>& src,
                     T* dest, int div, int mod) {
        U* address = compute_address(dest, div, mod);
        *address = src.head;
    }
};

template<typename T>
__device__
bool is_contiguous(int warp_id, const T* ptr) {
    int neighbor_idx = (warp_id == 0) ? 0 : warp_id-1;
    const T* neighbor_ptr = __shfl(ptr, neighbor_idx);
    bool neighbor_contiguous = (warp_id == 0) ? true : (ptr - neighbor_ptr == sizeof(T));
    bool result = __all(neighbor_contiguous);
    return result;
}


template<typename T>
__device__ typename enable_if<use_shfl<T>::value, T>::type
load_dispatch(const T* src) {
    int warp_id = threadIdx.x & WARP_MASK;
    // if (detail::is_contiguous(warp_id, src)) {
    //     return detail::load_warp_contiguous(src);
    // } else {
        typedef typename detail::dismember_type<T>::type U;
        typedef array<U, detail::aliased_size<T, U>::value> u_store;
        u_store loaded =
            detail::indexed_load<detail::aliased_size<T, U>::value, T>::impl(
                src,
                warp_id / address_constants<T>::m,
                warp_id % address_constants<T>::m);
        r2c_warp_transpose(loaded);
        return detail::fuse<T>(loaded);
    // }   
}



template<typename T>
__device__ typename enable_if<use_direct<T>::value, T>::type
load_dispatch(const T* src) {
    return detail::divergent_load(src);
}


template<typename T>
__device__ typename enable_if<use_shfl<T>::value>::type
store_dispatch(const T& data, T* dest) {
    int warp_id = threadIdx.x & WARP_MASK;
    // if (detail::is_contiguous(warp_id, dest)) {
    //     detail::store_warp_contiguous(data, dest);
    // } else {
        typedef typename detail::dismember_type<T>::type U;
        typedef array<U, detail::aliased_size<T, U>::value> u_store;
        u_store lysed = detail::lyse<U>(data);
        c2r_warp_transpose(lysed);
        detail::indexed_store<detail::aliased_size<T, U>::value, T>::impl(
            lysed, dest,
            warp_id / address_constants<T>::m,
            warp_id % address_constants<T>::m);
    // }
}

template<typename T>
__device__ typename enable_if<use_direct<T>::value>::type
store_dispatch(const T& data, T* dest) {
    detail::divergent_store(data, dest);
}

  
}

template<typename T>
__device__ T load(const T* src) {
    if (warp_converged()) {
        return detail::load_dispatch(src);
    } else {
        return detail::divergent_load(src);
    }
}

template<typename T>
__device__ void store(const T& data, T* dest) {
    if (warp_converged()) {
        detail::store_dispatch(data, dest);
    } else {
        detail::divergent_store(data, dest);
    }
}

}
