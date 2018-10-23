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
#include <trove/aos.h>
#include <trove/warp.h>

namespace trove {


template<int s, typename T, typename I>
__device__
trove::array<T, s> load_array_warp_contiguous(const T* src, const I& idx) {
    typedef trove::array<T, s> array_type;
    const array_type* src_ptr = (const array_type*)(src) + idx;
    return load_warp_contiguous(src_ptr);
}

template<int s, typename T, typename I>
__device__
trove::array<T, s> load_array(const T* src, const I& idx) {
    typedef trove::array<T, s> array_type;
    const array_type* src_ptr = (const array_type*)(src) + idx;
    return *src_ptr;
}

template<int s, typename T, typename I>
__device__
void store_array_warp_contiguous(T* dest, const I& idx, const trove::array<T, s>& src) {
    typedef trove::array<T, s> array_type;
    array_type* dest_ptr = (array_type*)(dest) + idx;
    store_warp_contiguous(src, dest_ptr);
}

template<int s, typename T, typename I>
__device__
void store_array(T* dest, const I& idx, const trove::array<T, s>& src) {
    typedef trove::array<T, s> array_type;
    array_type* dest_ptr = (array_type*)(dest) + idx;
    *dest_ptr = src;
}

}
