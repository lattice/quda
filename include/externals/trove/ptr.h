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

namespace trove {
namespace detail {

template<typename T>
struct coalesced_ref {
    T* m_ptr;
    __device__ explicit coalesced_ref(T* ptr) : m_ptr(ptr) {}
    
    __device__ operator T() {
        return trove::load(m_ptr);
    }
    __device__ coalesced_ref& operator=(const T& data) {
        trove::store(data, m_ptr);
        return *this;
    }

    __device__ coalesced_ref& operator=(const coalesced_ref& other) {
        if (warp_converged()) {
            T data = detail::load_dispatch(other.m_ptr);
            detail::store_dispatch(data, m_ptr);
        } else {
            T data = detail::divergent_load(other.m_ptr);
            detail::divergent_store(data, m_ptr);
        }
        return *this;
    }
};
}

template<typename T>
struct coalesced_ptr {
    T* m_ptr;
    __device__ coalesced_ptr(T* ptr) : m_ptr(ptr) {}
    __device__ coalesced_ptr() : m_ptr(nullptr) {}
    __device__ trove::detail::coalesced_ref<T> operator*() {
        return trove::detail::coalesced_ref<T>(m_ptr);
    }
    template<typename I>
    __device__ trove::detail::coalesced_ref<T> operator[](const I& idx) {
        return trove::detail::coalesced_ref<T>(m_ptr + idx);
    }
    __device__ operator T*() {
        return m_ptr;
    }
};





}
