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
#include <trove/detail/dismember.h>

namespace trove {
namespace detail {

template<int s, typename T>
struct divergent_loader {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, s> impl(const U* src) {
        return array<U, s>(*src,
                           divergent_loader<s-1, T>::impl(src+1));
    }
    __device__
    static array<U, s> impl(const T* src) {
        return impl((U*)src);
    }
};

template<typename T>
struct divergent_loader<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, 1> impl(const U* src) {
        return array<U, 1>(*src);
    }
    __device__
    static array<U, 1> impl(const T* src) {
        return impl((U*)src);
    }
};

template<typename T>
struct use_divergent {
    static const bool value = (sizeof(T) % 4) == 0;
};

template<typename T>
__device__
typename enable_if<use_divergent<T>::value, T>::type
divergent_load(const T* src) {
    typedef typename detail::dismember_type<T>::type U;
    typedef array<U, detail::aliased_size<T, U>::value> u_store;
    u_store loaded =
        detail::divergent_loader<detail::aliased_size<T, U>::value, T>::impl(
            src);
    return detail::fuse<T>(loaded);
}

template<typename T>
__device__
typename enable_if<!use_divergent<T>::value, T>::type
divergent_load(const T* src) {
    return *src;
}

template<int s, typename T>
struct divergent_storer {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, s>& data, U* dest) {
        *dest = data.head;
        divergent_storer<s-1, T>::impl(data.tail, dest+1);
    }
    __device__
    static void impl(const array<U, s>& data, const T* dest) {
        return impl(data, (U*)dest);
    }
};

template<typename T>
struct divergent_storer<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, 1>& data, U* dest) {
        *dest = data.head;
    }
    __device__
    static void impl(const array<U, 1>& data, const T* dest) {
        return impl(data, (U*)dest);
    }
};

template<typename T>
__device__
typename enable_if<use_divergent<T>::value>::type
divergent_store(const T& data, T* dest) {
    typedef typename detail::dismember_type<T>::type U;
    typedef array<U, detail::aliased_size<T, U>::value> u_store;
    u_store lysed = detail::lyse<U>(data);
    detail::divergent_storer<detail::aliased_size<T, U>::value, T>::impl(
        lysed, dest);
}

template<typename T>
__device__
typename enable_if<!use_divergent<T>::value>::type
divergent_store(const T& data, T* dest) {
    *dest = data;
}


}
}
