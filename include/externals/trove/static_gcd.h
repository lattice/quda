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
namespace trove {

template<int u, int v>
struct static_gcd;

namespace detail {

template<bool u_odd, bool v_odd, int u, int v>
struct static_gcd_helper {
    static const int value = static_gcd<(u>>1), (v>>1)>::value << 1;
};

template<int u, int v>
struct static_gcd_helper<false, true, u, v> {
    static const int value = static_gcd<(u>>1), v>::value;
};

template<int u, int v>
struct static_gcd_helper<true, false, u, v> {
    static const int value = static_gcd<u, (v>>1)>::value;
};

template<int u, int v>
struct static_gcd_helper<true, true, u, v> {
    static const int reduced_u = (u > v) ? ((u - v) >> 1) : ((v - u) >> 1);
    static const int reduced_v = (u > v) ? v : u;
    static const int value = static_gcd<reduced_u, reduced_v>::value;
};
}

template<int u, int v>
struct static_gcd {
    static const bool u_odd = (u & 0x1) == 1;
    static const bool v_odd = (v & 0x1) == 1;
    static const bool equal = u == v;
    static const int value = equal ? u : detail::static_gcd_helper<u_odd, v_odd, u, v>::value;
};

template<int v>
struct static_gcd<0, v> {
    static const bool value = v;
};

template<int u>
struct static_gcd<u, 0> {
    static const bool value = u;
};

}
