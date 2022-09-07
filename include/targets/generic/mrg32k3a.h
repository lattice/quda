/*
   An implementation of MRG32k3a based on constexpr.
   Original algorithm from
      Pierre L'Ecuyer
      Good Parameter Sets for Combined Multiple Recursive Random Number Generators
      Operations Research, 47, 1 (1999), 159-164.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace quda
{
  namespace target
  {
    namespace rng
    {
      template <typename E, int N> struct Arr {
        constexpr E &operator[](int i) { return d[i]; }
        constexpr const E &operator[](int i) const { return d[i]; }
        E d[N] = {};
      };

      template <typename E> struct Arr3 {
        constexpr Arr3() {};
        constexpr Arr3(E x, E y, E z) : d {x, y, z} {};
        constexpr E &operator[](int i) { return d[i]; }
        constexpr const E &operator[](int i) const { return d[i]; }
        constexpr bool operator==(const Arr3 &x) const { return d[0] == x.d[0] && d[1] == x.d[1] && d[2] == x.d[2]; }
        E d[3] = {};
      };

      template <typename E> inline std::ostream &operator<<(std::ostream &o, const Arr3<E> &a)
      {
        return o << '[' << a[0] << ' ' << a[1] << ' ' << a[2] << ']';
      }

      using State = Arr3<uint32_t>;
      using Trans = Arr3<State>;

      struct MRG32k3a {
        State s1, s2;
      };

      inline std::ostream &operator<<(std::ostream &o, const MRG32k3a &prn)
      {
        return o << "MRG32k3a(" << prn.s1 << ' ' << prn.s2 << ')';
      }

      constexpr Trans squaremod(Trans a, uint64_t m)
      {
        Trans x;
        for (int i = 0; i < 3; ++i) {
          Arr3<uint64_t> t {0u, 0u, 0u};
          for (int k = 0; k < 3; ++k) {
            uint64_t aik = static_cast<uint64_t>(a[i][k]);
            t[0] += aik * static_cast<uint64_t>(a[k][0]) % m;
            t[1] += aik * static_cast<uint64_t>(a[k][1]) % m;
            t[2] += aik * static_cast<uint64_t>(a[k][2]) % m;
          }
          x[i][0] = static_cast<uint32_t>(t[0] % m);
          x[i][1] = static_cast<uint32_t>(t[1] % m);
          x[i][2] = static_cast<uint32_t>(t[2] % m);
        }
        return x;
      }

      template <int p> constexpr Arr<Trans, p> squaremodarray(Trans a, uint64_t m)
      {
        Arr<Trans, p> x;
        x[0] = a;
        for (int i = 1; i < p; ++i) x[i] = squaremod(x[i - 1], m);
        return x;
      }

      constexpr void matvecmod(Trans a, State &v, uint64_t m)
      {
        uint64_t v0 = static_cast<uint64_t>(v[0]);
        uint64_t v1 = static_cast<uint64_t>(v[1]);
        uint64_t v2 = static_cast<uint64_t>(v[2]);
        v[0]
          = static_cast<uint32_t>(((static_cast<uint64_t>(a[0][0]) * v0) % m + (static_cast<uint64_t>(a[0][1]) * v1) % m
                                   + (static_cast<uint64_t>(a[0][2]) * v2) % m)
                                  % m);
        v[1]
          = static_cast<uint32_t>(((static_cast<uint64_t>(a[1][0]) * v0) % m + (static_cast<uint64_t>(a[1][1]) * v1) % m
                                   + (static_cast<uint64_t>(a[1][2]) * v2) % m)
                                  % m);
        v[2]
          = static_cast<uint32_t>(((static_cast<uint64_t>(a[2][0]) * v0) % m + (static_cast<uint64_t>(a[2][1]) * v1) % m
                                   + (static_cast<uint64_t>(a[2][2]) * v2) % m)
                                  % m);
      }

      constexpr double norm = 2.328306549295728e-10;
      constexpr double m1 = 4294967087.0;
      constexpr double m2 = 4294944443.0;
      constexpr double a12 = 1403580.0;
      constexpr double a13n = 810728.0;
      constexpr double a21 = 527612.0;
      constexpr double a23n = 1370589.0;
      constexpr uint32_t defaultSEED = 12345U;
      constexpr int subsequenceBase = 76;

      constexpr Trans a1 {State {0u, 1u, 0u}, State {0u, 0u, 1u},
                          State {static_cast<uint32_t>(m1 - a13n), static_cast<uint32_t>(a12), 0u}};
      constexpr Trans a2 {State {0u, 1u, 0u}, State {0u, 0u, 1u},
                          State {static_cast<uint32_t>(m2 - a23n), 0u, static_cast<uint32_t>(a21)}};

      constexpr int maxpower2 = 190;

      constexpr auto a1sq = squaremodarray<maxpower2>(a1, static_cast<uint64_t>(m1));
      constexpr auto a2sq = squaremodarray<maxpower2>(a2, static_cast<uint64_t>(m2));

      // device copies of a1sq and a2sq
      __device__ constexpr auto a1sq_d = a1sq;
      __device__ constexpr auto a2sq_d = a2sq;

      // default functor for grabbing the appropriate a1sq
      template <bool is_device> struct get_a1sq {
        constexpr auto operator()(int i) { return a1sq[i]; }
      };

      // device-specialized functor for grabbing the appropriate a1sq
      template <> struct get_a1sq<true> {
        constexpr auto operator()(int i) { return a1sq_d[i]; }
      };

      // default functor for grabbing the appropriate a2sq
      template <bool is_device> struct get_a2sq {
        constexpr auto operator()(int i) { return a2sq[i]; }
      };

      // device-specialized functor for grabbing the appropriate a2sq
      template <> struct get_a2sq<true> {
        constexpr auto operator()(int i) { return a2sq_d[i]; }
      };

      static_assert(a1sq_d[76]
                      == Trans {State {82758667u, 1871391091u, 4127413238u}, State {3672831523u, 69195019u, 1871391091u},
                                State {3672091415u, 3528743235u, 69195019u}},
                    "a1sq[76] wrong!");
      static_assert(a2sq[76]
                      == Trans {State {1511326704u, 3759209742u, 1610795712u},
                                State {4292754251u, 1511326704u, 3889917532u},
                                State {3859662829u, 4292754251u, 3708466080u}},
                    "a2sq[76] wrong!");

      constexpr void skip(MRG32k3a &prn, uint64_t offset, int base = 0)
      {
        int i = 0;
        uint64_t s = offset;
        while (s > 0) {
          if (s & 1u) {
            matvecmod(target::dispatch<get_a1sq>(base + i), prn.s1, static_cast<uint64_t>(m1));
            matvecmod(target::dispatch<get_a2sq>(base + i), prn.s2, static_cast<uint64_t>(m2));
          }
          s >>= 1u;
          ++i;
        }
      }

      constexpr void seed(MRG32k3a &prn, uint64_t seed, uint64_t subsequence)
      {
        if (seed) {
          const uint64_t d1
            = static_cast<uint64_t>(defaultSEED) * static_cast<uint64_t>((static_cast<uint32_t>(seed) ^ 0x55555555u));
          const uint64_t d2 = static_cast<uint64_t>(defaultSEED)
            * static_cast<uint64_t>((static_cast<uint32_t>(seed >> 32u) ^ 0xAAAAAAAAu));
          prn.s1[0] = static_cast<uint32_t>(d1 % static_cast<uint64_t>(m1));
          prn.s1[1] = static_cast<uint32_t>(d2 % static_cast<uint64_t>(m1));
          prn.s1[2] = static_cast<uint32_t>(d1 % static_cast<uint64_t>(m1));
          prn.s2[0] = static_cast<uint32_t>(d2 % static_cast<uint64_t>(m2));
          prn.s2[1] = static_cast<uint32_t>(d1 % static_cast<uint64_t>(m2));
          prn.s2[2] = static_cast<uint32_t>(d2 % static_cast<uint64_t>(m2));
        } else {
          prn.s1[0] = defaultSEED;
          prn.s1[1] = defaultSEED;
          prn.s1[2] = defaultSEED;
          prn.s2[0] = defaultSEED;
          prn.s2[1] = defaultSEED;
          prn.s2[2] = defaultSEED;
        }
        skip(prn, subsequence, subsequenceBase);
      }

      constexpr double uniform(MRG32k3a &prn)
      {
        double p1 = a12 * (double)(prn.s1[1]) - a13n * (double)(prn.s1[0]);
        p1 = std::fmod(p1, m1);
        if (p1 < 0.0) p1 += m1;
        prn.s1[0] = prn.s1[1];
        prn.s1[1] = prn.s1[2];
        prn.s1[2] = static_cast<uint32_t>(p1);

        double p2 = a21 * (double)(prn.s2[2]) - a23n * (double)(prn.s2[0]);
        p2 = std::fmod(p2, m2);
        if (p2 < 0.0) p2 += m2;
        prn.s2[0] = prn.s2[1];
        prn.s2[1] = prn.s2[2];
        prn.s2[2] = static_cast<uint32_t>(p2);

        if (p1 <= p2)
          return (p1 - p2 + m1) * norm;
        else
          return (p1 - p2) * norm;
      }

      template <typename R> inline void gaussian(MRG32k3a &prn, R &x, R &y)
      {
        constexpr R TINY = std::numeric_limits<R>::min();
        R v, p, r;
        v = (R)uniform(prn);
        p = (R)uniform(prn) * (R)2.0 * (R)3.141592653589793238462643383279502884;
        r = std::sqrt((R)(-2.0) * std::log(v + TINY));
        x = r * std::sin(p);
        y = r * std::cos(p);
      }
    } // namespace rng
  }   // namespace target
} // namespace quda
