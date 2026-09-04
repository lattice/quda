#pragma once

#include <cstdint>

/**
 * @brief Minimal xoroshiro128++ generator, meeting the standard
 * UniformRandomBitGenerator requirements so that it can be used with the
 * std::*_distribution types.  The state is 16 bytes, as opposed to the 2504
 * bytes of std::mt19937_64: since the test harness holds one generator per
 * lattice site, the latter needs ~200 GiB at a 96^4 volume.
 */
class host_rng_t
{
  uint64_t s[2];

  static constexpr uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  /**
     @brief SplitMix64, used to expand a single seed into the full state.  Any
     seed, zero included, gives a state that is safely far from all-zero.
   */
  static constexpr uint64_t splitmix64(uint64_t &x)
  {
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }

public:
  using result_type = uint64_t;

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~result_type(0); }

  constexpr host_rng_t(uint64_t seed = 0) : s {splitmix64(seed), splitmix64(seed)} { }

  constexpr result_type operator()()
  {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const result_type result = rotl(s0 + s1, 17) + s0;
    s1 ^= s0;
    s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21);
    s[1] = rotl(s1, 28);
    return result;
  }
};
