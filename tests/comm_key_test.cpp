#include <comm_key.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>

using quda::CommKey;

constexpr bool check_ordering()
{
  constexpr CommKey a {1, 1, 1, 4};
  constexpr CommKey b {2, 1, 1, 2};
  if (!(a < b) || b < a || !(b > a) || a > b) return false;
  if (!(a == a) || a == b || a < a || a > a) return false;
  // The first differing dimension wins, even when later dimensions disagree.
  for (int d = 0; d < CommKey::n_dim; d++) {
    CommKey lower {1, 1, 1, 1}, upper {1, 1, 1, 1};
    upper[d] = 2;
    for (int e = d + 1; e < CommKey::n_dim; e++) lower[e] = 4;
    if (!(lower < upper) || upper < lower || !(upper > lower) || lower > upper) return false;
    if (lower == upper) return false;
  }
  return true;
}

static_assert(check_ordering());
constexpr CommKey volume {2, 3, 4, 5};
static_assert(volume.product() == 120);
static_assert(quda::product(volume) == 120);

int main()
{
  // Exhaustively check ordering and map lookup across all 2^4 split layouts,
  // using std::array's independent lexicographic comparison as the oracle.
  std::map<CommKey, int> cache;
  for (int i = 15; i >= 0; i--) {
    const auto key = quda::coordinate_from_index(i, {2, 2, 2, 2}) + CommKey {1, 1, 1, 1};
    cache.emplace(key, i);
    cache.emplace(key, -1); // Equivalent keys must not create another entry.
  }
  if (cache.size() != 16) return 1;
  for (int i = 0; i < 16; i++) {
    const auto a = quda::coordinate_from_index(i, {2, 2, 2, 2}) + CommKey {1, 1, 1, 1};
    const auto found = cache.find(a);
    if (found == cache.end() || found->second != i) return 1;
    for (int j = 0; j < 16; j++) {
      const auto b = quda::coordinate_from_index(j, {2, 2, 2, 2}) + CommKey {1, 1, 1, 1};
      const std::array<int, 4> aa {a[0], a[1], a[2], a[3]}, bb {b[0], b[1], b[2], b[3]};
      if ((a < b) != (aa < bb) || (a > b) != (aa > bb) || (a == b) != (aa == bb)) return 1;
    }
  }
  if (!std::is_sorted(cache.begin(), cache.end(), [](const auto &a, const auto &b) { return a.first < b.first; }))
    return 1;
  std::puts("CommKey ordering, map lookup, and constexpr checks passed");
  return 0;
}
