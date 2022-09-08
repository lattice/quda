#pragma once

#include <string>
#include <map>
#include <stack>

namespace quda {

  template <typename T>
  struct FieldKey {
    std::string volume;
    std::string aux;

    FieldKey(const T &a) : volume(a.VolString()), aux(a.AuxString()) { }

    bool operator<(const FieldKey<T> &other) const
    {
      if (volume < other.volume) {
        return true;
      } else if (volume == other.volume) {
        return aux < other.aux ? true : false;
      }
      return false;
    }
  };

  template <typename T>
  class FieldTmp {
    static std::map<FieldKey<T>, std::stack<T>> cache;
    T tmp;

  public:
    operator T&() { return tmp; }
    FieldTmp(const T &a);
    ~FieldTmp();

    static void destroy();
  };

  template <typename T> auto getTmp(const T &a) { return FieldTmp<T>(a); }

}
