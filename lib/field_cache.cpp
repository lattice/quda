#include <field_cache.h>
#include <color_spinor_field.h>

namespace quda {

  template <typename T> std::map<FieldKey<T>, std::stack<T>> FieldTmp<T>::cache;

  template <typename T> FieldTmp<T>::FieldTmp(const T &a) : key(FieldKey(a))
  {
    auto it = cache.find(key);

    if (it != cache.end() && it->second.size()) { // found an entry
      tmp = std::move(it->second.top());
      it->second.pop(); // pop the defunct object
    } else { // no entry found, we must allocate a new field
      typename T::param_type param(a);
      param.create = QUDA_ZERO_FIELD_CREATE;
      tmp = T(param);
    }
  }

  template <typename T> FieldTmp<T>::FieldTmp(const FieldKey<T> &key, const typename T::param_type &param) : key(key)
  {
    auto it = cache.find(key);

    if (it != cache.end() && it->second.size()) { // found an entry
      tmp = std::move(it->second.top());
      it->second.pop(); // pop the defunct object
    } else {            // no entry found, we must allocate a new field
      tmp = T(param);
    }
  }

  template <typename T> FieldTmp<T>::~FieldTmp()
  {
    // don't cache the field if it's empty (e.g., has been moved)
    if (tmp.Bytes() == 0) return;
    cache[key].push(std::move(tmp));
  }

  template <typename T> void FieldTmp<T>::destroy()
  {
    cache.clear();
  }

  template class FieldTmp<ColorSpinorField>;
}
