#pragma once

#include <string>
#include <map>
#include <stack>
#include <vector>
#include <reference_wrapper_helper.h>

namespace quda {

  /**
     FieldKey is a container for a key for a std::map to cache
     allocated field instances.
     @tparam T The field type
   */
  template <typename T>
  struct FieldKey {
    std::string volume; /** volume kstring */
    std::string aux;    /** auxiliary string */

    FieldKey() = default;

    /**
       @brief Constructor for FieldKey
       @param[in] a Field whose key we wish to generate
    */
    FieldKey(const T &a) : volume(a.VolString()), aux(a.AuxString()) { }

    /**
       @brief Less than operator used for ordering in the container
     */
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

  /**
     FieldTmp is a wrapper for a cached field.
     @tparam T The field type
   */
  template <typename T>
  class FieldTmp {
    static std::map<FieldKey<T>, std::stack<T>> cache; /** Field Cache */
    T tmp;                                             /** The temporary field instance */
    FieldKey<T> key;                                   /** Key associated with this instance */

  public:
    /**
       @brief Allow FieldTmp<T> to be used in lieu of T
    */
    operator T&() { return tmp; }

    /**
       @brief Create a field temporary that is identical to the field
       instance argument.  If a matching field is present in the cache,
       it will be popped from the cache.  If no such temporary exists, a
       temporary will be allocated.
       @param[in] a Field we wish to create a matching temporary for
    */
    FieldTmp(const T &a);

    /**
       @brief Create a field temporary that corresponds to the key
       parameter.  If a matching field is present in the cache, it
       will be popped from the cache.  If no such temporary exists an
       error will be thrown.  Note that this constructor allows for a
       custom key that doesn't necessarily correspond to the key
       return by FieldKey<T>(const T&).
       @param[in] key Key corresponding to the field instance we
       require
       @param[in] param Parameter structure used to allocated the temporary
     */
    FieldTmp(const FieldKey<T> &key, const typename T::param_type &param);

    /**
       @brief Copy constructor is deleted to prevent accidental cache
       bloat
    */
    FieldTmp(const FieldTmp<T> &a) = delete;

    /**
       @brief Move constructor is noexcept to ensure it is used by STL
     */
    FieldTmp(FieldTmp<T> &&a) noexcept = default;

    /**
       @brief Push the temporary onto the cache, where it will be
       available for subsequent reuse.
    */
    ~FieldTmp();

    /** @brief Flush the cache and frees all temporary allocations */
    static void destroy();
  };

  /**
     @brief Get a field temporary that is identical to the field
     instance argument.  If a matching field is present in the cache,
     it will be popped from the cache.  If no such temporary exists, a
     temporary will be allocated.  When the destructor for the
     FieldTmp is called, e.g., the returned object goes out of scope,
     the temporary will be pushed onto the cache.

     @param[in] a Field we wish to create a matching temporary for
   */
  template <typename T> auto getFieldTmp(const T &a) { return FieldTmp<T>(a); }

  /**
     @brief Get a vector of field temporaries that are identical to
     the vector instance argument.  If enough matching fields are
     present in the cache, they will be popped from the cache.  If an
     insufficient number of temporaries exist, enough will be
     allocated.  When the destructor is called, e.g.,
     the returned object goes out of scope, the temporaries will be
     pushed onto the cache.

     @param[in] a Vector of fields we wish to create a matching
     temporary for
   */
  template <typename T> auto getFieldTmp(const vector_ref<T> &a)
  {
    std::vector<FieldTmp<T>> tmp;
    tmp.reserve(a.size());
    for (auto i = 0u; i < a.size(); i++) tmp.push_back(std::move(getFieldTmp(a[i])));
    return tmp;
  }
}
