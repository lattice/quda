#pragma once

#include <functional>
#include <initializer_list>
#include <enum_quda.h>
#include <util_quda.h>

namespace quda
{

  /**
     @brief Helper function to statically determine if a type is a QUDA field.  Specialized as needed.
   */
  template <typename T> struct is_field : std::false_type {
  };

  template <typename T> constexpr bool is_field_v = is_field<T>::value;

  template <typename T, class U> struct unwrap_impl {
    using type = U;
  };

  template <typename T, class U> struct unwrap_impl<std::reference_wrapper<T>, U> {
    using type = T;
  };

  /**
     @brief Helper to unwrapping the underlying type of a possibly reference_wrapped object.
     @tparam T the type we want to unwrap
   */
  template <typename T> using unwrap_t = typename unwrap_impl<std::decay_t<T>, T>::type;

  /**
     @brief Create a std::vector<std::reference_wrapper<T>> from input
     std::vector<T>
     @param[in] v Input vector whose elements we will wrap
     @return Reference wrapped vector
  */
  template <typename T, typename std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type>> &&
    !std::is_pointer_v<typename T::value_type>> * = nullptr>
  auto make_set_impl(T &v)
  {
    using V = unwrap_t<typename T::value_type>;
    return std::vector<std::reference_wrapper<V>>(v.begin(), v.end());
  }

  /**
     @brief Create a std::vector<std::reference_wrapper<T>> from input
     std::vector<T>
     @param[in] v Input vector whose elements we will wrap
     @return Reference wrapped vector
  */
  template <typename T, typename std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type>> &&
    !std::is_pointer_v<typename T::value_type>> * = nullptr>
  auto make_cset_impl(T &v)
  {
    using V = unwrap_t<typename T::value_type>;
    return std::vector<std::reference_wrapper<const V>>(v.begin(), v.end());
  }

  /**
     @brief Create a std::vector<std::reference_wrappers<T>> from a
     std::vector<T*>.
     @param[in] vp Vector of pointers
     @return Vector of reference_wrappers
  */
  template <typename T, typename std::enable_if_t<std::is_same_v<std::remove_const_t<T>, std::vector<typename T::value_type>> &&
    std::is_pointer_v<typename T::value_type>> * = nullptr>
  auto make_set_impl(T &vp)
  {
    using V = std::remove_pointer_t<typename T::value_type>;
    std::vector<std::reference_wrapper<V>> v;
    for (auto &vi : vp) v.push_back(*vi);
    return v;
  }

  /**
     @brief Create a std::vector<std::reference_wrapper<T>> from an
     input instance of a QUDA field
     @param[in] v Input T that we will wrap
     @return Reference wrapped vector
  */
  template <typename T, typename std::enable_if_t<is_field_v<T>> * = nullptr> auto make_set_impl(T &v)
  {
    return std::vector<std::reference_wrapper<T>> {v};
  }

  /**
     @brief Create a std::vector<std::reference_wrapper<T>> from an
     input instance of a QUDA field
     @param[in] v Input T that we will wrap
     @return Reference wrapped vector
  */
  template <typename T, typename std::enable_if_t<is_field_v<T>> * = nullptr> auto make_cset_impl(T &v)
  {
    return std::vector<std::reference_wrapper<const T>> {v};
  }

  /**
     @brief Create a std::vector of std::reference_wrappers the input argument
     @param[in] v Input to be wrapped in a vector
     @return Vector of input
  */
  template <typename T> inline auto make_set(T &&v)
  {
    auto v_set = make_set_impl(v);
    return v_set;
  }

  /**
     @brief Create a std::vector of std::reference_wrappers the input argument
     @param[in] v Input to be wrapped in a vector
     @return Vector of input
  */
  template <typename T> inline auto make_cset(T &&v)
  {
    auto v_set = make_cset_impl(v);
    return v_set;
  }

  /**
     @brief Create a std::vector of std::reference_wrappers containing
     both arguments

     @param[in] v1 First input to be wrapped in a vector
     @param[in] v2 Second input to be wrapped in a vector
     @return Superset vector containing v1 and v2
  */
  template <typename T1, typename T2> inline auto make_set(T1 &&v1, T2 &&v2)
  {
    auto v1_set = make_set_impl(v1);
    auto v2_set = make_set_impl(v2);

    v1_set.reserve(v1_set.size() + v2_set.size());
    v1_set.insert(v1_set.end(), v2_set.begin(), v2_set.end());

    return v1_set;
  }

  /** trait that can be used to determine if a type is an iterator */
  template <class T, class = void> struct is_iterator : std::false_type { };

  template <class T>
  struct is_iterator<T,
    std::void_t<typename std::iterator_traits<T>::difference_type,
    typename std::iterator_traits<T>::pointer,
    typename std::iterator_traits<T>::reference,
    typename std::iterator_traits<T>::value_type,
    typename std::iterator_traits<T>::iterator_category>> : std::true_type
    { };

  template <class T> constexpr bool is_iterator_v = is_iterator<T>::value;

  /** trait that can be used to determine if a type is an initializer_list */
  template <typename T> struct is_initializer_list : std::false_type {};
  template <typename T> struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
  template <typename T> static constexpr bool is_initializer_list_v = is_initializer_list<T>::value;

  class ColorSpinorField;

  /**
     Derived specializaton of std::vector<std::reference_wrapper<T>>
     which allows us to write generic multi-field functions.
   */
  template <class T>
  class vector_ref : public std::vector<std::reference_wrapper<T>> {
    using vector = std::vector<std::reference_wrapper<T>>;
    using value_type = T;

    /**
       make_set is a helper function that creates a vector of
       reference wrapped objects from the input reference argument.
       This is the default base case that is for a simple object
       @param[in] v Object reference we wish to wrap
     */
    template <typename U> std::enable_if_t<!is_initializer_list_v<U>, vector> make_set(U &v)
    {
      return vector(1, static_cast<T &>(v));
    }

    /**
       make_set is a helper function that creates a vector of
       reference wrapped objects from the input reference argument.
       This is the specialized overload that handles a vector of
       objects.
       @param[in] v Vector argument we wish to wrap
     */
    template <typename U> vector make_set(std::vector<U> &v) { return vector{v.begin(), v.end()}; }

    /**
       make_set is a helper function that creates a vector of
       reference wrapped objects from the input reference argument.
       This is the specialized overload that handles a vector_ref of
       objects.  Used to convert a non-const set to a const set.
       @param[in] v Vector argument we wish to wrap
     */
    template <typename U> vector make_set(vector_ref<U> &v) { return vector {v.begin(), v.end()}; }

    /**
       make_set is a helper function that creates a vector of
       reference wrapped objects from the input reference argument.
       This is the specialized overload that handles an
       initializer_list of a pair of iterators.
       @param[in] v initializer_list
     */
    template <typename U>
    std::enable_if_t<is_initializer_list_v<U> && is_iterator_v<typename U::value_type>, vector> make_set(U &v)
    {
      if (v.size() != 2) errorQuda("this constructor requires a size=2 initializer list"); // static_assert is flakey
      return vector{*(v.begin() + 0), *(v.begin() + 1)}; // need to dereference the iterators
    }

  public:
    vector_ref() = default;
    vector_ref(const vector_ref &) = default;
    vector_ref(vector_ref &&) = default;

    /**
       Unary constructor
       @param[in] v Object to which we are constructing a vector_ref around
     */
    template <class U> vector_ref(U &v)
    {
      auto vset = make_set(v);
      vector::reserve(vset.size());
      vector::insert(vector::end(), vset.begin(), vset.end());
    }

    /**
       Constructor from pair of iterators
       @param[in] first Begin iterator
       @param[in] last End iterator
     */
    template <class U, std::enable_if_t<is_iterator_v<U>>* = nullptr>
    vector_ref(U first, U last)
    {
      vector::reserve(last - first);
      for (auto it = first; it != last; it++) vector::push_back(*it);
    }

    /**
       Constructor from a set of non-iterator references constructing a
       vector_ref that is a union of this set
       @param[in] u first reference
       @param[in] args other references
     */
    template <class U, class... Args, std::enable_if_t<!is_iterator_v<U>>* = nullptr>
    vector_ref(U &u, Args &...args)
    {
      auto uset = make_set(u);
      auto tmp = vector_ref(args...);
      vector::reserve(uset.size() + tmp.size());
      vector::insert(vector::end(), uset.begin(), uset.end());
      vector::insert(vector::end(), tmp.begin(), tmp.end());
    }

    /**
       @brief This overload allows us to directly access the
       underlying reference without needing to invoke get() like would
       do for the parent method.  Moreover, we intentionally mark this
       function as const, since it will allow us to return non-const
       references from a constant container if the underlying
       references are themselves non-const.
       @param[in] idx The location index we are requesting
       @return The underlying object reference
     */
    T& operator[](size_t idx) const { return vector::operator[](idx).get(); }

    template <class U = T>
    std::enable_if_t<std::is_same_v<std::remove_const_t<U>, ColorSpinorField>, vector_ref<T>> Even() const
    {
      vector_ref<T> even;
      even.reserve(vector::size());
      for (auto i = 0u; i < vector::size(); i++) even.push_back(operator[](i).Even());
      return even;
    }

    template <class U = T>
    std::enable_if_t<std::is_same_v<std::remove_const_t<U>, ColorSpinorField>, vector_ref<T>> Odd() const
    {
      vector_ref<T> odd;
      odd.reserve(vector::size());
      for (auto i = 0u; i < vector::size(); i++) odd.push_back(operator[](i).Odd());
      return odd;
    }
  };

  template <class T> using cvector_ref = const vector_ref<T>;

  /**
     @brief Create a vector_ref of the parity subset requested.
     @param[in] in The input set
     @param[in] parity The desired parity subset
   */
  template <class T> auto make_parity_subset(T &in, QudaParity parity)
  {
    if (parity != QUDA_EVEN_PARITY && parity != QUDA_ODD_PARITY) errorQuda("Invalid parity %d requested", parity);
    vector_ref<typename T::value_type> out;
    out.reserve(in.size());
    for (auto i = 0u; i < in.size(); i++) out.push_back(parity == QUDA_EVEN_PARITY ? in[i].Even() : in[i].Odd());
    return out;
  }

} // namespace quda
