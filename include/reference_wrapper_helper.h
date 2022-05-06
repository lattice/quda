#include <functional>

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
  template <typename T, typename std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type>>> * = nullptr>
  auto make_set_impl(T &v)
  {
    using V = unwrap_t<typename T::value_type>;
    return std::vector<std::reference_wrapper<V>>(v.begin(), v.end());
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

} // namespace quda
