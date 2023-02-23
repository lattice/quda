#pragma once

//#define SLM

namespace quda
{

  template <typename T, int n> struct thread_array : array<T, n> {};

#if 0
  /**
     @brief Class that provides indexable per-thread storage.  The
     shared memory version maps to assigning each thread a unique
     window of shared memory.  The default version uses a stack array.
   */

  // default stack version
  template <typename T, int n = 0, bool _ = true> struct thread_array : array<T, n> {
    static_assert(!_);  // always fails, check for use of default version
  };

  // shared memory specialization
  template <typename O> struct thread_array<O,0,(bool)isOpThreadArray<O>> {
    using T = typename O::ElemT;
    static constexpr int n = O::n;
    static_assert(std::is_same_v<T,int>);
    static_assert(n==4);
    using array_t = array<T,n>;
#ifdef SLM
    sycl::local_ptr<array_t> array_ptr;
#else
    array_t array_;
#endif
    template <typename ...U, typename ...Ts>
    inline thread_array(const SpecialOps<U...> *ops, Ts ...t)
    {
#ifdef SLM
      int offset = target::thread_idx_linear<3>();
      auto op = getSpecialOp<op_thread_array<T,n>>(ops);
      sycl::local_ptr<void> v(op.smem);
      sycl::local_ptr<array_t> p(v);
      array_ptr = &p[offset];
      (*array_ptr) = array_t { t... };
#else
      static_assert(hasSpecialOpType<op_thread_array<T,n>,U...>);
      array_ = array_t { t... };
#endif
    }
#ifdef SLM
    inline T &operator[](int i) { return (*array_ptr)[i]; }
    inline const T &operator[](int i) const { return (*array_ptr)[i]; }
#else
    inline T &operator[](int i) { return array_[i]; }
    inline const T &operator[](int i) const { return array_[i]; }
#endif
  };
#endif

} // namespace quda
