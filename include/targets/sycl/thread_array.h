#pragma once

#define SLM

namespace quda
{

  /**
     @brief Class that provides indexable per-thread storage.  The
     dynamic version maps to using assigning each thread a unique
     window of shared memory.  The static version just uses an stack
     array.
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
#ifndef SLM
    array_t arr;
#endif
    sycl::local_ptr<array_t> array_;
    template <typename ...U, typename ...Ts>
    thread_array(const SpecialOps<U...> *ops, Ts ...t)
    {
#ifdef SLM
      //int offset = (target::thread_idx().z * target::block_dim().y + target::thread_idx().y)
      //* target::block_dim().x + target::thread_idx().x;
      int offset = target::thread_idx_linear<3>();
      auto op = getSpecialOp<op_thread_array<T,n>>(ops);
      sycl::local_ptr<void> v(op.smem);
      sycl::local_ptr<array_t> p(v);
      array_ = &p[offset];
#else
      array_ = &arr;
#endif
      //if constexpr (sizeof...(Ts) != 0) {
      (*array_) = array_t { t... };
      //}
    }
    T &operator[](int i) { return (*array_)[i]; }
    const T &operator[](int i) const { return (*array_)[i]; }
  };

} // namespace quda
