#pragma once

#include <quda_api.h>
#include <quda_target.h>
#include <float_vector.h>
//#include <cub_helper.cuh>
#include <target_device.h>
#include <reducer.h>
#include <kernel_helper.h>
#include <atomic.cuh>

#ifdef QUAD_SUM
using device_reduce_t = doubledouble;
#else
using device_reduce_t = double;
#endif

#ifdef HETEROGENEOUS_ATOMIC
#include <cuda/std/atomic>
using count_t = cuda::atomic<unsigned int, cuda::thread_scope_device>;
#else
//using count_t = unsigned int;
using count_t = int;
#endif

namespace quda
{

  namespace reducer
  {
    /** returns the reduce buffer size allocated */
    size_t buffer_size();

    void *get_device_buffer();
    void *get_mapped_buffer();
    void *get_host_buffer();
    count_t *get_count();
    qudaEvent_t &get_event();
  } // namespace reducer

  constexpr int max_n_reduce() { return QUDA_MAX_MULTI_REDUCE; }

  /**
     @brief The initialization value we used to check for completion
   */
  template <typename T> constexpr T init_value() { return -std::numeric_limits<T>::infinity(); }

  /**
     @brief The termination value we use to prevent a possible hang in
     case the computed reduction is equal to the initialization
  */
  template <typename T> constexpr T terminate_value() { return std::numeric_limits<T>::infinity(); }

  /**
     @brief The atomic word size we use for a given reduction type.
     This type should be lock-free to guarantee correct behaviour on
     platforms that are not coherent with respect to the host
   */
  template <typename T> struct atomic_type {
    using type = device_reduce_t;
  };
  template <> struct atomic_type<float> {
    using type = float;
  };

  template <typename T, bool use_kernel_arg = true> struct ReduceArg : kernel_param<use_kernel_arg> {

    template <int, int, typename Reducer, typename Arg, typename I>
    friend __device__  void reduce(Arg&, const Reducer &, const I &, const int);
    qudaError_t launch_error; // only do complete if no launch error to avoid hang

  private:
    const int n_reduce; // number of reductions of length n_item
    bool reset = false; // reset the counter post completion (required for multiple calls with the same arg instance
#ifdef HETEROGENEOUS_ATOMIC
    using system_atomic_t = typename atomic_type<T>::type;
    static constexpr int n_item = sizeof(T) / sizeof(system_atomic_t);
    cuda::atomic<T, cuda::thread_scope_device> *partial;
    // for heterogeneous atomics we need to use lock-free atomics -> operate on scalars
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_d;
    cuda::atomic<system_atomic_t, cuda::thread_scope_system> *result_h;
#else
    T *partial;
    T *result_d;
    T *result_h;
#endif
    count_t *count;
    bool consumed; // check to ensure that we don't complete more than once unless we explicitly reset

  public:
    /**
       @brief Constructor for ReduceArg
       @param[in] n_reduce The number of reductions of length n_item
       @param[in] reset Whether to reset the atomics after the
       reduction has completed; required if the same ReduceArg
       instance will be used for multiple reductions.
    */
    ReduceArg(dim3 threads, int n_reduce = 1, bool reset = false) :
      kernel_param<use_kernel_arg>(threads),
      launch_error(QUDA_ERROR_UNINITIALIZED),
      n_reduce(n_reduce),
      reset(reset),
      partial(static_cast<decltype(partial)>(reducer::get_device_buffer())),
      result_d(static_cast<decltype(result_d)>(reducer::get_mapped_buffer())),
      result_h(static_cast<decltype(result_h)>(reducer::get_host_buffer())),
      count {reducer::get_count()},
      consumed(false)
    {
      // check reduction buffers are large enough if requested
      auto max_reduce_blocks = 2 * device::processor_count();
      auto reduce_size = max_reduce_blocks * n_reduce * sizeof(*partial);
      if (reduce_size > reducer::buffer_size())
        errorQuda("Requested reduction requires a larger buffer %lu than allocated %lu", reduce_size,
                  reducer::buffer_size());

#ifdef HETEROGENEOUS_ATOMIC
      if (!commAsyncReduction()) {
        // initialize the result buffer so we can test for completion
        for (int i = 0; i < n_reduce * n_item; i++) {
          new (result_h + i) cuda::atomic<system_atomic_t, cuda::thread_scope_system> {init_value<system_atomic_t>()};
        }
        std::atomic_thread_fence(std::memory_order_release);
      } else {
        // write reduction to GPU memory if asynchronous (reuse partial)
        result_d = nullptr;
      }
#else
      if (commAsyncReduction()) result_d = partial;
#endif
    }

    /**
       @brief Finalize the reduction, returning the computed reduction
       into result.  With heterogeneous atomics this means we poll the
       atomics until their value differs from the init_value.  The
       alternate legacy path posts an event after the kernel and then
       polls on completion of the event.
       @param[out] result The reduction result is copied here
       @param[in] stream The stream on which we the reduction is being done
     */
#ifdef HETEROGENEOUS_ATOMIC
    template <typename host_t, typename device_t = host_t>
    void complete(std::vector<host_t> &result, const qudaStream_t = device::get_default_stream())
    {
      if (launch_error == QUDA_ERROR) return; // kernel launch failed so return
      if (launch_error == QUDA_ERROR_UNINITIALIZED) errorQuda("No reduction kernel appears to have been launched");
      if (consumed) errorQuda("Cannot call complete more than once for each construction");

      for (int i = 0; i < n_reduce * n_item; i++) {
        while (result_h[i].load(cuda::std::memory_order_relaxed) == init_value<system_atomic_t>()) {}
      }

      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        errorQuda("result vector length %lu does not match n_reduce %d", result.size(), n_element);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];

      if (!reset) {
        consumed = true;
      } else {
        // reset the atomic counter - this allows multiple calls to complete with ReduceArg construction
        for (int i = 0; i < n_reduce * n_item; i++) {
          result_h[i].store(init_value<system_atomic_t>(), cuda::std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_release);
      }
    }
#else
    template <typename host_t, typename device_t = host_t>
    void complete(std::vector<host_t> &result, const qudaStream_t stream = device::get_default_stream())
    {
      if (launch_error == QUDA_ERROR) return; // kernel launch failed so return
      if (launch_error == QUDA_ERROR_UNINITIALIZED) errorQuda("No reduction kernel appears to have been launched");
      //auto event = reducer::get_event();
      //qudaEventRecord(event, stream);
      //while (!qudaEventQuery(event)) {}
      auto q = device::get_target_stream(stream);
      q.wait();

      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(T) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        errorQuda("result vector length %lu does not match n_reduce %d", result.size(), n_element);
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];
    }
#endif

    void debug()
    {
      warningQuda("r2d count: %i", count[0]);
      warningQuda("r2d result_h: %g", ((double*)result_h)[0]);
    }

  };

#ifdef HETEROGENEOUS_ATOMIC
    /**
       @brief Generic reduction function that reduces block-distributed
       data "in" per thread to a single value.  This is the
       heterogeneous-atomic version which uses std::atomic to signal the
       completion of the reduction to the host.

       @param in The input per-thread data to be reduced
       @param idx In the case of multiple reductions, idx identifies
       which reduction this thread block corresponds to.  Typically idx
       will be constant along constant blockIdx.y and blockIdx.z.
    */
    template <int block_size_x, int block_size_y = 1, typename Reducer, typename Arg, typename T>
    __device__ inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx = 0)
    {
      using BlockReduce = cub::BlockReduce<T, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y, 1, __COMPUTE_CAPABILITY__>;
      __shared__ typename BlockReduce::TempStorage cub_tmp;
      __shared__ bool isLastBlockDone;

      T aggregate = Reducer::do_sum ? BlockReduce(cub_tmp).Sum(in) : BlockReduce(cub_tmp).Reduce(in, r);

      if (threadIdx.x == 0 && threadIdx.y == 0) {
        // need to call placement new constructor since partial is not necessarily constructed
        new (arg.partial + idx * gridDim.x + blockIdx.x) cuda::atomic<T, cuda::thread_scope_device> {aggregate};

        // increment global block counter for this reduction
        auto value = arg.count[idx].fetch_add(1, cuda::std::memory_order_release);

        // determine if last block
        isLastBlockDone = (value == (gridDim.x - 1));
      }

      __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        auto i = threadIdx.y * block_size_x + threadIdx.x;
        T sum = arg.init();
        while (i < gridDim.x) {
          sum = r(sum, arg.partial[idx * gridDim.x + i].load(cuda::std::memory_order_relaxed));
          i += block_size_x * block_size_y;
        }

        sum = (Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

        // write out the final reduced value
        if (threadIdx.y * block_size_x + threadIdx.x == 0) {
          if (arg.result_d) { // write to host mapped memory
            using atomic_t = typename atomic_type<T>::type;
            constexpr size_t n = sizeof(T) / sizeof(atomic_t);
            atomic_t sum_tmp[n];
            memcpy(sum_tmp, &sum, sizeof(sum));
#pragma unroll
            for (int i = 0; i < n; i++) {
              // catch the case where the computed value is equal to the init_value
              sum_tmp[i] = sum_tmp[i] == init_value<atomic_t>() ? terminate_value<atomic_t>() : sum_tmp[i];
              arg.result_d[n * idx + i].store(sum_tmp[i], cuda::std::memory_order_relaxed);
            }
          } else { // write to device memory
            arg.partial[idx].store(sum, cuda::std::memory_order_relaxed);
          }
          // TODO in principle we could remove this final atomic store
          // if we use a sense reversal barrier, avoiding the need to
          // reset the count at the end
          arg.count[idx].store(0, cuda::std::memory_order_relaxed); // set to zero for next time
        }
      }
    }

#else

  inline float toGroupReduceType(const float in) {
    return in;
  }

  inline double toGroupReduceType(const double in) {
    return in;
  }
  inline sycl::double2 toGroupReduceType(const vec2<double> in) {
    return sycl::double2{in.x, in.y};
  }
  inline sycl::double3 toGroupReduceType(const vec3<double> in) {
    return sycl::double3{in.x, in.y, in.z};
  }
  inline sycl::double4 toGroupReduceType(const vec4<double> in) {
    return sycl::double4{in.x, in.y, in.z, in.w};
  }
  inline double toGroupReduceType(const quda::vector_type<double, 1> in) {
    return in[0];
  }
  inline sycl::double2
  toGroupReduceType(const quda::vector_type<double,2> in) {
    return sycl::double2{in[0], in[1]};
  }
  inline sycl::double3
  toGroupReduceType(const quda::vector_type<double, 3> in) {
    return sycl::double3{in[0], in[1], in[2]};
  }
  inline sycl::double4
  toGroupReduceType(const quda::vector_type<double, 4> in) {
    return sycl::double4{in[0], in[1], in[2], in[3]};
  }
  inline auto toGroupReduceType(const quda::vector_type<double, 8> in) {
    return sycl::vec{in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7]};
  }
  inline auto toGroupReduceType(const quda::vector_type<double, 16> &in) {
    return *reinterpret_cast<const sycl::vec<double,16>*>(&in);
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>, 1> &in) {
    //return *reinterpret_cast<const sycl::vec<double,2>*>(&in);
    //return sycl::vec{in[0].x,in[0].y};
    return sycl::double2{in[0].x,in[0].y};
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>, 2> &in) {
    //return *reinterpret_cast<const sycl::vec<double,4>*>(&in);
    return sycl::vec{in[0].x,in[0].y,in[1].x,in[1].y};
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>, 3> &in) {
    //return sycl::vec{in[0],in[1],in[2]};
    return *reinterpret_cast<const sycl::vec<double,8>*>(&in);
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>, 4> &in) {
    //return sycl::vec{in[0],in[1],in[2],in[3]};
    return *reinterpret_cast<const sycl::vec<double,8>*>(&in);
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>, 8> &in) {
    //return *reinterpret_cast<const sycl::vec<double,16>*>(&in);
    //return sycl::vec{in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7]};
    return sycl::vec{in[0].x,in[0].y,in[1].x,in[1].y,in[2].x,in[2].y,in[3].x,in[3].y,
      in[4].x,in[4].y,in[5].x,in[5].y,in[6].x,in[6].y,in[7].x,in[7].y};
  }
  inline auto toGroupReduceType(const quda::vector_type<vec2<double>,16> &in) {
    //auto a = sycl::double4{in[0], in[1], in[2], in[3]};
    //auto b = sycl::double4{in[4], in[5], in[6], in[7]};
    //auto c = sycl::double4{in[8], in[9], in[10], in[11]};
    //auto d = sycl::double4{in[12], in[13], in[14], in[15]};
    //return sycl::vec{a,b,c,d};
    return reinterpret_cast<const sycl::vec<double,16>*>(&in);
  }

  template<typename T> inline T fromGroupReduceType(float in) {
    return T{in};
  }
  template<typename T> inline T fromGroupReduceType(double in) {
    return T{in};
  }
  template<typename T> inline T fromGroupReduceType(sycl::double2 &in) {
    //return T{in.x(), in.y()};
    return *reinterpret_cast<T*>(&in);
  }
  template<typename T> inline T fromGroupReduceType(sycl::double3 &in) {
    //return T{in.x(), in.y(), in.z()};
    return *reinterpret_cast<T*>(&in);
  }
  template<typename T> inline T fromGroupReduceType(sycl::double4 &in) {
    //return T{in.x(), in.y(), in.z(), in.w()};
    return *reinterpret_cast<T*>(&in);
  }
  template<typename T> inline T fromGroupReduceType(sycl::vec<double,8> &in) {
    //return T{in.x(), in.y(), in.z(), in.w()};
    return *reinterpret_cast<T*>(&in);
  }
  template<typename T> inline T fromGroupReduceType(sycl::vec<double,16> &in) {
    //return T{in.x(), in.y(), in.z(), in.w()};
    return *reinterpret_cast<T*>(&in);
  }



  template <typename T, typename U>
  void blockReduceSum(sycl::group<3> &grp, T &agg, const U &in)
  {
    //auto t = sycl::ONEAPI::reduce(grp, in, sycl::ONEAPI::plus<>());
    auto t = sycl::reduce_over_group(grp, in, sycl::plus<>());
    agg = fromGroupReduceType<T>(t);
  }

  template <typename T>
  void blockReduceSum(sycl::group<3> &grp, T &agg,
		      const sycl::vec<double,16> *in)
  {
    //auto t0 = sycl::ONEAPI::reduce(grp, in[0], sycl::ONEAPI::plus<>());
    //auto t1 = sycl::ONEAPI::reduce(grp, in[1], sycl::ONEAPI::plus<>());
    auto t0 = sycl::reduce_over_group(grp, in[0], sycl::plus<>());
    auto t1 = sycl::reduce_over_group(grp, in[1], sycl::plus<>());
    sycl::vec<double,16> x[2] = {t0,t1};
    agg = *reinterpret_cast<T*>(x);
  }

  template <typename T, typename U, int N>
  void blockReduceSum(sycl::group<3> &grp, T &agg,
		      const vector_type<quda::complex<U>, N> &in)
  {
    for(int i=0; i<in.size(); i++) {
      auto t = in[i];
      auto v = sycl::vec{t.real(),t.imag()};
      //auto r = sycl::ONEAPI::reduce(grp, v, sycl::ONEAPI::plus<>());
      auto r = sycl::reduce_over_group(grp, v, sycl::plus<>());
      agg[i].real(r[0]);
      agg[i].imag(r[1]);
    }
    //auto t = sycl::ONEAPI::reduce(grp, in, sycl::ONEAPI::plus<>());
    //agg = fromGroupReduceType<T>(t);
  }

  template <typename T, typename U>
  void blockReduceSum(T &agg, const U &in)
  {
    auto grp = getGroup();
    blockReduceSum(grp, agg, in);
  }


  template <typename T, typename U>
  void blockReduceMax(sycl::group<3> &grp, T &agg, const U &in)
  {
    //auto t = sycl::ONEAPI::reduce(grp, in, sycl::ONEAPI::maximum<>());
    auto t = sycl::reduce_over_group(grp, in, sycl::maximum<>());
    agg = fromGroupReduceType<T>(t);
  }


  template <typename T>
  inline void blockReduce(sycl::group<3> &grp, T &agg, const T &in,
			  const quda::plus<T> &r)
  {
    auto inX = toGroupReduceType(in);
    blockReduceSum(grp, agg, inX);
  }

  template <typename T>
  inline void blockReduce(sycl::group<3> &grp, T &agg, const T &in,
			  const quda::maximum<T> &r)
  {
    auto inX = toGroupReduceType(in);
    blockReduceMax(grp, agg, inX);
  }

  template <typename T, typename R>
  inline void blockReduce(sycl::group<3> &grp, T &agg, const T &in,
			  const R &r)
  {
    if (R::do_sum) {
      auto inX = toGroupReduceType(in);
      blockReduceSum(grp, agg, inX);
    } else {
      // FIXME: custom operation
      auto inX = toGroupReduceType(in);
      blockReduceSum(grp, agg, inX);
    }
  }

  //template <typename T, typename U>
  //void blockReduce(sycl::group<3> grp, T agg, T in, quda::blas::MultiReduce_<U,U> r)
  //{
    //auto inX = toGroupReduceType(in);
    //blockReduceMax(grp, agg, inX, r);
  //}

  //template<typename T, typename U> inline T fromGroupReduceType(sycl::vec<U,2> in) {
  //return T{fromGroupReduceType(in[0]), fromGroupReduceType(in[1])};
  //}
    /**
       @brief Generic reduction function that reduces block-distributed
       data "in" per thread to a single value.  This is the legacy
       variant which require explicit host-device synchronization to
       signal the completion of the reduction to the host.

       @param in The input per-thread data to be reduced
       @param idx In the case of multiple reductions, idx identifies
       which reduction this thread block corresponds to.  Typically idx
       will be constant along constant blockIdx.y and blockIdx.z.
    */
    template <int block_size_x, int block_size_y = 1,
	      typename Reducer, typename Arg, typename T>
    inline void reduce(Arg &arg, const Reducer &r, const T &in, const int idx = 0)
    {
      auto ndi = getNdItem();
      auto grp = getGroup();
      T aggregate;
      //T tin = in;
      blockReduce(grp, aggregate, in, r);
      //blockReduce(grp, aggregate, in, r);

      auto llid = ndi.get_local_linear_id();
      auto grpRg = ndi.get_group_range(0);
      auto grpId = ndi.get_group(0);

      bool isLastBlockDone = false;
      if (llid==0) {
	arg.partial[idx*grpRg + grpId] = aggregate;
	sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::device);
	// increment global block counter
	unsigned int prev = atomicAdd(&arg.count[idx], 1);
	// determine if last block
	isLastBlockDone = (prev == (grpRg-1));
      }
      //isLastBlockDone = sycl::ONEAPI::broadcast(grp, isLastBlockDone);
      //isLastBlockDone = sycl::ONEAPI::any_of(grp, isLastBlockDone);
      isLastBlockDone = group_broadcast(grp, isLastBlockDone);
      //ndi.barrier();
      //isLastBlockDone = sycl::ONEAPI::broadcast(grp, isLastBlockDone);

      // finish the reduction if last block
      if (isLastBlockDone) {
	auto nloc = ndi.get_local_range(0) * ndi.get_local_range(1);
	uint i = llid;
        T sum = arg.init();
	sycl::atomic_fence(sycl::memory_order::acquire,sycl::memory_scope::device);
	while (i<grpRg) {
	  sum = r(sum, arg.partial[idx*grpRg + i]);
	  //i += block_size_x*block_size_y;
	  i += nloc;
	}
	blockReduce(grp, sum, sum, r);

	// write out the final reduced value
	if (llid == 0) {
	  arg.result_d[idx] = sum;
	  arg.count[idx] = 0; // set to zero for next time
	}
      }
    }
#endif

} // namespace quda
