#include <task_graph.h>
#include <quda_api.h>
#include <quda_cuda_api.h>
#include <timer.h>
#include <target_device.h>

#define CHECK_CUDA_ERROR(func)                                          \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

namespace quda {

  /**
     @brief Wrapper around cudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] tp TuneParam containing the launch parameters
     @param[in] arg Host address of argument struct
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel(const void *func, const TuneParam &tp, const qudaStream_t &stream, const void *arg);

  bool use_task_graph()
  {
    static bool enabled = false;
    static bool init = false;

    if (!init) {
      char *enable_task_graph = getenv("QUDA_ENABLE_TASK_GRAPH");
      if (enable_task_graph && strcmp(enable_task_graph, "1") == 0) {
        warningQuda("CUDA graphs enabled");
        enabled = true;
      }

      init = true;
    }
    return enabled;
  }

    // From fast-hash. See https://github.com/ztanml/fast-hash
    inline uint64_t fasthash64(uint64_t h)
    {
      h ^= h >> 23;
      h *= 0x2127599bf4325c37ull;
      h ^= h >> 47;
      return h;
    }

    uint64_t fasthash64(const uint64_t* data, size_t size, uint64_t seed)
    {
      //auto start = data;
      const uint64_t m = 0x880355f21e6d1965ull;
      const uint64_t* end = data + size;
      uint64_t h = seed ^ (size * m);
      while (data != end) {
        h ^= fasthash64(*data++);
        h *= m;
        //std::cout << "offset = " << (char*)data - (char*)start << " hash = "  << h << std::endl;
      }
      return fasthash64(h);
    }

  void copy_arg(void *, void *, size_t, cudaStream_t stream);

  namespace task_graph {

    static std::map<graph_key_t, cudaGraphExec_t> graph_cache;

    // cache of pinned memory 
    static std::map<uint64_t, void*> arg_cache;

    static TimeProfile profile("Graph");

    qudaError_t launch_kernel(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream, const void *arg, size_t arg_bytes,
                              bool use_kernel_arg, void *constant_buffer)
    {
      profile.TPSTART(QUDA_PROFILE_TOTAL);

      qudaError_t launch_error = QUDA_SUCCESS;

      uint64_t arg_hash;
      profile.TPSTART(QUDA_PROFILE_GRAPH_KEY);
      arg_hash = fasthash64(reinterpret_cast<const uint64_t*>(arg), arg_bytes / sizeof(uint64_t));
      graph_key_t key{tp, kernel, arg_hash};
      profile.TPSTOP(QUDA_PROFILE_GRAPH_KEY);

      profile.TPSTART(QUDA_PROFILE_GRAPH_FIND);
      auto it = graph_cache.find(key);
      profile.TPSTOP(QUDA_PROFILE_GRAPH_FIND);

      cudaStream_t cuda_stream = target::cuda::get_stream(stream);

      if (it == graph_cache.end()) {
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        profile.TPSTART(QUDA_PROFILE_GRAPH_CAPTURE);

        void *arg_buffer = nullptr;
        if (!use_kernel_arg) {
          auto it = arg_cache.find(arg_hash);
          if (it == arg_cache.end()) {
            arg_buffer = pool_pinned_malloc(arg_bytes);
            arg_cache.insert(std::make_pair(arg_hash, arg_buffer));
          } else {
            arg_buffer = it->second;
          }
          memcpy(arg_buffer, arg, arg_bytes);
        }

        CHECK_CUDA_ERROR(cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal));
#if 1
        if (!use_kernel_arg) copy_arg(constant_buffer, arg_buffer, arg_bytes, cuda_stream);
#else
        if (!use_kernel_arg) qudaMemcpyAsync(constant_buffer, arg_buffer, arg_bytes, qudaMemcpyHostToDevice, stream);
#endif
        launch_error = qudaLaunchKernel(kernel.func, tp, stream, static_cast<const void *>(arg));
        CHECK_CUDA_ERROR(cudaStreamEndCapture(cuda_stream, &graph));

        profile.TPSTOP(QUDA_PROFILE_GRAPH_CAPTURE);

        profile.TPSTART(QUDA_PROFILE_GRAPH_INSTANTIATE);
        CHECK_CUDA_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
        CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
        graph_cache.insert(std::make_pair(key, instance)); // add this graph to the cache
        profile.TPSTOP(QUDA_PROFILE_GRAPH_INSTANTIATE);

        profile.TPSTART(QUDA_PROFILE_GRAPH_LAUNCH);
        CHECK_CUDA_ERROR(cudaGraphLaunch(instance, cuda_stream));
        profile.TPSTOP(QUDA_PROFILE_GRAPH_LAUNCH);
      } else {
        profile.TPSTART(QUDA_PROFILE_GRAPH_LAUNCH);
        CHECK_CUDA_ERROR(cudaGraphLaunch(it->second, cuda_stream));
        profile.TPSTOP(QUDA_PROFILE_GRAPH_LAUNCH);
      }
    
      profile.TPSTOP(QUDA_PROFILE_TOTAL);

      return launch_error;
    }

    void destroy()
    {
      if (getVerbosity() > QUDA_SUMMARIZE) profile.Print();

      // free pinned arg allocations
      for (auto &it : arg_cache) pool_pinned_free(it.second);
      arg_cache.clear();

      // free allocated executable graphs
      for (auto &it : graph_cache) cudaGraphExecDestroy(it.second);
      graph_cache.clear();
    }

  }
  
}
