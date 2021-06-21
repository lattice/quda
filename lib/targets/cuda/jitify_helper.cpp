#include <tune_quda.h>
#include <quda_api.h>
#include <quda_cuda_api.h>

#ifdef HOST_DEBUG

// display debugging info
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE        1
#define JITIFY_PRINT_LOG           1
#define JITIFY_PRINT_PTX           1
#define JITIFY_PRINT_LINKER_LOG    1
#define JITIFY_PRINT_LAUNCH        1
#define JITIFY_PRINT_HEADER_PATHS  1

#else // !HOST_DEBUG

// hide debugging info
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE        0
#ifdef DEVEL
#define JITIFY_PRINT_LOG           1
#else
#define JITIFY_PRINT_LOG           0
#endif
#define JITIFY_PRINT_PTX           0
#define JITIFY_PRINT_LINKER_LOG    0
#define JITIFY_PRINT_LAUNCH        0
#define JITIFY_PRINT_HEADER_PATHS  0

#endif // !HOST_DEBUG

#include "jitify_options.hpp"
#include <jitify_helper2.cuh>

namespace quda {

#ifdef JITIFY

  static jitify::JitCache *kernel_cache = nullptr;
  static bool jitify_init = false;

  static std::map<std::string, jitify::Program*> program_map;

  void create_jitify_program_v2(const std::string &file, const std::vector<std::string> extra_options = {})
  {
    if (!jitify_init) {
      kernel_cache = new jitify::JitCache;
    }
    
    if (program_map.find(file) == program_map.end()) {

      std::vector<std::string> options = {"-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-remove-unused-globals"};

#ifdef DEVICE_DEBUG
      options.push_back(std::string("-G"));
#endif

#if __cplusplus >= 201703L
      options.push_back(std::string("-std=c++17"));
#else
      options.push_back(std::string("-std=c++14"));
#endif

      // add an extra compilation options specific to this instance
      for (auto option : extra_options) options.push_back(option);

      jitify::Program *program = new jitify::Program(kernel_cache->program(file, 0, options));
      program_map[file] = program;
    }
  }

  qudaError_t launch_jitify(const std::string &file, const std::string &kernel,
                            const std::vector<std::string> &template_args,
                            const TuneParam &tp, const qudaStream_t &stream,
                            const std::vector<constant_param_t> &param,
                            std::vector<void*> arg_ptrs, jitify::detail::vector<std::string> arg_types)
  {
    std::string kernel_file(std::string("kernels/") + file);
    create_jitify_program_v2(kernel_file);

    std::string kernel_name(std::string("quda::") + kernel);
    auto instance = program_map[kernel_file]->kernel(kernel_name).instantiate(template_args);

    if (tp.set_max_shared_bytes && device::max_dynamic_shared_memory() > device::max_default_shared_memory()) {
      instance.set_func_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
      auto shared_size = instance.get_func_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
      instance.set_func_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  device::max_dynamic_shared_memory() - shared_size);
    }

    for (unsigned int i=0; i < param.size(); i++) {
      auto device_ptr = instance.get_constant_ptr(param[i].device_name);
      qudaMemcpyAsync((void*)device_ptr, param[i].host, param[i].bytes, qudaMemcpyHostToDevice, stream);
    }

    auto configured_instance = instance.configure(tp.grid, tp.block, tp.shared_bytes, device::get_cuda_stream(stream));
    auto error = configured_instance.launch(arg_ptrs, arg_types);

    if (error != CUDA_SUCCESS)
      cuda::set_driver_error(error, __func__, __func__, __FILE__, __STRINGIFY__(__LINE__), activeTuning());

    return error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
  }

#endif

}
