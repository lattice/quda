#pragma once

/**
   @file jitify_helper.cuh

   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#include <quda_define.h>
#if defined(JITIFY) && !defined(QUDA_TARGET_CUDA)
#error "Jitify compilation cannot be enabled unless targeting CUDA"
#endif

#if defined(JITIFY)

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

#include <jitify.hpp>
#include <device.h>
#include <kernel_helper.h>

#endif

namespace quda {

#ifdef JITIFY

  qudaError_t launch_jitify(const std::string &file, const std::string &kernel,
                            const std::vector<std::string> &template_args,
                            const TuneParam &tp, const qudaStream_t &stream,
                            std::vector<void*> arg_ptrs, jitify::detail::vector<std::string> arg_types);

  template <template <typename> class Functor, bool grid_stride, typename Arg, bool template_block_size = false>
  qudaError_t launch_jitify(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream,
                            const Arg &arg)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();
    std::vector<std::string> template_args = template_block_size ?
      std::vector<std::string>{reflect((int)tp.block.x), reflect((int)tp.block.y), Functor_naked, Arg_reflect, reflect(grid_stride)} :
      std::vector<std::string>{Functor_naked, Arg_reflect, reflect(grid_stride)};

    std::vector<void*> arg_ptrs{(void*)&arg};
    jitify::detail::vector<std::string> arg_types{Arg_reflect};

    return launch_jitify(Functor<Arg>::filename(), kernel, template_args,
                         //{reflect((int)tp.block.x), reflect((int)tp.block.y), Functor_naked, Arg_reflect, reflect(grid_stride)},
                         tp, stream, arg_ptrs, arg_types);
  }

  // FIXME merge the functionality of these launchers
  template <template <int, typename> class Functor, typename Arg>
  qudaError_t launch_jitify_block(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream,
                                  const Arg &arg)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<1, Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();

    std::vector<void*> arg_ptrs{(void*)&arg};
    jitify::detail::vector<std::string> arg_types{Arg_reflect};

    return launch_jitify(Functor<1, Arg>::filename(), kernel,
                         {reflect((int)tp.block.x), Functor_naked, Arg_reflect},
                         tp, stream, arg_ptrs, arg_types);
  }

#endif

} // namespace quda
