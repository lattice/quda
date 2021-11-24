#pragma once

/**
   @file jitify_helper.h

   @brief Helper file when using jitify run-time compilation.  This
   file should be included in source code, and not jitify.hpp
   directly.
*/

#include <quda_define.h>
#if defined(JITIFY) && !defined(QUDA_TARGET_CUDA)
#error "Jitify compilation cannot be enabled unless targeting CUDA"
#endif

#if defined(JITIFY)

#include <jitify.hpp>
#include <device.h>
#include <kernel_helper.h>

#endif

namespace quda
{

#ifdef JITIFY

  qudaError_t launch_jitify(const std::string &file, const std::string &kernel,
                            const std::vector<std::string> &template_args, const TuneParam &tp,
                            const qudaStream_t &stream, std::vector<void *> &arg_ptrs,
                            jitify::detail::vector<std::string> &arg_types, std::vector<size_t> &arg_sizes,
                            bool use_kernel_arg);

  template <template <typename> class Functor, bool grid_stride, typename Arg>
  qudaError_t launch_jitify(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream, const Arg &arg)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();
    std::vector<std::string> template_args = std::vector<std::string> {Functor_naked, Arg_reflect, reflect(grid_stride)};

    std::vector<void *> arg_ptrs {(void *)&arg};
    jitify::detail::vector<std::string> arg_types {Arg_reflect};
    std::vector<size_t> arg_sizes {sizeof(arg)};

    return launch_jitify(Functor<Arg>::filename(), kernel, template_args, tp, stream, arg_ptrs, arg_types, arg_sizes,
                         device::use_kernel_arg<Arg>());
  }

#endif

} // namespace quda
