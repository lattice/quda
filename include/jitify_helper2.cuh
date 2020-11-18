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
#include <jitify.hpp>
#include <device.h>
#include <kernel_helper.h>
#endif

namespace quda {

#ifdef JITIFY

  CUresult launch_jitify(const std::string &file, const std::string &kernel,
                         const std::vector<std::string> &template_args,
                         const TuneParam &tp, const qudaStream_t &stream,
                         const std::vector<constant_param_t> &param,
                         std::vector<void*> arg_ptrs, jitify::detail::vector<std::string> arg_types);

  template <template <typename> class Functor, typename Arg>
  CUresult launch_jitify(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream,
                         const Arg &arg, const std::vector<constant_param_t> &param = dummy_param)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();

    std::vector<void*> arg_ptrs{(void*)&arg};
    jitify::detail::vector<std::string> arg_types{Arg_reflect};

    return launch_jitify(Functor<Arg>::filename(), kernel, {Functor_naked, Arg_reflect},
                         tp, stream, param, arg_ptrs, arg_types);    
  }

  template <template <typename> class Functor, template <typename> class Reducer, typename Arg>
  CUresult launch_jitify(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream,
                         const Arg &arg, const std::vector<constant_param_t> &param = dummy_param)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Reducer_instance = reflect<Reducer<Arg>>();
    auto Reducer_naked = Reducer_instance.substr(0, Reducer_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();

    std::vector<void*> arg_ptrs{(void*)&arg};
    jitify::detail::vector<std::string> arg_types{Arg_reflect};

    return launch_jitify(Functor<Arg>::filename(), kernel,
                         {reflect((int)tp.block.x), reflect((int)tp.block.y), Functor_naked, Reducer_naked, Arg_reflect},
                         tp, stream, param, arg_ptrs, arg_types);
  }

  // FIXME merge the functionality of these launchers
  template <template <typename> class Functor, typename Arg>
  CUresult launch_jitify_block(const std::string &kernel, const TuneParam &tp, const qudaStream_t &stream,
                               const Arg &arg, const std::vector<constant_param_t> &param = dummy_param)
  {
    // we need this hackery to get the naked unbound template class parameters
    using namespace jitify::reflection;
    auto Functor_instance = reflect<Functor<Arg>>();
    auto Functor_naked = Functor_instance.substr(0, Functor_instance.find("<"));
    auto Arg_reflect = reflect<Arg>();

    std::vector<void*> arg_ptrs{(void*)&arg};
    jitify::detail::vector<std::string> arg_types{Arg_reflect};

    return launch_jitify(Functor<Arg>::filename(), kernel,
                         {reflect((int)tp.block.x), Functor_naked, Arg_reflect},
                         tp, stream, param, arg_ptrs, arg_types);
  }

#endif

} // namespace quda
