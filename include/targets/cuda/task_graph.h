#include <tune_quda.h>
#include <kernel_helper.h>

namespace quda {

  bool use_task_graph();

  namespace task_graph {

    struct graph_key_t {
      TuneParam launch_param;
      kernel_t kernel;
      //TuneKey tune_key;    
      uint64_t arg_hash;

      bool operator<(const graph_key_t &other) const
      {
        if (kernel < other.kernel) {
          return true;
        } else if (kernel == other.kernel) {
          if (arg_hash < other.arg_hash) return true;
          else if (arg_hash == other.arg_hash) {
            if (launch_param < other.launch_param) return true;
          }
        }
        return false;
      }

      friend std::ostream &operator<<(std::ostream &output, const graph_key_t &key)
      {
        output << key.launch_param << std::endl;
        output << "kernel name = " << key.kernel.name << std::endl;
        output << "kernel func = " << key.kernel.func << std::endl;
        output << "arg_hash = " << key.arg_hash << std::endl;
        return output;
      }

    };

    qudaError_t launch_kernel(const kernel_t &kernel, const TuneParam &tp, const qudaStream_t &stream,
                              const void *arg, size_t arg_bytes, bool use_kernel_arg, void *constant_buffer);


    void destroy();

  }
}
