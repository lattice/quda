#pragma once

namespace quda {

  class Worker {

  public:
    Worker() { }
    virtual ~Worker() { }
    virtual void apply(const cudaStream_t &stream) = 0;
    virtual void async_global_reduction(const cudaStream_t &stream) = 0;
    
  };

};
