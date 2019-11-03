#pragma once

namespace quda {

  class Worker {

  public:
    Worker() { }
    virtual ~Worker() { }
    virtual void apply(const hipStream_t &stream) = 0;
    
  };

};
