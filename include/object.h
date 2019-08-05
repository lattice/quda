/**
   @file object.h

   @section DESCRIPTION

   Abstract parent class for all classes in QUDA.  This parent class
   defines the new/delete methods to use QUDA's memory allocators.
   This gives us memory leak checking on these object instances.
*/

#pragma once

#include <malloc_quda.h>

namespace quda {
  
  class Object {
    
    
  public:
    inline Object() { }
    inline virtual ~Object() { }
    
    inline static void* operator new(std::size_t size) {
      return safe_malloc(size);
    }
    
    inline static void operator delete(void* p) {
      host_free(p);
    }
  
    inline static void* operator new[](std::size_t size) {
      return safe_malloc(size);
    }
  
    inline static void operator delete[](void* p) {
      host_free(p);
    }
  };

} // namespace quda
