#include <quda_internal.h>

namespace quda {

  /*
    Here we use traits to define the mapping between storage type and
    register type:
    double -> double
    float -> float
    short -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
  template<> struct mapper<double> { typedef double type; };
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };

  template<typename T1, typename T2> __host__ __device__ inline void copy (T1 &a, const T2 &b) { a = b; }
  template<> __host__ __device__ inline void copy(float &a, const short &b) { a = (float)b/MAX_SHORT; }
  template<> __host__ __device__ inline void copy(short &a, const float &b) { a = (short)(b*MAX_SHORT); }

  /**
     Generic wrapper for Trig functions
  */
  template <bool isHalf>
    struct Trig {
      template<typename T> 
      __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b); }
      template<typename T> 
      __device__ __host__ static T Sin( const T &a ) { return sin(a); }
      template<typename T> 
      __device__ __host__ static T Cos( const T &a ) { return cos(a); }
    };
  
  /**
     Specialization of Trig functions using shorts
   */
  template <>
    struct Trig<true> {
    template<typename T> 
      __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b)/M_PI; }
    template<typename T> 
      __device__ __host__ static T Sin( const T &a ) { return sin(a*M_PI); }
    template<typename T> 
      __device__ __host__ static T Cos( const T &a ) { return cos(a*M_PI); }
  };

} // namespace quda
