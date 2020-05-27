#ifndef _QUDA_INTERNAL_H
#define _QUDA_INTERNAL_H

#include <quda_cuda_api.h>
#include <string>
#include <complex>
#include <vector>

#if ((defined(QMP_COMMS) || defined(MPI_COMMS)) && !defined(MULTI_GPU))
#error "MULTI_GPU must be enabled to use MPI or QMP"
#endif

#if (!defined(QMP_COMMS) && !defined(MPI_COMMS) && defined(MULTI_GPU))
#error "MPI or QMP must be enabled to use MULTI_GPU"
#endif

#ifdef QMP_COMMS
#include <qmp.h>
#endif

// these are helper macros used to enable spin-1, spin-2 and spin-4 building blocks as needed
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_CLOVER_DIRAC)                           \
  || defined(GPU_TWISTED_MASS_DIRAC) || defined(GPU_TWISTED_CLOVER_DIRAC) || defined(GPU_NDEG_TWISTED_MASS_DIRAC)      \
  || defined(GPU_CLOVER_HASENBUSCH_TWIST) || defined(GPU_COVDEV)
#define NSPIN4
#endif

#if defined(GPU_MULTIGRID)
#define NSPIN2
#endif

#if defined(GPU_STAGGERED_DIRAC)
#define NSPIN1
#endif

// this is a helper macro for stripping the path information from
// __FILE__.  FIXME - convert this into a consexpr routine
#define KERNEL_FILE                                                                                                    \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 :                                                               \
                            strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
#include <enum_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  struct QUDA_DiracField{
    void *field; /**< Pointer to a ColorSpinorField */
  };

  extern cudaDeviceProp deviceProp;
  extern qudaStream_t *streams;

#ifdef __cplusplus
}
#endif

namespace quda {

  struct alignas(8) char8 {
    char4 x;
    char4 y;
  };

  struct alignas(16) short8 {
    short4 x;
    short4 y;
  };

  struct alignas(32) float8 {
    float4 x;
    float4 y;
  };

  struct alignas(64) double8 {
    double4 x;
    double4 y;
  };

  typedef std::complex<double> Complex;

  /**
   * Traits for determining the maximum and inverse maximum
   * value of a (signed) char and short. Relevant for
   * fixed-precision types.
   */
  template< typename T > struct fixedMaxValue{ static constexpr float value = 0.0f; };
  template<> struct fixedMaxValue<short>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<short2>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<short4>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<short8>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<char>{ static constexpr float value = 127.0f; };
  template<> struct fixedMaxValue<char2>{ static constexpr float value = 127.0f; };
  template<> struct fixedMaxValue<char4>{ static constexpr float value = 127.0f; };
  template<> struct fixedMaxValue<char8>{ static constexpr float value = 127.0f; };

  template <typename T> struct fixedInvMaxValue {
    static constexpr float value = 3.402823e+38f;
  };
  template <> struct fixedInvMaxValue<short> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short2> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short4> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<short8> {
    static constexpr float value = 3.0518509476e-5f;
  };
  template <> struct fixedInvMaxValue<char> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char2> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char4> {
    static constexpr float value = 7.874015748031e-3f;
  };
  template <> struct fixedInvMaxValue<char8> {
    static constexpr float value = 7.874015748031e-3f;
  };

  const int Nstream = 9;

  /**
   * Check that the resident gauge field is compatible with the requested inv_param
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  bool canReuseResidentGauge(QudaInvertParam *inv_param);

}

#include <timer.h>


#endif // _QUDA_INTERNAL_H
