/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file thrust/system/cuda/error.h
 *  \brief CUDA-specific error reporting
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>
#include <thrust/system/cuda/detail/guarded_driver_types.h>

namespace thrust
{

namespace system
{

namespace cuda
{

/*! \addtogroup system
 *  \{
 */

// To construct an error_code after a CUDA Runtime error:
//
//   error_code(::hipGetLastError(), cuda_category())

// XXX N3000 prefers enum class errc { ... }
namespace errc
{

/*! \p errc_t enumerates the kinds of CUDA Runtime errors.
 */
enum  	hipError_t { 
  hipSuccess = 0, 
  hipErrorOutOfMemory = 2, 
  hipErrorNotInitialized = 3, 
  hipErrorDeinitialized = 4, 
  hipErrorProfilerDisabled = 5, 
  hipErrorProfilerNotInitialized = 6, 
  hipErrorProfilerAlreadyStarted = 7, 
  hipErrorProfilerAlreadyStopped = 8, 
  hipErrorInvalidImage = 200, 
  hipErrorInvalidContext = 201, 
  hipErrorContextAlreadyCurrent = 202, 
  hipErrorMapFailed = 205, 
  hipErrorUnmapFailed = 206, 
  hipErrorArrayIsMapped = 207, 
  hipErrorAlreadyMapped = 208, 
  hipErrorNoBinaryForGpu = 209, 
  hipErrorAlreadyAcquired = 210, 
  hipErrorNotMapped = 211, 
  hipErrorNotMappedAsArray = 212, 
  hipErrorNotMappedAsPointer = 213, 
  hipErrorECCNotCorrectable = 214, 
  hipErrorUnsupportedLimit = 215, 
  hipErrorContextAlreadyInUse = 216, 
  hipErrorPeerAccessUnsupported = 217, 
  hipErrorInvalidKernelFile = 218, 
  hipErrorInvalidGraphicsContext = 219, 
  hipErrorInvalidSource = 300, 
  hipErrorFileNotFound = 301, 
  hipErrorSharedObjectSymbolNotFound = 302, 
  hipErrorSharedObjectInitFailed = 303, 
  hipErrorOperatingSystem = 304, 
  hipErrorInvalidHandle = 400, 
  hipErrorNotFound = 500, 
  hipErrorIllegalAddress = 700, 
  hipErrorInvalidSymbol = 701, 
  hipErrorMissingConfiguration = 1001, 
  hipErrorMemoryAllocation = 1002, 
  hipErrorInitializationError = 1003, 
  hipErrorLaunchFailure = 1004, 
  hipErrorPriorLaunchFailure = 1005, 
  hipErrorLaunchTimeOut = 1006, 
  hipErrorLaunchOutOfResources = 1007, 
  hipErrorInvalidDeviceFunction = 1008, 
  hipErrorInvalidConfiguration = 1009, 
  hipErrorInvalidDevice = 1010, 
  hipErrorInvalidValue = 1011, 
  hipErrorInvalidDevicePointer = 1017, 
  hipErrorInvalidMemcpyDirection = 1021, 
  hipErrorUnknown = 1030, 
  hipErrorInvalidResourceHandle = 1033, 
  hipErrorNotReady = 1034, 
  hipErrorNoDevice = 1038, 
  hipErrorPeerAccessAlreadyEnabled = 1050, 
  hipErrorPeerAccessNotEnabled = 1051, 
  hipErrorRuntimeMemory = 1052, 
  hipErrorRuntimeOther = 1053, 
  hipErrorHostMemoryAlreadyRegistered = 1061, 
  hipErrorHostMemoryNotRegistered = 1062, 
  hipErrorMapBufferObjectFailed = 1071, 
  hipErrorTbd 
};
enum errc_t
{
  // from cuda/include/driver_types.h
  // mirror their order
  success                            = hipSuccess,
  missing_configuration              = hipErrorMissingConfiguration,
  memory_allocation                  = hipErrorMemoryAllocation,
  initialization_error               = hipErrorInitializationError,
  launch_failure                     = hipErrorLaunchFailure,
  prior_launch_failure               = hipErrorPriorLaunchFailure,
  launch_timeout                     = hipErrorLaunchTimeOut,
  launch_out_of_resources            = hipErrorLaunchOutOfResources,
  invalid_device_function            = hipErrorInvalidDeviceFunction,
  invalid_configuration              = hipErrorInvalidConfiguration,
  invalid_device                     = hipErrorInvalidDevice,
  invalid_value                      = hipErrorInvalidValue, 
#if 0
  invalid_pitch_value                = hipErrorInvalidPitchValue,
invalid_symbol                     = hipErrorInvalidSymbol,
  map_buffer_object_failed           = hipErrorMapBufferObjectFailed,
  unmap_buffer_object_failed         = hipErrorUnmapBufferObjectFailed,
  invalid_host_pointer               = hipErrorInvalidHostPointer,
  
  invalid_device_pointer             = hipErrorInvalidDevicePointer,
  
  invalid_texture                    = hipErrorInvalidTexture,
  invalid_texture_binding            = hipErrorInvalidTextureBinding,
  invalid_channel_descriptor         = hipErrorInvalidChannelDescriptor,
  invalid_memcpy_direction           = hipErrorInvalidMemcpyDirection,
  address_of_constant_error          = hipErrorAddressOfConstant,
  texture_fetch_failed               = hipErrorTextureFetchFailed,
  texture_not_bound                  = hipErrorTextureNotBound,
  synchronization_error              = hipErrorSynchronizationError,
  invalid_filter_setting             = hipErrorInvalidFilterSetting,
  invalid_norm_setting               = hipErrorInvalidNormSetting,
  mixed_device_execution             = hipErrorMixedDeviceExecution,
  cuda_runtime_unloading             = hipErrorCudartUnloading,
  unknown                            = hipErrorUnknown,
  not_yet_implemented                = hipErrorNotYetImplemented,
  memory_value_too_large             = hipErrorMemoryValueTooLarge,
  invalid_resource_handle            = hipErrorInvalidResourceHandle,
  not_ready                          = hipErrorNotReady,
  insufficient_driver                = hipErrorInsufficientDriver,
  set_on_active_process_error        = hipErrorSetOnActiveProcess,
  no_device                          = hipErrorNoDevice,
  ecc_uncorrectable                  = hipErrorECCUncorrectable,

#if (defined(__NVCC__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 4020)) || defined(__HCC__) 
  shared_object_symbol_not_found     = hipErrorSharedObjectSymbolNotFound,
  shared_object_init_failed          = hipErrorSharedObjectInitFailed,
  unsupported_limit                  = hipErrorUnsupportedLimit,
  duplicate_variable_name            = hipErrorDuplicateVariableName,
  duplicate_texture_name             = hipErrorDuplicateTextureName,
  duplicate_surface_name             = hipErrorDuplicateSurfaceName,
  devices_unavailable                = hipErrorDevicesUnavailable,
  invalid_kernel_image               = hipErrorInvalidKernelImage,
  no_kernel_image_for_device         = hipErrorNoKernelImageForDevice,
  incompatible_driver_context        = hipErrorIncompatibleDriverContext,
  peer_access_already_enabled        = hipErrorPeerAccessAlreadyEnabled,
  peer_access_not_enabled            = hipErrorPeerAccessNotEnabled,
  device_already_in_use              = hipErrorDeviceAlreadyInUse,
  profiler_disabled                  = hipErrorProfilerDisabled,
  assert_triggered                   = hipErrorAssert,
  too_many_peers                     = hipErrorTooManyPeers,
  host_memory_already_registered     = hipErrorHostMemoryAlreadyRegistered ,
  host_memory_not_registered         = hipErrorHostMemoryNotRegistered,
  operating_system_error             = hipErrorOperatingSystem,
#endif

#if (defined(__NVCC__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 5000)) || defined(__HCC__) 
  peer_access_unsupported            = hipErrorPeerAccessUnsupported,
  launch_max_depth_exceeded          = hipErrorLaunchMaxDepthExceeded,
  launch_file_scoped_texture_used    = hipErrorLaunchFileScopedTex,
  launch_file_scoped_surface_used    = hipErrorLaunchFileScopedSurf,
  sync_depth_exceeded                = hipErrorSyncDepthExceeded,
  attempted_operation_not_permitted  = hipErrorNotPermitted,
  attempted_operation_not_supported  = hipErrorNotSupported,
#endif


  startup_failure                    = hipErrorStartupFailure
#endif
}; // end errc_t


} // end namespace errc

} // end namespace cuda

/*! \return A reference to an object of a type derived from class \p thrust::error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified
 *        for the class \p thrust::error_category. The object's \p name virtual function shall
 *        return a pointer to the string <tt>"cuda"</tt>. The object's
 *        \p default_error_condition virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a CUDA error value, the function
 *        shall return <tt>error_condition(ev,cuda_category())</tt>.
 *        Otherwise, the function shall return <tt>system_category.default_error_condition(ev)</tt>.
 */
inline const error_category &cuda_category(void);


// XXX N3000 prefers is_error_code_enum<cuda::errc>

/*! Specialization of \p is_error_code_enum for \p cuda::errc::errc_t
 */
template<> struct is_error_code_enum<cuda::errc::errc_t> : thrust::detail::true_type {};


// XXX replace cuda::errc::errc_t with cuda::errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), cuda::error_category())</tt>
 */
inline error_code make_error_code(cuda::errc::errc_t e);


// XXX replace cuda::errc::errc_t with cuda::errc upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), cuda::error_category())</tt>.
 */
inline error_condition make_error_condition(cuda::errc::errc_t e);

/*! \} // end system
 */


} // end system

namespace cuda
{

// XXX replace with using system::cuda_errc upon c++0x
namespace errc = system::cuda::errc;

} // end cuda

using system::cuda_category;

} // end namespace thrust

#include <thrust/system/cuda/detail/error.inl>

