#.rst:
# FindCUDAWrapper
# --------
#
# wrapper calls to help port cuda_add_executable / cuda_add_library over to
# the new cmake cuda first class support

# Find the CUDA_INCLUDE_DIRS and CUDA_TOOLKIT_INCLUDE like FindCUDA does
find_path(CUDAToolkit_INCLUDE_DIRS
  device_functions.h # Header included in toolkit
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES include ../include
  NO_DEFAULT_PATH
  )
mark_as_advanced(CUDAToolkit_INCLUDE_DIRS)
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${CUDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)

find_package(CUDALibs)