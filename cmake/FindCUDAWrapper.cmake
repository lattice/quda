#.rst:
# FindCUDAWrapper
# --------
#
# wrapper calls to help port cuda_add_executable / cuda_add_library over to
# the new cmake cuda first class support

# FindCUDAWrapper.cmake

#Very important the first step is to enable the CUDA language.
enable_language(CUDA)

# Find the CUDA_INCLUDE_DIRS and CUDA_TOOLKIT_INCLUDE like FindCUDA does
find_path(CUDA_TOOLKIT_INCLUDE
  device_functions.h # Header included in toolkit
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES include ../include
  NO_DEFAULT_PATH
  )
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${CUDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})


# Setup CUDA_LIBRARIES
set(CUDA_LIBRARIES ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that
  # the static cuda runtime can find it at runtime.
  list(APPEND CUDA_LIBRARIES -Wl,-rpath,/usr/local/cuda/lib)
endif()

# wrapper for cuda_add_library
# Issues:
#
function(cuda_add_library)
  add_library(${ARGV})
  target_include_directories(${ARGV0} PUBLIC
                             ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${ARGV0} ${CUDA_LIBRARIES})
  set_target_properties(${ARGV0} PROPERTIES LINKER_LANGUAGE CUDA)
endfunction()


# wrapper for cuda_add_library
# Issues:
#
function(cuda_add_executable)
  add_executable(${ARGV})
  target_include_directories(${ARGV0} PUBLIC
                             ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${ARGV0} ${CUDA_LIBRARIES})
  set_target_properties(${ARGV0} PROPERTIES LINKER_LANGUAGE CUDA)
endfunction()


find_package(CUDALibs)