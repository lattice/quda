#.rst:
# FindCUDALibs
# --------
#
# Finds libraries provided by the CUDA Toolkit
#
#
# The script will prompt the user to specify CUDA_TOOLKIT_ROOT_DIR if
# the prefix cannot be determined by the location of nvcc in the system
# path and REQUIRED is specified to find_package().  To use a different
# installed version of the toolkit set the environment variable
# CUDA_BIN_PATH before running cmake (e.g.
# CUDA_BIN_PATH=/usr/local/cuda1.0 instead of the default
# /usr/local/cuda) or set CUDA_TOOLKIT_ROOT_DIR after configuring.  If
# you change the value of CUDA_TOOLKIT_ROOT_DIR, various components that
# depend on the path will be relocated.
#
# It might be necessary to set CUDA_TOOLKIT_ROOT_DIR manually on certain
# platforms, or to use a cuda runtime not installed in the default
# location.  In newer versions of the toolkit the cuda library is
# included with the graphics driver- be sure that the driver version
# matches what is needed by the cuda runtime version.
#
#
#
# The modules defines the following variables::
#   CUDA_VERSION_MAJOR    -- The major version of cuda as reported by nvcc.
#   CUDA_VERSION_MINOR    -- The minor version.
#   CUDA_VERSION
#   CUDA_VERSION_STRING   -- CUDA_VERSION_MAJOR.CUDA_VERSION_MINOR
#
#   CUDA_cupti_LIBRARY    -- CUDA Profiling Tools Interface library.
#                            Only available for CUDA version 4.0+.
#   CUDA_curand_LIBRARY   -- CUDA Random Number Generation library.
#                            Only available for CUDA version 3.2+.
#   CUDA_cusolver_LIBRARY -- CUDA Direct Solver library.
#                            Only available for CUDA version 7.0+.
#   CUDA_cusparse_LIBRARY -- CUDA Sparse Matrix library.
#                            Only available for CUDA version 3.2+.
#   CUDA_npp_LIBRARY      -- NVIDIA Performance Primitives lib.
#                            Only available for CUDA version 4.0+.
#   CUDA_nppc_LIBRARY     -- NVIDIA Performance Primitives lib (core).
#                            Only available for CUDA version 5.5+.
#   CUDA_nppi_LIBRARY     -- NVIDIA Performance Primitives lib (image processing).
#                            Only available for CUDA version 5.5+.
#   CUDA_npps_LIBRARY     -- NVIDIA Performance Primitives lib (signal processing).
#                            Only available for CUDA version 5.5+.
#   CUDA_nvcuvenc_LIBRARY -- CUDA Video Encoder library.
#                            Only available for CUDA version 3.2+.
#                            Windows only.
#   CUDA_nvcuvid_LIBRARY  -- CUDA Video Decoder library.
#                            Only available for CUDA version 3.2+.
#                            Windows only.
#

#   James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#   Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
#
#   Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#   Copyright (c) 2007-2009
#   Scientific Computing and Imaging Institute, University of Utah
#
#   This code is licensed under the MIT License.  See the FindCUDA.cmake script
#   for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindCUDALibs.cmake

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
endif()

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")


macro(cuda_unset_include_and_libraries)
  unset(CUDA_cublas_LIBRARY CACHE)
  unset(CUDA_cublas_device_LIBRARY CACHE)
  unset(CUDA_cufft_LIBRARY CACHE)
  unset(CUDA_cupti_LIBRARY CACHE)
  unset(CUDA_curand_LIBRARY CACHE)
  unset(CUDA_cusolver_LIBRARY CACHE)
  unset(CUDA_cusparse_LIBRARY CACHE)
  unset(CUDA_npp_LIBRARY CACHE)
  unset(CUDA_nppc_LIBRARY CACHE)
  unset(CUDA_nppi_LIBRARY CACHE)
  unset(CUDA_npps_LIBRARY CACHE)
  unset(CUDA_nvcuvenc_LIBRARY CACHE)
  unset(CUDA_nvcuvid_LIBRARY CACHE)
endmacro()

# Check to see if the CUDA_TOOLKIT_ROOT_DIR and CUDA_SDK_ROOT_DIR have changed,
# if they have then clear the cache variables, so that will be detected again.
if(NOT "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR_INTERNAL}")
  unset(CUDA_TOOLKIT_TARGET_DIR CACHE)
  unset(CUDA_NVCC_EXECUTABLE CACHE)
  cuda_unset_include_and_libraries()
endif()

if(NOT "${CUDA_TOOLKIT_TARGET_DIR}" STREQUAL "${CUDA_TOOLKIT_TARGET_DIR_INTERNAL}")
  cuda_unset_include_and_libraries()
endif()

macro(cuda_find_library_local_first_with_path_ext _var _names _doc _path_ext )
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # CUDA 3.2+ on Windows moved the library directories, so we need the new
    # and old paths.
    set(_cuda_64bit_lib_dir "${_path_ext}lib/x64" "${_path_ext}lib64" "${_path_ext}libx64" )
  endif()

  # CUDA 3.2+ on Windows moved the library directories, so we need to new
  # (lib/Win32) and the old path (lib).
  find_library(${_var}
    NAMES ${_names}
    PATHS "${CUDA_TOOLKIT_TARGET_DIR}" "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
    ENV CUDA_PATH
    ENV CUDA_LIB_PATH
    PATH_SUFFIXES ${_cuda_64bit_lib_dir} "lib" "${_path_ext}lib/Win32" "${_path_ext}lib" "${_path_ext}libWin32"
    DOC ${_doc}
    NO_DEFAULT_PATH
    )
  if (NOT CMAKE_CROSSCOMPILING)
    # Search default search paths, after we search our own set of paths.
    find_library(${_var}
      NAMES ${_names}
      PATHS "/usr/lib/nvidia-current"
      DOC ${_doc}
      )
  endif()
endmacro()

macro(cuda_find_library_local_first _var _names _doc)
  cuda_find_library_local_first_with_path_ext( "${_var}" "${_names}" "${_doc}" "" )
endmacro()

#######################
# Look for some of the toolkit helper libraries
macro(find_cuda_helper_libs _name)
  cuda_find_library_local_first(CUDA_${_name}_LIBRARY ${_name} "\"${_name}\" library")
  mark_as_advanced(CUDA_${_name}_LIBRARY)
endmacro()

# Search for the cuda distribution.
if(NOT CUDA_TOOLKIT_ROOT_DIR AND NOT CMAKE_CROSSCOMPILING)
  # Search in the CUDA_BIN_PATH first.

  get_filename_component(cuda_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)


  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS "${cuda_dir}"
      ENV CUDA_TOOLKIT_ROOT
      ENV CUDA_PATH
      ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin bin64
    DOC "CUDA Toolkit location."
    NO_DEFAULT_PATH
    )
endif()

if(CMAKE_CROSSCOMPILING)
  SET (CUDA_TOOLKIT_ROOT $ENV{CUDA_TOOLKIT_ROOT})
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    # Support for NVPACK
    set (CUDA_TOOLKIT_TARGET_NAME "armv7-linux-androideabi")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    # Support for arm cross compilation
    set(CUDA_TOOLKIT_TARGET_NAME "armv7-linux-gnueabihf")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Support for aarch64 cross compilation
    if (ANDROID_ARCH_NAME STREQUAL "arm64")
      set(CUDA_TOOLKIT_TARGET_NAME "aarch64-linux-androideabi")
    else()
      set(CUDA_TOOLKIT_TARGET_NAME "aarch64-linux")
    endif (ANDROID_ARCH_NAME STREQUAL "arm64")
  endif()

  if (EXISTS "${CUDA_TOOLKIT_ROOT}/targets/${CUDA_TOOLKIT_TARGET_NAME}")
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT}/targets/${CUDA_TOOLKIT_TARGET_NAME}" CACHE PATH "CUDA Toolkit target location.")
    SET (CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT})
    mark_as_advanced(CUDA_TOOLKIT_TARGET_DIR)
  endif()
endif()

# CUPTI library showed up in cuda toolkit 4.0
if(NOT CUDA_VERSION VERSION_LESS "4.0")
  cuda_find_library_local_first_with_path_ext(CUDA_cupti_LIBRARY cupti "\"cupti\" library" "extras/CUPTI/")
  mark_as_advanced(CUDA_cupti_LIBRARY)
endif()

find_cuda_helper_libs(cufft)
find_cuda_helper_libs(cublas)
if(NOT CUDA_VERSION VERSION_LESS "3.2")
  # cusparse showed up in version 3.2
  find_cuda_helper_libs(cusparse)
  find_cuda_helper_libs(curand)
  if (WIN32)
    find_cuda_helper_libs(nvcuvenc)
    find_cuda_helper_libs(nvcuvid)
  endif()
endif()
if(CUDA_VERSION VERSION_GREATER "5.0")
  find_cuda_helper_libs(cublas_device)
  # In CUDA 5.5 NPP was splitted onto 3 separate libraries.
  find_cuda_helper_libs(nppc)
  find_cuda_helper_libs(nppi)
  find_cuda_helper_libs(npps)
  set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}")
elseif(NOT CUDA_VERSION VERSION_LESS "4.0")
  find_cuda_helper_libs(npp)
endif()
if(NOT CUDA_VERSION VERSION_LESS "7.0")
  # cusolver showed up in version 7.0
  find_cuda_helper_libs(cusolver)
endif()
