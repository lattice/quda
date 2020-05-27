# enable utils build a common library for all utils
add_library(
  quda_utils STATIC
  command_line_params.cpp
  face_gauge.cpp
  host_blas.cpp
  host_utils.cpp
  llfat_utils.cpp
  misc.cpp
  set_params.cpp
  staggered_gauge_utils.cpp
  staggered_host_utils.cpp)

target_include_directories(quda_utils PUBLIC .)
target_link_libraries(quda_utils PRIVATE quda)

if(QUDA_QIO
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QIO_FOUND)
  add_dependencies(quda_utils QIO)
endif()

if(QUDA_QMP
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QMP_FOUND)
  add_dependencies(quda_utils QMP)
endif()
