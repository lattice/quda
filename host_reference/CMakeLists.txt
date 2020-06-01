# enable utils build a common library for all utils
add_library(
  quda_reference STATIC
  clover_reference.cpp
  covdev_reference.cpp
  domain_wall_dslash_reference.cpp
  dslash_reference.cpp
  dslash_test_helpers.cpp
  gauge_force_reference.cpp
  hisq_force_reference.cpp
  hisq_force_reference2.cpp
  staggered_dslash_reference.cpp
  wilson_dslash_reference.cpp)

target_include_directories(quda_reference PUBLIC .)
target_include_directories(quda_reference PRIVATE ../utils)
target_link_libraries(quda_reference PRIVATE quda)

if(QUDA_QIO
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QIO_FOUND)
  add_dependencies(quda_reference QIO)
endif()
if(QUDA_QMP
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QMP_FOUND)
  add_dependencies(quda_reference QMP)
endif()
