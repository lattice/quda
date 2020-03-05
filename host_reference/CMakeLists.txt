include_directories(../utils)

# enable utils build a common library for all utils
set(QUDA_REFERENCE_COMMON blas_reference.cpp    covdev_reference.cpp              dslash_test_helpers.cpp    hisq_force_reference.cpp   llfat_reference.cpp             wilson_dslash_reference.cpp
clover_reference.cpp  domain_wall_dslash_reference.cpp  gauge_force_reference.cpp  hisq_force_reference2.cpp  staggered_dslash_reference.cpp)
cuda_add_library(quda_reference STATIC ${QUDA_REFERENCE_COMMON})
#cuda_add_library(quda_utils STATIC ../utils/libquda_utils.a)
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QUDA_QMPHOME)
  add_dependencies(quda_reference QMP)
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QIO AND QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
  add_dependencies(quda_reference QIO)
endif()
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
  add_dependencies(quda_reference QMP)
endif()

set(REFERENCE_LIBS quda quda_reference)

macro(QUDA_CHECKBUILDREFERENCE mytarget qudabuildreference)
  if(NOT ${qudabuildreference})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  endif()

  if(QUDA_QIO AND QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
    add_dependencies(${mytarget} QIO)
  endif()
  if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
    add_dependencies(${mytarget} QMP)
  endif()
endmacro()

if(QUDA_ARPACK)
  list(APPEND REFERENCE_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND REFERENCE_LIBS ${PARPACK})
  endif()
endif()
