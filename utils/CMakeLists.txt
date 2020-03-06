include_directories(../host_reference)

# enable utils build a common library for all utils
set(QUDA_UTILS_COMMON misc.cpp face_gauge.cpp host_utils.cpp command_line_params.cpp staggered_gauge_utils.cpp)
cuda_add_library(quda_utils STATIC ${QUDA_UTILS_COMMON})
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QUDA_QMPHOME)
  add_dependencies(quda_utils QMP)
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QIO AND QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
  add_dependencies(quda_utils QIO)
endif()
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
  add_dependencies(quda_utils QMP)
endif()

set(UTILS_LIBS quda quda_utils quda_reference)

macro(QUDA_CHECKBUILDUTILS mytarget qudabuildutils)
  if(NOT ${qudabuildutils})
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
  list(APPEND UTILS_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND UTILS_LIBS ${PARPACK})
  endif()
endif()
