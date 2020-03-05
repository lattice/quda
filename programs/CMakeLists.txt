# enable programs build a common library for all programs
#set(QUDA_PROGRAMS_COMMON )
cuda_add_library(quda_programs STATIC ../tests/libquda_test.a)
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QUDA_QMPHOME)
  add_dependencies(quda_programs QMP)
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QIO AND QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
  add_dependencies(quda_programs QIO)
endif()
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
  add_dependencies(quda_programs QMP)
endif()

set(PROGRAMS_LIBS quda quda_programs)

macro(QUDA_CHECKBUILDPROGRAMS mytarget qudabuildprograms)
  if(NOT ${qudabuildprograms})
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
  list(APPEND PROGRAMS_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND PROGRAMS_LIBS ${PARPACK})
  endif()
endif()

# define programs

# if we build with QDP JIT the programs cannot run
if(QUDA_QDPJIT)
  set(QUDA_BUILD_ALL_PROGRAMS OFF)
endif()

cuda_add_executable(stoch_laph_quark_smear stoch_laph_quark_smear.cpp)
target_link_libraries(stoch_laph_quark_smear ${PROGRAMS_LIBS})
quda_checkbuildprograms(stoch_laph_quark_smear QUDA_BUILD_ALL_PROGRAMS)
