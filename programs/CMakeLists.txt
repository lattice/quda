include_directories(. ../utils ../host_reference ../../quda/include/externals/)

# enable programs build a common library for all programs
set(QUDA_PROG_COMMON stoch_laph_quark_smear.h spin_dilute.cu evec_project.cu)
add_library(quda_prog STATIC ${QUDA_PROG_COMMON})
set_target_properties(quda_prog PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QUDA_QMPHOME)
  add_dependencies(quda_prog QMP)
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_QIO AND QUDA_DOWNLOAD_USQCD AND NOT QIO_FOUND)
  add_dependencies(quda_prog QIO)
endif()
if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QMP_FOUND)
  add_dependencies(quda_prog QMP)
endif()

set(PROG_LIBS quda quda_prog quda_reference quda_utils)

macro(QUDA_CHECKBUILDPROG mytarget qudabuildprogs)
  if(NOT ${qudabuildprogs})
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
  list(APPEND PROG_LIBS ${ARPACK})
  if(QUDA_MPI OR QUDA_QMP)
    list(APPEND PROG_LIBS ${PARPACK})
  endif()
endif()

# define programs

# if we build with QDP JIT the programs cannot run anyway
if(QUDA_QDPJIT)
  set(QUDA_BUILD_ALL_PROGRAMS OFF)
endif()

add_executable(stoch_laph_quark_smear stoch_laph_quark_smear_driver.cpp  stoch_laph_quark_smear_utils.cpp)
set_property(TARGET stoch_laph_quark_smear PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(stoch_laph_quark_smear PRIVATE ${PROG_LIBS})
quda_checkbuildtest(stoch_laph_quark_smear QUDA_BUILD_ALL_PROGRAMS)
