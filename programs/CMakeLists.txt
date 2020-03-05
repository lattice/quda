include_directories( ../utils)

# enable programs build a common library for all programs
if(QUDA_QDPJIT)
  target_link_libraries(quda
    INTERFACE ${QDP_LDFLAGS} ${QDP_LIB} ${QDP_LIBS} ${QIO_LIB} ${LIME_LIB} ${QUDA_QMP_LDFLAGS}
    ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

set(PROGRAMS_LIBS quda quda_utils)

# define programs
# if we build with QDP JIT the programs cannot run
if(QUDA_QDPJIT)
  set(QUDA_BUILD_ALL_PROGRAMS OFF)
endif()

cuda_add_executable(stoch_laph_quark_smear stoch_laph_quark_smear.cpp)
target_link_libraries(stoch_laph_quark_smear ${PROGRAMS_LIBS})
