# add interface files / options
target_sources(quda_cpp PRIVATE 
  blas_interface.cpp
  laph_interface.cpp
  slaph_interface.cpp)

if(BUILD_FORTRAN_INTERFACE)
  target_sources(quda_cpp PRIVATE fortran_interface.cpp)
endif()
