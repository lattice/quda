# generate an object library for all target specific files
add_library(quda_blas_interface OBJECT blas_interface.cpp)
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_blas_interface PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

# this part needs cleanup, thinking about what is needed here
target_compile_definitions(quda_blas_interface PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_DEFINITIONS>)
target_include_directories(quda_blas_interface PRIVATE $<TARGET_PROPERTY:quda_cpp,INCLUDE_DIRECTORIES>)
target_compile_options(quda_blas_interface PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_OPTIONS>)

# add alias
add_library(quda_interface ALIAS quda_blas_interface)
