# generate an object library for all target specific files 
add_library(quda_cuda_programs OBJECT programs_quda.cpp color_cross.cu color_contract.cu evec_project_sum.cu)
if(QUDA_BUILD_SHAREDLIB)
  set_target_properties(quda_cuda_programs PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

#target_include_directories(quda_cuda_programs PUBLIC ./include)

# this part needs cleanup, thinking about what is needed here
target_compile_definitions(quda_cuda_programs PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_DEFINITIONS>)
target_include_directories(quda_cuda_programs PRIVATE $<TARGET_PROPERTY:quda_cpp,INCLUDE_DIRECTORIES> ./include)
target_compile_options(quda_cuda_programs PRIVATE $<TARGET_PROPERTY:quda_cpp,COMPILE_OPTIONS>)

#add alias
add_library(quda_programs ALIAS quda_cuda_programs)
