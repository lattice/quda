# add reference files to quda_test
target_sources(
  quda_test PRIVATE
  clover_reference.cpp
  covdev_reference.cpp
  blas_reference.cpp
  domain_wall_dslash_reference.cpp
  dslash_reference.cpp
  dslash_test_helpers.cpp
  gauge_force_reference.cpp
  hisq_force_reference.cpp
  staggered_dslash_reference.cpp
  wilson_dslash_reference.cpp)

target_include_directories(quda_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quda_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(quda_test PRIVATE ${CMAKE_BINARY_DIR}/include)
