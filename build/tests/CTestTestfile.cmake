# CMake generated Testfile for 
# Source directory: /home/alex/ssd-disk/workspace/OneAPI/quda/tests
# Build directory: /home/alex/ssd-disk/workspace/OneAPI/quda/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(blas_test_parity_wilson "/opt/openmpi-4.0.0-gcc-7.4.0-dyn/bin/mpiexec" "-n" "4" "/home/alex/ssd-disk/workspace/OneAPI/quda/build/tests/blas_test" "--dim" "2" "4" "6" "8" "--solve-type" "direct-pc" "--gtest_output=xml:blas_test_parity.xml")
set_tests_properties(blas_test_parity_wilson PROPERTIES  _BACKTRACE_TRIPLES "/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;253;add_test;/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;0;")
add_test(blas_test_full_wilson "/opt/openmpi-4.0.0-gcc-7.4.0-dyn/bin/mpiexec" "-n" "4" "/home/alex/ssd-disk/workspace/OneAPI/quda/build/tests/blas_test" "--dim" "2" "4" "6" "8" "--solve-type" "direct" "--gtest_output=xml:blas_test_full.xml")
set_tests_properties(blas_test_full_wilson PROPERTIES  _BACKTRACE_TRIPLES "/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;258;add_test;/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;0;")
add_test(dslash_wilson-policytune "/opt/openmpi-4.0.0-gcc-7.4.0-dyn/bin/mpiexec" "-n" "4" "/home/alex/ssd-disk/workspace/OneAPI/quda/build/tests/dslash_ctest" "--dslash-type" "wilson" "--test" "MatPCDagMatPC" "--dim" "2" "4" "6" "8" "--gtest_output=xml:dslash_wilson_test_poltune.xml")
set_tests_properties(dslash_wilson-policytune PROPERTIES  _BACKTRACE_TRIPLES "/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;308;add_test;/home/alex/ssd-disk/workspace/OneAPI/quda/tests/CMakeLists.txt;0;")
