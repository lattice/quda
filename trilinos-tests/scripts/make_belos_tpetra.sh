#!/bin/bash

cmake  	-D CMAKE_CXX_COMPILER=/opt/openmpi-3.0.0-gcc-6.4.0-dyn/bin/mpicxx \
	-D CMAKE_C_COMPILER=/opt/openmpi-3.0.0-gcc-6.4.0-dyn/bin/mpicc \
	-D CMAKE_Fortran_COMPILER=/opt/openmpi-3.0.0-gcc-6.4.0-dyn/bin/mpif77 \
	-D CMAKE_CXX_FLAGS="-Wall --pedantic -O2 -std=c++11" \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_VERBOSE_MAKEFILE=ON \
        -D CMAKE_INSTALL_PREFIX="/home/astrel/Work/Libs/install" \
        -D BUILD_SHARED_LIBS=ON \
	-D Trilinos_CXX11_FLAGS="--expt-extended-lambda" \
	-D TPL_ENABLE_MPI=ON \
	-D TPL_ENABLE_CUDA=ON \
	-D Kokkos_ENABLE_Cuda=ON \
	-D KOKKOS_ARCH=Pascal61 \
	-D Kokkos_ENABLE_Cuda_UVM=ON \
	-D Kokkos_ENABLE_Cuda_Lambda=ON \
        -D Teuchos_ENABLE_COMPLEX=ON \
        -D Teuchos_ENABLE_DEBUG=ON \
        -D Trilinos_ENABLE_DEBUG=ON \
        -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
        -D Trilinos_ENABLE_Belos=ON \
        -D Trilinos_ENABLE_Tpetra=ON \
        -D Tpetra_INST_COMPLEX_DOUBLE=ON \
        -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
        -D Trilinos_ENABLE_Epetra=OFF \
	-D Trilinos_ENABLE_Fortran=OFF \
	-D TPL_ENABLE_Boost=OFF \
	-D TPL_ENABLE_BoostLib=OFF \
	-D Trilinos_ENABLE_AztecOO=OFF \
	-D Trilinos_ENABLE_Ifpack=OFF \
	-D Trilinos_ENABLE_MueLu=OFF \
	-D MueLu_ENABLE_Epetra=OFF \
	-D Trilinos_ENABLE_OpenMP=ON \
	-D Trilinos_SHOW_DEPRECATED_WARNINGS=OFF \
	-D Trilinos_ENABLE_TESTS=ON \
	-D TPL_BLAS_LIBRARIES="/usr/lib/libblas.so.3" \
	-D TPL_LAPACK_LIBRARIES="/usr/lib/liblapack.so.3" \
	$TRILINOS_PATH 

make -j4

#       -D CMAKE_BUILD_TYPE=DEBUG \

