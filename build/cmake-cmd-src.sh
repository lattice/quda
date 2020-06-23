#!/bin/bash

export QUDA_GPU_ARCH=sm_75

cmake .. --debug-trycompile -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_EXE_LINKER_FLAGS_INIT="-L/usr/local/cuda-10.1/lib64 -lnvrtc  -lnvToolsExt" \
        -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda-10.1/include \
        -DCUDA_cublas_LIBRARY=/usr/local/cuda-10.1/lib64/libcublas.so \
        -DCUDA_cuda_LIBRARY=/usr/local/cuda-10.1/lib64/stubs/libcuda.so \
        -DQUDA_DIRAC_WILSON=ON \
        -DQUDA_DIRAC_CLOVER=OFF \
        -DQUDA_DIRAC_STAGGERED=OFF \
        -DQUDA_DIRAC_DOMAIN_WALL=OFF \
        -DQUDA_DIRAC_TWISTED_MASS=OFF \
        -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF \
        -DQUDA_BUILD_SHAREDLIB=ON \
	-DQUDA_LINK_HISQ=OFF \
        -DQUDA_FORCE_GAUGE=OFF \
        -DQUDA_FORCE_HISQ=OFF \
        -DQUDA_MPI=ON \
        -DQUDA_QMP=OFF \
        -DQUDA_QIO=OFF \
        -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
        -DQUDA_MULTIGRID=OFF\
        -DQUDA_INTERFACE_MILC=OFF \
        -DQUDA_USE_EIGEN=ON \
        -DEIGEN_INCLUDE_DIR=/opt/eigen-3.3.7/include/eigen3 \
        -DQUDA_DOWNLOAD_EIGEN=ON \
        -DQUDA_JITIFY=OFF \
        -DQUDA_TEX=OFF \
        -DCMAKE_BUILD_TYPE=HOSTDEBUG
