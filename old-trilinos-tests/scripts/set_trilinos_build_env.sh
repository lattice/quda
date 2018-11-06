export TRILINOS_PATH=/home/astrel/Work/Libs/Trilinos-devel-v042018
export TRILINOS_BUILD=$TRILINOS_PATH/build-belos-tpetra-cc61-2


export OMPI_CXX=$TRILINOS_PATH/packages/kokkos/bin/nvcc_wrapper
export NVCC_WRAPPER_DEFAULT_COMPILER=/opt/gcc-6.4.0/bin/g++
export CUDA_LAUNCH_BLOCKING=1

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
