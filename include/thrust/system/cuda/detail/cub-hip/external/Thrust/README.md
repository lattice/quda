# Thrust
HIP back-end for Thrust (alpha release).

### Introduction
Thrust is a parallel algorithm library. This library has been ported to HIP/ROCm platform. This repository contains the HIP port of Thrust. The HIP ported library works on both HIP/CUDA and HIP/ROCm platforms.


### Pre-requisites

### Hardware
Visit the following link for ROCm hardware requirements:
https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#supported-cpus


### Installation

AMD ROCm Installation

$ wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -

$ sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

$ sudo apt-get update

$ sudo apt-get install rocm-dev

### Thrust

$ git clone  https://github.com/ROCmSoftwarePlatform/Thrust.git

$ cd Thrust

### Executable generation

$ export HIP_PLATFORM=hcc (For HCC Platform )

$ export HIP_PLATFORM=nvcc ( For NVCC Platform)				

$ cd examples									

Change the extension of the application file from .cu to .cpp

$ cp sum.cu sum.cpp

To recursively change the extension of all the applications in the Thrust/examples run the  cu_to_cpp.sh script

 $ ./cu_to_cpp.sh

For NVCC or HCC platform, build application using hipcc compiler for example:  

$ /opt/rocm/bin/hipcc sum.cpp  -I../ -o sum.out 							

To recursively generate executables to all the applications in Thrust/examples,run script_compile_testing_hcc.sh script

$ ./script_compile_testing_hcc.sh


### Unit Test												
 Run the following commands to generate executables for test cases in Thrust/testing.	

 Single file

$ cp device_ptr.cu device_ptr.cpp

$ /opt/rocm/bin/hipcc device_ptr.cpp testframework.cpp -I. -I../ -o device_ptr.cpp.out									

 Multiple files(recursively)

$ ./cu_to_cpp.sh

$ ./script_compile_testing_hcc.sh												

### Known Issues:
A linker error is being faced while trying to generate executables for test cases in Thrust/testing/backend/cuda,thereby allowing to generate only object files. The compile only option “-c” alone works while compiling test cases in backend/cuda.					
