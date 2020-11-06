# HIP Release notes
## 11/5/2020: First running Dslash on HIP

### Missing features: 
* GaugeFixOVR (Coulomp and Landau overrelaxation) relied too much on thrust. It works fine in the QUDA build but not in the HIP one
* All IPC Communications (P2P) has been removed since it causes link failures under HIP. The removal has been 
    guarded by the CMake variable QUDA_ENABLE_P2P. HIP builds should invoke CMake wiht -DQUDA_ENABLE_P2P=OFF. QUDA
    builds can use -DQUDA_ENABLE_P2P=ON (FIXME: Ensure this is default for CUDA)
* At last trial enabling the autotuner on HIP gave segfaults. This may be due to a poor choice for tuning params and needs debugging

### To Do Tasks:
* CMake improvements so HIP builds can automatically choose hiprand, hipfft, hipcub and hipblas (also if needed rocrand, rocfft, rocblas)
* Figure out about P2P linkage issues
* Figure out about autotuner
* Ensure currently not enabled kernels work (Staggered, DWF, TM, etc.)
* Ensure multigrid functions
* Optimize on H/W
* Try for mulit-node

### Building
I have teste HIPCC from ROCm 3.9.0. A reasonable CMake version is required

```  
module load rocm/3.9.0
module load gcc
module load cmake/3.18.2

export SM="sm_70" # not actually used
export INSTALLDIR=$HOME/install/quda
export HIP_CXXFLAGS="-D__gfx906__ -I${ROCM_PATH}/hiprand/include -I${ROCM_PATH}/rocrand/include -I${ROCM_PATH}/hipblas/include -I${ROCM_PATH}/hipcub/include -I${ROCM_PATH}/rocprim/include --amdgpu-target=gfx906" 
export HIP_LDFLAGS="-D__gfx906 --amdgpu-target=gfx906 -Wl,-rpath=${ROCM_PATH}/hiprand/lib -L${ROCM_PATH}/hiprand/lib -Wl,-rpath=${ROCM_PATH}/rocfft/lib -L${ROCM_PATH}/rocfft/lib -lhiprand -lrocfft -lrocfft-device -Wl,-rpath=${ROCM_PATH}/hipblas/lib -L${ROCM_PATH}/hipblas/lib -lhipblas -Wl,-rpath=${ROCM_PATH}/rocblas/lib -L${ROCM_PATH}/rocblas/lib -lrocblas -Wl,-rpath=${ROCMP_PATH}/hip/lib -Wl,-rpath=${ROCM_PATH}/rccl/lib -L${ROCM_PATH}/rccl/lib -lrccl"

cmake <path-to-parent-dir-of-your-quda-source>/quda \
        -G "Unix Makefiles" \
	-DQUDA_TARGET_TYPE="HIP" \
        -DQUDA_DIRAC_CLOVER=ON \
        -QUDA_DIRAC_CLOVER_HASENBUSCH=ON \
        -DQUDA_DIRAC_DOMAIN_WALL=ON \
        -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF \
        -DQUDA_DIRAC_STAGGERED=OFF \
        -DQUDA_DIRAC_TWISTED_MASS=OFF \
        -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
        -DQUDA_DIRAC_WILSON=ON \
        -DQUDA_DYNAMIC_CLOVER=OFF \
        -DQUDA_FORCE_GAUGE=OFF \
        -DQUDA_FORCE_HISQ=OFF \
        -DQUDA_GAUGE_ALG=ON \
        -DQUDA_GAUGE_TOOLS=OFF \
        -DQUDA_GPU_ARCH=${SM} \
        -DQUDA_QDPJIT=OFF \
        -DQUDA_INTERFACE_QDPJIT=OFF \
        -DQUDA_INTERFACE_MILC=OFF \
        -DQUDA_INTERFACE_CPS=OFF \
        -DQUDA_INTERFACE_QDP=ON \
        -DQUDA_INTERFACE_TIFR=OFF \
        -DQUDA_MAGMA=OFF        \
        -DQUDA_QMP=OFF \
        -DQUDA_OPENMP=OFF \
        -DQUDA_MULTIGRID=ON \
        -DQUDA_MAX_MULTI_BLAS_N=9 \
        -DQUDA_DOWNLOAD_EIGEN=ON \
        -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/quda \
        -DCMAKE_BUILD_TYPE=STRICT \
        -DCMAKE_CXX_COMPILER="hipcc"\
        -DCMAKE_CXX_FLAGS=" -std=c++14 ${CXXSYSTEM_INCLUDES} ${HIP_CXXFLAGS} -Wno-error=unused-variable " \
        -DCMAKE_C_COMPILER="hipcc" \
        -DCMAKE_EXE_LINKER_FLAGS="${HIP_LDFLAGS}  -lstdc++" \
        -DQUDA_BUILD_SHAREDLIB=ON \
        -DQUDA_BUILD_ALL_TESTS=ON \
	-DQUDA_ENABLE_P2P=OFF
```

