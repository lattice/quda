# CUDA Specific CMake
# Enable the language first so find_dependency uses the toolkit of
# CMAKE_CUDA_COMPILER, not whichever nvcc CUDA_PATH or PATH finds.

enable_language(CUDA)

find_dependency(CUDAToolkit REQUIRED)

