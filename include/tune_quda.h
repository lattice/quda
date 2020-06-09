#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cfloat>
#include <stdarg.h>
#include <map>
#include <algorithm>
#include <typeinfo>

#include <tune_key.h>
#include <quda_internal.h>

// this file has some workarounds to allow compilation using nvrtc of kernels that include this file
#ifdef __CUDACC_RTC__
#define CUresult bool
#define CUDA_SUCCESS true
#endif

namespace quda {

  class TuneParam {

  public:
    dim3 block;
    dim3 grid;
    int shared_bytes;
    int4 aux; // free parameter that can be used as an arbitrary autotuning dimension outside of launch parameters

    std::string comment;
    float time;
    long long n_calls;

    inline TuneParam() : block(32, 1, 1), grid(1, 1, 1), shared_bytes(0), aux(), time(FLT_MAX), n_calls(0) {
      aux = make_int4(1,1,1,1);
    }

    inline TuneParam(const TuneParam &param)
      : block(param.block), grid(param.grid), shared_bytes(param.shared_bytes), aux(param.aux), comment(param.comment), time(param.time), n_calls(param.n_calls) { }

    inline TuneParam& operator=(const TuneParam &param) {
      if (&param != this) {
	block = param.block;
	grid = param.grid;
	shared_bytes = param.shared_bytes;
	aux = param.aux;
	comment = param.comment;
	time = param.time;
	n_calls = param.n_calls;
      }
      return *this;
    }

#ifndef __CUDACC_RTC__
    friend std::ostream& operator<<(std::ostream& output, const TuneParam& param) {
      output << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      output << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
      output << "shared_bytes=" << param.shared_bytes;
      output << ", aux=(" << param.aux.x << "," << param.aux.y << "," << param.aux.z << "," << param.aux.w << ")";
      return output;
    }
#endif
  };

#ifndef __CUDACC_RTC__
  /**
   * @brief Returns a reference to the tunecache map
   * @return tunecache reference
   */
  const std::map<TuneKey, TuneParam> &getTuneCache();
#endif

  class Tunable {

  protected:
    virtual long long flops() const = 0;
    virtual long long bytes() const { return 0; } // FIXME

    // the minimum number of shared bytes per thread
    virtual unsigned int sharedBytesPerThread() const = 0;

    // the minimum number of shared bytes per thread block
    virtual unsigned int sharedBytesPerBlock(const TuneParam &param) const = 0;

    // override this if a specific thread count is required (e.g., if not grid size tuning)
    virtual unsigned int minThreads() const { return 1; }
    virtual bool tuneGridDim() const { return true; }
    virtual bool tuneAuxDim() const { return false; }
    virtual bool tuneSharedBytes() const { return true; }

    virtual bool advanceGridDim(TuneParam &param) const
    {
      if (tuneGridDim()) {
        const int step = gridStep();
        param.grid.x += step;
        if (param.grid.x > maxGridSize()) {
          param.grid.x = minGridSize();
	  return false;
        } else {
          return true;
        }
      } else {
	return false;
      }
    }

    virtual unsigned int maxBlockSize(const TuneParam &param) const { return deviceProp.maxThreadsPerBlock / (param.block.y*param.block.z); }
    virtual unsigned int maxGridSize() const { return 2*deviceProp.multiProcessorCount; }
    virtual unsigned int minGridSize() const { return 1; }

    /**
       @brief gridStep sets the step size when iterating the grid size
       in advanceGridDim.
       @return Grid step size
    */
    virtual int gridStep() const { return 1; }

    virtual int blockStep() const { return deviceProp.warpSize; }
    virtual int blockMin() const { return deviceProp.warpSize; }

    virtual void resetBlockDim(TuneParam &param) const {
      if (tuneGridDim()) {
        param.block.x = blockMin();
      } else { // not tuning the grid dimension so have to set a valid grid size
        const auto step = blockStep();
        const auto max_threads = maxBlockSize(param);
        const auto max_blocks = deviceProp.maxGridSize[0];

        // ensure the blockDim is large enough given the limit on gridDim
        param.block.x = (minThreads()+max_blocks-1)/max_blocks;
	param.block.x = ((param.block.x+step-1)/step)*step; // round up to nearest step size
	if (param.block.x > max_threads && param.block.y == 1 && param.block.z == 1)
	  errorQuda("Local lattice volume is too large for device");
      }
    }

    virtual bool advanceBlockDim(TuneParam &param) const
    {
      const unsigned int max_threads = maxBlockSize(param);
      const unsigned int max_shared = maxSharedBytesPerBlock();
      bool ret;

      param.block.x += blockStep();
      int nthreads = param.block.x*param.block.y*param.block.z;
      if (param.block.x > max_threads || sharedBytesPerThread() * nthreads > max_shared
          || sharedBytesPerBlock(param) > max_shared) {
        resetBlockDim(param);
	ret = false;
      } else {
        ret = true;
      }

      if (!tuneGridDim())
	param.grid = dim3((minThreads()+param.block.x-1)/param.block.x, 1, 1);

      return ret;
    }

    /**
     * @brief Returns the maximum number of simultaneously resident
     * blocks per SM.  We can directly query this of CUDA 11, but
     * previously this needed to be hand coded.
     * @return The maximum number of simultaneously resident blocks per SM
     */
    unsigned int maxBlocksPerSM() const
    {
#if CUDA_VERSION >= 11000
      static int max_blocks_per_sm = 0;
      if (!max_blocks_per_sm) cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, comm_gpuid());
      return max_blocks_per_sm;
#else
      // these variables are taken from Table 14 of the CUDA 10.2 prgramming guide
      switch (deviceProp.major) {
      case 2:
	return 8;
      case 3:
	return 16;
      case 5:
      case 6: return 32;
      case 7:
        switch (deviceProp.minor) {
        case 0: return 32;
        case 2: return 32;
        case 5: return 16;
        }
      default:
        warningQuda("Unknown SM architecture %d.%d - assuming limit of 32 blocks per SM\n",
                    deviceProp.major, deviceProp.minor);
        return 32;
      }
#endif
    }

    /**
     * @brief Enable the maximum dynamic shared bytes for the kernel
     * "func" (values given by maxDynamicSharedBytesPerBlock()).
     * @param[in] func Function pointer to the kernel we want to
     * enable max shared memory per block for
     */
    template <typename F> inline void setMaxDynamicSharedBytesPerBlock(F *func) const
    {
#if CUDA_VERSION >= 9000
      qudaFuncSetAttribute(
          (const void *)func, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
      qudaFuncSetAttribute(
          (const void *)func, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynamicSharedBytesPerBlock());
#endif
    }

    /**
     * @brief Returns the maximum dynamic shared memory per block.
     * @return The maximum dynamic shared memory to CUDA thread block
     */
    unsigned int maxDynamicSharedBytesPerBlock() const
    {
#if CUDA_VERSION >= 9000
      static int max_shared_bytes = 0;
      if (!max_shared_bytes) cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, comm_gpuid());
      return max_shared_bytes;
#else
      // these variables are taken from Table 14 of the CUDA 10.2 prgramming guide
      switch (deviceProp.major) {
      case 2:
      case 3:
      case 5:
      case 6: return 48 * 1024;
      default:
        warningQuda("Unknown SM architecture %d.%d - assuming limit of 48 KiB per SM\n",
                    deviceProp.major, deviceProp.minor);
        return 48 * 1024;
      }
#endif
    }

    /**
     * @brief The maximum shared memory that a CUDA thread block can
     * use in the autotuner.  This isn't necessarily the same as
     * maxDynamicSharedMemoryPerBlock since that may need explicit opt
     * in to enable (by calling setMaxDynamicSharedBytes for the
     * kernel in question).  If the CUDA kernel in question does this
     * opt in then this function can be overloaded to return
     * maxDynamicSharedBytesPerBlock.
     * @return The maximum shared bytes limit per block the autotung
     * will utilize.
     */
    virtual unsigned int maxSharedBytesPerBlock() const { return deviceProp.sharedMemPerBlock; }

    /**
     * The goal here is to throttle the number of thread blocks per SM
     * by over-allocating shared memory (in order to improve L2
     * utilization, etc.).  We thus request the smallest amount of
     * dynamic shared memory that guarantees throttling to a given
     * number of blocks, in order to allow some extra leeway.
     */
    virtual bool advanceSharedBytes(TuneParam &param) const
    {
      if (tuneSharedBytes()) {
        const int max_shared = maxSharedBytesPerBlock();
        const int max_blocks_per_sm = std::min(deviceProp.maxThreadsPerMultiProcessor / (param.block.x*param.block.y*param.block.z), maxBlocksPerSM());
	int blocks_per_sm = max_shared / (param.shared_bytes ? param.shared_bytes : 1);
	if (blocks_per_sm > max_blocks_per_sm) blocks_per_sm = max_blocks_per_sm;
	param.shared_bytes = (blocks_per_sm > 0 ? max_shared / blocks_per_sm + 1 : max_shared + 1);

	if (param.shared_bytes > max_shared) {
	  TuneParam next(param);
	  advanceBlockDim(next); // to get next blockDim
	  int nthreads = next.block.x * next.block.y * next.block.z;
          param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(next) ?
              sharedBytesPerThread() * nthreads :
              sharedBytesPerBlock(next);
          return false;
	} else {
	  return true;
	}
      } else {
	return false;
      }
    }

    virtual bool advanceAux(TuneParam &param) const { return false; }

    char aux[TuneKey::aux_n];

    int writeAuxString(const char *format, ...) {
      int n = 0;
#ifndef __CUDACC_RTC__
      va_list arguments;
      va_start(arguments, format);
      n = vsnprintf(aux, TuneKey::aux_n, format, arguments);
      if (n < 0 || n >= TuneKey::aux_n) errorQuda("Error writing auxiliary string");
#endif
      return n;
    }

    /** This is the return result from kernels launched using jitify */
    CUresult jitify_error;

    /**
       @brief Whether the present instance has already been tuned or not
       @return True if tuned, false if not
    */
    bool tuned()
    {
#ifndef __CUDACC_RTC__
      // not tuning is equivalent to already tuned
      if (!getTuning()) return true;

      TuneKey key = tuneKey();
      if (use_managed_memory()) strcat(key.aux, ",managed");
      // if key is present in cache then already tuned
      return getTuneCache().find(key) != getTuneCache().end();
#else
      return true;
#endif
    }

  public:
    Tunable() : jitify_error(CUDA_SUCCESS) { aux[0] = '\0'; }
    virtual ~Tunable() { }
    virtual TuneKey tuneKey() const = 0;
    virtual void apply(const qudaStream_t &stream) = 0;
    virtual void preTune() { }
    virtual void postTune() { }
    virtual int tuningIter() const { return 1; }

#ifndef __CUDACC_RTC__
    virtual std::string paramString(const TuneParam &param) const
    {
      std::stringstream ps;
      ps << param;
      return ps.str();
    }

    virtual std::string perfString(float time) const
    {
      float gflops = flops() / (1e9 * time);
      float gbytes = bytes() / (1e9 * time);
      std::stringstream ss;
      ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << gflops << " Gflop/s, ";
      ss << gbytes << " GB/s";
      return ss.str();
    }
#endif

    virtual void initTuneParam(TuneParam &param) const
    {
      const unsigned int max_threads = deviceProp.maxThreadsDim[0];
      const unsigned int max_blocks = deviceProp.maxGridSize[0];
      const int min_grid_size = minGridSize();
      const int min_block_size = blockMin();

      if (tuneGridDim()) {
	param.block = dim3(min_block_size,1,1);

	param.grid = dim3(min_grid_size,1,1);
      } else {
	// find the minimum valid blockDim
	param.block = dim3((minThreads()+max_blocks-1)/max_blocks, 1, 1);
	param.block.x = ((param.block.x+min_block_size-1) / min_block_size) * min_block_size; // round up to the nearest multiple of desired minimum block size
	if (param.block.x > max_threads) errorQuda("Local lattice volume is too large for device");

	param.grid = dim3((minThreads()+param.block.x-1)/param.block.x, 1, 1);
      }
      int nthreads = param.block.x*param.block.y*param.block.z;
      param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
	sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      initTuneParam(param);
      if (tuneGridDim()) param.grid.x = maxGridSize(); // don't set y and z in case derived initTuneParam has
    }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      return advanceSharedBytes(param) || advanceBlockDim(param) || advanceGridDim(param) || advanceAux(param);
    }

    /**
     * Check the launch parameters of the kernel to ensure that they are
     * valid for the current device.
     */
    void checkLaunchParam(TuneParam &param) {

      if (param.block.x*param.block.y*param.block.z > (unsigned)deviceProp.maxThreadsPerBlock)
        errorQuda("Requested block size %dx%dx%d=%d greater than hardware limit %d",
                  param.block.x, param.block.y, param.block.z, param.block.x*param.block.y*param.block.z, deviceProp.maxThreadsPerBlock);

      if (param.block.x > (unsigned int)deviceProp.maxThreadsDim[0])
	errorQuda("Requested X-dimension block size %d greater than hardware limit %d",
		  param.block.x, deviceProp.maxThreadsDim[0]);

      if (param.block.y > (unsigned int)deviceProp.maxThreadsDim[1])
	errorQuda("Requested Y-dimension block size %d greater than hardware limit %d",
		  param.block.y, deviceProp.maxThreadsDim[1]);

      if (param.block.z > (unsigned int)deviceProp.maxThreadsDim[2])
	errorQuda("Requested Z-dimension block size %d greater than hardware limit %d",
		  param.block.z, deviceProp.maxThreadsDim[2]);

      if (param.grid.x > (unsigned int)deviceProp.maxGridSize[0])
	errorQuda("Requested X-dimension grid size %d greater than hardware limit %d",
		  param.grid.x, deviceProp.maxGridSize[0]);

      if (param.grid.y > (unsigned int)deviceProp.maxGridSize[1])
	errorQuda("Requested Y-dimension grid size %d greater than hardware limit %d",
		  param.grid.y, deviceProp.maxGridSize[1]);

      if (param.grid.z > (unsigned int)deviceProp.maxGridSize[2])
	errorQuda("Requested Z-dimension grid size %d greater than hardware limit %d",
		  param.grid.z, deviceProp.maxGridSize[2]);
    }

    CUresult jitifyError() const { return jitify_error; }
    CUresult& jitifyError() { return jitify_error; }
  };


  /**
     This derived class is for algorithms that deploy parity across
     the y dimension of the thread block with no shared memory tuning.
     The x threads will typically correspond to the checkboarded
     volume.
   */
  class TunableLocalParity : public Tunable {

  protected:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    // don't tune the grid dimension
    virtual bool tuneGridDim() const { return false; }

    /**
       The maximum block size in the x dimension is the total number
       of threads divided by the size of the y dimension
     */
    unsigned int maxBlockSize(const TuneParam &param) const { return deviceProp.maxThreadsPerBlock / 2; }

  public:
    bool advanceBlockDim(TuneParam &param) const {
      bool rtn = Tunable::advanceBlockDim(param);
      param.block.y = 2;
      return rtn;
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.block.y = 2;
    }

    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.block.y = 2;
    }

  };

  /**
     This derived class is for algorithms that deploy a vector of
     computations across the y dimension of both the threads block and
     grid.  For example this could be parity in the y dimension and
     checkerboarded volume in x.
   */
  class TunableVectorY : public Tunable {

  protected:
    virtual unsigned int sharedBytesPerThread() const { return 0; }
    virtual unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    mutable unsigned int vector_length_y;
    mutable unsigned int step_y;
    bool tune_block_x;

  public:
  TunableVectorY(unsigned int vector_length_y) : vector_length_y(vector_length_y),
      step_y(1), tune_block_x(true) { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = tune_block_x ? Tunable::advanceBlockDim(param) : false;
      param.block.y = block.y;
      param.grid.y = grid.y;

      if (ret) {
	return true;
      } else { // block.x (spacetime) was reset

	// we can advance spin/block-color since this is valid
	if (param.block.y < vector_length_y && param.block.y < (unsigned int)deviceProp.maxThreadsDim[1] &&
	    param.block.x*(param.block.y+step_y)*param.block.z <= (unsigned int)deviceProp.maxThreadsPerBlock) {
	  param.block.y += step_y;
	  param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
	  return true;
	} else { // we have run off the end so let's reset
	  param.block.y = step_y;
	  param.grid.y = (vector_length_y + param.block.y - 1) / param.block.y;
	  return false;
	}
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = step_y;
      param.grid.y = (vector_length_y + step_y - 1) / step_y;
    }

    void resizeVector(int y) const { vector_length_y = y; }
    void resizeStep(int y) const { step_y = y; }
  };

  class TunableVectorYZ : public TunableVectorY {

    mutable unsigned vector_length_z;
    mutable unsigned step_z;
    bool tune_block_y;

  public:
    TunableVectorYZ(unsigned int vector_length_y, unsigned int vector_length_z)
      : TunableVectorY(vector_length_y), vector_length_z(vector_length_z),
      step_z(1), tune_block_y(true) { }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = tune_block_y ? TunableVectorY::advanceBlockDim(param) : tune_block_x ? Tunable::advanceBlockDim(param) : false;
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (ret) {
	// we advanced the block.x / block.y so we're done
	return true;
      } else { // block.x/block.y (spacetime) was reset

	// we can advance spin/block-color since this is valid
	if (param.block.z < vector_length_z && param.block.z < (unsigned int)deviceProp.maxThreadsDim[2] &&
	    param.block.x*param.block.y*(param.block.z+step_z) <= (unsigned int)deviceProp.maxThreadsPerBlock) {
	  param.block.z += step_z;
	  param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
	  return true;
	} else { // we have run off the end so let's reset
	  param.block.z = step_z;
	  param.grid.z = (vector_length_z + param.block.z - 1) / param.block.z;
	  return false;
	}
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorY::initTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorY::defaultTuneParam(param);
      param.block.z = step_z;
      param.grid.z = (vector_length_z + step_z - 1) / step_z;
    }

    void resizeVector(int y, int z) const { vector_length_z = z;  TunableVectorY::resizeVector(y); }
    void resizeStep(int y, int z) const { step_z = z;  TunableVectorY::resizeStep(y); }
  };

  /**
     @brief query if tuning is in progress
     @return tuning in progress?
  */
  bool activeTuning();

  void loadTuneCache();
  void saveTuneCache(bool error = false);

  /**
   * @brief Save profile to disk.
   */
  void saveProfile(const std::string label = "");

  /**
   * @brief Flush profile contents, setting all counts to zero.
   */
  void flushProfile();

  TuneParam& tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity);

  /**
   * @brief Post an event in the trace, recording where it was posted
   */
  void postTrace_(const char *func, const char *file, int line);

  /**
   * @brief Enable the profile kernel counting
   */
  void enableProfileCount();

  /**
   * @brief Disable the profile kernel counting
   */
  void disableProfileCount();

  /**
   * @brief Enable / disable whether are tuning a policy
   */
  void setPolicyTuning(bool);

} // namespace quda

// undo jit-safe modifications
#ifdef __CUDACC_RTC__
#undef CUresult
#undef CUDA_SUCCESS
#endif

#define postTrace() quda::postTrace_(__func__, quda::file_name(__FILE__), __LINE__)
