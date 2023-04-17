#pragma once

#include <string>
#include <iostream>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <cstdarg>
#include <iomanip>
#include <typeinfo>
#include <map>

#include <tune_key.h>
#include <quda_internal.h>
#include <device.h>
#include <uint_to_char.h>

namespace quda {

  class TuneParam {

  public:
    dim3 block;
    dim3 grid;
    unsigned int shared_bytes;
    bool set_max_shared_bytes; // whether to opt in to max shared bytes per thread block
    int4 aux; // free parameter that can be used as an arbitrary autotuning dimension outside of launch parameters

    std::string comment;
    float time;
    long long n_calls;

    TuneParam();
    TuneParam(const TuneParam &) = default;
    TuneParam(TuneParam &&) = default;
    TuneParam &operator=(const TuneParam &) = default;
    TuneParam &operator=(TuneParam &&) = default;

    friend std::ostream& operator<<(std::ostream& output, const TuneParam& param) {
      output << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      output << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
      output << "shared_bytes=" << param.shared_bytes;
      output << ", aux=(" << param.aux.x << "," << param.aux.y << "," << param.aux.z << "," << param.aux.w << ")";
      return output;
    }
  };

  /**
   * @brief Returns a reference to the tunecache map
   * @return tunecache reference
   */
  const std::map<TuneKey, TuneParam> &getTuneCache();

  class Tunable {

  protected:
    virtual long long flops() const { return 0; }
    virtual long long bytes() const { return 0; }

    // the minimum number of shared bytes per thread
    virtual unsigned int sharedBytesPerThread() const { return 0; }

    // the minimum number of shared bytes per thread block
    virtual unsigned int sharedBytesPerBlock(const TuneParam &) const{ return 0; }

    // override this if a specific thread count is required (e.g., if not grid size tuning)
    virtual unsigned int minThreads() const { return 1; }
    virtual bool tuneGridDim() const { return true; }
    virtual bool tuneAuxDim() const { return false; }

    virtual bool tuneSharedBytes() const
    {
      static bool tune_shared = true;
      static bool init = false;

      if (!init) {
        char *enable_shared_env = getenv("QUDA_ENABLE_TUNING_SHARED");
        if (enable_shared_env) {
          if (strcmp(enable_shared_env, "0") == 0) { tune_shared = false; }
        }
        init = true;
      }
      return tune_shared;
    }

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

    /**
       @brief Return the maximum block size in the x dimension
       explored by the autotuner.
     */
    virtual unsigned int maxBlockSize(const TuneParam &param) const
    {
      return device::max_threads_per_block() / (param.block.y * param.block.z);
    }

    /**
       @brief Return the maximum grid size in the x dimension explored
       by the autotuner.  This defaults to twice the number of
       processors on the GPU, since it's unlikely a large grid size
       will help (if a kernels needs more parallelism, the autotuner
       will find this through increased block size.
     */
    virtual unsigned int maxGridSize() const { return 2 * device::processor_count(); }

    /**
       @brief Return the minimum grid size in the x dimension explored
       by the autotuner.  Default is 1, but it may be desirable to
       increase this to pare down the tuning dimension size.
    */
    virtual unsigned int minGridSize() const { return 1; }

    /**
       @brief gridStep sets the step size when iterating the grid size
       in advanceGridDim.
       @return Grid step size
    */
    virtual int gridStep() const { return 1; }

    virtual int blockStep() const;
    virtual int blockMin() const;

    virtual void resetBlockDim(TuneParam &param) const {
      if (tuneGridDim()) {
        param.block.x = blockMin();
      } else { // not tuning the grid dimension so have to set a valid grid size
        const auto step = blockStep();
        const auto max_threads = maxBlockSize(param);
        const auto max_blocks = device::max_grid_size(0);

        // ensure the blockDim is large enough given the limit on gridDim
        param.block.x = (minThreads() + max_blocks - 1) / max_blocks;
        param.block.x = ((param.block.x + step - 1) / step) * step; // round up to nearest step size
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
      int nthreads = param.block.x * param.block.y * param.block.z;
      param.shared_bytes = std::max(sharedBytesPerThread() * nthreads, sharedBytesPerBlock(param));

      if (param.block.x > max_threads || param.shared_bytes > max_shared
          || param.block.x * param.block.y * param.block.z > device::max_threads_per_block()) {
        resetBlockDim(param);
        int nthreads = param.block.x * param.block.y * param.block.z;
        param.shared_bytes = std::max(sharedBytesPerThread() * nthreads, sharedBytesPerBlock(param));
        ret = false;
      } else {
        ret = true;
      }

      if (!tuneGridDim()) param.grid.x = (minThreads() + param.block.x - 1) / param.block.x;

      return ret;
    }

    /**
     * @brief Returns the maximum dynamic shared memory per block.
     * @return The maximum dynamic shared memory to CUDA thread block
     */
    unsigned int maxDynamicSharedBytesPerBlock() const { return device::max_dynamic_shared_memory(); }

    /**
     * @brief The maximum shared memory that a CUDA thread block can
     * use in the autotuner.  This isn't necessarily the same as
     * maxDynamicSharedMemoryPerBlock since that may need explicit opt
     * in to enable.  If the kernel in question does this opt in then
     * this function can be overloaded to return
     * maxDynamicSharedBytesPerBlock.
     * @return The maximum shared bytes limit per block the autotung
     * will utilize.
     */
    virtual unsigned int maxSharedBytesPerBlock() const { return device::max_default_shared_memory(); }

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
        const auto max_shared = maxSharedBytesPerBlock();
        const int max_blocks_per_sm
          = std::min(device::max_threads_per_processor() / (param.block.x * param.block.y * param.block.z),
                     device::max_blocks_per_processor());
        int blocks_per_sm = max_shared / (param.shared_bytes ? param.shared_bytes : 1);
	if (blocks_per_sm > max_blocks_per_sm) blocks_per_sm = max_blocks_per_sm;
	param.shared_bytes = (blocks_per_sm > 0 ? max_shared / blocks_per_sm + 1 : max_shared + 1);

	if (param.shared_bytes > max_shared) {
	  TuneParam next(param);
	  advanceBlockDim(next); // to get next blockDim
	  int nthreads = next.block.x * next.block.y * next.block.z;
          param.shared_bytes = std::max(sharedBytesPerThread() * nthreads, sharedBytesPerBlock(next));
          return false;
	} else {
	  return true;
	}
      } else {
	return false;
      }
    }

    virtual bool advanceAux(TuneParam &) const { return false; }

    char vol[TuneKey::volume_n];
    char aux[TuneKey::aux_n];

    /** This is the return result from kernels launched using jitify,
        and can also be used to allow user invalidation of a tuning
        configuration */
    qudaError_t launch_error;

    /**
       @brief Whether the present instance has already been tuned or not
       @return True if tuned, false if not
    */
    bool tuned()
    {
      // not tuning is equivalent to already tuned
      if (!getTuning()) return true;

      TuneKey key = tuneKey();
      if (use_managed_memory()) strcat(key.aux, ",managed");
      // if key is present in cache then already tuned
      return getTuneCache().find(key) != getTuneCache().end();
    }

  public:
    Tunable() : launch_error(QUDA_SUCCESS) { aux[0] = '\0'; }
    virtual ~Tunable() = default;
    virtual TuneKey tuneKey() const = 0;
    virtual void apply(const qudaStream_t &stream) = 0;
    virtual void preTune() { }
    virtual void postTune() { }

    /**
     * @brief Number of iterations used in the 1st phase of tuning, i.e. finding the candidates for the 2nd phase/
     *
     * @return number of iterations
     */
    virtual int candidate_iter() const { return 2; }

    /**
     * @brief Number of candidates to be identified in the 1st phase for the 2nd tuning phase
     *
     * @return number of candidates
     */
    virtual size_t num_candidates() const { return 10; }

    /**
     * @brief Parameter to control the number of iteration used in the 2nd phase of tuning, i.e. for the candidates.
     *
     * @return minimum number of iterations
     */
    virtual int min_tune_iter() const { return 3; }

    /**
     * @brief Time parameter to control the number of iteration used in the 2nd phase of tuning, i.e. for the candidates.
     * This controls that the measured time for the number of iterations is at least this long based on the time from phase 1.
     *
     * @return minimum time that should be measured
     */
    virtual float min_tune_time() const { return 1e-3; }

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

    virtual std::string miscString(const TuneParam &) const { return std::string(); }

    virtual void initTuneParam(TuneParam &param) const
    {
      const unsigned int max_threads = device::max_threads_per_block_dim(0);
      const unsigned int max_blocks = device::max_grid_size(0);
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
      param.shared_bytes = std::max(sharedBytesPerThread() * nthreads, sharedBytesPerBlock(param));
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
    void checkLaunchParam(TuneParam &tp)
    {
      if (tp.block.x * tp.block.y * tp.block.z > device::max_threads_per_block())
        errorQuda("Requested block size %dx%dx%d=%d greater than max %d", tp.block.x, tp.block.y, tp.block.z,
                  tp.block.x * tp.block.y * tp.block.z, device::max_threads_per_block());

      if (tp.block.x > device::max_threads_per_block_dim(0))
        errorQuda("Requested X-dimension block size %d greater than max %d", tp.block.x,
                  device::max_threads_per_block_dim(0));

      if (tp.block.y > device::max_threads_per_block_dim(1))
        errorQuda("Requested Y-dimension block size %d greater than max %d", tp.block.y,
                  device::max_threads_per_block_dim(1));

      if (tp.block.z > device::max_threads_per_block_dim(2))
        errorQuda("Requested Z-dimension block size %d greater than max %d", tp.block.z,
                  device::max_threads_per_block_dim(2));

      if (tp.grid.x > device::max_grid_size(0))
        errorQuda("Requested X-dimension grid size %d greater than max %d", tp.grid.x, device::max_grid_size(0));

      if (tp.grid.y > device::max_grid_size(1))
        errorQuda("Requested Y-dimension grid size %d greater than max %d", tp.grid.y, device::max_grid_size(1));

      if (tp.grid.z > device::max_grid_size(2))
        errorQuda("Requested Z-dimension grid size %d greater than max %d", tp.grid.z, device::max_grid_size(2));

      if (tuneAuxDim() && tp.aux.x == -1 && tp.aux.y == -1 && tp.aux.z == -1 && tp.aux.w == -1)
        errorQuda("aux tuning enabled but param.aux is not initialized");
    }

    /**
     * @brief Return the rank on which kernel tuning is performed.
     * This will default to 0, but can be globally overriden with the
     * QUDA_TUNING_RANK environment variable.
     */
    virtual int32_t getTuneRank() const;

    qudaError_t launchError() const { return launch_error; }
    qudaError_t &launchError() { return launch_error; }
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

  /**
   * @brief Launch the autotuner.  If the tunable instance has already
   * been tuned, the launch parameters will be returned immediately.
   * If not, autotuner will commence, if enabled, else default launch
   * parameters will be returned.
   * @param[in,out] tunable The instance tunable we are tuning
   * @param[in] Whether tuning is enabled (if not then just return the
   * default parameters specificed in Tunable::defaultTuneParam()
   * @param[in] verbosity What verbosity to use during tuning?
   * @return The tuned launch parameters
   */
  TuneParam tuneLaunch(Tunable &tunable, QudaTune enabled = getTuning(), QudaVerbosity verbosity = getVerbosity());

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

  /**
   * @brief Query whether we are currently tuning a policy
   */
  bool policyTuning();

  /**
   * @brief Enable / disable whether we are tuning an uber kernel
   */
  void setUberTuning(bool);

  /**
   * @brief Query whether we are tuning an uber kernel
   */
  bool uberTuning();

} // namespace quda

#define postTrace() quda::postTrace_(__func__, quda::file_name(__FILE__), __LINE__)
