#pragma once

#include <cstdlib>
#include <quda_internal.h>

namespace quda {
  // forward declaration of class TestEnv for later use
  class TestEnv;


  /**
   * @brief Singleton class to hold global quda configuration variables ususally set through environement variables
   * @details See https://github.com/lattice/quda/wiki/QUDA-Environment-Variables for a list of environment variables.
   */
  class QudaEnv {
    friend class TestEnv;

   public:
    static QudaEnv &getInstance() {
      static QudaEnv    instance; // Guaranteed to be destroyed.
      // Instantiated on first use.
      return instance;
    }

   private:
    /**
    * try to read an integer value from an environement variable
    * @param envname
    * @param defaultValue
    */
    int getenvInt(const char *envname, const int defaultvalue) {
      const char *env = std::getenv(envname);

      return env ? atoi(env) : defaultvalue;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     *
     * @param envname [description]
     * @param defaultvalue [description]
     *
     * @return [description]
     */
    bool getenvBool(const char *envname, const int defaultvalue) {
      const char *env = std::getenv(envname);

      return env ? atoi(env) == 1 : defaultvalue == 1;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     *
     * @param envname [description]
     * @return [description]
     */
    char *getenvChar(const char *envname) {
      char *env = std::getenv(envname);
      return env ? env : nullptr;
    }

    // MPI related
    int enable_p2p;
    bool enable_gdr;
    char *gdr_blacklist;
    bool enable_mps;

    // dslash policy
    char *dslash_policy;
    bool dslash_pack;
    bool dslash_interior;
    bool dslash_exterior;
    bool dslash_copy;
    bool dslash_comms;

    // interface
    bool allow_jit;
    bool enable_numa;
    char *reorder_location;
    bool device_reset;

    // malloc
    bool device_memory_pool;
    bool pinned_memory_pool;

    char *resource_path;
    char *profile_output_base;

    char *rank_verbosity;
    bool enable_tuning;

    QudaEnv() {

      enable_p2p = getenvInt("QUDA_ENABLE_P2P", 3);
      enable_gdr = (getenvInt("QUDA_ENABLE_GDR", 0) == 1);
      gdr_blacklist = getenvChar("QUDA_ENABLE_GDR_BLACKLIST");
      enable_mps = (getenvInt("QUDA_ENABLE_MPS", 0) == 1);

      dslash_policy = getenvChar("QUDA_ENABLE_DSLASH_POLICY");
      dslash_pack = getenvBool("QUDA_ENABLE_DSLASH_PACK", 1);
      dslash_interior = getenvBool("QUDA_ENABLE_DSLASH_INTERIOR", 1);
      dslash_exterior = getenvBool("QUDA_ENABLE_DSLASH_EXTERIOR", 1);
      dslash_copy = getenvBool("QUDA_ENABLE_DSLASH_COPY", 1);
      dslash_comms = getenvBool("QUDA_ENABLE_DSLASH_COMMS", 1);

      allow_jit = getenvBool("QUDA_ALLOW_JIT", 0);
      enable_numa = getenvBool("QUDA_ENABLE_NUMA", 0);
      reorder_location = getenvChar("QUDA_REORDER_LOCATION");
      device_reset = getenvBool("QUDA_DEVICE_RESET", 0);

      device_memory_pool = getenvBool("QUDA_ENABLE_DEVICE_MEMORY_POOL", 1);
      pinned_memory_pool = getenvBool("QUDA_ENABLE_PINNED_MEMORY_POOL", 1);

      resource_path = getenvChar("QUDA_RESOURCE_PATH");
      profile_output_base = getenvChar("QUDA_PROFILE_OUTPUT_BASE");

      rank_verbosity = getenvChar("QUDA_RANK_VERBOSITY");
      enable_tuning = getenvBool("QUDA_ENABLE_TUNING", 1);

    }

   public:
    QudaEnv(QudaEnv const &)         = delete;
    void operator=(QudaEnv const &)  = delete;

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    int get_enable_p2p() const {
      return enable_p2p;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_gdr() const {
      return enable_gdr;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_mps() const {
      return enable_mps;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */char *get_enable_gdr_blacklist() const {
      return gdr_blacklist;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    char *get_enable_dslash_policy() const {
      return dslash_policy;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_dslash_pack() const {
      return dslash_pack;
    }
    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_dslash_interior() const {
      return dslash_interior;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_dslash_exterior()const {
      return dslash_exterior;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_dslash_copy()const {
      return dslash_copy;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_dslash_comms()const {
      return dslash_comms;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_allow_jit() const {
      return allow_jit;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_numa() const {
      return enable_numa;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    char *get_reorder_location() const {
      return reorder_location;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_device_reset() const {
      return device_reset;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_device_memory_pool() const {
      return device_memory_pool;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_pinned_memory_pool() const {
      return pinned_memory_pool;
    }

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    char *get_resource_path() const {
      return resource_path;
    }

    /**
     * @brief
     *
     * @details [long description]
     * @return [description]
     */
    char *get_profile_output_base() const {
      return profile_output_base;
    }


    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    char *get_rank_verbosity() const {
      return rank_verbosity;
    };

    /**
     * @brief [brief description]
     * @details [long description]
     * @return [description]
     */
    bool get_enable_tuning() const {
      return enable_tuning;
    }
   protected:
    // may add setters here

  };

} // namespace quda