#pragma once

#include <cstdlib>
#include <quda_internal.h>

namespace quda {
  // forward declaration of class TestEnv for later use
	class TestEnv;

  /**
  * Singleton
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
  	int getenvInt(const char* envname, const int defaultvalue){
  		const char* env = std::getenv(envname);

  		return env ? atoi(env) : defaultvalue;
  	}

  	bool getenvBool(const char* envname, const int defaultvalue){
  		const char* env = std::getenv(envname);

  		return env ? atoi(env)==1 : defaultvalue==1;
  	}

  	char* getenvChar(const char* envname){
  		char* env = std::getenv(envname);
  		return env ? env : nullptr;
  	}

		// MPI related
  	int enable_p2p;
  	bool enable_gdr;
  	char* gdr_blacklist;
  	bool enable_mps;

		// dslash policy
  	char* dslash_policy;
  	bool dslash_pack;
  	bool dslash_interior;
  	bool dslash_exterior;
  	bool dslash_copy;
  	bool dslash_comms;

  	// interface
  	bool allow_jit;
  	bool enable_numa;
  	char* reorder_location;
  	bool device_reset;


  	QudaEnv() {

  		enable_p2p = getenvInt("QUDA_ENABLE_P2P",3);
  		enable_gdr = (getenvInt("QUDA_ENABLE_GDR",0)==1);
  		gdr_blacklist = getenvChar("QUDA_ENABLE_GDR_BLACKLIST");
  		enable_mps = (getenvInt("QUDA_ENABLE_MPS",0)==1);

  		dslash_policy = getenvChar("QUDA_ENABLE_DSLASH_POLICY");
  		dslash_pack = getenvBool("QUDA_ENABLE_DSLASH_PACK",1);
  		dslash_interior = getenvBool("QUDA_ENABLE_DSLASH_INTERIOR",1);
  		dslash_exterior = getenvBool("QUDA_ENABLE_DSLASH_EXTERIOR",1);
  		dslash_copy = getenvBool("QUDA_ENABLE_DSLASH_COPY",1);
  		dslash_comms = getenvBool("QUDA_ENABLE_DSLASH_COMMS",1);

  		allow_jit = getenvBool("QUDA_ALLOW_JIT",0);
  		enable_numa =getenvBool("QUDA_ENABLE_NUMA",0);
  		reorder_location=getenvChar("QUDA_REORDER_LOCATION");
  		device_reset=getenvBool("QUDA_DEVICE_RESET",0);

    } 


		// char *device_reset_env = getenv("QUDA_DEVICE_RESET");


		// //malloc.cpp:	c
		// char *enable_device_pool = getenv("QUDA_ENABLE_DEVICE_MEMORY_POOL");
		// char *enable_pinned_pool = getenv("QUDA_ENABLE_PINNED_MEMORY_POOL");

		// //milc_interface.cpp
		// char *quda_reconstruct = getenv("QUDA_MILC_HISQ_RECONSTRUCT");
		// char *quda_solver = getenv("QUDA_MILC_CLOVER_SOLVER");

		// //tune.cpp
		// char *path = getenv("QUDA_RESOURCE_PATH");
		// char *profile_fname = getenv("QUDA_PROFILE_OUTPUT_BASE");

		// //util_quda
		// static char *rank_verbosity_env = getenv("QUDA_RANK_VERBOSITY");
		// char *enable_tuning = getenv("QUDA_ENABLE_TUNING");

  public:
  	QudaEnv(QudaEnv const &)         = delete;
  	void operator=(QudaEnv const &)  = delete;

  	int get_enable_p2p() const{
  		return enable_p2p;
  	}

  	bool get_enable_gdr() const{
  		return enable_gdr;
  	}

  	bool get_enable_mps() const{
  		return enable_mps;
  	}

  	char* get_enable_gdr_blacklist() const{
  		return gdr_blacklist;
  	}

  	char* get_enable_dslash_policy() const{
  		return dslash_policy;
  	}

  	bool get_enable_dslash_pack() const{
  		return dslash_pack;
  	}

  	bool get_enable_dslash_interior() const{
  		return dslash_interior;
  	}

  	bool get_enable_dslash_exterior()const {
  		return dslash_exterior;
  	}

  	bool get_enable_dslash_copy()const {
  		return dslash_copy;
  	}

  	bool get_enable_dslash_comms()const {
  		return dslash_comms;
  	}

  	bool get_allow_jit() const {
  		return allow_jit;
  	}

  	bool get_enable_numa() const{
  		return enable_numa;
  	}

  	char* get_reorder_location() const{
  		return reorder_location;
  	}

  	bool get_device_reset() const{
  		return device_reset;
  	}

  protected:
  	// may add setters here

  };

} // namespace quda