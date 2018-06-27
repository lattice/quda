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
		bool dslash_policy;
		bool dslash_pack;
		bool dslash_interior;
		bool dslash_exterior;
		bool dslash_copy;
		bool dslash_comms;


    QudaEnv() {

    	//
    	enable_p2p = getenvInt("QUDA_ENABLE_P2P",3);
    	enable_gdr = (getenvInt("QUDA_ENABLE_GDR",0)==1);
    	blacklist_env = getenvChar("QUDA_ENABLE_GDR_BLACKLIST");
    	enable_mps = (getenvInt("QUDA_ENABLE_MPS",0)==1);

    	dslash_policy = getenvChar("QUDA_ENABLE_DSLASH_POLICY");
			dslash_pack = getenvBool("QUDA_ENABLE_DSLASH_PACK");
			dslash_interior = getenvBool("QUDA_ENABLE_DSLASH_INTERIOR")
			dslash_exterior = getenvBool("QUDA_ENABLE_DSLASH_EXTERIOR")
			dslash_copy = getenvBool("QUDA_ENABLE_DSLASH_COPY")
			dslash_comms = getenvBool("QUDA_ENABLE_DSLASH_COMMS")
    }                    // Constructor? (the {} brackets) are needed here.


    


		// //dslash_policy.cuh
		// static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_POLICY");
		// static char *dslash_pack_env = getenv("QUDA_ENABLE_DSLASH_PACK");
		// static char *dslash_interior_env = getenv("QUDA_ENABLE_DSLASH_INTERIOR");
		// static char *dslash_exterior_env = getenv("QUDA_ENABLE_DSLASH_EXTERIOR");
		// static char *dslash_copy_env = getenv("QUDA_ENABLE_DSLASH_COPY");
		// static char *dslash_comms_env = getenv("QUDA_ENABLE_DSLASH_COMMS");

		// //interface_quda.cpp:  
		// char *cni_str = getenv("CUDA_NIC_INTEROP");
		// char *allow_jit_env = getenv("QUDA_ALLOW_JIT");
		// char *enable_numa_env = getenv("QUDA_ENABLE_NUMA");
		// char *reorder_str = getenv("QUDA_REORDER_LOCATION");
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
    QudaEnv(QudaEnv const &)               = delete;
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
    	return blacklist_env;
    }


   protected:
//    void setx(int _x) {
//      x = _x;
//    }


  };

} // namespace quda