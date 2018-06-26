#pragma once

#include <cstdlib>
#include <quda_internal.h>

namespace quda {
  // forward declaration
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

		int enable_p2p;
		int enable_gdr;


    QudaEnv() {
    	enable_p2p = getenvInt("QUDA_ENABLE_P2P",3);
    	enable_gdr = getenvInt("QUDA_ENABLE_GDR",0);


    }                    // Constructor? (the {} brackets) are needed here.


    
    // comm_common
		// char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");
		// char *enable_gdr_env = getenv("QUDA_ENABLE_GDR");
		// char *blacklist_env = getenv("QUDA_ENABLE_GDR_BLACKLIST");
		
		// // comm_mpi
		// char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
		// // comm_qmp
		// char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
		// //dslash_coarse.cu
		// static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_COARSE_POLICY");

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

    int get_enable_gdr() const{
    	return enable_gdr;
    }


   protected:
//    void setx(int _x) {
//      x = _x;
//    }


  };

} // namespace quda