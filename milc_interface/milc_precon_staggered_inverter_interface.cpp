#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <test_util.h>
#include "../tests/blas_reference.h" // What do I need here?
#include "../tests/staggered_dslash_reference.h" // What do I need here?

#include <quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "external_headers/quda_milc_interface.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/milc_utilities.h"
#include "include/milc_inverter_utilities.h"

using namespace quda;

namespace milc_interface {
  namespace domain_decomposition {


    void* getQuarkPointer(void* field, QudaPrecision precision, QudaParity parity, int volume)
    {
      const int quark_offset = getColorVectorOffset(parity, false, volume);
      void* quark_pointer;
      if(precision == QUDA_SINGLE_PRECISION){
        quark_pointer = (float*)field + quark_offset;
      }else if(precision == QUDA_DOUBLE_PRECISION){
        quark_pointer = (double*)field + quark_offset;
      }else{
        errorQuda("Unrecognised precision");
      }
      return quark_pointer;
    }


    void assignExtendedMILCGaugeField(const int dim[4],
        const int domain_overlap[4],
        QudaPrecision precision,
        const void* const src,
        void* const dst)
    {
      const int matrix_size = 18*getRealSize(precision);
      const int site_size = 4*matrix_size;  
      const int volume = getVolume(dim);
      const int half_volume = volume/2;

      int extended_dim[4];
      for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir] + 2*domain_overlap[dir];
      const int extended_volume = getVolume(extended_dim);

      const int half_dim0 = extended_dim[0]/2;
      const int half_extended_volume = extended_volume/2;

      for(int i=0; i<extended_volume; ++i){
        int site_id = i;
        int odd_bit = 0;

        if(i >= half_extended_volume){
          site_id -= half_extended_volume;
          odd_bit = 1;
        }
        // x1h = site_id % half_dim0;
        // x2  = (site_id/half_dim0) % extended_dim[1];
        // x3  = (site_id/(extended_dim[1]*half_dim0)) % extended_dim[2];
        // x4  =  site_id/(extended_dim[2]*extended_dim[1]*half_dim0));
        int za  = site_id/half_dim0;
        int x1h = site_id - za*half_dim0;      
        int zb  = za/extended_dim[1];
        int x2  = za - zb*extended_dim[1];
        int x4  = zb/extended_dim[2];
        int x3  = zb - x4*extended_dim[2];
        int x1odd = (x2 + x3 + x4 + odd_bit) & 1;
        int x1  = 2*x1h + x1odd;


        x1 = (x1 - domain_overlap[0] + dim[0]) % dim[0];
        x2 = (x2 - domain_overlap[1] + dim[1]) % dim[1];
        x3 = (x3 - domain_overlap[2] + dim[2]) % dim[2];
        x4 = (x4 - domain_overlap[3] + dim[3]) % dim[3];

        int little_index = (x4*dim[2]*dim[1]*dim[0] + x3*dim[1]*dim[0] + x2*dim[0] + x1) >> 1;
        if(odd_bit){ little_index += half_volume; }

        memcpy((char*)dst + i*site_size, (char*)src + little_index*site_size, site_size); 
      } // loop over extended volume
      return;
    } // assignExtendedMILCGaugeField



    class FieldHandle{
      private:
        void* field_ptr;
      public:
        FieldHandle(void* fp) : field_ptr(fp) {}
        ~FieldHandle(){ free(field_ptr); } 
        void* get(){ return field_ptr; }
    }; // RAII

  } // namespace domain_decomposition
} // namespace milc_interface



void qudaDDInvert(int external_precision, 
    int quda_precision,
    double mass,
    QudaInvertArgs_t inv_args,
    double target_residual,
    double target_fermilab_residual,
    const int* const domain_overlap,
    const void* const fatlink,
    const void* const longlink,
    const double tadpole,
    void* source,
    void* solution,
    double* const final_residual,
    double* const final_fermilab_residual,
    int* num_iters){

  using namespace milc_interface;
  using namespace milc_interface::domain_decomposition;

  // check to see if the domain overlaps are even
  for(int dir=0; dir<4; ++dir){
    if((domain_overlap[dir] % 2) != 0){
      errorQuda("Odd overlap dimensions not yet supported");
      return;
    }
  }


  printfQuda("overlap dimensions = %d %d %d %d\n", domain_overlap[0], domain_overlap[1], domain_overlap[2], domain_overlap[3]);



  Layout layout;
  const int* local_dim = layout.getLocalDim();
  int* extended_dim;
  for(int dir=0; dir<4; ++dir){ extended_dim[dir] = local_dim[dir] + 2*domain_overlap[dir]; }
  const int extended_volume = getVolume(extended_dim);

  printfQuda("extended_dim = %d %d %d %d\n", extended_dim[0], extended_dim[1], extended_dim[2], extended_dim[3]);

  const QudaVerbosity verbosity = QUDA_VERBOSE;	

  const QudaPrecision precision = (external_precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  const int link_size = 18*getRealSize(precision);
  const QudaPrecision device_sloppy_precision = device_precision; // HACK!!
  const QudaPrecision device_precon_precision = device_sloppy_precision;

  { // load the gauge fields
    QudaGaugeParam gaugeParam;
    setGaugeParams(local_dim, precision, device_precision, device_sloppy_precision, device_precon_precision, tadpole, &gaugeParam);
    // load the precise and sloppy gauge fields onto the device
    const int fat_pad = getFatLinkPadding(local_dim);
    const int long_pad = 3*fat_pad;

    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam);
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);

    FieldHandle extended_fatlink(malloc(extended_volume*4*link_size)); // RAII => free is called upon destruction of 
    //         extended_fatlink
    // Extend the fat gauge field
    assignExtendedMILCGaugeField(local_dim, domain_overlap, precision, fatlink, extended_fatlink.get());
    exchange_cpu_sitelink_ex(const_cast<int*>(local_dim), const_cast<int*>(domain_overlap), (void**)(extended_fatlink.get()), QUDA_MILC_GAUGE_ORDER, precision, 1); 

    setGaugeParams(extended_dim, precision, device_precision, device_sloppy_precision, device_precon_precision, tadpole, &gaugeParam);
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = getFatLinkPadding(extended_dim);
    loadPreconGaugeQuda(extended_fatlink.get(), &gaugeParam);
  }

  // set up the inverter
  {
    QudaInvertParam invertParam;
    setInvertParams(local_dim, precision, device_precision, device_sloppy_precision, device_precon_precision, 
        mass, target_residual, inv_args.max_iter, 1e-1, inv_args.evenodd,
        verbosity, QUDA_PCG_INVERTER, &invertParam); // preconditioned inverter

    ColorSpinorParam csParam;
    setColorSpinorParams(local_dim, precision, &csParam);

    // Set the pointers to the source and solution parity fields
    const int volume = getVolume(local_dim);
    void* src_pointer = getQuarkPointer(source, precision, inv_args.evenodd, volume);
    void* sln_pointer = getQuarkPointer(solution, precision, inv_args.evenodd, volume);

    invertQuda(sln_pointer, src_pointer, &invertParam);

    *num_iters = invertParam.iter;
    *final_residual = invertParam.true_res;
    *final_fermilab_residual = invertParam.true_res_hq; 
  }
  freeGaugeQuda(); // free up the gauge-field objects allocated in loadGaugeQuda
  return;
} // qudaDDInvert

