#include "include/milc_utilities.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <cuda_runtime.h> // Needed for cudaMallocHost
#include <string.h>
#include <omp.h>

namespace milc_interface {

  QudaVerbosity PersistentData::verbosity = QUDA_SILENT;


  void Layout::setLocalDim(const int X[4])
  {
    for(int dir=0; dir<4; ++dir) local_dim[dir] = X[dir];
  }

  void Layout::setGridDim(const int X[4])
  {
    for(int dir=0; dir<4; ++dir) grid_dim[dir] = X[dir];
  }



  int Layout::local_dim[4] = {1,1,1,1};
  int Layout::grid_dim[4] = {1,1,1,1};




  void GridInfo::setDim(const int d[4]){
    volume = 1;
    for(int dir=0; dir<4; ++dir){
      dim[dir] = d[dir];
      volume *= dim[dir];
    }
    return;
  }


  //int (&GridInfo::getDim() const)[4]{
  //  return dim;
  //}

  int GridInfo::getVolume() const {
    return volume;
  }

  int GridInfo::getSliceVolume(int i) const{
    return volume/dim[i];
  }


  int GridInfo::getArea(int i, int j) const {
    assert(i != j);
    return dim[i]*dim[j];
  }


  int GridInfo::getMaxArea() const {

    int max_area = 1;
    for(int i=0; i<4; ++i){
      for(int j=i+1; j<4; ++j){
        int area = dim[i]*dim[j];
        if(area > max_area) max_area = area;
      }
    }
    return max_area;
  }



  int getVolume(const int dim[4]){
    int volume = 1;
    for(int dir=0; dir<4; ++dir){ 
      assert(dim[dir] > 0);
      volume *= dim[dir]; 
    }
    return volume;
  }

  int getRealSize(QudaPrecision prec){
    return (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  }


  // Functions used to load the gauge fields from Milc
  template<class MilcReal, class QudaReal>
    static void
    reorderMilcLinks(const MilcReal* const src, int volume, QudaReal* const dst[4])
    {
      for(int i=0; i<volume; ++i){
        for(int dir=0; dir<4; ++dir){
          for(int j=0; j<18; ++j){
            dst[dir][i*18+j] = src[(i*4+dir)*18+j];
          }  
        }
      }
      return;
    }


  template<class MilcReal, class QudaReal>
    static void 
    reorderMilcLinks(const MilcReal* const src, bool exchange_parity, int volume, QudaReal* const dst[4])
    {
      const size_t offset = (exchange_parity) ? volume/2 : 0;

      for(int i=0; i<volume/2; ++i){
        for(int dir=0; dir<4; ++dir){
          for(int j=0; j<18; ++j){
            dst[dir][i*18+j] = src[((i+offset)*4+dir)*18+j];
          }
        }
      }

      for(int i=volume/2; i<volume; ++i){
        for(int dir=0; dir<4; ++dir){
          for(int j=0; j<18; ++j){
            dst[dir][i*18+j] = src[((i-offset)*4+dir)*18+j];
          }
        }
      }
      return;
    }




  MilcFieldLoader::MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam)
    : milc_precision(milc_prec), quda_precision(gaugeParam.cpu_prec), volume(1), exchange_parity(false)
  {
    for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
  }

  MilcFieldLoader::MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam, bool exchange)
    : milc_precision(milc_prec), quda_precision(gaugeParam.cpu_prec), volume(1), exchange_parity(exchange)
  {
    for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
  }

#include <util_quda.h>
  void MilcFieldLoader::loadGaugeField(const void* const milc_field, void* quda_field[4]) const
  {
    if(milc_precision == quda_precision && milc_precision == QUDA_SINGLE_PRECISION){
      reorderMilcLinks((float*)milc_field, exchange_parity, volume, (float**)quda_field); 
    }else if(milc_precision == quda_precision && milc_precision == QUDA_DOUBLE_PRECISION){
      reorderMilcLinks((double*)milc_field, exchange_parity, volume, (double**)quda_field); 
    }else if(milc_precision == QUDA_SINGLE_PRECISION && quda_precision == QUDA_DOUBLE_PRECISION){
      reorderMilcLinks((float*)milc_field, exchange_parity, volume, (double**)quda_field);
    }else if(milc_precision == QUDA_DOUBLE_PRECISION && quda_precision == QUDA_SINGLE_PRECISION){
      reorderMilcLinks((double*)milc_field, exchange_parity, volume, (float**)quda_field);
    }else{
      errorQuda("Invalid precision\n");
    }
    return;
  }



  //********************************************************
  // 
  // Used in the multi-gpu fattening, fermion-force, and 
  // gauge-force code
  // 
  //********************************************************


  void allocateColorField(int volume, QudaPrecision prec, bool usePinnedMemory, void*& field)
  {
    const int realSize = getRealSize(prec);
    int siteSize = 18;
    if(usePinnedMemory){
      cudaMallocHost((void**)&field, volume*siteSize*realSize);
    }else{
      field = (void*)malloc(volume*siteSize*realSize);
    }
    if(field == NULL){
      errorQuda("ERROR: allocateColorField failed\n");
    }
    return;
  }




  void assignExtendedQDPGaugeField(const int dim[4], QudaPrecision precision, const void* const src,  void** const dst)
  {

    const int matrix_size = 18*getRealSize(precision);
    const int volume = getVolume(dim);

    int extended_dim[4];
    for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir]+4;

    const int extended_volume = getVolume(extended_dim);


    const int half_dim0 = extended_dim[0]/2;
    const int half_extended_volume = extended_volume/2;
    const int half_volume = volume/2;
    for(int t=0; t<extended_dim[3]; ++t){
      int x4 = (t - 2 + dim[3]) % dim[3];
      int off4 = x4*dim[2];
   //   int ex_off4 = t*extended_dim[2];
      for(int z=0; z<extended_dim[2]; ++z){
        int x3 = (z - 2 + dim[2]) % dim[2];
        int off3 = (off4 + x3)*dim[1];
   //     int ex_off3 = (ex_off4 + z)*extended_dim[1];
        for(int y=0; y<extended_dim[1]; ++y){
          int x2 = (y - 2 + dim[1]) % dim[1];
          int off2 = (off3 + x2)*dim[0];
    //      int ex_off2 = (ex_off3 + y)*extended_dim[0];
          for(int x=0; x<extended_dim[0]; ++x){
            int x1 = (x - 2 + dim[0]) % dim[0];
            int offset = (off2 + x1)>>1;
    //        int ex_offset = (ex_off2 + x)>>1;

            const int odd_bit = (x + y + z + t) & 1; 

            int in_index = offset + odd_bit*half_volume;
      //      int out_index = ex_offset + odd_bit*half_extended_volume;
       
           int out_index = ((((t*extended_dim[2] + z)*extended_dim[1] + y)*extended_dim[0] + x) >> 1) + odd_bit*half_extended_volume;

            for(int dir=0; dir<4; ++dir){
              char* dst_ptr = (char*)dst[dir];
              memcpy(dst_ptr + out_index*matrix_size, (char*)src + (in_index*4 + dir)*matrix_size, matrix_size);
            } // end loop over directions
          } // x
        } // y
      } // z
    } // t

    /*
       for(int i=0; i<extended_volume; ++i){
       int site_id = i;
       int odd_bit = 0;

       if(i >= half_extended_volume){
       site_id -= half_extended_volume;
       odd_bit  = 1;
       }

       int za     = site_id/half_dim0;
       int x1h    = site_id - za*half_dim0;
       int zb     = za/extended_dim[1];
       int x2     = za - zb*extended_dim[1];
       int x4     = zb/extended_dim[2];
       int x3     = zb - x4*extended_dim[2];
       int x1odd  = (x2 + x3 + x4 + odd_bit) & 1;
       int x1     = 2*x1h + x1odd;

       x1 = (x1 - 2 + dim[0]) % dim[0];
       x2 = (x2 - 2 + dim[1]) % dim[1];
       x3 = (x3 - 2 + dim[2]) % dim[2];
       x4 = (x4 - 2 + dim[3]) % dim[3];

       int full_index = (x4*dim[2]*dim[1]*dim[0] + x3*dim[1]*dim[0] + x2*dim[0] + x1)>>1;
       if(odd_bit){ full_index += volume/2; }

       for(int dir=0; dir<4; ++dir){
       char* dst_ptr = (char*)dst[dir];
       memcpy(dst_ptr + i*matrix_size, (char*)src + (full_index*4 + dir)*matrix_size, matrix_size);
       } // end loop over directions
       } // loop over the extended volume
       */
    return;
  } // assignExtendedQDPGaugeField


  // update boundaries (pretty inefficient).
  void updateExtendedQDPBorders(const int dim[4], QudaPrecision precision, void** const qdp_field)
  {

    const int matrix_size = 18*getRealSize(precision);

    int extended_dim[4];
    for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir]+4;

    const int extended_volume = getVolume(extended_dim);


    const int half_dim0 = extended_dim[0]/2;
    const int half_extended_volume = extended_volume/2;

    for(int i=0; i<extended_volume; ++i){
      int site_id = i;
      int odd_bit = 0;

      if(i >= half_extended_volume){
        site_id -= half_extended_volume;
        odd_bit  = 1;
      }

      int za     = site_id/half_dim0;
      int x1h    = site_id - za*half_dim0;
      int zb     = za/extended_dim[1];
      int x2     = za - zb*extended_dim[1];
      int x4     = zb/extended_dim[2];
      int x3     = zb - x4*extended_dim[2];
      int x1odd  = (x2 + x3 + x4 + odd_bit) & 1;
      int x1     = 2*x1h + x1odd;


      int y1, y2, y3, y4;
      int y1h = x1h;
      y1 = x1;
      y2 = x2; 
      y3 = x3; 
      y4 = x4;
      bool boundary = false;
      if(x1 < 2 || x1 > 1+dim[0]  ){
        y1 = ((2 + ( (x1 - 2 + dim[0]) % dim[0])));
        y1h = y1 >> 1;
        boundary = true;
      } 

      if(x2 < 2 || x2 > 1+dim[1] ){
        y2 = 2 + ( (x2 - 2 + dim[1]) % dim[1]);
        boundary = true;
      }

      if(x3 < 2 || x3 > 1+dim[2] ){
        y3 = 2 + ( (x3 - 2 + dim[2]) % dim[2]);
        boundary = true;
      }

      if(x4 < 2 || x4 > 1+dim[3] ){
        y4 = 2 + ( (x4 - 2 + dim[3]) % dim[3]);
        boundary = true;
      }


      if(boundary){
        int interior_index = (  y4*extended_dim[2]*extended_dim[1]*extended_dim[0]/2 
            + y3*extended_dim[1]*extended_dim[0]/2 
            + y2*extended_dim[0]/2
            + y1h 
            + odd_bit*half_extended_volume );



        for(int dir=0; dir<4; ++dir){
          char* field_ptr = (char*)qdp_field[dir];
          memcpy(field_ptr + i*matrix_size, field_ptr + interior_index*matrix_size, matrix_size);
        }
      } // if(boundary)
    } // loop over the extended volume

    // see if it makes a difference
    return;
  } // updateExtendedQDPBorders

} // namespace milc_interface
