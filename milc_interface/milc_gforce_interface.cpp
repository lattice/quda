#include <cstdlib>
#include <cstdio>
#include <cstring> // needed for memcpy

#include <quda.h>         // contains initQuda
#include <dslash_quda.h>  // contains initDslashConstants
#include <gauge_field.h>  
#include <gauge_force_quda.h> 
//#include <vector>
#include <gauge_force_reference.h>
#include "external_headers/quda_milc_interface.h"
// The Symanzik-improved gauge action 
// used by the MILC collaboration consists of 
// plaquettes, 6-link rectangular loops, and 
// "chair" loops.
// The gauge action involves three couplings - 
// one for each type of loop.
//
// Each momentum component receives 48 contributions:
// 6 from the plaquettes
// 18 from the 6-link rectangles
// 24 from the "chair" loops.
// 

#include "include/milc_timer.h"
#include "include/milc_utilities.h"

extern int V;
extern int Vh;
extern int Z[4];


namespace milc_interface {

void
print_su3_matrix(void *_a, int prec)
{
  int i, j;
  for(i=0;i < 3; i++){
    for(j=0;j < 3;j++){
      if(prec == QUDA_DOUBLE_PRECISION){
	double* a = (double*)_a;
	printf("(%f %f)\t", a[(i*3+j)*2], a[(i*3+j)*2+1]);
      }else{
	float* a = (float*)_a;
	printf("(%f %f)\t", a[(i*3+j)*2], a[(i*3+j)*2+1]);
	
      }
    }
    printf("\n");
  }
  
}

void
print_mom(void *_a, int prec)
{
  
  int i, j;
  for(i=0;i < 5; i++){
    if(prec == QUDA_DOUBLE_PRECISION){
      double* a = (double*)_a;
      printf("(%f %f)\t", a[i*2], a[i*2+1]);      
    }else{
      float* a = (float*)_a;
      printf("(%f %f)\t", a[i*2], a[i*2+1]);      
    }
  }
  printf("\n");

}


static 
void setDims(const int X[4]) {
    V = 1;
      for (int d=0; d< 4; d++) {
            V *= X[d];
                Z[d] = X[d];
                  }
        Vh = V/2;
}

template<class Real>
void setLoopCoeffs(const double milc_loop_coeff[3],
                         Real loop_coeff[48])
{
  // 6 plaquette terms
  for(int i=0; i<6; ++i){
    loop_coeff[i] = milc_loop_coeff[0];      
  }
  
  // 18 rectangle terms
  for(int i=0; i<18; ++i){
    loop_coeff[i+6] = milc_loop_coeff[1];
  }

  for(int i=0; i<24; ++i){
    loop_coeff[i+24] = milc_loop_coeff[2];
  }

  return;
}


// is there a way to get rid of the length parameter

static void
setGaugeParams(QudaGaugeParam* gaugeParam,
               const int dim[4],
               QudaPrecision cpu_prec,
               QudaPrecision cuda_prec,
               QudaReconstructType link_recon = QUDA_RECONSTRUCT_12
               )
{
  for(int dir=0; dir<4; ++dir){
    gaugeParam->X[dir] = dim[dir];
  }
  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = cuda_prec;
  gaugeParam->reconstruct = link_recon;
  gaugeParam->type = QUDA_SU3_LINKS;
#ifdef MULTI_GPU // use the QDP ordering scheme for the internal host fields
  gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER;
#else
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
#endif
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = 1.0;
  gaugeParam->gauge_fix     = QUDA_GAUGE_FIXED_NO;
  return;
}

} // namespace milc_interface

void qudaGaugeForce( int precision,
                     int num_loop_types,
                     double milc_loop_coeff[3],
                     double eb3,
                     void* milc_sitelink,
                     void* milc_momentum
                   )
{
  using namespace milc_interface;

  Timer timer("qudaGaugeForce");
#ifndef TIME_INTERFACE
  timer.mute();
#endif

  Layout layout; // example of the Monostate pattern

  const int* dim = layout.getLocalDim();
  setDims(dim);


  QudaGaugeParam qudaGaugeParam;

  QudaPrecision cpu_precision, cuda_precision;
  if(precision==1){
    cpu_precision = cuda_precision = QUDA_SINGLE_PRECISION;
  }else if(precision==2){
    cpu_precision = cuda_precision = QUDA_DOUBLE_PRECISION;
  }else{
    errorQuda("qudaGaugeForce: unrecognised precision\n");
  }
  setGaugeParams(&qudaGaugeParam, dim, cpu_precision, cuda_precision);


  double d_loop_coeff[48];
  float  f_loop_coeff[48];
  setLoopCoeffs(milc_loop_coeff, d_loop_coeff);
  for(int i=0; i<48; ++i) f_loop_coeff[i] = d_loop_coeff[i];
  int length[48];
  for(int i=0; i<6; ++i) length[i] = 3;
  for(int i=6; i<48; ++i) length[i] = 5;

  int path_dir_x[][5] =
    {
      {1,	7,	6	},
      {6,	7,	1	},
      {2,	7,	5	},
      {5,	7,	2	},
      {3,	7,	4	},
      {4,	7,	3	},
      {0,	1,	7,	7,	6	},
      {1,	7,	7,	6,	0	},
      {6,	7,	7,	1,	0	},
      {0,	6,	7,	7,	1	},
      {0,	2,	7,	7,	5	},
      {2,	7,	7,	5,	0	},
      {5,	7,	7,	2,	0	},
      {0,	5,	7,	7,	2	},
      {0,	3,	7,	7,	4	},
      {3,	7,	7,	4,	0	},
      {4,	7,	7,	3,	0	},
      {0,	4,	7,	7,	3	},
      {6,	6,	7,	1,	1	},
      {1,	1,	7,	6,	6	},
      {5,	5,	7,	2,	2	},
      {2,	2,	7,	5,	5	},
      {4,	4,	7,	3,	3	},
      {3,	3,	7,	4,	4	},
      {1,	2,	7,	6,	5	},
      {5,	6,	7,	2,	1	},
      {1,	5,	7,	6,	2	},
      {2,	6,	7,	5,	1	},
      {6,	2,	7,	1,	5	},
      {5,	1,	7,	2,	6	},
      {6,	5,	7,	1,	2	},
      {2,	1,	7,	5,	6	},
      {1,	3,	7,	6,	4	},
      {4,	6,	7,	3,	1	},
      {1,	4,	7,	6,	3	},
      {3,	6,	7,	4,	1	},
      {6,	3,	7,	1,	4	},
      {4,	1,	7,	3,	6	},
      {6,	4,	7,	1,	3	},
      {3,	1,	7,	4,	6	},
      {2,	3,	7,	5,	4	},
      {4,	5,	7,	3,	2	},
      {2,	4,	7,	5,	3	},
      {3,	5,	7,	4,	2	},
      {5,	3,	7,	2,	4	},
      {4,	2,	7,	3,	5	},
      {5,	4,	7,	2,	3	},
      {3,	2,	7,	4,	5	}
    };
  
  int path_dir_y[][5] = {
    {2,	6,	5	},
    {5,	6,	2	},
    {3,	6,	4	},
    {4,	6,	3	},
    {0,	6,	7	},
    {7,	6,	0	},
    {1,	2,	6,	6,	5	},
    {2,	6,	6,	5,	1	},
    {5,	6,	6,	2,	1	},
    {1,	5,	6,	6,	2	},
    {1,	3,	6,	6,	4	},
    {3,	6,	6,	4,	1	},
    {4,	6,	6,	3,	1	},
    {1,	4,	6,	6,	3	},
    {1,	0,	6,	6,	7	},
    {0,	6,	6,	7,	1	},
    {7,	6,	6,	0,	1	},
    {1,	7,	6,	6,	0	},
    {5,	5,	6,	2,	2	},
    {2,	2,	6,	5,	5	},
    {4,	4,	6,	3,	3	},
    {3,	3,	6,	4,	4	},
    {7,	7,	6,	0,	0	},
    {0,	0,	6,	7,	7	},
    {2,	3,	6,	5,	4	},
    {4,	5,	6,	3,	2	},
    {2,	4,	6,	5,	3	},
    {3,	5,	6,	4,	2	},
    {5,	3,	6,	2,	4	},
    {4,	2,	6,	3,	5	},
    {5,	4,	6,	2,	3	},
    {3,	2,	6,	4,	5	},
    {2,	0,	6,	5,	7	},
    {7,	5,	6,	0,	2	},
    {2,	7,	6,	5,	0	},
    {0,	5,	6,	7,	2	},
    {5,	0,	6,	2,	7	},
    {7,	2,	6,	0,	5	},
    {5,	7,	6,	2,	0	},
    {0,	2,	6,	7,	5	},
    {3,	0,	6,	4,	7	},
    {7,	4,	6,	0,	3	},
    {3,	7,	6,	4,	0	},
    {0,	4,	6,	7,	3	},
    {4,	0,	6,	3,	7	},
    {7,	3,	6,	0,	4	},
    {4,	7,	6,	3,	0	},
    {0,	3,	6,	7,	4	}
  };
  
  
  int path_dir_z[][5] = {	
    {3,	5,	4	},
    {4,	5,	3	},
    {0,	5,	7	},
    {7,	5,	0	},
    {1,	5,	6	},
    {6,	5,	1	},
    {2,	3,	5,	5,	4	},
    {3,	5,	5,	4,	2	},
    {4,	5,	5,	3,	2	},
    {2,	4,	5,	5,	3	},
    {2,	0,	5,	5,	7	},
    {0,	5,	5,	7,	2	},
    {7,	5,	5,	0,	2	},
    {2,	7,	5,	5,	0	},
    {2,	1,	5,	5,	6	},
    {1,	5,	5,	6,	2	},
    {6,	5,	5,	1,	2	},
    {2,	6,	5,	5,	1	},
    {4,	4,	5,	3,	3	},
    {3,	3,	5,	4,	4	},
    {7,	7,	5,	0,	0	},
    {0,	0,	5,	7,	7	},
    {6,	6,	5,	1,	1	},
    {1,	1,	5,	6,	6	},
    {3,	0,	5,	4,	7	},
    {7,	4,	5,	0,	3	},
    {3,	7,	5,	4,	0	},
    {0,	4,	5,	7,	3	},
    {4,	0,	5,	3,	7	},
    {7,	3,	5,	0,	4	},
    {4,	7,	5,	3,	0	},
    {0,	3,	5,	7,	4	},
    {3,	1,	5,	4,	6	},
    {6,	4,	5,	1,	3	},
    {3,	6,	5,	4,	1	},
    {1,	4,	5,	6,	3	},
    {4,	1,	5,	3,	6	},
    {6,	3,	5,	1,	4	},
    {4,	6,	5,	3,	1	},
    {1,	3,	5,	6,	4	},
    {0,	1,	5,	7,	6	},
    {6,	7,	5,	1,	0	},
    {0,	6,	5,	7,	1	},
    {1,	7,	5,	6,	0	},
    {7,	1,	5,	0,	6	},
    {6,	0,	5,	1,	7	},
    {7,	6,	5,	0,	1	},
    {1,	0,	5,	6,	7	}
  };



  int path_dir_t[][5] = {
    {0,	4,	7	},
    {7,	4,	0	},
    {1,	4,	6	},
    {6,	4,	1	},
    {2,	4,	5	},
    {5,	4,	2	},
    {3,	0,	4,	4,	7	},
    {0,	4,	4,	7,	3	},
    {7,	4,	4,	0,	3	},
    {3,	7,	4,	4,	0	},
    {3,	1,	4,	4,	6	},
    {1,	4,	4,	6,	3	},
    {6,	4,	4,	1,	3	},
    {3,	6,	4,	4,	1	},
    {3,	2,	4,	4,	5	},
    {2,	4,	4,	5,	3	},
    {5,	4,	4,	2,	3	},
    {3,	5,	4,	4,	2	},
    {7,	7,	4,	0,	0	},
    {0,	0,	4,	7,	7	},
    {6,	6,	4,	1,	1	},
    {1,	1,	4,	6,	6	},
    {5,	5,	4,	2,	2	},
    {2,	2,	4,	5,	5	},
    {0,	1,	4,	7,	6	},
    {6,	7,	4,	1,	0	},
    {0,	6,	4,	7,	1	},
    {1,	7,	4,	6,	0	},
    {7,	1,	4,	0,	6	},
    {6,	0,	4,	1,	7	},
    {7,	6,	4,	0,	1	},
    {1,	0,	4,	6,	7	},
    {0,	2,	4,	7,	5	},
    {5,	7,	4,	2,	0	},
    {0,	5,	4,	7,	2	},
    {2,	7,	4,	5,	0	},
    {7,	2,	4,	0,	5	},
    {5,	0,	4,	2,	7	},
    {7,	5,	4,	0,	2	},
    {2,	0,	4,	5,	7	},
    {1,	2,	4,	6,	5	},
    {5,	6,	4,	2,	1	},
    {1,	5,	4,	6,	2	},
    {2,	6,	4,	5,	1	},
    {6,	2,	4,	1,	5	},
    {5,	1,	4,	2,	6	},
    {6,	5,	4,	1,	2	},
    {2,	1,	4,	5,	6	}
  };
  
  const int max_length = 6;
  
  int num_paths =48;
  int** input_path_buf[4];
  for(int dir =0; dir < 4; dir++){
    input_path_buf[dir] = (int**)malloc(num_paths*sizeof(int*));
    if (input_path_buf[dir] == NULL){
      printf("ERORR: malloc failed for input path\n");
      exit(1);
    }
    
    for(int i=0;i < num_paths;i++){
      input_path_buf[dir][i] = (int*)malloc(length[i]*sizeof(int));
      if (input_path_buf[dir][i] == NULL){
	printf("ERROR: malloc failed for input_path_buf[dir][%d]\n", i);
	exit(1);
      }
      if(dir == 0) memcpy(input_path_buf[dir][i], path_dir_x[i], length[i]*sizeof(int));
      else if(dir ==1) memcpy(input_path_buf[dir][i], path_dir_y[i], length[i]*sizeof(int));
      else if(dir ==2) memcpy(input_path_buf[dir][i], path_dir_z[i], length[i]*sizeof(int));
      else if(dir ==3) memcpy(input_path_buf[dir][i], path_dir_t[i], length[i]*sizeof(int));
    }
  }
  
  
  void* loop_coeff_ptr; 
  if(cpu_precision == QUDA_SINGLE_PRECISION){
    loop_coeff_ptr = (void*)f_loop_coeff;
  }else{
    loop_coeff_ptr = (void*)d_loop_coeff;
  }
  
  double timeinfo[3];
  memset(milc_momentum, 0, 4*V*10*qudaGaugeParam.cpu_prec);

#ifdef MULTI_GPU
  void* extended_link[4]; // extended field to hold the input sitelinks
  int extended_dim[4];
  for(int dir=0; dir<4; ++dir){ extended_dim[dir] = dim[dir]+4; }
  const int extended_volume = getVolume(extended_dim);

  for(int dir=0; dir<4; ++dir){
    allocateColorField(extended_volume,cpu_precision,false,extended_link[dir]);
  } 
  assignExtendedQDPGaugeField(dim, cpu_precision, milc_sitelink, extended_link);
  int R[4] = {2, 2, 2, 2};
  exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, extended_link,
             QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0);

  computeGaugeForceQuda(milc_momentum, extended_link,  input_path_buf, length,
			loop_coeff_ptr, num_paths, max_length, eb3,
			&qudaGaugeParam, timeinfo);

  for(int dir=0; dir<4; ++dir){ free(extended_link[dir]); }
#else
  computeGaugeForceQuda(milc_momentum, milc_sitelink,  input_path_buf, length,
			loop_coeff_ptr, num_paths, max_length, eb3,
			&qudaGaugeParam, timeinfo);
#endif  
  for(int dir = 0; dir < 4; dir++){
    for(int i=0;i < num_paths; i++){
      free(input_path_buf[dir][i]);
    }
    free(input_path_buf[dir]);
 }
  
  
  return;
}
