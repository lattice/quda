#ifndef __GAUGE_FORCE_REFERENCE__
#define __GAUGE_FORCE_REFERENCE__
#ifdef __cplusplus
extern "C"{
#endif
  
  void gauge_force_reference(void* refMom, double eb3, void** sitelink, void** sitelink_2d, QudaPrecision prec, 
			     int ***path_dir, int* length, void* loop_coeff, int num_paths);
  
#ifdef __cplusplus
}
#endif

#endif

