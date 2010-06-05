#ifndef __FERMION_FORCE_REFERENCE__
#define __FERMION_FORCE_REFERENCE__
#ifdef __cplusplus
extern "C"{
#endif    
    void fermion_force_reference(float eps, float weight1, float weight2, 
				 void* act_path_coeff, void* temp_x, void* sitelink, void* mom);
	
#ifdef __cplusplus
}
#endif

#endif

