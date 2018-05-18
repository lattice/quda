#ifndef __FERMION_FORCE_REFERENCE__
#define __FERMION_FORCE_REFERENCE__

    void fermion_force_reference(float eps, float weight1, float weight2, 
				 void* act_path_coeff, void* temp_x, void* sitelink, void* mom);

    void fermion_force_reference(double eps, double weight1, double weight2,
				 void* act_path_coeff, void* temp_x, void* sitelink, void* mom);

	

#endif

