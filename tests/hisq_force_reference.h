#ifndef _HISQ_FORCE_REFERENCE_H
#define _HISQ_FORCE_REFERENCE_H
#ifdef __cplusplus
extern "C"{
#endif    
    void halfwilson_hisq_force_reference(float eps, float weight1, void* act_path_coeff, void* temp_x, void* sitelink, void* mom);

    void color_matrix_hisq_force_reference(float eps, float weight,
                          void* act_path_coeff, void* temp_xx,
                          void* sitelink, void* mom);


    void computeHisqOuterProduct(void* src, void* dst);		

    void computeLinkOrderedOuterProduct(void *src, void* dest);



#ifdef __cplusplus
}
#endif

#endif

