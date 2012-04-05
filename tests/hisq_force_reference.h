#ifndef _HISQ_FORCE_REFERENCE_H
#define _HISQ_FORCE_REFERENCE_H

#include<enum_quda.h>
    void halfwilson_hisq_force_reference(float eps, float weight1, void* act_path_coeff, void* temp_x, void* sitelink, void* mom);
    void halfwilson_hisq_force_reference(double eps, double weight1, void* act_path_coeff, void* temp_x, void* sitelink, void* mom);

    void color_matrix_hisq_force_reference(float eps, float weight,
                          void* act_path_coeff, void* temp_xx,
                          void* sitelink, void* mom);


    void computeHisqOuterProduct(void* src, void* dst, QudaPrecision precision);		

void computeLinkOrderedOuterProduct(void *src, void* dest, QudaPrecision precision, int gauge_order);

void computeLinkOrderedOuterProduct(void *src, void* dest, QudaPrecision precision, size_t separation, int gauge_order);

class cpuGaugeField;


void hisqStaplesForceCPU(const double* path_coeff,
			 const QudaGaugeParam &param,
			 cpuGaugeField  &oprod,
			 cpuGaugeField  &link,
			 cpuGaugeField* newOprod);

void hisqCompleteForceCPU(const QudaGaugeParam &param,
			  cpuGaugeField &oprod,
			  cpuGaugeField &link,
			  cpuGaugeField* mom);

void hisqLongLinkForceCPU(double coeff,
			  const QudaGaugeParam &param,
			  cpuGaugeField &oprod,
			  cpuGaugeField &link,
			  cpuGaugeField *newOprod);


#endif

