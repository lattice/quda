#ifndef _HISQ_LINKS_QUDA_H
#define _HISQ_LINKS_QUDA_H

#include <gauge_field.h>

namespace hisq {

void setUnitarizeLinksPadding(int input_padding, 
			     int output_padding);

void setUnitarizeLinksConstants(double unitarize_eps, double max_det_error, 
				bool allow_svd, bool svd_only,
				double svd_rel_error, double svd_abs_error);


void unitarizeLinksCuda(const QudaGaugeParam& param,
			cudaGaugeField& infield,
			cudaGaugeField* outfield, 
			int* num_failures);

void unitarizeLinksCPU(const QudaGaugeParam& param,
		       cpuGaugeField& infield,
		       cpuGaugeField* outfield);

} // namespace hisq


#endif // _HISQ_LINKS_H
