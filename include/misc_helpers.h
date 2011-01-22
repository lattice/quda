#ifndef __MISC_HELPERS_H__
#define __MISC_HELPERS_H__


void link_format_cpu_to_gpu(double* dst, double* src,
			    int reconstruct, int bytes, int Vh, int pad, int Vsh);
void link_format_gpu_to_cpu(double* dst, double* src, 
			    int reconstruct, int bytes, int Vh, int stride);
#endif
