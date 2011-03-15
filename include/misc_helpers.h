#ifndef __MISC_HELPERS_H__
#define __MISC_HELPERS_H__


void link_format_cpu_to_gpu(void* dst, void* src,
			    int reconstruct, int bytes, int Vh, int pad, int Vsh, 
			    QudaPrecision prec);
void link_format_gpu_to_cpu(void* dst, void* src, 
			    int bytes, int Vh, int stride, QudaPrecision prec);

void collectGhostSpinor(void *in, const void *inNorm,
			void* ghost_spinor_gpu,		   
			int dir, int whichway,
			const int parity, cudaColorSpinorField* inSpinor);
#endif
