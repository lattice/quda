#ifndef __EXCHANGE_FACE_H__
#define __EXCHANGE_FACE_H__

#ifdef __cplusplus
extern "C" {
#endif
  void exchange_cpu_links(int* X,
			  void** fatlink, void* ghost_fatlink, 
			  void** longlink, void* ghost_longlink,
			  QudaPrecision gPrecision);
  void exchange_cpu_spinor(int* X,
			   void* spinorField, void* fwd_nbr_spinor, void* back_nbr_spinor,
			   QudaPrecision sPrecision);
  void exchange_gpu_spinor(void* _cudaSpinor, cudaStream_t* stream);
  void exchange_gpu_spinor_start(void* _cudaSpinor, cudaStream_t* stream);
  void exchange_gpu_spinor_wait(void* _cudaSpinor, cudaStream_t* stream);
  void exchange_cpu_sitelink(int* X,
			     void** sitelink, void* ghost_sitelink,
			     QudaPrecision gPrecision);
  void exchange_cpu_staple(int* X,
			 void* staple, void* ghost_staple,
			   QudaPrecision gPrecision);
  void exchange_llfat_init(FullStaple* cudaStaple);
  void exchange_gpu_staple(int*X, void* _cudaStaple, cudaStream_t* stream);
  void exchange_gpu_staple_start(int* X, void* _cudaStaple, cudaStream_t * stream);
  void exchange_gpu_staple_wait(int* X, void* _cudaStaple, cudaStream_t * stream);
  void exchange_cleanup(void);
  
#define TDIFF(t1, t0) ((t1.tv_sec - t0.tv_sec) + 1e-6*(t1.tv_usec -t0.tv_usec))
  
#ifdef __cplusplus
}
#endif

#endif



