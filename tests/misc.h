#ifndef __MISC_H__
#define __MISC_H__

#ifdef __cplusplus
extern "C" {
#endif
    
    void display_spinor(void* spinor, int len, int precision);
    void display_link(void* link, int len, int precision);
    int link_sanity_check(void* link, int len, int precision, int dir, QudaGaugeParam* gaugeParam);
    int site_link_sanity_check(void* link, int len, int precision, QudaGaugeParam* gaugeParam);

    QudaReconstructType get_recon(char* s);
    QudaPrecision   get_prec(char* s);
    const char* get_prec_str(QudaPrecision prec);
    const char* get_recon_str(QudaReconstructType recon);
    const char* get_test_type(int t);
    void quda_set_verbose(int );
#ifdef __cplusplus
}
#endif

extern int verbose;

#define PRINTF(fmt,...) do{                                             \
	if (verbose){							\
	    printf("GPU:"fmt, ##__VA_ARGS__);				\
	}								\
    } while(0) 

#endif


