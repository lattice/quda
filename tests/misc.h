#ifndef __MISC_H__
#define __MISC_H__

#include <quda.h>

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


#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir)    (7-(dir))
#define GOES_FORWARDS(dir) (dir<=3)
#define GOES_BACKWARDS(dir) (dir>3)


#endif


