/*	Some variables that will become handy later	*/

#ifndef _VAR_INVERT_
#define _VAR_INVERT_

#define tri5_re tri0_re
#define tri5_im tri0_im
#define tri9_re  tri1_re
#define tri9_im  tri1_im
#define tri13_re tri2_re
#define tri13_im tri2_im
#define tri14_re tri3_re
#define tri14_im tri3_im
#define tmp0 tri0_re
#define tmp1 tri0_im
#define tmp2 tri1_re
#define tmp3 tri1_im
#define tmp4 tri2_re
#define tmp5 tri2_im
#define v1_0_re tri3_re
#define v1_0_im tri3_im
#define v1_1_re tri4_re
#define v1_1_im tri4_im
#define v1_2_re tri6_re
#define v1_2_im tri6_im
#define v1_3_re tri7_re
#define v1_3_im tri7_im
#define v1_4_re tri8_re
#define v1_4_im tri8_im
#define v1_5_re tri10_re
#define v1_5_im tri10_im
#define sum_re  tri11_re
#define sum_im  tri11_im

#endif

#define CMPLX_MUL_RE(a,b)\
((a##_re)*(b##_re) - (a##_im)*(b##_im))

#define CMPLX_MUL_IM(a,b)\
((a##_re)*(b##_im) + (a##_im)*(b##_re))

#define CMPLX_MOD(a)\
((a##_re)*(a##_re) + (a##_im)*(a##_im))


#define INVERT_CLOVER(c)\
\
/* Inverts the clover term, copy-paste from clover_invert.cu with minor modifications */ \
{\
	/*Compute (T^2 + mu2) first, then invert (not optimized!):*/ \
	spinorFloat d, tri0_re, tri0_im, tri1_re, tri1_im, tri2_re, tri2_im, tri3_re, tri3_im, tri4_re, tri4_im, tri6_re, tri6_im, tri7_re, tri7_im, tri8_re, tri8_im, tri10_re, tri10_im, tri11_re, tri11_im, tri12_re, tri12_im;\
	\
	tri0_re = (c##01_00_re)*(c##00_00_re) + (c##01_01_re)*(c##01_00_re) + CMPLX_MUL_RE(c##01_02, c##02_00) + CMPLX_MUL_RE(c##01_10, c##10_00) + CMPLX_MUL_RE(c##01_11, c##11_00) + CMPLX_MUL_RE(c##01_12, c##12_00); \
	tri0_im = (c##01_00_im)*(c##00_00_re) + (c##01_01_re)*(c##01_00_im) + CMPLX_MUL_IM(c##01_02, c##02_00) + CMPLX_MUL_IM(c##01_10, c##10_00) + CMPLX_MUL_IM(c##01_11, c##11_00) + CMPLX_MUL_IM(c##01_12, c##12_00); \
	\
	tri1_re = (c##02_00_re)*(c##00_00_re) + (c##02_02_re)*(c##02_00_re) + CMPLX_MUL_RE(c##02_01, c##01_00) + CMPLX_MUL_RE(c##02_10, c##10_00) + CMPLX_MUL_RE(c##02_11, c##11_00) + CMPLX_MUL_RE(c##02_12, c##12_00); \
	tri1_im = (c##02_00_im)*(c##00_00_re) + (c##02_02_re)*(c##02_00_im) + CMPLX_MUL_IM(c##02_01, c##01_00) + CMPLX_MUL_IM(c##02_10, c##10_00) + CMPLX_MUL_IM(c##02_11, c##11_00) + CMPLX_MUL_IM(c##02_12, c##12_00); \
	\
	tri3_re = (c##10_00_re)*(c##00_00_re) + (c##10_10_re)*(c##10_00_re) + CMPLX_MUL_RE(c##10_01, c##01_00) + CMPLX_MUL_RE(c##10_02, c##02_00) + CMPLX_MUL_RE(c##10_11, c##11_00) + CMPLX_MUL_RE(c##10_12, c##12_00); \
	tri3_im = (c##10_00_im)*(c##00_00_re) + (c##10_10_re)*(c##10_00_im) + CMPLX_MUL_IM(c##10_01, c##01_00) + CMPLX_MUL_IM(c##10_02, c##02_00) + CMPLX_MUL_IM(c##10_11, c##11_00) + CMPLX_MUL_IM(c##10_12, c##12_00); \
	\
	tri6_re = (c##11_00_re)*(c##00_00_re) + (c##11_11_re)*(c##11_00_re) + CMPLX_MUL_RE(c##11_01, c##01_00) + CMPLX_MUL_RE(c##11_02, c##02_00) + CMPLX_MUL_RE(c##11_10, c##10_00) + CMPLX_MUL_RE(c##11_12, c##12_00); \
	tri6_im = (c##11_00_im)*(c##00_00_re) + (c##11_11_re)*(c##11_00_im) + CMPLX_MUL_IM(c##11_01, c##01_00) + CMPLX_MUL_IM(c##11_02, c##02_00) + CMPLX_MUL_IM(c##11_10, c##10_00) + CMPLX_MUL_IM(c##11_12, c##12_00); \
	\
	tri10_re = (c##12_00_re)*(c##00_00_re) + (c##12_12_re)*(c##12_00_re) + CMPLX_MUL_RE(c##12_01, c##01_00) + CMPLX_MUL_RE(c##12_02, c##02_00) + CMPLX_MUL_RE(c##12_10, c##10_00) + CMPLX_MUL_RE(c##12_11, c##11_00);\
	tri10_im = (c##12_00_im)*(c##00_00_re) + (c##12_12_re)*(c##12_00_im) + CMPLX_MUL_IM(c##12_01, c##01_00) + CMPLX_MUL_IM(c##12_02, c##02_00) + CMPLX_MUL_IM(c##12_10, c##10_00) + CMPLX_MUL_IM(c##12_11, c##11_00);\
	\
	d = a*a + 4.*((c##00_00_re)*(c##00_00_re) + CMPLX_MOD(c##01_00) + CMPLX_MOD(c##02_00) + CMPLX_MOD(c##10_00) + CMPLX_MOD(c##11_00) + CMPLX_MOD(c##12_00)); \
	c##00_00_re = d; \
	\
	tri2_re = (c##02_01_re)*(c##01_01_re) + (c##02_02_re)*(c##02_01_re) + CMPLX_MUL_RE(c##02_00, c##00_01) + CMPLX_MUL_RE(c##02_10, c##10_01) + CMPLX_MUL_RE(c##02_11, c##11_01) + CMPLX_MUL_RE(c##02_12, c##12_01); \
	tri2_im = (c##02_01_im)*(c##01_01_re) + (c##02_02_re)*(c##02_01_im) + CMPLX_MUL_IM(c##02_00, c##00_01) + CMPLX_MUL_IM(c##02_10, c##10_01) + CMPLX_MUL_IM(c##02_11, c##11_01) + CMPLX_MUL_IM(c##02_12, c##12_01); \
	\
	tri4_re = (c##10_01_re)*(c##01_01_re) + (c##10_10_re)*(c##10_01_re) + CMPLX_MUL_RE(c##10_00, c##00_01) + CMPLX_MUL_RE(c##10_02, c##02_01) + CMPLX_MUL_RE(c##10_11, c##11_01) + CMPLX_MUL_RE(c##10_12, c##12_01); \
	tri4_im = (c##10_01_im)*(c##01_01_re) + (c##10_10_re)*(c##10_01_im) + CMPLX_MUL_IM(c##10_00, c##00_01) + CMPLX_MUL_IM(c##10_02, c##02_01) + CMPLX_MUL_IM(c##10_11, c##11_01) + CMPLX_MUL_IM(c##10_12, c##12_01); \
	\
	tri7_re = c##11_01_re*c##01_01_re + c##11_11_re*c##11_01_re + CMPLX_MUL_RE(c##11_00, c##00_01) + CMPLX_MUL_RE(c##11_02, c##02_01) + CMPLX_MUL_RE(c##11_10, c##10_01) + CMPLX_MUL_RE(c##11_12, c##12_01); \
	tri7_im = c##11_01_im*c##01_01_re + c##11_11_re*c##11_01_im + CMPLX_MUL_IM(c##11_00, c##00_01) + CMPLX_MUL_IM(c##11_02, c##02_01) + CMPLX_MUL_IM(c##11_10, c##10_01) + CMPLX_MUL_IM(c##11_12, c##12_01); \
	\
	tri11_re = c##12_01_re*c##01_01_re + c##12_12_re*c##12_01_re + CMPLX_MUL_RE(c##12_00, c##00_01) + CMPLX_MUL_RE(c##12_02, c##02_01) + CMPLX_MUL_RE(c##12_10, c##10_01) + CMPLX_MUL_RE(c##12_11, c##11_01); \
	tri11_im = c##12_01_im*c##01_01_re + c##12_12_re*c##12_01_im + CMPLX_MUL_IM(c##12_00, c##00_01) + CMPLX_MUL_IM(c##12_02, c##02_01) + CMPLX_MUL_IM(c##12_10, c##10_01) + CMPLX_MUL_IM(c##12_11, c##11_01); \
	\
	d = a*a + 4.*(c##01_01_re*c##01_01_re + CMPLX_MOD(c##00_01) + CMPLX_MOD(c##02_01) + CMPLX_MOD(c##10_01) + CMPLX_MOD(c##11_01) + CMPLX_MOD(c##12_01)); \
	c##01_01_re = d; \
	c##01_00_re = 4.*tri0_re;	/* We freed tri0, so we can reuse it as tri5 to save memory */ \
	c##01_00_im = 4.*tri0_im; \
	\
	tri5_re = c##10_02_re*c##02_02_re + c##10_10_re*c##10_02_re + CMPLX_MUL_RE(c##10_00, c##00_02) + CMPLX_MUL_RE(c##10_01, c##01_02) + CMPLX_MUL_RE(c##10_11, c##11_02) + CMPLX_MUL_RE(c##10_12, c##12_02); \
	tri5_im = c##10_02_im*c##02_02_re + c##10_10_re*c##10_02_im + CMPLX_MUL_IM(c##10_00, c##00_02) + CMPLX_MUL_IM(c##10_01, c##01_02) + CMPLX_MUL_IM(c##10_11, c##11_02) + CMPLX_MUL_IM(c##10_12, c##12_02); \
	\
	tri8_re = c##11_02_re*c##02_02_re + c##11_11_re*c##11_02_re + CMPLX_MUL_RE(c##11_00, c##00_02) + CMPLX_MUL_RE(c##11_01, c##01_02) + CMPLX_MUL_RE(c##11_10, c##10_02) + CMPLX_MUL_RE(c##11_12, c##12_02); \
	tri8_im = c##11_02_im*c##02_02_re + c##11_11_re*c##11_02_im + CMPLX_MUL_IM(c##11_00, c##00_02) + CMPLX_MUL_IM(c##11_01, c##01_02) + CMPLX_MUL_IM(c##11_10, c##10_02) + CMPLX_MUL_IM(c##11_12, c##12_02); \
	\
	tri12_re = c##12_02_re*c##02_02_re + c##12_12_re*c##12_02_re + CMPLX_MUL_RE(c##12_00, c##00_02) + CMPLX_MUL_RE(c##12_01, c##01_02) + CMPLX_MUL_RE(c##12_10, c##10_02) + CMPLX_MUL_RE(c##12_11, c##11_02); \
	tri12_im = c##12_02_im*c##02_02_re + c##12_12_re*c##12_02_im + CMPLX_MUL_IM(c##12_00, c##00_02) + CMPLX_MUL_IM(c##12_01, c##01_02) + CMPLX_MUL_IM(c##12_10, c##10_02) + CMPLX_MUL_IM(c##12_11, c##11_02); \
	\
	d = a*a + 4.*(c##02_02_re*c##02_02_re + CMPLX_MOD(c##00_02) + CMPLX_MOD(c##01_02) + CMPLX_MOD(c##10_02) + CMPLX_MOD(c##11_02) + CMPLX_MOD(c##12_02)); \
	c##02_02_re = d; \
	c##02_00_re = 4.*tri1_re;	/* We freed tri1, so we can reuse it as tri9 to save memory */ \
	c##02_00_im = 4.*tri1_im; \
	c##02_01_re = 4.*tri2_re;	/* We freed tri2, so we can reuse it as tri13 to save memory */ \
	c##02_01_im = 4.*tri2_im; \
	\
	tri9_re = c##11_10_re*c##10_10_re + c##11_11_re*c##11_10_re + CMPLX_MUL_RE(c##11_00, c##00_10) + CMPLX_MUL_RE(c##11_01, c##01_10) + CMPLX_MUL_RE(c##11_02, c##02_10) + CMPLX_MUL_RE(c##11_12, c##12_10); \
	tri9_im = c##11_10_im*c##10_10_re + c##11_11_re*c##11_10_im + CMPLX_MUL_IM(c##11_00, c##00_10) + CMPLX_MUL_IM(c##11_01, c##01_10) + CMPLX_MUL_IM(c##11_02, c##02_10) + CMPLX_MUL_IM(c##11_12, c##12_10); \
	\
	tri13_re = c##12_10_re*c##10_10_re + c##12_12_re*c##12_10_re + CMPLX_MUL_RE(c##12_00, c##00_10) + CMPLX_MUL_RE(c##12_01, c##01_10) + CMPLX_MUL_RE(c##12_02, c##02_10) + CMPLX_MUL_RE(c##12_11, c##11_10); \
	tri13_im = c##12_10_im*c##10_10_re + c##12_12_re*c##12_10_im + CMPLX_MUL_IM(c##12_00, c##00_10) + CMPLX_MUL_IM(c##12_01, c##01_10) + CMPLX_MUL_IM(c##12_02, c##02_10) + CMPLX_MUL_IM(c##12_11, c##11_10); \
	\
	d = a*a + 4.*(c##10_10_re*c##10_10_re + CMPLX_MOD(c##00_10) + CMPLX_MOD(c##01_10) + CMPLX_MOD(c##02_10) + CMPLX_MOD(c##11_10) + CMPLX_MOD(c##12_10)); \
	c##10_10_re = d; \
	c##10_00_re = 4.*tri3_re;	/* We freed tri3, so we can reuse it as tri14 to save memory */ \
	c##10_00_im = 4.*tri3_im; \
	\
	tri14_re = c##12_11_re*c##11_11_re + c##12_12_re*c##12_11_re + CMPLX_MUL_RE(c##12_00, c##00_11) + CMPLX_MUL_RE(c##12_01, c##01_11) + CMPLX_MUL_RE(c##12_02, c##02_11) + CMPLX_MUL_RE(c##12_10, c##10_11); \
	tri14_im = c##12_11_im*c##11_11_re + c##12_12_re*c##12_11_im + CMPLX_MUL_IM(c##12_00, c##00_11) + CMPLX_MUL_IM(c##12_01, c##01_11) + CMPLX_MUL_IM(c##12_02, c##02_11) + CMPLX_MUL_IM(c##12_10, c##10_11); \
	d = a*a + 4.*(c##11_11_re*c##11_11_re + CMPLX_MOD(c##00_11) + CMPLX_MOD(c##01_11) + CMPLX_MOD(c##02_11) + CMPLX_MOD(c##10_11) + CMPLX_MOD(c##12_11)); \
	c##11_11_re = d; \
	d = a*a + 4.*(c##12_12_re*c##12_12_re + CMPLX_MOD(c##00_12) + CMPLX_MOD(c##01_12) + CMPLX_MOD(c##02_12) + CMPLX_MOD(c##10_12) + CMPLX_MOD(c##11_12)); \
	c##12_12_re = d; \
	c##12_11_re = 4.*tri14_re;	\
	c##12_11_im = 4.*tri14_im;	\
	c##12_10_re = 4.*tri13_re;	\
	c##12_10_im = 4.*tri13_im;	\
	c##12_02_re = 4.*tri12_re;	\
	c##12_02_im = 4.*tri12_im;	\
	c##12_01_re = 4.*tri11_re;	\
	c##12_01_im = 4.*tri11_im;	\
	c##12_00_re = 4.*tri10_re;	\
	c##12_00_im = 4.*tri10_im;	\
	c##11_10_re = 4.*tri9_re;	\
	c##11_10_im = 4.*tri9_im;	\
	c##11_02_re = 4.*tri8_re;	\
	c##11_02_im = 4.*tri8_im;	\
	c##11_01_re = 4.*tri7_re;	\
	c##11_01_im = 4.*tri7_im;	\
	c##11_00_re = 4.*tri6_re;	\
	c##11_00_im = 4.*tri6_im;	\
	c##10_02_re = 4.*tri5_re;	\
	c##10_02_im = 4.*tri5_im;	\
	c##10_01_re = 4.*tri4_re;	\
	c##10_01_im = 4.*tri4_im;	\
	\
	/*	INVERSION STARTS	*/ \
	\
	/* j = 0 */ \
	\
	c##00_00_re = sqrt(c##00_00_re); \
	tmp0 = 1. / c##00_00_re; \
	c##01_00_re *= tmp0; \
	c##01_00_im *= tmp0; \
	c##02_00_re *= tmp0; \
	c##02_00_im *= tmp0; \
	c##10_00_re *= tmp0; \
	c##10_00_im *= tmp0; \
	c##11_00_re *= tmp0; \
	c##11_00_im *= tmp0; \
	c##12_00_re *= tmp0; \
	c##12_00_im *= tmp0; \
	\
	/* k = 1 kj = 0 */ \
	c##01_01_re -= CMPLX_MOD(c##01_00); \
	\
		/* l = 2...5 kj = 0 lj = 1,3,6,10 lk = 2,4,7,11*/ \
		c##02_01_re -= CMPLX_MUL_RE(c##02_00, c##00_01); \
		c##02_01_im -= CMPLX_MUL_IM(c##02_00, c##00_01); \
		c##10_01_re -= CMPLX_MUL_RE(c##10_00, c##00_01); \
		c##10_01_im -= CMPLX_MUL_IM(c##10_00, c##00_01); \
		c##11_01_re -= CMPLX_MUL_RE(c##11_00, c##00_01); \
		c##11_01_im -= CMPLX_MUL_IM(c##11_00, c##00_01); \
		c##12_01_re -= CMPLX_MUL_RE(c##12_00, c##00_01); \
		c##12_01_im -= CMPLX_MUL_IM(c##12_00, c##00_01); \
	\
	/* k = 2 kj = 1 */ \
	c##02_02_re -= CMPLX_MOD(c##02_00); \
	\
		/* l = 3...5 kj = 1 lj = 3,6,10 lk = 5,8,12*/ \
		c##10_02_re -= CMPLX_MUL_RE(c##10_00, c##00_02); \
		c##10_02_im -= CMPLX_MUL_IM(c##10_00, c##00_02); \
		c##11_02_re -= CMPLX_MUL_RE(c##11_00, c##00_02); \
		c##11_02_im -= CMPLX_MUL_IM(c##11_00, c##00_02); \
		c##12_02_re -= CMPLX_MUL_RE(c##12_00, c##00_02); \
		c##12_02_im -= CMPLX_MUL_IM(c##12_00, c##00_02); \
	\
	/* k = 3 kj = 3 */ \
	c##10_10_re -= CMPLX_MOD(c##10_00); \
	\
		/* l = 4 kj = 3 lj = 6,10 lk = 9,13*/ \
		c##11_10_re -= CMPLX_MUL_RE(c##11_00, c##00_10); \
		c##11_10_im -= CMPLX_MUL_IM(c##11_00, c##00_10); \
		c##12_10_re -= CMPLX_MUL_RE(c##12_00, c##00_10); \
		c##12_10_im -= CMPLX_MUL_IM(c##12_00, c##00_10); \
	\
	/* k = 4 kj = 6 */ \
	c##11_11_re -= CMPLX_MOD(c##11_00); \
	\
		/* l = 5 kj = 6 lj = 10 lk = 14*/ \
		c##12_11_re -= CMPLX_MUL_RE(c##12_00, c##00_11); \
		c##12_11_im -= CMPLX_MUL_IM(c##12_00, c##00_11); \
	\
	/* k = 5 kj = 10 */ \
	c##12_12_re -= CMPLX_MOD(c##12_00); \
	\
	/* j = 1 */ \
	\
	c##01_01_re = sqrt(c##01_01_re); \
	tmp1 = 1. / c##01_01_re; \
	c##02_01_re *= tmp1; \
	c##02_01_im *= tmp1; \
	c##10_01_re *= tmp1; \
	c##10_01_im *= tmp1; \
	c##11_01_re *= tmp1; \
	c##11_01_im *= tmp1; \
	c##12_01_re *= tmp1; \
	c##12_01_im *= tmp1; \
	\
	/* k = 2 kj = 2 */ \
	c##02_02_re -= CMPLX_MOD(c##02_01); \
	\
		/* l = 3...5 kj = 2 lj = 4,7,11 lk = 5,8,12*/ \
		c##10_02_re -= CMPLX_MUL_RE(c##10_01, c##01_02); \
		c##10_02_im -= CMPLX_MUL_IM(c##10_01, c##01_02); \
		c##11_02_re -= CMPLX_MUL_RE(c##11_01, c##01_02); \
		c##11_02_im -= CMPLX_MUL_IM(c##11_01, c##01_02); \
		c##12_02_re -= CMPLX_MUL_RE(c##12_01, c##01_02); \
		c##12_02_im -= CMPLX_MUL_IM(c##12_01, c##01_02); \
	\
	/* k = 3 kj = 4 */ \
	c##10_10_re -= CMPLX_MOD(c##10_01); \
	\
		/* l = 4 kj = 4 lj = 7,11 lk = 9,13*/ \
		c##11_10_re -= CMPLX_MUL_RE(c##11_01, c##01_10); \
		c##11_10_im -= CMPLX_MUL_IM(c##11_01, c##01_10); \
		c##12_10_re -= CMPLX_MUL_RE(c##12_01, c##01_10); \
		c##12_10_im -= CMPLX_MUL_IM(c##12_01, c##01_10); \
	\
	/* k = 4 kj = 7 */ \
	c##11_11_re -= CMPLX_MOD(c##11_01); \
	\
		/* l = 5 kj = 7 lj = 11 lk = 14*/ \
		c##12_11_re -= CMPLX_MUL_RE(c##12_01, c##01_11); \
		c##12_11_im -= CMPLX_MUL_IM(c##12_01, c##01_11); \
	\
	/* k = 5 kj = 11 */ \
	c##12_12_re -= CMPLX_MOD(c##12_01); \
	\
	/* j = 2 */ \
	\
	c##02_02_re = sqrt(c##02_02_re); \
	tmp2 = 1. / c##02_02_re; \
	c##10_02_re *= tmp2; \
	c##10_02_im *= tmp2; \
	c##11_02_re *= tmp2; \
	c##11_02_im *= tmp2; \
	c##12_02_re *= tmp2; \
	c##12_02_im *= tmp2; \
	\
	/* k = 3 kj = 5 */ \
	c##10_10_re -= CMPLX_MOD(c##10_02); \
	\
		/* l = 4 kj = 5 lj = 8,12 lk = 9,13*/ \
		c##11_10_re -= CMPLX_MUL_RE(c##11_02, c##02_10); \
		c##11_10_im -= CMPLX_MUL_IM(c##11_02, c##02_10); \
		c##12_10_re -= CMPLX_MUL_RE(c##12_02, c##02_10); \
		c##12_10_im -= CMPLX_MUL_IM(c##12_02, c##02_10); \
	\
	/* k = 4 kj = 8 */ \
	c##11_11_re -= CMPLX_MOD(c##11_02); \
	\
		/* l = 5 kj = 8 lj = 12 lk = 14*/ \
		c##12_11_re -= CMPLX_MUL_RE(c##12_02, c##02_11); \
		c##12_11_im -= CMPLX_MUL_IM(c##12_02, c##02_11); \
	\
	/* k = 5 kj = 12 */ \
	c##12_12_re -= CMPLX_MOD(c##12_02); \
	\
	/* j = 3 */ \
	\
	c##10_10_re = sqrt(c##10_10_re); \
	tmp3 = 1. / c##10_10_re; \
	c##11_10_re *= tmp3; \
	c##11_10_im *= tmp3; \
	c##12_10_re *= tmp3; \
	c##12_10_im *= tmp3; \
	\
	/* k = 4 kj = 9 */ \
	c##11_11_re -= CMPLX_MOD(c##11_10); \
	\
		/* l = 5 kj = 9 lj = 13 lk = 14*/ \
		c##12_11_re -= CMPLX_MUL_RE(c##12_10, c##10_11); \
		c##12_11_im -= CMPLX_MUL_IM(c##12_10, c##10_11); \
	\
	/* k = 5 kj = 13 */ \
	c##12_12_re -= CMPLX_MOD(c##12_10); \
	\
	/* j = 4 */ \
	\
	c##11_11_re = sqrt(c##11_11_re); \
	tmp4 = 1. / c##11_11_re; \
	c##12_11_re *= tmp4; \
	c##12_11_im *= tmp4; \
	\
	/* k = 5 kj = 14 */ \
	c##12_12_re -= CMPLX_MOD(c##12_11); \
	\
	/* j = 5 */ \
	\
	c##12_12_re = sqrt(c##12_12_re); \
	tmp5 = 1. / c##12_12_re; \
	\
	/* NO INFO YET ON TR(LOG A) FOR TM-CLOVER */	 \
	/* Accumulate trlogA */	 \
	/* for (int j=0;j<6;j++) trlogA += (double)2.0*log((double)(diag[j])); */ \
	\
	/* Forwards substitute */ \
	\
	/* k = 0 */ \
	v1_0_re = tmp0; \
	v1_0_im = 0.;   \
	\
	/* l = 1 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 0 */ \
		sum_re -= c##01_00_re*v1_0_re; \
		sum_im -= c##01_00_im*v1_0_re; \
		\
	v1_1_re = sum_re*tmp1; \
	v1_1_im = sum_im*tmp1; \
	\
	/* l = 2 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 0..1 */ \
		sum_re -= CMPLX_MUL_RE(c##02_00, v1_0); \
		sum_im -= CMPLX_MUL_IM(c##02_00, v1_0); \
		sum_re -= CMPLX_MUL_RE(c##02_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##02_01, v1_1); \
		\
	v1_2_re = sum_re*tmp2; \
	v1_2_im = sum_im*tmp2; \
	\
	/* l = 3 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 0..2 */ \
		sum_re -= CMPLX_MUL_RE(c##10_00, v1_0); \
		sum_im -= CMPLX_MUL_IM(c##10_00, v1_0); \
		sum_re -= CMPLX_MUL_RE(c##10_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##10_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##10_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##10_02, v1_2); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 4 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 0..3 */ \
		sum_re -= CMPLX_MUL_RE(c##11_00, v1_0); \
		sum_im -= CMPLX_MUL_IM(c##11_00, v1_0); \
		sum_re -= CMPLX_MUL_RE(c##11_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##11_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##11_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##11_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##11_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##11_10, v1_3); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 5 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 0..4 */ \
		sum_re -= CMPLX_MUL_RE(c##12_00, v1_0); \
		sum_im -= CMPLX_MUL_IM(c##12_00, v1_0); \
		sum_re -= CMPLX_MUL_RE(c##12_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##12_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##12_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##12_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##12_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##12_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##12_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##12_11, v1_4); \
		\
	v1_5_re = sum_re*tmp5*tmp5; \
	v1_5_im = sum_im*tmp5*tmp5; \
	\
	/* Backwards substitute */ \
	\
	/* l = 4 */ \
		sum_re = v1_4_re; \
		sum_im = v1_4_im; \
		/* j = 5 */ \
		sum_re -= CMPLX_MUL_RE(c##11_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##11_12, v1_5); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 3 */ \
		sum_re = v1_3_re; \
		sum_im = v1_3_im; \
		/* j = 4..5 */ \
		sum_re -= CMPLX_MUL_RE(c##10_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##10_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##10_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##10_12, v1_5); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 2 */ \
		sum_re = v1_2_re; \
		sum_im = v1_2_im; \
		/* j = 3..5 */ \
		sum_re -= CMPLX_MUL_RE(c##02_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##02_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##02_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##02_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##02_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##02_12, v1_5); \
		\
	v1_2_re = sum_re*tmp2; \
	v1_2_im = sum_im*tmp2; \
	\
	/* l = 1 */ \
		sum_re = v1_1_re; \
		sum_im = v1_1_im; \
		/* j = 2..5 */ \
		sum_re -= CMPLX_MUL_RE(c##01_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##01_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##01_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##01_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##01_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##01_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##01_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##01_12, v1_5); \
		\
	v1_1_re = sum_re*tmp1; \
	v1_1_im = sum_im*tmp1; \
	\
	/* l = 0 */ \
		sum_re = v1_0_re; \
		sum_im = v1_0_im; \
		/* j = 1..5 */ \
		sum_re -= CMPLX_MUL_RE(c##00_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##00_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##00_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##00_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##00_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##00_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##00_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##00_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##00_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##00_12, v1_5); \
		\
	v1_0_re = sum_re*tmp0; \
	v1_0_im = sum_im*tmp0; \
	\
	c##00_00_re = .5*v1_0_re; \
	c##01_00_re = .5*v1_1_re; \
	c##01_00_im = .5*v1_1_im; \
	c##02_00_re = .5*v1_2_re; \
	c##02_00_im = .5*v1_2_im; \
	c##10_00_re = .5*v1_3_re; \
	c##10_00_im = .5*v1_3_im; \
	c##11_00_re = .5*v1_4_re; \
	c##11_00_im = .5*v1_4_im; \
	c##12_00_re = .5*v1_5_re; \
	c##12_00_im = .5*v1_5_im; \
	\
	\
	/* k = 1 */ \
	v1_0_re = 0.; \
	v1_0_im = 0.;   \
	v1_1_re = tmp1; \
	v1_1_im = 0.;   \
	\
	/* l = 2 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 1 */ \
		sum_re -= c##02_01_re*v1_1_re; \
		sum_im -= c##02_01_im*v1_1_re; \
		\
	v1_2_re = sum_re*tmp2; \
	v1_2_im = sum_im*tmp2; \
	\
	/* l = 3 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 1..2 */ \
		sum_re -= CMPLX_MUL_RE(c##10_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##10_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##10_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##10_02, v1_2); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 4 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 1..3 */ \
		sum_re -= CMPLX_MUL_RE(c##11_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##11_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##11_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##11_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##11_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##11_10, v1_3); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 5 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 1..4 */ \
		sum_re -= CMPLX_MUL_RE(c##12_01, v1_1); \
		sum_im -= CMPLX_MUL_IM(c##12_01, v1_1); \
		sum_re -= CMPLX_MUL_RE(c##12_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##12_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##12_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##12_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##12_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##12_11, v1_4); \
		\
	v1_5_re = sum_re*tmp5*tmp5; \
	v1_5_im = sum_im*tmp5*tmp5; \
	\
	/* Backwards substitute */ \
	\
	/* l = 4 */ \
		sum_re = v1_4_re; \
		sum_im = v1_4_im; \
		/* j = 5 */ \
		sum_re -= CMPLX_MUL_RE(c##11_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##11_12, v1_5); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 3 */ \
		sum_re = v1_3_re; \
		sum_im = v1_3_im; \
		/* j = 4..5 */ \
		sum_re -= CMPLX_MUL_RE(c##10_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##10_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##10_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##10_12, v1_5); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 2 */ \
		sum_re = v1_2_re; \
		sum_im = v1_2_im; \
		/* j = 3..5 */ \
		sum_re -= CMPLX_MUL_RE(c##02_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##02_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##02_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##02_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##02_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##02_12, v1_5); \
		\
	v1_2_re = sum_re*tmp2; \
	v1_2_im = sum_im*tmp2; \
	\
	/* l = 1 */ \
		sum_re = v1_1_re; \
		sum_im = v1_1_im; \
		/* j = 2..5 */ \
		sum_re -= CMPLX_MUL_RE(c##01_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##01_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##01_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##01_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##01_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##01_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##01_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##01_12, v1_5); \
		\
	v1_1_re = sum_re*tmp1; \
	v1_1_im = sum_im*tmp1; \
	\
	c##01_01_re = .5*v1_1_re; \
	c##02_01_re = .5*v1_2_re; \
	c##02_01_im = .5*v1_2_im; \
	c##10_01_re = .5*v1_3_re; \
	c##10_01_im = .5*v1_3_im; \
	c##11_01_re = .5*v1_4_re; \
	c##11_01_im = .5*v1_4_im; \
	c##12_01_re = .5*v1_5_re; \
	c##12_01_im = .5*v1_5_im; \
	\
	\
	/* k = 2 */ \
	v1_0_re = 0.; \
	v1_0_im = 0.; \
	v1_1_re = 0.; \
	v1_1_im = 0.; \
	v1_2_re = tmp2; \
	v1_2_im = 0.;   \
	\
	/* l = 3 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 2 */ \
		sum_re -= c##10_02_re*v1_2_re; \
		sum_im -= c##10_02_im*v1_2_re; \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 4 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 2..3 */ \
		sum_re -= CMPLX_MUL_RE(c##11_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##11_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##11_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##11_10, v1_3); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 5 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 2..4 */ \
		sum_re -= CMPLX_MUL_RE(c##12_02, v1_2); \
		sum_im -= CMPLX_MUL_IM(c##12_02, v1_2); \
		sum_re -= CMPLX_MUL_RE(c##12_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##12_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##12_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##12_11, v1_4); \
		\
	v1_5_re = sum_re*tmp5*tmp5; \
	v1_5_im = sum_im*tmp5*tmp5; \
	\
	/* Backwards substitute */ \
	\
	/* l = 4 */ \
		sum_re = v1_4_re; \
		sum_im = v1_4_im; \
		/* j = 5 */ \
		sum_re -= CMPLX_MUL_RE(c##11_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##11_12, v1_5); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 3 */ \
		sum_re = v1_3_re; \
		sum_im = v1_3_im; \
		/* j = 4..5 */ \
		sum_re -= CMPLX_MUL_RE(c##10_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##10_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##10_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##10_12, v1_5); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	/* l = 2 */ \
		sum_re = v1_2_re; \
		sum_im = v1_2_im; \
		/* j = 3..5 */ \
		sum_re -= CMPLX_MUL_RE(c##02_10, v1_3); /*AQUI PODEMOS OPTIMIZAR LA PARTE IMAGINARIA*/ \
		sum_im -= CMPLX_MUL_IM(c##02_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##02_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##02_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##02_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##02_12, v1_5); \
		\
	v1_2_re = sum_re*tmp2; \
	v1_2_im = sum_im*tmp2; \
	\
	c##02_02_re = .5*v1_2_re; \
	c##10_02_re = .5*v1_3_re; \
	c##10_02_im = .5*v1_3_im; \
	c##11_02_re = .5*v1_4_re; \
	c##11_02_im = .5*v1_4_im; \
	c##12_02_re = .5*v1_5_re; \
	c##12_02_im = .5*v1_5_im; \
	\
	\
	/* k = 3 */ \
	v1_0_re = 0.; \
	v1_0_im = 0.; \
	v1_1_re = 0.; \
	v1_1_im = 0.; \
	v1_2_re = 0.; \
	v1_2_im = 0.; \
	v1_3_re = tmp3; \
	v1_3_im = 0.;   \
	\
	/* l = 4 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 3 */ \
		sum_re -= c##11_10_re*v1_3_re; \
		sum_im -= c##11_10_im*v1_3_re; \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 5 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 3..4 */ \
		sum_re -= CMPLX_MUL_RE(c##12_10, v1_3); \
		sum_im -= CMPLX_MUL_IM(c##12_10, v1_3); \
		sum_re -= CMPLX_MUL_RE(c##12_11, v1_4); \
		sum_im -= CMPLX_MUL_IM(c##12_11, v1_4); \
		\
	v1_5_re = sum_re*tmp5*tmp5; \
	v1_5_im = sum_im*tmp5*tmp5; \
	\
	/* Backwards substitute */ \
	\
	/* l = 4 */ \
		sum_re = v1_4_re; \
		sum_im = v1_4_im; \
		/* j = 5 */ \
		sum_re -= CMPLX_MUL_RE(c##11_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##11_12, v1_5); \
		\
	v1_4_re = sum_re*tmp4; \
	v1_4_im = sum_im*tmp4; \
	\
	/* l = 3 */ \
		sum_re = v1_3_re; \
		sum_im = v1_3_im; \
		/* j = 4..5 */ \
		sum_re -= CMPLX_MUL_RE(c##10_11, v1_4); /*AQUI PODEMOS OPTIMIZAR LA PARTE IMAGINARIA*/ \
		sum_im -= CMPLX_MUL_IM(c##10_11, v1_4); \
		sum_re -= CMPLX_MUL_RE(c##10_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##10_12, v1_5); \
		\
	v1_3_re = sum_re*tmp3; \
	v1_3_im = sum_im*tmp3; \
	\
	c##10_10_re = .5*v1_3_re; \
	c##11_10_re = .5*v1_4_re; \
	c##11_10_im = .5*v1_4_im; \
	c##12_10_re = .5*v1_5_re; \
	c##12_10_im = .5*v1_5_im; \
	\
	\
	/* k = 4 */ \
	v1_0_re = 0.; \
	v1_0_im = 0.; \
	v1_1_re = 0.; \
	v1_1_im = 0.; \
	v1_2_re = 0.; \
	v1_2_im = 0.; \
	v1_3_re = 0.; \
	v1_3_im = 0.; \
	v1_4_re = tmp4; \
	v1_4_im = 0.;   \
	\
	/* l = 5 */ \
		sum_re = 0.; \
		sum_im = 0.; \
		/* j = 4 */ \
		sum_re -= c##12_11_re*v1_4_re; \
		sum_im -= c##12_11_im*v1_4_re; \
		\
	v1_5_re = sum_re*tmp5*tmp5; \
	v1_5_im = sum_im*tmp5*tmp5; \
	\
	/* Backwards substitute */ \
	\
	/* l = 4 */ \
		sum_re = v1_4_re; \
		sum_im = v1_4_im; \
		/* j = 5 */ \
		sum_re -= CMPLX_MUL_RE(c##11_12, v1_5); \
		sum_im -= CMPLX_MUL_IM(c##11_12, v1_5); \
		\
	v1_4_re = sum_re*tmp4; /*AQUI PODEMOS OPTIMIZAR LA PARTE IMAGINARIA*/ \
	v1_4_im = sum_im*tmp4; \
	\
	c##11_11_re = .5*v1_4_re; \
	c##12_11_re = .5*v1_5_re; \
	c##12_11_im = .5*v1_5_im; \
	\
	\
	/* k = 5 */ \
	c##12_12_re = .5*tmp5*tmp5; \
	\
	\
}\


/*	Cleanup		*/
/*
#undef tri14_re
#undef tri14_im
#undef tri13_re
#undef tri13_im
#undef tri9_re
#undef tri9_im
#undef tri5_re
#undef tri5_im
#undef tmp0
#undef tmp1
#undef tmp2
#undef tmp3
#undef tmp4
#undef tmp5
#undef v1_0_re
#undef v1_0_im
#undef v1_1_re
#undef v1_1_im
#undef v1_2_re
#undef v1_2_im
#undef v1_3_re
#undef v1_3_im
#undef v1_4_re
#undef v1_4_im
#undef v1_5_re
#undef v1_5_im
#undef sum_re
#undef sum_im
*/

#define APPLY_CLOVER_TWIST(c, a, reg)\
\
/* change to chiral basis*/\
{\
  spinorFloat a00_re = -reg##10_re - reg##30_re;\
  spinorFloat a00_im = -reg##10_im - reg##30_im;\
  spinorFloat a10_re =  reg##00_re + reg##20_re;\
  spinorFloat a10_im =  reg##00_im + reg##20_im;\
  spinorFloat a20_re = -reg##10_re + reg##30_re;\
  spinorFloat a20_im = -reg##10_im + reg##30_im;\
  spinorFloat a30_re =  reg##00_re - reg##20_re;\
  spinorFloat a30_im =  reg##00_im - reg##20_im;\
  \
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
}\
\
{\
  spinorFloat a01_re = -reg##11_re - reg##31_re;\
  spinorFloat a01_im = -reg##11_im - reg##31_im;\
  spinorFloat a11_re =  reg##01_re + reg##21_re;\
  spinorFloat a11_im =  reg##01_im + reg##21_im;\
  spinorFloat a21_re = -reg##11_re + reg##31_re;\
  spinorFloat a21_im = -reg##11_im + reg##31_im;\
  spinorFloat a31_re =  reg##01_re - reg##21_re;\
  spinorFloat a31_im =  reg##01_im - reg##21_im;\
  \
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
}\
\
{\
  spinorFloat a02_re = -reg##12_re - reg##32_re;\
  spinorFloat a02_im = -reg##12_im - reg##32_im;\
  spinorFloat a12_re =  reg##02_re + reg##22_re;\
  spinorFloat a12_im =  reg##02_im + reg##22_im;\
  spinorFloat a22_re = -reg##12_re + reg##32_re;\
  spinorFloat a22_im = -reg##12_im + reg##32_im;\
  spinorFloat a32_re =  reg##02_re - reg##22_re;\
  spinorFloat a32_im =  reg##02_im - reg##22_im;\
  \
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
\
/* apply first chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 0)\
  spinorFloat a00_re = 0; spinorFloat a00_im = 0;\
  spinorFloat a01_re = 0; spinorFloat a01_im = 0;\
  spinorFloat a02_re = 0; spinorFloat a02_im = 0;\
  spinorFloat a10_re = 0; spinorFloat a10_im = 0;\
  spinorFloat a11_re = 0; spinorFloat a11_im = 0;\
  spinorFloat a12_re = 0; spinorFloat a12_im = 0;\
  \
  a00_re += c##00_00_re * reg##00_re;\
  a00_im += c##00_00_re * reg##00_im;\
  a00_re += c##00_01_re * reg##01_re;\
  a00_re -= c##00_01_im * reg##01_im;\
  a00_im += c##00_01_re * reg##01_im;\
  a00_im += c##00_01_im * reg##01_re;\
  a00_re += c##00_02_re * reg##02_re;\
  a00_re -= c##00_02_im * reg##02_im;\
  a00_im += c##00_02_re * reg##02_im;\
  a00_im += c##00_02_im * reg##02_re;\
  a00_re += c##00_10_re * reg##10_re;\
  a00_re -= c##00_10_im * reg##10_im;\
  a00_im += c##00_10_re * reg##10_im;\
  a00_im += c##00_10_im * reg##10_re;\
  a00_re += c##00_11_re * reg##11_re;\
  a00_re -= c##00_11_im * reg##11_im;\
  a00_im += c##00_11_re * reg##11_im;\
  a00_im += c##00_11_im * reg##11_re;\
  a00_re += c##00_12_re * reg##12_re;\
  a00_re -= c##00_12_im * reg##12_im;\
  a00_im += c##00_12_re * reg##12_im;\
  a00_im += c##00_12_im * reg##12_re;\
  \
  a01_re += c##01_00_re * reg##00_re;\
  a01_re -= c##01_00_im * reg##00_im;\
  a01_im += c##01_00_re * reg##00_im;\
  a01_im += c##01_00_im * reg##00_re;\
  a01_re += c##01_01_re * reg##01_re;\
  a01_im += c##01_01_re * reg##01_im;\
  a01_re += c##01_02_re * reg##02_re;\
  a01_re -= c##01_02_im * reg##02_im;\
  a01_im += c##01_02_re * reg##02_im;\
  a01_im += c##01_02_im * reg##02_re;\
  a01_re += c##01_10_re * reg##10_re;\
  a01_re -= c##01_10_im * reg##10_im;\
  a01_im += c##01_10_re * reg##10_im;\
  a01_im += c##01_10_im * reg##10_re;\
  a01_re += c##01_11_re * reg##11_re;\
  a01_re -= c##01_11_im * reg##11_im;\
  a01_im += c##01_11_re * reg##11_im;\
  a01_im += c##01_11_im * reg##11_re;\
  a01_re += c##01_12_re * reg##12_re;\
  a01_re -= c##01_12_im * reg##12_im;\
  a01_im += c##01_12_re * reg##12_im;\
  a01_im += c##01_12_im * reg##12_re;\
  \
  a02_re += c##02_00_re * reg##00_re;\
  a02_re -= c##02_00_im * reg##00_im;\
  a02_im += c##02_00_re * reg##00_im;\
  a02_im += c##02_00_im * reg##00_re;\
  a02_re += c##02_01_re * reg##01_re;\
  a02_re -= c##02_01_im * reg##01_im;\
  a02_im += c##02_01_re * reg##01_im;\
  a02_im += c##02_01_im * reg##01_re;\
  a02_re += c##02_02_re * reg##02_re;\
  a02_im += c##02_02_re * reg##02_im;\
  a02_re += c##02_10_re * reg##10_re;\
  a02_re -= c##02_10_im * reg##10_im;\
  a02_im += c##02_10_re * reg##10_im;\
  a02_im += c##02_10_im * reg##10_re;\
  a02_re += c##02_11_re * reg##11_re;\
  a02_re -= c##02_11_im * reg##11_im;\
  a02_im += c##02_11_re * reg##11_im;\
  a02_im += c##02_11_im * reg##11_re;\
  a02_re += c##02_12_re * reg##12_re;\
  a02_re -= c##02_12_im * reg##12_im;\
  a02_im += c##02_12_re * reg##12_im;\
  a02_im += c##02_12_im * reg##12_re;\
  \
  a10_re += c##10_00_re * reg##00_re;\
  a10_re -= c##10_00_im * reg##00_im;\
  a10_im += c##10_00_re * reg##00_im;\
  a10_im += c##10_00_im * reg##00_re;\
  a10_re += c##10_01_re * reg##01_re;\
  a10_re -= c##10_01_im * reg##01_im;\
  a10_im += c##10_01_re * reg##01_im;\
  a10_im += c##10_01_im * reg##01_re;\
  a10_re += c##10_02_re * reg##02_re;\
  a10_re -= c##10_02_im * reg##02_im;\
  a10_im += c##10_02_re * reg##02_im;\
  a10_im += c##10_02_im * reg##02_re;\
  a10_re += c##10_10_re * reg##10_re;\
  a10_im += c##10_10_re * reg##10_im;\
  a10_re += c##10_11_re * reg##11_re;\
  a10_re -= c##10_11_im * reg##11_im;\
  a10_im += c##10_11_re * reg##11_im;\
  a10_im += c##10_11_im * reg##11_re;\
  a10_re += c##10_12_re * reg##12_re;\
  a10_re -= c##10_12_im * reg##12_im;\
  a10_im += c##10_12_re * reg##12_im;\
  a10_im += c##10_12_im * reg##12_re;\
  \
  a11_re += c##11_00_re * reg##00_re;\
  a11_re -= c##11_00_im * reg##00_im;\
  a11_im += c##11_00_re * reg##00_im;\
  a11_im += c##11_00_im * reg##00_re;\
  a11_re += c##11_01_re * reg##01_re;\
  a11_re -= c##11_01_im * reg##01_im;\
  a11_im += c##11_01_re * reg##01_im;\
  a11_im += c##11_01_im * reg##01_re;\
  a11_re += c##11_02_re * reg##02_re;\
  a11_re -= c##11_02_im * reg##02_im;\
  a11_im += c##11_02_re * reg##02_im;\
  a11_im += c##11_02_im * reg##02_re;\
  a11_re += c##11_10_re * reg##10_re;\
  a11_re -= c##11_10_im * reg##10_im;\
  a11_im += c##11_10_re * reg##10_im;\
  a11_im += c##11_10_im * reg##10_re;\
  a11_re += c##11_11_re * reg##11_re;\
  a11_im += c##11_11_re * reg##11_im;\
  a11_re += c##11_12_re * reg##12_re;\
  a11_re -= c##11_12_im * reg##12_im;\
  a11_im += c##11_12_re * reg##12_im;\
  a11_im += c##11_12_im * reg##12_re;\
  \
  a12_re += c##12_00_re * reg##00_re;\
  a12_re -= c##12_00_im * reg##00_im;\
  a12_im += c##12_00_re * reg##00_im;\
  a12_im += c##12_00_im * reg##00_re;\
  a12_re += c##12_01_re * reg##01_re;\
  a12_re -= c##12_01_im * reg##01_im;\
  a12_im += c##12_01_re * reg##01_im;\
  a12_im += c##12_01_im * reg##01_re;\
  a12_re += c##12_02_re * reg##02_re;\
  a12_re -= c##12_02_im * reg##02_im;\
  a12_im += c##12_02_re * reg##02_im;\
  a12_im += c##12_02_im * reg##02_re;\
  a12_re += c##12_10_re * reg##10_re;\
  a12_re -= c##12_10_im * reg##10_im;\
  a12_im += c##12_10_re * reg##10_im;\
  a12_im += c##12_10_im * reg##10_re;\
  a12_re += c##12_11_re * reg##11_re;\
  a12_re -= c##12_11_im * reg##11_im;\
  a12_im += c##12_11_re * reg##11_im;\
  a12_im += c##12_11_im * reg##11_re;\
  a12_re += c##12_12_re * reg##12_re;\
  a12_im += c##12_12_re * reg##12_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a00_re = a00_re - .5*a* reg##00_im;  a00_im = a00_im + .5*a* reg##00_re;\
  a01_re = a01_re - .5*a* reg##01_im;  a01_im = a01_im + .5*a* reg##01_re;\
  a02_re = a02_re - .5*a* reg##02_im;  a02_im = a02_im + .5*a* reg##02_re;\
  a10_re = a10_re - .5*a* reg##10_im;  a10_im = a10_im + .5*a* reg##10_re;\
  a11_re = a11_re - .5*a* reg##11_im;  a11_im = a11_im + .5*a* reg##11_re;\
  a12_re = a12_re - .5*a* reg##12_im;  a12_im = a12_im + .5*a* reg##12_re;\
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  \
}\
\
/* apply second chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 1)\
  spinorFloat a20_re = 0; spinorFloat a20_im = 0;\
  spinorFloat a21_re = 0; spinorFloat a21_im = 0;\
  spinorFloat a22_re = 0; spinorFloat a22_im = 0;\
  spinorFloat a30_re = 0; spinorFloat a30_im = 0;\
  spinorFloat a31_re = 0; spinorFloat a31_im = 0;\
  spinorFloat a32_re = 0; spinorFloat a32_im = 0;\
  \
  a20_re += c##20_20_re * reg##20_re;\
  a20_im += c##20_20_re * reg##20_im;\
  a20_re += c##20_21_re * reg##21_re;\
  a20_re -= c##20_21_im * reg##21_im;\
  a20_im += c##20_21_re * reg##21_im;\
  a20_im += c##20_21_im * reg##21_re;\
  a20_re += c##20_22_re * reg##22_re;\
  a20_re -= c##20_22_im * reg##22_im;\
  a20_im += c##20_22_re * reg##22_im;\
  a20_im += c##20_22_im * reg##22_re;\
  a20_re += c##20_30_re * reg##30_re;\
  a20_re -= c##20_30_im * reg##30_im;\
  a20_im += c##20_30_re * reg##30_im;\
  a20_im += c##20_30_im * reg##30_re;\
  a20_re += c##20_31_re * reg##31_re;\
  a20_re -= c##20_31_im * reg##31_im;\
  a20_im += c##20_31_re * reg##31_im;\
  a20_im += c##20_31_im * reg##31_re;\
  a20_re += c##20_32_re * reg##32_re;\
  a20_re -= c##20_32_im * reg##32_im;\
  a20_im += c##20_32_re * reg##32_im;\
  a20_im += c##20_32_im * reg##32_re;\
  \
  a21_re += c##21_20_re * reg##20_re;\
  a21_re -= c##21_20_im * reg##20_im;\
  a21_im += c##21_20_re * reg##20_im;\
  a21_im += c##21_20_im * reg##20_re;\
  a21_re += c##21_21_re * reg##21_re;\
  a21_im += c##21_21_re * reg##21_im;\
  a21_re += c##21_22_re * reg##22_re;\
  a21_re -= c##21_22_im * reg##22_im;\
  a21_im += c##21_22_re * reg##22_im;\
  a21_im += c##21_22_im * reg##22_re;\
  a21_re += c##21_30_re * reg##30_re;\
  a21_re -= c##21_30_im * reg##30_im;\
  a21_im += c##21_30_re * reg##30_im;\
  a21_im += c##21_30_im * reg##30_re;\
  a21_re += c##21_31_re * reg##31_re;\
  a21_re -= c##21_31_im * reg##31_im;\
  a21_im += c##21_31_re * reg##31_im;\
  a21_im += c##21_31_im * reg##31_re;\
  a21_re += c##21_32_re * reg##32_re;\
  a21_re -= c##21_32_im * reg##32_im;\
  a21_im += c##21_32_re * reg##32_im;\
  a21_im += c##21_32_im * reg##32_re;\
  \
  a22_re += c##22_20_re * reg##20_re;\
  a22_re -= c##22_20_im * reg##20_im;\
  a22_im += c##22_20_re * reg##20_im;\
  a22_im += c##22_20_im * reg##20_re;\
  a22_re += c##22_21_re * reg##21_re;\
  a22_re -= c##22_21_im * reg##21_im;\
  a22_im += c##22_21_re * reg##21_im;\
  a22_im += c##22_21_im * reg##21_re;\
  a22_re += c##22_22_re * reg##22_re;\
  a22_im += c##22_22_re * reg##22_im;\
  a22_re += c##22_30_re * reg##30_re;\
  a22_re -= c##22_30_im * reg##30_im;\
  a22_im += c##22_30_re * reg##30_im;\
  a22_im += c##22_30_im * reg##30_re;\
  a22_re += c##22_31_re * reg##31_re;\
  a22_re -= c##22_31_im * reg##31_im;\
  a22_im += c##22_31_re * reg##31_im;\
  a22_im += c##22_31_im * reg##31_re;\
  a22_re += c##22_32_re * reg##32_re;\
  a22_re -= c##22_32_im * reg##32_im;\
  a22_im += c##22_32_re * reg##32_im;\
  a22_im += c##22_32_im * reg##32_re;\
  \
  a30_re += c##30_20_re * reg##20_re;\
  a30_re -= c##30_20_im * reg##20_im;\
  a30_im += c##30_20_re * reg##20_im;\
  a30_im += c##30_20_im * reg##20_re;\
  a30_re += c##30_21_re * reg##21_re;\
  a30_re -= c##30_21_im * reg##21_im;\
  a30_im += c##30_21_re * reg##21_im;\
  a30_im += c##30_21_im * reg##21_re;\
  a30_re += c##30_22_re * reg##22_re;\
  a30_re -= c##30_22_im * reg##22_im;\
  a30_im += c##30_22_re * reg##22_im;\
  a30_im += c##30_22_im * reg##22_re;\
  a30_re += c##30_30_re * reg##30_re;\
  a30_im += c##30_30_re * reg##30_im;\
  a30_re += c##30_31_re * reg##31_re;\
  a30_re -= c##30_31_im * reg##31_im;\
  a30_im += c##30_31_re * reg##31_im;\
  a30_im += c##30_31_im * reg##31_re;\
  a30_re += c##30_32_re * reg##32_re;\
  a30_re -= c##30_32_im * reg##32_im;\
  a30_im += c##30_32_re * reg##32_im;\
  a30_im += c##30_32_im * reg##32_re;\
  \
  a31_re += c##31_20_re * reg##20_re;\
  a31_re -= c##31_20_im * reg##20_im;\
  a31_im += c##31_20_re * reg##20_im;\
  a31_im += c##31_20_im * reg##20_re;\
  a31_re += c##31_21_re * reg##21_re;\
  a31_re -= c##31_21_im * reg##21_im;\
  a31_im += c##31_21_re * reg##21_im;\
  a31_im += c##31_21_im * reg##21_re;\
  a31_re += c##31_22_re * reg##22_re;\
  a31_re -= c##31_22_im * reg##22_im;\
  a31_im += c##31_22_re * reg##22_im;\
  a31_im += c##31_22_im * reg##22_re;\
  a31_re += c##31_30_re * reg##30_re;\
  a31_re -= c##31_30_im * reg##30_im;\
  a31_im += c##31_30_re * reg##30_im;\
  a31_im += c##31_30_im * reg##30_re;\
  a31_re += c##31_31_re * reg##31_re;\
  a31_im += c##31_31_re * reg##31_im;\
  a31_re += c##31_32_re * reg##32_re;\
  a31_re -= c##31_32_im * reg##32_im;\
  a31_im += c##31_32_re * reg##32_im;\
  a31_im += c##31_32_im * reg##32_re;\
  \
  a32_re += c##32_20_re * reg##20_re;\
  a32_re -= c##32_20_im * reg##20_im;\
  a32_im += c##32_20_re * reg##20_im;\
  a32_im += c##32_20_im * reg##20_re;\
  a32_re += c##32_21_re * reg##21_re;\
  a32_re -= c##32_21_im * reg##21_im;\
  a32_im += c##32_21_re * reg##21_im;\
  a32_im += c##32_21_im * reg##21_re;\
  a32_re += c##32_22_re * reg##22_re;\
  a32_re -= c##32_22_im * reg##22_im;\
  a32_im += c##32_22_re * reg##22_im;\
  a32_im += c##32_22_im * reg##22_re;\
  a32_re += c##32_30_re * reg##30_re;\
  a32_re -= c##32_30_im * reg##30_im;\
  a32_im += c##32_30_re * reg##30_im;\
  a32_im += c##32_30_im * reg##30_re;\
  a32_re += c##32_31_re * reg##31_re;\
  a32_re -= c##32_31_im * reg##31_im;\
  a32_im += c##32_31_re * reg##31_im;\
  a32_im += c##32_31_im * reg##31_re;\
  a32_re += c##32_32_re * reg##32_re;\
  a32_im += c##32_32_re * reg##32_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a20_re = a20_re + .5*a* reg##20_im;  a20_im = a20_im - .5*a* reg##20_re;\
  a21_re = a21_re + .5*a* reg##21_im;  a21_im = a21_im - .5*a* reg##21_re;\
  a22_re = a22_re + .5*a* reg##22_im;  a22_im = a22_im - .5*a* reg##22_re;\
  a30_re = a30_re + .5*a* reg##30_im;  a30_im = a30_im - .5*a* reg##30_re;\
  a31_re = a31_re + .5*a* reg##31_im;  a31_im = a31_im - .5*a* reg##31_re;\
  a32_re = a32_re + .5*a* reg##32_im;  a32_im = a32_im - .5*a* reg##32_re;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
  \
}\
\
/* change back from chiral basis*/\
/* (note: required factor of 1/2 is included in clover term normalization)*/\
{\
  spinorFloat a00_re =  reg##10_re + reg##30_re;\
  spinorFloat a00_im =  reg##10_im + reg##30_im;\
  spinorFloat a10_re = -reg##00_re - reg##20_re;\
  spinorFloat a10_im = -reg##00_im - reg##20_im;\
  spinorFloat a20_re =  reg##10_re - reg##30_re;\
  spinorFloat a20_im =  reg##10_im - reg##30_im;\
  spinorFloat a30_re = -reg##00_re + reg##20_re;\
  spinorFloat a30_im = -reg##00_im + reg##20_im;\
  \
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
}\
\
{\
  spinorFloat a01_re =  reg##11_re + reg##31_re;\
  spinorFloat a01_im =  reg##11_im + reg##31_im;\
  spinorFloat a11_re = -reg##01_re - reg##21_re;\
  spinorFloat a11_im = -reg##01_im - reg##21_im;\
  spinorFloat a21_re =  reg##11_re - reg##31_re;\
  spinorFloat a21_im =  reg##11_im - reg##31_im;\
  spinorFloat a31_re = -reg##01_re + reg##21_re;\
  spinorFloat a31_im = -reg##01_im + reg##21_im;\
  \
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
}\
\
{\
  spinorFloat a02_re =  reg##12_re + reg##32_re;\
  spinorFloat a02_im =  reg##12_im + reg##32_im;\
  spinorFloat a12_re = -reg##02_re - reg##22_re;\
  spinorFloat a12_im = -reg##02_im - reg##22_im;\
  spinorFloat a22_re =  reg##12_re - reg##32_re;\
  spinorFloat a22_im =  reg##12_im - reg##32_im;\
  spinorFloat a32_re = -reg##02_re + reg##22_re;\
  spinorFloat a32_im = -reg##02_im + reg##22_im;\
  \
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
\

#define APPLY_CLOVER_TWIST_INV(c, cinv, a, reg)\
\
/* change to chiral basis*/\
{\
  spinorFloat a00_re = -reg##10_re - reg##30_re;\
  spinorFloat a00_im = -reg##10_im - reg##30_im;\
  spinorFloat a10_re =  reg##00_re + reg##20_re;\
  spinorFloat a10_im =  reg##00_im + reg##20_im;\
  spinorFloat a20_re = -reg##10_re + reg##30_re;\
  spinorFloat a20_im = -reg##10_im + reg##30_im;\
  spinorFloat a30_re =  reg##00_re - reg##20_re;\
  spinorFloat a30_im =  reg##00_im - reg##20_im;\
  \
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
}\
\
{\
  spinorFloat a01_re = -reg##11_re - reg##31_re;\
  spinorFloat a01_im = -reg##11_im - reg##31_im;\
  spinorFloat a11_re =  reg##01_re + reg##21_re;\
  spinorFloat a11_im =  reg##01_im + reg##21_im;\
  spinorFloat a21_re = -reg##11_re + reg##31_re;\
  spinorFloat a21_im = -reg##11_im + reg##31_im;\
  spinorFloat a31_re =  reg##01_re - reg##21_re;\
  spinorFloat a31_im =  reg##01_im - reg##21_im;\
  \
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
}\
\
{\
  spinorFloat a02_re = -reg##12_re - reg##32_re;\
  spinorFloat a02_im = -reg##12_im - reg##32_im;\
  spinorFloat a12_re =  reg##02_re + reg##22_re;\
  spinorFloat a12_im =  reg##02_im + reg##22_im;\
  spinorFloat a22_re = -reg##12_re + reg##32_re;\
  spinorFloat a22_im = -reg##12_im + reg##32_im;\
  spinorFloat a32_re =  reg##02_re - reg##22_re;\
  spinorFloat a32_im =  reg##02_im - reg##22_im;\
  \
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
\
/* apply first chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 0)\
  spinorFloat a00_re = 0; spinorFloat a00_im = 0;\
  spinorFloat a01_re = 0; spinorFloat a01_im = 0;\
  spinorFloat a02_re = 0; spinorFloat a02_im = 0;\
  spinorFloat a10_re = 0; spinorFloat a10_im = 0;\
  spinorFloat a11_re = 0; spinorFloat a11_im = 0;\
  spinorFloat a12_re = 0; spinorFloat a12_im = 0;\
  \
  a00_re += c##00_00_re * reg##00_re;\
  a00_im += c##00_00_re * reg##00_im;\
  a00_re += c##00_01_re * reg##01_re;\
  a00_re -= c##00_01_im * reg##01_im;\
  a00_im += c##00_01_re * reg##01_im;\
  a00_im += c##00_01_im * reg##01_re;\
  a00_re += c##00_02_re * reg##02_re;\
  a00_re -= c##00_02_im * reg##02_im;\
  a00_im += c##00_02_re * reg##02_im;\
  a00_im += c##00_02_im * reg##02_re;\
  a00_re += c##00_10_re * reg##10_re;\
  a00_re -= c##00_10_im * reg##10_im;\
  a00_im += c##00_10_re * reg##10_im;\
  a00_im += c##00_10_im * reg##10_re;\
  a00_re += c##00_11_re * reg##11_re;\
  a00_re -= c##00_11_im * reg##11_im;\
  a00_im += c##00_11_re * reg##11_im;\
  a00_im += c##00_11_im * reg##11_re;\
  a00_re += c##00_12_re * reg##12_re;\
  a00_re -= c##00_12_im * reg##12_im;\
  a00_im += c##00_12_re * reg##12_im;\
  a00_im += c##00_12_im * reg##12_re;\
  \
  a01_re += c##01_00_re * reg##00_re;\
  a01_re -= c##01_00_im * reg##00_im;\
  a01_im += c##01_00_re * reg##00_im;\
  a01_im += c##01_00_im * reg##00_re;\
  a01_re += c##01_01_re * reg##01_re;\
  a01_im += c##01_01_re * reg##01_im;\
  a01_re += c##01_02_re * reg##02_re;\
  a01_re -= c##01_02_im * reg##02_im;\
  a01_im += c##01_02_re * reg##02_im;\
  a01_im += c##01_02_im * reg##02_re;\
  a01_re += c##01_10_re * reg##10_re;\
  a01_re -= c##01_10_im * reg##10_im;\
  a01_im += c##01_10_re * reg##10_im;\
  a01_im += c##01_10_im * reg##10_re;\
  a01_re += c##01_11_re * reg##11_re;\
  a01_re -= c##01_11_im * reg##11_im;\
  a01_im += c##01_11_re * reg##11_im;\
  a01_im += c##01_11_im * reg##11_re;\
  a01_re += c##01_12_re * reg##12_re;\
  a01_re -= c##01_12_im * reg##12_im;\
  a01_im += c##01_12_re * reg##12_im;\
  a01_im += c##01_12_im * reg##12_re;\
  \
  a02_re += c##02_00_re * reg##00_re;\
  a02_re -= c##02_00_im * reg##00_im;\
  a02_im += c##02_00_re * reg##00_im;\
  a02_im += c##02_00_im * reg##00_re;\
  a02_re += c##02_01_re * reg##01_re;\
  a02_re -= c##02_01_im * reg##01_im;\
  a02_im += c##02_01_re * reg##01_im;\
  a02_im += c##02_01_im * reg##01_re;\
  a02_re += c##02_02_re * reg##02_re;\
  a02_im += c##02_02_re * reg##02_im;\
  a02_re += c##02_10_re * reg##10_re;\
  a02_re -= c##02_10_im * reg##10_im;\
  a02_im += c##02_10_re * reg##10_im;\
  a02_im += c##02_10_im * reg##10_re;\
  a02_re += c##02_11_re * reg##11_re;\
  a02_re -= c##02_11_im * reg##11_im;\
  a02_im += c##02_11_re * reg##11_im;\
  a02_im += c##02_11_im * reg##11_re;\
  a02_re += c##02_12_re * reg##12_re;\
  a02_re -= c##02_12_im * reg##12_im;\
  a02_im += c##02_12_re * reg##12_im;\
  a02_im += c##02_12_im * reg##12_re;\
  \
  a10_re += c##10_00_re * reg##00_re;\
  a10_re -= c##10_00_im * reg##00_im;\
  a10_im += c##10_00_re * reg##00_im;\
  a10_im += c##10_00_im * reg##00_re;\
  a10_re += c##10_01_re * reg##01_re;\
  a10_re -= c##10_01_im * reg##01_im;\
  a10_im += c##10_01_re * reg##01_im;\
  a10_im += c##10_01_im * reg##01_re;\
  a10_re += c##10_02_re * reg##02_re;\
  a10_re -= c##10_02_im * reg##02_im;\
  a10_im += c##10_02_re * reg##02_im;\
  a10_im += c##10_02_im * reg##02_re;\
  a10_re += c##10_10_re * reg##10_re;\
  a10_im += c##10_10_re * reg##10_im;\
  a10_re += c##10_11_re * reg##11_re;\
  a10_re -= c##10_11_im * reg##11_im;\
  a10_im += c##10_11_re * reg##11_im;\
  a10_im += c##10_11_im * reg##11_re;\
  a10_re += c##10_12_re * reg##12_re;\
  a10_re -= c##10_12_im * reg##12_im;\
  a10_im += c##10_12_re * reg##12_im;\
  a10_im += c##10_12_im * reg##12_re;\
  \
  a11_re += c##11_00_re * reg##00_re;\
  a11_re -= c##11_00_im * reg##00_im;\
  a11_im += c##11_00_re * reg##00_im;\
  a11_im += c##11_00_im * reg##00_re;\
  a11_re += c##11_01_re * reg##01_re;\
  a11_re -= c##11_01_im * reg##01_im;\
  a11_im += c##11_01_re * reg##01_im;\
  a11_im += c##11_01_im * reg##01_re;\
  a11_re += c##11_02_re * reg##02_re;\
  a11_re -= c##11_02_im * reg##02_im;\
  a11_im += c##11_02_re * reg##02_im;\
  a11_im += c##11_02_im * reg##02_re;\
  a11_re += c##11_10_re * reg##10_re;\
  a11_re -= c##11_10_im * reg##10_im;\
  a11_im += c##11_10_re * reg##10_im;\
  a11_im += c##11_10_im * reg##10_re;\
  a11_re += c##11_11_re * reg##11_re;\
  a11_im += c##11_11_re * reg##11_im;\
  a11_re += c##11_12_re * reg##12_re;\
  a11_re -= c##11_12_im * reg##12_im;\
  a11_im += c##11_12_re * reg##12_im;\
  a11_im += c##11_12_im * reg##12_re;\
  \
  a12_re += c##12_00_re * reg##00_re;\
  a12_re -= c##12_00_im * reg##00_im;\
  a12_im += c##12_00_re * reg##00_im;\
  a12_im += c##12_00_im * reg##00_re;\
  a12_re += c##12_01_re * reg##01_re;\
  a12_re -= c##12_01_im * reg##01_im;\
  a12_im += c##12_01_re * reg##01_im;\
  a12_im += c##12_01_im * reg##01_re;\
  a12_re += c##12_02_re * reg##02_re;\
  a12_re -= c##12_02_im * reg##02_im;\
  a12_im += c##12_02_re * reg##02_im;\
  a12_im += c##12_02_im * reg##02_re;\
  a12_re += c##12_10_re * reg##10_re;\
  a12_re -= c##12_10_im * reg##10_im;\
  a12_im += c##12_10_re * reg##10_im;\
  a12_im += c##12_10_im * reg##10_re;\
  a12_re += c##12_11_re * reg##11_re;\
  a12_re -= c##12_11_im * reg##11_im;\
  a12_im += c##12_11_re * reg##11_im;\
  a12_im += c##12_11_im * reg##11_re;\
  a12_re += c##12_12_re * reg##12_re;\
  a12_im += c##12_12_re * reg##12_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a00_re = a00_re - .5*a* reg##00_im;  a00_im = a00_im + .5*a* reg##00_re;\
  a01_re = a01_re - .5*a* reg##01_im;  a01_im = a01_im + .5*a* reg##01_re;\
  a02_re = a02_re - .5*a* reg##02_im;  a02_im = a02_im + .5*a* reg##02_re;\
  a10_re = a10_re - .5*a* reg##10_im;  a10_im = a10_im + .5*a* reg##10_re;\
  a11_re = a11_re - .5*a* reg##11_im;  a11_im = a11_im + .5*a* reg##11_re;\
  a12_re = a12_re - .5*a* reg##12_im;  a12_im = a12_im + .5*a* reg##12_re;\
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
}\
/*Apply inverse clover*/\
{\
  ASSN_CLOVER(TM_INV_CLOVERTEX, 0) \
  spinorFloat a00_re = 0; spinorFloat a00_im = 0;\
  spinorFloat a01_re = 0; spinorFloat a01_im = 0;\
  spinorFloat a02_re = 0; spinorFloat a02_im = 0;\
  spinorFloat a10_re = 0; spinorFloat a10_im = 0;\
  spinorFloat a11_re = 0; spinorFloat a11_im = 0;\
  spinorFloat a12_re = 0; spinorFloat a12_im = 0;\
  \
  a00_re += cinv##00_00_re * reg##00_re;\
  a00_im += cinv##00_00_re * reg##00_im;\
  a00_re += cinv##00_01_re * reg##01_re;\
  a00_re -= cinv##00_01_im * reg##01_im;\
  a00_im += cinv##00_01_re * reg##01_im;\
  a00_im += cinv##00_01_im * reg##01_re;\
  a00_re += cinv##00_02_re * reg##02_re;\
  a00_re -= cinv##00_02_im * reg##02_im;\
  a00_im += cinv##00_02_re * reg##02_im;\
  a00_im += cinv##00_02_im * reg##02_re;\
  a00_re += cinv##00_10_re * reg##10_re;\
  a00_re -= cinv##00_10_im * reg##10_im;\
  a00_im += cinv##00_10_re * reg##10_im;\
  a00_im += cinv##00_10_im * reg##10_re;\
  a00_re += cinv##00_11_re * reg##11_re;\
  a00_re -= cinv##00_11_im * reg##11_im;\
  a00_im += cinv##00_11_re * reg##11_im;\
  a00_im += cinv##00_11_im * reg##11_re;\
  a00_re += cinv##00_12_re * reg##12_re;\
  a00_re -= cinv##00_12_im * reg##12_im;\
  a00_im += cinv##00_12_re * reg##12_im;\
  a00_im += cinv##00_12_im * reg##12_re;\
  \
  a01_re += cinv##01_00_re * reg##00_re;\
  a01_re -= cinv##01_00_im * reg##00_im;\
  a01_im += cinv##01_00_re * reg##00_im;\
  a01_im += cinv##01_00_im * reg##00_re;\
  a01_re += cinv##01_01_re * reg##01_re;\
  a01_im += cinv##01_01_re * reg##01_im;\
  a01_re += cinv##01_02_re * reg##02_re;\
  a01_re -= cinv##01_02_im * reg##02_im;\
  a01_im += cinv##01_02_re * reg##02_im;\
  a01_im += cinv##01_02_im * reg##02_re;\
  a01_re += cinv##01_10_re * reg##10_re;\
  a01_re -= cinv##01_10_im * reg##10_im;\
  a01_im += cinv##01_10_re * reg##10_im;\
  a01_im += cinv##01_10_im * reg##10_re;\
  a01_re += cinv##01_11_re * reg##11_re;\
  a01_re -= cinv##01_11_im * reg##11_im;\
  a01_im += cinv##01_11_re * reg##11_im;\
  a01_im += cinv##01_11_im * reg##11_re;\
  a01_re += cinv##01_12_re * reg##12_re;\
  a01_re -= cinv##01_12_im * reg##12_im;\
  a01_im += cinv##01_12_re * reg##12_im;\
  a01_im += cinv##01_12_im * reg##12_re;\
  \
  a02_re += cinv##02_00_re * reg##00_re;\
  a02_re -= cinv##02_00_im * reg##00_im;\
  a02_im += cinv##02_00_re * reg##00_im;\
  a02_im += cinv##02_00_im * reg##00_re;\
  a02_re += cinv##02_01_re * reg##01_re;\
  a02_re -= cinv##02_01_im * reg##01_im;\
  a02_im += cinv##02_01_re * reg##01_im;\
  a02_im += cinv##02_01_im * reg##01_re;\
  a02_re += cinv##02_02_re * reg##02_re;\
  a02_im += cinv##02_02_re * reg##02_im;\
  a02_re += cinv##02_10_re * reg##10_re;\
  a02_re -= cinv##02_10_im * reg##10_im;\
  a02_im += cinv##02_10_re * reg##10_im;\
  a02_im += cinv##02_10_im * reg##10_re;\
  a02_re += cinv##02_11_re * reg##11_re;\
  a02_re -= cinv##02_11_im * reg##11_im;\
  a02_im += cinv##02_11_re * reg##11_im;\
  a02_im += cinv##02_11_im * reg##11_re;\
  a02_re += cinv##02_12_re * reg##12_re;\
  a02_re -= cinv##02_12_im * reg##12_im;\
  a02_im += cinv##02_12_re * reg##12_im;\
  a02_im += cinv##02_12_im * reg##12_re;\
  \
  a10_re += cinv##10_00_re * reg##00_re;\
  a10_re -= cinv##10_00_im * reg##00_im;\
  a10_im += cinv##10_00_re * reg##00_im;\
  a10_im += cinv##10_00_im * reg##00_re;\
  a10_re += cinv##10_01_re * reg##01_re;\
  a10_re -= cinv##10_01_im * reg##01_im;\
  a10_im += cinv##10_01_re * reg##01_im;\
  a10_im += cinv##10_01_im * reg##01_re;\
  a10_re += cinv##10_02_re * reg##02_re;\
  a10_re -= cinv##10_02_im * reg##02_im;\
  a10_im += cinv##10_02_re * reg##02_im;\
  a10_im += cinv##10_02_im * reg##02_re;\
  a10_re += cinv##10_10_re * reg##10_re;\
  a10_im += cinv##10_10_re * reg##10_im;\
  a10_re += cinv##10_11_re * reg##11_re;\
  a10_re -= cinv##10_11_im * reg##11_im;\
  a10_im += cinv##10_11_re * reg##11_im;\
  a10_im += cinv##10_11_im * reg##11_re;\
  a10_re += cinv##10_12_re * reg##12_re;\
  a10_re -= cinv##10_12_im * reg##12_im;\
  a10_im += cinv##10_12_re * reg##12_im;\
  a10_im += cinv##10_12_im * reg##12_re;\
  \
  a11_re += cinv##11_00_re * reg##00_re;\
  a11_re -= cinv##11_00_im * reg##00_im;\
  a11_im += cinv##11_00_re * reg##00_im;\
  a11_im += cinv##11_00_im * reg##00_re;\
  a11_re += cinv##11_01_re * reg##01_re;\
  a11_re -= cinv##11_01_im * reg##01_im;\
  a11_im += cinv##11_01_re * reg##01_im;\
  a11_im += cinv##11_01_im * reg##01_re;\
  a11_re += cinv##11_02_re * reg##02_re;\
  a11_re -= cinv##11_02_im * reg##02_im;\
  a11_im += cinv##11_02_re * reg##02_im;\
  a11_im += cinv##11_02_im * reg##02_re;\
  a11_re += cinv##11_10_re * reg##10_re;\
  a11_re -= cinv##11_10_im * reg##10_im;\
  a11_im += cinv##11_10_re * reg##10_im;\
  a11_im += cinv##11_10_im * reg##10_re;\
  a11_re += cinv##11_11_re * reg##11_re;\
  a11_im += cinv##11_11_re * reg##11_im;\
  a11_re += cinv##11_12_re * reg##12_re;\
  a11_re -= cinv##11_12_im * reg##12_im;\
  a11_im += cinv##11_12_re * reg##12_im;\
  a11_im += cinv##11_12_im * reg##12_re;\
  \
  a12_re += cinv##12_00_re * reg##00_re;\
  a12_re -= cinv##12_00_im * reg##00_im;\
  a12_im += cinv##12_00_re * reg##00_im;\
  a12_im += cinv##12_00_im * reg##00_re;\
  a12_re += cinv##12_01_re * reg##01_re;\
  a12_re -= cinv##12_01_im * reg##01_im;\
  a12_im += cinv##12_01_re * reg##01_im;\
  a12_im += cinv##12_01_im * reg##01_re;\
  a12_re += cinv##12_02_re * reg##02_re;\
  a12_re -= cinv##12_02_im * reg##02_im;\
  a12_im += cinv##12_02_re * reg##02_im;\
  a12_im += cinv##12_02_im * reg##02_re;\
  a12_re += cinv##12_10_re * reg##10_re;\
  a12_re -= cinv##12_10_im * reg##10_im;\
  a12_im += cinv##12_10_re * reg##10_im;\
  a12_im += cinv##12_10_im * reg##10_re;\
  a12_re += cinv##12_11_re * reg##11_re;\
  a12_re -= cinv##12_11_im * reg##11_im;\
  a12_im += cinv##12_11_re * reg##11_im;\
  a12_im += cinv##12_11_im * reg##11_re;\
  a12_re += cinv##12_12_re * reg##12_re;\
  a12_im += cinv##12_12_re * reg##12_im;\
  \
  /*store  the result*/\
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  \
}\
\
/* apply second chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 1)\
  spinorFloat a20_re = 0; spinorFloat a20_im = 0;\
  spinorFloat a21_re = 0; spinorFloat a21_im = 0;\
  spinorFloat a22_re = 0; spinorFloat a22_im = 0;\
  spinorFloat a30_re = 0; spinorFloat a30_im = 0;\
  spinorFloat a31_re = 0; spinorFloat a31_im = 0;\
  spinorFloat a32_re = 0; spinorFloat a32_im = 0;\
  \
  a20_re += c##20_20_re * reg##20_re;\
  a20_im += c##20_20_re * reg##20_im;\
  a20_re += c##20_21_re * reg##21_re;\
  a20_re -= c##20_21_im * reg##21_im;\
  a20_im += c##20_21_re * reg##21_im;\
  a20_im += c##20_21_im * reg##21_re;\
  a20_re += c##20_22_re * reg##22_re;\
  a20_re -= c##20_22_im * reg##22_im;\
  a20_im += c##20_22_re * reg##22_im;\
  a20_im += c##20_22_im * reg##22_re;\
  a20_re += c##20_30_re * reg##30_re;\
  a20_re -= c##20_30_im * reg##30_im;\
  a20_im += c##20_30_re * reg##30_im;\
  a20_im += c##20_30_im * reg##30_re;\
  a20_re += c##20_31_re * reg##31_re;\
  a20_re -= c##20_31_im * reg##31_im;\
  a20_im += c##20_31_re * reg##31_im;\
  a20_im += c##20_31_im * reg##31_re;\
  a20_re += c##20_32_re * reg##32_re;\
  a20_re -= c##20_32_im * reg##32_im;\
  a20_im += c##20_32_re * reg##32_im;\
  a20_im += c##20_32_im * reg##32_re;\
  \
  a21_re += c##21_20_re * reg##20_re;\
  a21_re -= c##21_20_im * reg##20_im;\
  a21_im += c##21_20_re * reg##20_im;\
  a21_im += c##21_20_im * reg##20_re;\
  a21_re += c##21_21_re * reg##21_re;\
  a21_im += c##21_21_re * reg##21_im;\
  a21_re += c##21_22_re * reg##22_re;\
  a21_re -= c##21_22_im * reg##22_im;\
  a21_im += c##21_22_re * reg##22_im;\
  a21_im += c##21_22_im * reg##22_re;\
  a21_re += c##21_30_re * reg##30_re;\
  a21_re -= c##21_30_im * reg##30_im;\
  a21_im += c##21_30_re * reg##30_im;\
  a21_im += c##21_30_im * reg##30_re;\
  a21_re += c##21_31_re * reg##31_re;\
  a21_re -= c##21_31_im * reg##31_im;\
  a21_im += c##21_31_re * reg##31_im;\
  a21_im += c##21_31_im * reg##31_re;\
  a21_re += c##21_32_re * reg##32_re;\
  a21_re -= c##21_32_im * reg##32_im;\
  a21_im += c##21_32_re * reg##32_im;\
  a21_im += c##21_32_im * reg##32_re;\
  \
  a22_re += c##22_20_re * reg##20_re;\
  a22_re -= c##22_20_im * reg##20_im;\
  a22_im += c##22_20_re * reg##20_im;\
  a22_im += c##22_20_im * reg##20_re;\
  a22_re += c##22_21_re * reg##21_re;\
  a22_re -= c##22_21_im * reg##21_im;\
  a22_im += c##22_21_re * reg##21_im;\
  a22_im += c##22_21_im * reg##21_re;\
  a22_re += c##22_22_re * reg##22_re;\
  a22_im += c##22_22_re * reg##22_im;\
  a22_re += c##22_30_re * reg##30_re;\
  a22_re -= c##22_30_im * reg##30_im;\
  a22_im += c##22_30_re * reg##30_im;\
  a22_im += c##22_30_im * reg##30_re;\
  a22_re += c##22_31_re * reg##31_re;\
  a22_re -= c##22_31_im * reg##31_im;\
  a22_im += c##22_31_re * reg##31_im;\
  a22_im += c##22_31_im * reg##31_re;\
  a22_re += c##22_32_re * reg##32_re;\
  a22_re -= c##22_32_im * reg##32_im;\
  a22_im += c##22_32_re * reg##32_im;\
  a22_im += c##22_32_im * reg##32_re;\
  \
  a30_re += c##30_20_re * reg##20_re;\
  a30_re -= c##30_20_im * reg##20_im;\
  a30_im += c##30_20_re * reg##20_im;\
  a30_im += c##30_20_im * reg##20_re;\
  a30_re += c##30_21_re * reg##21_re;\
  a30_re -= c##30_21_im * reg##21_im;\
  a30_im += c##30_21_re * reg##21_im;\
  a30_im += c##30_21_im * reg##21_re;\
  a30_re += c##30_22_re * reg##22_re;\
  a30_re -= c##30_22_im * reg##22_im;\
  a30_im += c##30_22_re * reg##22_im;\
  a30_im += c##30_22_im * reg##22_re;\
  a30_re += c##30_30_re * reg##30_re;\
  a30_im += c##30_30_re * reg##30_im;\
  a30_re += c##30_31_re * reg##31_re;\
  a30_re -= c##30_31_im * reg##31_im;\
  a30_im += c##30_31_re * reg##31_im;\
  a30_im += c##30_31_im * reg##31_re;\
  a30_re += c##30_32_re * reg##32_re;\
  a30_re -= c##30_32_im * reg##32_im;\
  a30_im += c##30_32_re * reg##32_im;\
  a30_im += c##30_32_im * reg##32_re;\
  \
  a31_re += c##31_20_re * reg##20_re;\
  a31_re -= c##31_20_im * reg##20_im;\
  a31_im += c##31_20_re * reg##20_im;\
  a31_im += c##31_20_im * reg##20_re;\
  a31_re += c##31_21_re * reg##21_re;\
  a31_re -= c##31_21_im * reg##21_im;\
  a31_im += c##31_21_re * reg##21_im;\
  a31_im += c##31_21_im * reg##21_re;\
  a31_re += c##31_22_re * reg##22_re;\
  a31_re -= c##31_22_im * reg##22_im;\
  a31_im += c##31_22_re * reg##22_im;\
  a31_im += c##31_22_im * reg##22_re;\
  a31_re += c##31_30_re * reg##30_re;\
  a31_re -= c##31_30_im * reg##30_im;\
  a31_im += c##31_30_re * reg##30_im;\
  a31_im += c##31_30_im * reg##30_re;\
  a31_re += c##31_31_re * reg##31_re;\
  a31_im += c##31_31_re * reg##31_im;\
  a31_re += c##31_32_re * reg##32_re;\
  a31_re -= c##31_32_im * reg##32_im;\
  a31_im += c##31_32_re * reg##32_im;\
  a31_im += c##31_32_im * reg##32_re;\
  \
  a32_re += c##32_20_re * reg##20_re;\
  a32_re -= c##32_20_im * reg##20_im;\
  a32_im += c##32_20_re * reg##20_im;\
  a32_im += c##32_20_im * reg##20_re;\
  a32_re += c##32_21_re * reg##21_re;\
  a32_re -= c##32_21_im * reg##21_im;\
  a32_im += c##32_21_re * reg##21_im;\
  a32_im += c##32_21_im * reg##21_re;\
  a32_re += c##32_22_re * reg##22_re;\
  a32_re -= c##32_22_im * reg##22_im;\
  a32_im += c##32_22_re * reg##22_im;\
  a32_im += c##32_22_im * reg##22_re;\
  a32_re += c##32_30_re * reg##30_re;\
  a32_re -= c##32_30_im * reg##30_im;\
  a32_im += c##32_30_re * reg##30_im;\
  a32_im += c##32_30_im * reg##30_re;\
  a32_re += c##32_31_re * reg##31_re;\
  a32_re -= c##32_31_im * reg##31_im;\
  a32_im += c##32_31_re * reg##31_im;\
  a32_im += c##32_31_im * reg##31_re;\
  a32_re += c##32_32_re * reg##32_re;\
  a32_im += c##32_32_re * reg##32_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a20_re = a20_re + .5*a* reg##20_im;  a20_im = a20_im - .5*a* reg##20_re;\
  a21_re = a21_re + .5*a* reg##21_im;  a21_im = a21_im - .5*a* reg##21_re;\
  a22_re = a22_re + .5*a* reg##22_im;  a22_im = a22_im - .5*a* reg##22_re;\
  a30_re = a30_re + .5*a* reg##30_im;  a30_im = a30_im - .5*a* reg##30_re;\
  a31_re = a31_re + .5*a* reg##31_im;  a31_im = a31_im - .5*a* reg##31_re;\
  a32_re = a32_re + .5*a* reg##32_im;  a32_im = a32_im - .5*a* reg##32_re;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
/*Apply inverse clover*/\
{\
  ASSN_CLOVER(TM_INV_CLOVERTEX, 1) \
  spinorFloat a20_re = 0; spinorFloat a20_im = 0;\
  spinorFloat a21_re = 0; spinorFloat a21_im = 0;\
  spinorFloat a22_re = 0; spinorFloat a22_im = 0;\
  spinorFloat a30_re = 0; spinorFloat a30_im = 0;\
  spinorFloat a31_re = 0; spinorFloat a31_im = 0;\
  spinorFloat a32_re = 0; spinorFloat a32_im = 0;\
  \
  a20_re += cinv##20_20_re * reg##20_re;\
  a20_im += cinv##20_20_re * reg##20_im;\
  a20_re += cinv##20_21_re * reg##21_re;\
  a20_re -= cinv##20_21_im * reg##21_im;\
  a20_im += cinv##20_21_re * reg##21_im;\
  a20_im += cinv##20_21_im * reg##21_re;\
  a20_re += cinv##20_22_re * reg##22_re;\
  a20_re -= cinv##20_22_im * reg##22_im;\
  a20_im += cinv##20_22_re * reg##22_im;\
  a20_im += cinv##20_22_im * reg##22_re;\
  a20_re += cinv##20_30_re * reg##30_re;\
  a20_re -= cinv##20_30_im * reg##30_im;\
  a20_im += cinv##20_30_re * reg##30_im;\
  a20_im += cinv##20_30_im * reg##30_re;\
  a20_re += cinv##20_31_re * reg##31_re;\
  a20_re -= cinv##20_31_im * reg##31_im;\
  a20_im += cinv##20_31_re * reg##31_im;\
  a20_im += cinv##20_31_im * reg##31_re;\
  a20_re += cinv##20_32_re * reg##32_re;\
  a20_re -= cinv##20_32_im * reg##32_im;\
  a20_im += cinv##20_32_re * reg##32_im;\
  a20_im += cinv##20_32_im * reg##32_re;\
  \
  a21_re += cinv##21_20_re * reg##20_re;\
  a21_re -= cinv##21_20_im * reg##20_im;\
  a21_im += cinv##21_20_re * reg##20_im;\
  a21_im += cinv##21_20_im * reg##20_re;\
  a21_re += cinv##21_21_re * reg##21_re;\
  a21_im += cinv##21_21_re * reg##21_im;\
  a21_re += cinv##21_22_re * reg##22_re;\
  a21_re -= cinv##21_22_im * reg##22_im;\
  a21_im += cinv##21_22_re * reg##22_im;\
  a21_im += cinv##21_22_im * reg##22_re;\
  a21_re += cinv##21_30_re * reg##30_re;\
  a21_re -= cinv##21_30_im * reg##30_im;\
  a21_im += cinv##21_30_re * reg##30_im;\
  a21_im += cinv##21_30_im * reg##30_re;\
  a21_re += cinv##21_31_re * reg##31_re;\
  a21_re -= cinv##21_31_im * reg##31_im;\
  a21_im += cinv##21_31_re * reg##31_im;\
  a21_im += cinv##21_31_im * reg##31_re;\
  a21_re += cinv##21_32_re * reg##32_re;\
  a21_re -= cinv##21_32_im * reg##32_im;\
  a21_im += cinv##21_32_re * reg##32_im;\
  a21_im += cinv##21_32_im * reg##32_re;\
  \
  a22_re += cinv##22_20_re * reg##20_re;\
  a22_re -= cinv##22_20_im * reg##20_im;\
  a22_im += cinv##22_20_re * reg##20_im;\
  a22_im += cinv##22_20_im * reg##20_re;\
  a22_re += cinv##22_21_re * reg##21_re;\
  a22_re -= cinv##22_21_im * reg##21_im;\
  a22_im += cinv##22_21_re * reg##21_im;\
  a22_im += cinv##22_21_im * reg##21_re;\
  a22_re += cinv##22_22_re * reg##22_re;\
  a22_im += cinv##22_22_re * reg##22_im;\
  a22_re += cinv##22_30_re * reg##30_re;\
  a22_re -= cinv##22_30_im * reg##30_im;\
  a22_im += cinv##22_30_re * reg##30_im;\
  a22_im += cinv##22_30_im * reg##30_re;\
  a22_re += cinv##22_31_re * reg##31_re;\
  a22_re -= cinv##22_31_im * reg##31_im;\
  a22_im += cinv##22_31_re * reg##31_im;\
  a22_im += cinv##22_31_im * reg##31_re;\
  a22_re += cinv##22_32_re * reg##32_re;\
  a22_re -= cinv##22_32_im * reg##32_im;\
  a22_im += cinv##22_32_re * reg##32_im;\
  a22_im += cinv##22_32_im * reg##32_re;\
  \
  a30_re += cinv##30_20_re * reg##20_re;\
  a30_re -= cinv##30_20_im * reg##20_im;\
  a30_im += cinv##30_20_re * reg##20_im;\
  a30_im += cinv##30_20_im * reg##20_re;\
  a30_re += cinv##30_21_re * reg##21_re;\
  a30_re -= cinv##30_21_im * reg##21_im;\
  a30_im += cinv##30_21_re * reg##21_im;\
  a30_im += cinv##30_21_im * reg##21_re;\
  a30_re += cinv##30_22_re * reg##22_re;\
  a30_re -= cinv##30_22_im * reg##22_im;\
  a30_im += cinv##30_22_re * reg##22_im;\
  a30_im += cinv##30_22_im * reg##22_re;\
  a30_re += cinv##30_30_re * reg##30_re;\
  a30_im += cinv##30_30_re * reg##30_im;\
  a30_re += cinv##30_31_re * reg##31_re;\
  a30_re -= cinv##30_31_im * reg##31_im;\
  a30_im += cinv##30_31_re * reg##31_im;\
  a30_im += cinv##30_31_im * reg##31_re;\
  a30_re += cinv##30_32_re * reg##32_re;\
  a30_re -= cinv##30_32_im * reg##32_im;\
  a30_im += cinv##30_32_re * reg##32_im;\
  a30_im += cinv##30_32_im * reg##32_re;\
  \
  a31_re += cinv##31_20_re * reg##20_re;\
  a31_re -= cinv##31_20_im * reg##20_im;\
  a31_im += cinv##31_20_re * reg##20_im;\
  a31_im += cinv##31_20_im * reg##20_re;\
  a31_re += cinv##31_21_re * reg##21_re;\
  a31_re -= cinv##31_21_im * reg##21_im;\
  a31_im += cinv##31_21_re * reg##21_im;\
  a31_im += cinv##31_21_im * reg##21_re;\
  a31_re += cinv##31_22_re * reg##22_re;\
  a31_re -= cinv##31_22_im * reg##22_im;\
  a31_im += cinv##31_22_re * reg##22_im;\
  a31_im += cinv##31_22_im * reg##22_re;\
  a31_re += cinv##31_30_re * reg##30_re;\
  a31_re -= cinv##31_30_im * reg##30_im;\
  a31_im += cinv##31_30_re * reg##30_im;\
  a31_im += cinv##31_30_im * reg##30_re;\
  a31_re += cinv##31_31_re * reg##31_re;\
  a31_im += cinv##31_31_re * reg##31_im;\
  a31_re += cinv##31_32_re * reg##32_re;\
  a31_re -= cinv##31_32_im * reg##32_im;\
  a31_im += cinv##31_32_re * reg##32_im;\
  a31_im += cinv##31_32_im * reg##32_re;\
  \
  a32_re += cinv##32_20_re * reg##20_re;\
  a32_re -= cinv##32_20_im * reg##20_im;\
  a32_im += cinv##32_20_re * reg##20_im;\
  a32_im += cinv##32_20_im * reg##20_re;\
  a32_re += cinv##32_21_re * reg##21_re;\
  a32_re -= cinv##32_21_im * reg##21_im;\
  a32_im += cinv##32_21_re * reg##21_im;\
  a32_im += cinv##32_21_im * reg##21_re;\
  a32_re += cinv##32_22_re * reg##22_re;\
  a32_re -= cinv##32_22_im * reg##22_im;\
  a32_im += cinv##32_22_re * reg##22_im;\
  a32_im += cinv##32_22_im * reg##22_re;\
  a32_re += cinv##32_30_re * reg##30_re;\
  a32_re -= cinv##32_30_im * reg##30_im;\
  a32_im += cinv##32_30_re * reg##30_im;\
  a32_im += cinv##32_30_im * reg##30_re;\
  a32_re += cinv##32_31_re * reg##31_re;\
  a32_re -= cinv##32_31_im * reg##31_im;\
  a32_im += cinv##32_31_re * reg##31_im;\
  a32_im += cinv##32_31_im * reg##31_re;\
  a32_re += cinv##32_32_re * reg##32_re;\
  a32_im += cinv##32_32_re * reg##32_im;\
  \
  /*store  the result*/\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
  \
}\
\
/* change back from chiral basis*/\
/* (note: required factor of 1/2 is included in clover term normalization)*/\
{\
  spinorFloat a00_re =  reg##10_re + reg##30_re;\
  spinorFloat a00_im =  reg##10_im + reg##30_im;\
  spinorFloat a10_re = -reg##00_re - reg##20_re;\
  spinorFloat a10_im = -reg##00_im - reg##20_im;\
  spinorFloat a20_re =  reg##10_re - reg##30_re;\
  spinorFloat a20_im =  reg##10_im - reg##30_im;\
  spinorFloat a30_re = -reg##00_re + reg##20_re;\
  spinorFloat a30_im = -reg##00_im + reg##20_im;\
  \
  reg##00_re = a00_re*2.;  reg##00_im = a00_im*2.;\
  reg##10_re = a10_re*2.;  reg##10_im = a10_im*2.;\
  reg##20_re = a20_re*2.;  reg##20_im = a20_im*2.;\
  reg##30_re = a30_re*2.;  reg##30_im = a30_im*2.;\
}\
\
{\
  spinorFloat a01_re =  reg##11_re + reg##31_re;\
  spinorFloat a01_im =  reg##11_im + reg##31_im;\
  spinorFloat a11_re = -reg##01_re - reg##21_re;\
  spinorFloat a11_im = -reg##01_im - reg##21_im;\
  spinorFloat a21_re =  reg##11_re - reg##31_re;\
  spinorFloat a21_im =  reg##11_im - reg##31_im;\
  spinorFloat a31_re = -reg##01_re + reg##21_re;\
  spinorFloat a31_im = -reg##01_im + reg##21_im;\
  \
  reg##01_re = a01_re*2.;  reg##01_im = a01_im*2.;\
  reg##11_re = a11_re*2.;  reg##11_im = a11_im*2.;\
  reg##21_re = a21_re*2.;  reg##21_im = a21_im*2.;\
  reg##31_re = a31_re*2.;  reg##31_im = a31_im*2.;\
}\
\
{\
  spinorFloat a02_re =  reg##12_re + reg##32_re;\
  spinorFloat a02_im =  reg##12_im + reg##32_im;\
  spinorFloat a12_re = -reg##02_re - reg##22_re;\
  spinorFloat a12_im = -reg##02_im - reg##22_im;\
  spinorFloat a22_re =  reg##12_re - reg##32_re;\
  spinorFloat a22_im =  reg##12_im - reg##32_im;\
  spinorFloat a32_re = -reg##02_re + reg##22_re;\
  spinorFloat a32_im = -reg##02_im + reg##22_im;\
  \
  reg##02_re = a02_re*2.;  reg##02_im = a02_im*2.;\
  reg##12_re = a12_re*2.;  reg##12_im = a12_im*2.;\
  reg##22_re = a22_re*2.;  reg##22_im = a22_im*2.;\
  reg##32_re = a32_re*2.;  reg##32_im = a32_im*2.;\
}\


#define APPLY_CLOVER_TWIST_DYN_INV(c, a, reg)\
\
/* change to chiral basis*/\
{\
  spinorFloat a00_re = -reg##10_re - reg##30_re;\
  spinorFloat a00_im = -reg##10_im - reg##30_im;\
  spinorFloat a10_re =  reg##00_re + reg##20_re;\
  spinorFloat a10_im =  reg##00_im + reg##20_im;\
  spinorFloat a20_re = -reg##10_re + reg##30_re;\
  spinorFloat a20_im = -reg##10_im + reg##30_im;\
  spinorFloat a30_re =  reg##00_re - reg##20_re;\
  spinorFloat a30_im =  reg##00_im - reg##20_im;\
  \
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
}\
\
{\
  spinorFloat a01_re = -reg##11_re - reg##31_re;\
  spinorFloat a01_im = -reg##11_im - reg##31_im;\
  spinorFloat a11_re =  reg##01_re + reg##21_re;\
  spinorFloat a11_im =  reg##01_im + reg##21_im;\
  spinorFloat a21_re = -reg##11_re + reg##31_re;\
  spinorFloat a21_im = -reg##11_im + reg##31_im;\
  spinorFloat a31_re =  reg##01_re - reg##21_re;\
  spinorFloat a31_im =  reg##01_im - reg##21_im;\
  \
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
}\
\
{\
  spinorFloat a02_re = -reg##12_re - reg##32_re;\
  spinorFloat a02_im = -reg##12_im - reg##32_im;\
  spinorFloat a12_re =  reg##02_re + reg##22_re;\
  spinorFloat a12_im =  reg##02_im + reg##22_im;\
  spinorFloat a22_re = -reg##12_re + reg##32_re;\
  spinorFloat a22_im = -reg##12_im + reg##32_im;\
  spinorFloat a32_re =  reg##02_re - reg##22_re;\
  spinorFloat a32_im =  reg##02_im - reg##22_im;\
  \
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
\
/* apply first chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 0)\
  spinorFloat a00_re = 0; spinorFloat a00_im = 0;\
  spinorFloat a01_re = 0; spinorFloat a01_im = 0;\
  spinorFloat a02_re = 0; spinorFloat a02_im = 0;\
  spinorFloat a10_re = 0; spinorFloat a10_im = 0;\
  spinorFloat a11_re = 0; spinorFloat a11_im = 0;\
  spinorFloat a12_re = 0; spinorFloat a12_im = 0;\
  \
  a00_re += c##00_00_re * reg##00_re;\
  a00_im += c##00_00_re * reg##00_im;\
  a00_re += c##00_01_re * reg##01_re;\
  a00_re -= c##00_01_im * reg##01_im;\
  a00_im += c##00_01_re * reg##01_im;\
  a00_im += c##00_01_im * reg##01_re;\
  a00_re += c##00_02_re * reg##02_re;\
  a00_re -= c##00_02_im * reg##02_im;\
  a00_im += c##00_02_re * reg##02_im;\
  a00_im += c##00_02_im * reg##02_re;\
  a00_re += c##00_10_re * reg##10_re;\
  a00_re -= c##00_10_im * reg##10_im;\
  a00_im += c##00_10_re * reg##10_im;\
  a00_im += c##00_10_im * reg##10_re;\
  a00_re += c##00_11_re * reg##11_re;\
  a00_re -= c##00_11_im * reg##11_im;\
  a00_im += c##00_11_re * reg##11_im;\
  a00_im += c##00_11_im * reg##11_re;\
  a00_re += c##00_12_re * reg##12_re;\
  a00_re -= c##00_12_im * reg##12_im;\
  a00_im += c##00_12_re * reg##12_im;\
  a00_im += c##00_12_im * reg##12_re;\
  \
  a01_re += c##01_00_re * reg##00_re;\
  a01_re -= c##01_00_im * reg##00_im;\
  a01_im += c##01_00_re * reg##00_im;\
  a01_im += c##01_00_im * reg##00_re;\
  a01_re += c##01_01_re * reg##01_re;\
  a01_im += c##01_01_re * reg##01_im;\
  a01_re += c##01_02_re * reg##02_re;\
  a01_re -= c##01_02_im * reg##02_im;\
  a01_im += c##01_02_re * reg##02_im;\
  a01_im += c##01_02_im * reg##02_re;\
  a01_re += c##01_10_re * reg##10_re;\
  a01_re -= c##01_10_im * reg##10_im;\
  a01_im += c##01_10_re * reg##10_im;\
  a01_im += c##01_10_im * reg##10_re;\
  a01_re += c##01_11_re * reg##11_re;\
  a01_re -= c##01_11_im * reg##11_im;\
  a01_im += c##01_11_re * reg##11_im;\
  a01_im += c##01_11_im * reg##11_re;\
  a01_re += c##01_12_re * reg##12_re;\
  a01_re -= c##01_12_im * reg##12_im;\
  a01_im += c##01_12_re * reg##12_im;\
  a01_im += c##01_12_im * reg##12_re;\
  \
  a02_re += c##02_00_re * reg##00_re;\
  a02_re -= c##02_00_im * reg##00_im;\
  a02_im += c##02_00_re * reg##00_im;\
  a02_im += c##02_00_im * reg##00_re;\
  a02_re += c##02_01_re * reg##01_re;\
  a02_re -= c##02_01_im * reg##01_im;\
  a02_im += c##02_01_re * reg##01_im;\
  a02_im += c##02_01_im * reg##01_re;\
  a02_re += c##02_02_re * reg##02_re;\
  a02_im += c##02_02_re * reg##02_im;\
  a02_re += c##02_10_re * reg##10_re;\
  a02_re -= c##02_10_im * reg##10_im;\
  a02_im += c##02_10_re * reg##10_im;\
  a02_im += c##02_10_im * reg##10_re;\
  a02_re += c##02_11_re * reg##11_re;\
  a02_re -= c##02_11_im * reg##11_im;\
  a02_im += c##02_11_re * reg##11_im;\
  a02_im += c##02_11_im * reg##11_re;\
  a02_re += c##02_12_re * reg##12_re;\
  a02_re -= c##02_12_im * reg##12_im;\
  a02_im += c##02_12_re * reg##12_im;\
  a02_im += c##02_12_im * reg##12_re;\
  \
  a10_re += c##10_00_re * reg##00_re;\
  a10_re -= c##10_00_im * reg##00_im;\
  a10_im += c##10_00_re * reg##00_im;\
  a10_im += c##10_00_im * reg##00_re;\
  a10_re += c##10_01_re * reg##01_re;\
  a10_re -= c##10_01_im * reg##01_im;\
  a10_im += c##10_01_re * reg##01_im;\
  a10_im += c##10_01_im * reg##01_re;\
  a10_re += c##10_02_re * reg##02_re;\
  a10_re -= c##10_02_im * reg##02_im;\
  a10_im += c##10_02_re * reg##02_im;\
  a10_im += c##10_02_im * reg##02_re;\
  a10_re += c##10_10_re * reg##10_re;\
  a10_im += c##10_10_re * reg##10_im;\
  a10_re += c##10_11_re * reg##11_re;\
  a10_re -= c##10_11_im * reg##11_im;\
  a10_im += c##10_11_re * reg##11_im;\
  a10_im += c##10_11_im * reg##11_re;\
  a10_re += c##10_12_re * reg##12_re;\
  a10_re -= c##10_12_im * reg##12_im;\
  a10_im += c##10_12_re * reg##12_im;\
  a10_im += c##10_12_im * reg##12_re;\
  \
  a11_re += c##11_00_re * reg##00_re;\
  a11_re -= c##11_00_im * reg##00_im;\
  a11_im += c##11_00_re * reg##00_im;\
  a11_im += c##11_00_im * reg##00_re;\
  a11_re += c##11_01_re * reg##01_re;\
  a11_re -= c##11_01_im * reg##01_im;\
  a11_im += c##11_01_re * reg##01_im;\
  a11_im += c##11_01_im * reg##01_re;\
  a11_re += c##11_02_re * reg##02_re;\
  a11_re -= c##11_02_im * reg##02_im;\
  a11_im += c##11_02_re * reg##02_im;\
  a11_im += c##11_02_im * reg##02_re;\
  a11_re += c##11_10_re * reg##10_re;\
  a11_re -= c##11_10_im * reg##10_im;\
  a11_im += c##11_10_re * reg##10_im;\
  a11_im += c##11_10_im * reg##10_re;\
  a11_re += c##11_11_re * reg##11_re;\
  a11_im += c##11_11_re * reg##11_im;\
  a11_re += c##11_12_re * reg##12_re;\
  a11_re -= c##11_12_im * reg##12_im;\
  a11_im += c##11_12_re * reg##12_im;\
  a11_im += c##11_12_im * reg##12_re;\
  \
  a12_re += c##12_00_re * reg##00_re;\
  a12_re -= c##12_00_im * reg##00_im;\
  a12_im += c##12_00_re * reg##00_im;\
  a12_im += c##12_00_im * reg##00_re;\
  a12_re += c##12_01_re * reg##01_re;\
  a12_re -= c##12_01_im * reg##01_im;\
  a12_im += c##12_01_re * reg##01_im;\
  a12_im += c##12_01_im * reg##01_re;\
  a12_re += c##12_02_re * reg##02_re;\
  a12_re -= c##12_02_im * reg##02_im;\
  a12_im += c##12_02_re * reg##02_im;\
  a12_im += c##12_02_im * reg##02_re;\
  a12_re += c##12_10_re * reg##10_re;\
  a12_re -= c##12_10_im * reg##10_im;\
  a12_im += c##12_10_re * reg##10_im;\
  a12_im += c##12_10_im * reg##10_re;\
  a12_re += c##12_11_re * reg##11_re;\
  a12_re -= c##12_11_im * reg##11_im;\
  a12_im += c##12_11_re * reg##11_im;\
  a12_im += c##12_11_im * reg##11_re;\
  a12_re += c##12_12_re * reg##12_re;\
  a12_im += c##12_12_re * reg##12_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a00_re = a00_re - .5*a* reg##00_im;  a00_im = a00_im + .5*a* reg##00_re;\
  a01_re = a01_re - .5*a* reg##01_im;  a01_im = a01_im + .5*a* reg##01_re;\
  a02_re = a02_re - .5*a* reg##02_im;  a02_im = a02_im + .5*a* reg##02_re;\
  a10_re = a10_re - .5*a* reg##10_im;  a10_im = a10_im + .5*a* reg##10_re;\
  a11_re = a11_re - .5*a* reg##11_im;  a11_im = a11_im + .5*a* reg##11_re;\
  a12_re = a12_re - .5*a* reg##12_im;  a12_im = a12_im + .5*a* reg##12_re;\
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
}\
/*Apply inverse clover*/\
{\
  /*ASSN_CLOVER(TMCLOVERTEX, 0)*/ /* PUEDO EIMINARLO PORQUE YA LO HEMOS LEIDO */ \
  INVERT_CLOVER(c) \
  spinorFloat a00_re = 0; spinorFloat a00_im = 0;\
  spinorFloat a01_re = 0; spinorFloat a01_im = 0;\
  spinorFloat a02_re = 0; spinorFloat a02_im = 0;\
  spinorFloat a10_re = 0; spinorFloat a10_im = 0;\
  spinorFloat a11_re = 0; spinorFloat a11_im = 0;\
  spinorFloat a12_re = 0; spinorFloat a12_im = 0;\
  \
  a00_re += c##00_00_re * reg##00_re;\
  a00_im += c##00_00_re * reg##00_im;\
  a00_re += c##00_01_re * reg##01_re;\
  a00_re -= c##00_01_im * reg##01_im;\
  a00_im += c##00_01_re * reg##01_im;\
  a00_im += c##00_01_im * reg##01_re;\
  a00_re += c##00_02_re * reg##02_re;\
  a00_re -= c##00_02_im * reg##02_im;\
  a00_im += c##00_02_re * reg##02_im;\
  a00_im += c##00_02_im * reg##02_re;\
  a00_re += c##00_10_re * reg##10_re;\
  a00_re -= c##00_10_im * reg##10_im;\
  a00_im += c##00_10_re * reg##10_im;\
  a00_im += c##00_10_im * reg##10_re;\
  a00_re += c##00_11_re * reg##11_re;\
  a00_re -= c##00_11_im * reg##11_im;\
  a00_im += c##00_11_re * reg##11_im;\
  a00_im += c##00_11_im * reg##11_re;\
  a00_re += c##00_12_re * reg##12_re;\
  a00_re -= c##00_12_im * reg##12_im;\
  a00_im += c##00_12_re * reg##12_im;\
  a00_im += c##00_12_im * reg##12_re;\
  \
  a01_re += c##01_00_re * reg##00_re;\
  a01_re -= c##01_00_im * reg##00_im;\
  a01_im += c##01_00_re * reg##00_im;\
  a01_im += c##01_00_im * reg##00_re;\
  a01_re += c##01_01_re * reg##01_re;\
  a01_im += c##01_01_re * reg##01_im;\
  a01_re += c##01_02_re * reg##02_re;\
  a01_re -= c##01_02_im * reg##02_im;\
  a01_im += c##01_02_re * reg##02_im;\
  a01_im += c##01_02_im * reg##02_re;\
  a01_re += c##01_10_re * reg##10_re;\
  a01_re -= c##01_10_im * reg##10_im;\
  a01_im += c##01_10_re * reg##10_im;\
  a01_im += c##01_10_im * reg##10_re;\
  a01_re += c##01_11_re * reg##11_re;\
  a01_re -= c##01_11_im * reg##11_im;\
  a01_im += c##01_11_re * reg##11_im;\
  a01_im += c##01_11_im * reg##11_re;\
  a01_re += c##01_12_re * reg##12_re;\
  a01_re -= c##01_12_im * reg##12_im;\
  a01_im += c##01_12_re * reg##12_im;\
  a01_im += c##01_12_im * reg##12_re;\
  \
  a02_re += c##02_00_re * reg##00_re;\
  a02_re -= c##02_00_im * reg##00_im;\
  a02_im += c##02_00_re * reg##00_im;\
  a02_im += c##02_00_im * reg##00_re;\
  a02_re += c##02_01_re * reg##01_re;\
  a02_re -= c##02_01_im * reg##01_im;\
  a02_im += c##02_01_re * reg##01_im;\
  a02_im += c##02_01_im * reg##01_re;\
  a02_re += c##02_02_re * reg##02_re;\
  a02_im += c##02_02_re * reg##02_im;\
  a02_re += c##02_10_re * reg##10_re;\
  a02_re -= c##02_10_im * reg##10_im;\
  a02_im += c##02_10_re * reg##10_im;\
  a02_im += c##02_10_im * reg##10_re;\
  a02_re += c##02_11_re * reg##11_re;\
  a02_re -= c##02_11_im * reg##11_im;\
  a02_im += c##02_11_re * reg##11_im;\
  a02_im += c##02_11_im * reg##11_re;\
  a02_re += c##02_12_re * reg##12_re;\
  a02_re -= c##02_12_im * reg##12_im;\
  a02_im += c##02_12_re * reg##12_im;\
  a02_im += c##02_12_im * reg##12_re;\
  \
  a10_re += c##10_00_re * reg##00_re;\
  a10_re -= c##10_00_im * reg##00_im;\
  a10_im += c##10_00_re * reg##00_im;\
  a10_im += c##10_00_im * reg##00_re;\
  a10_re += c##10_01_re * reg##01_re;\
  a10_re -= c##10_01_im * reg##01_im;\
  a10_im += c##10_01_re * reg##01_im;\
  a10_im += c##10_01_im * reg##01_re;\
  a10_re += c##10_02_re * reg##02_re;\
  a10_re -= c##10_02_im * reg##02_im;\
  a10_im += c##10_02_re * reg##02_im;\
  a10_im += c##10_02_im * reg##02_re;\
  a10_re += c##10_10_re * reg##10_re;\
  a10_im += c##10_10_re * reg##10_im;\
  a10_re += c##10_11_re * reg##11_re;\
  a10_re -= c##10_11_im * reg##11_im;\
  a10_im += c##10_11_re * reg##11_im;\
  a10_im += c##10_11_im * reg##11_re;\
  a10_re += c##10_12_re * reg##12_re;\
  a10_re -= c##10_12_im * reg##12_im;\
  a10_im += c##10_12_re * reg##12_im;\
  a10_im += c##10_12_im * reg##12_re;\
  \
  a11_re += c##11_00_re * reg##00_re;\
  a11_re -= c##11_00_im * reg##00_im;\
  a11_im += c##11_00_re * reg##00_im;\
  a11_im += c##11_00_im * reg##00_re;\
  a11_re += c##11_01_re * reg##01_re;\
  a11_re -= c##11_01_im * reg##01_im;\
  a11_im += c##11_01_re * reg##01_im;\
  a11_im += c##11_01_im * reg##01_re;\
  a11_re += c##11_02_re * reg##02_re;\
  a11_re -= c##11_02_im * reg##02_im;\
  a11_im += c##11_02_re * reg##02_im;\
  a11_im += c##11_02_im * reg##02_re;\
  a11_re += c##11_10_re * reg##10_re;\
  a11_re -= c##11_10_im * reg##10_im;\
  a11_im += c##11_10_re * reg##10_im;\
  a11_im += c##11_10_im * reg##10_re;\
  a11_re += c##11_11_re * reg##11_re;\
  a11_im += c##11_11_re * reg##11_im;\
  a11_re += c##11_12_re * reg##12_re;\
  a11_re -= c##11_12_im * reg##12_im;\
  a11_im += c##11_12_re * reg##12_im;\
  a11_im += c##11_12_im * reg##12_re;\
  \
  a12_re += c##12_00_re * reg##00_re;\
  a12_re -= c##12_00_im * reg##00_im;\
  a12_im += c##12_00_re * reg##00_im;\
  a12_im += c##12_00_im * reg##00_re;\
  a12_re += c##12_01_re * reg##01_re;\
  a12_re -= c##12_01_im * reg##01_im;\
  a12_im += c##12_01_re * reg##01_im;\
  a12_im += c##12_01_im * reg##01_re;\
  a12_re += c##12_02_re * reg##02_re;\
  a12_re -= c##12_02_im * reg##02_im;\
  a12_im += c##12_02_re * reg##02_im;\
  a12_im += c##12_02_im * reg##02_re;\
  a12_re += c##12_10_re * reg##10_re;\
  a12_re -= c##12_10_im * reg##10_im;\
  a12_im += c##12_10_re * reg##10_im;\
  a12_im += c##12_10_im * reg##10_re;\
  a12_re += c##12_11_re * reg##11_re;\
  a12_re -= c##12_11_im * reg##11_im;\
  a12_im += c##12_11_re * reg##11_im;\
  a12_im += c##12_11_im * reg##11_re;\
  a12_re += c##12_12_re * reg##12_re;\
  a12_im += c##12_12_re * reg##12_im;\
  \
  /*store  the result*/\
  reg##00_re = a00_re;  reg##00_im = a00_im;\
  reg##01_re = a01_re;  reg##01_im = a01_im;\
  reg##02_re = a02_re;  reg##02_im = a02_im;\
  reg##10_re = a10_re;  reg##10_im = a10_im;\
  reg##11_re = a11_re;  reg##11_im = a11_im;\
  reg##12_re = a12_re;  reg##12_im = a12_im;\
  \
}\
\
/* apply second chiral block*/\
{\
  ASSN_CLOVER(TMCLOVERTEX, 1)\
  spinorFloat a20_re = 0; spinorFloat a20_im = 0;\
  spinorFloat a21_re = 0; spinorFloat a21_im = 0;\
  spinorFloat a22_re = 0; spinorFloat a22_im = 0;\
  spinorFloat a30_re = 0; spinorFloat a30_im = 0;\
  spinorFloat a31_re = 0; spinorFloat a31_im = 0;\
  spinorFloat a32_re = 0; spinorFloat a32_im = 0;\
  \
  a20_re += c##20_20_re * reg##20_re;\
  a20_im += c##20_20_re * reg##20_im;\
  a20_re += c##20_21_re * reg##21_re;\
  a20_re -= c##20_21_im * reg##21_im;\
  a20_im += c##20_21_re * reg##21_im;\
  a20_im += c##20_21_im * reg##21_re;\
  a20_re += c##20_22_re * reg##22_re;\
  a20_re -= c##20_22_im * reg##22_im;\
  a20_im += c##20_22_re * reg##22_im;\
  a20_im += c##20_22_im * reg##22_re;\
  a20_re += c##20_30_re * reg##30_re;\
  a20_re -= c##20_30_im * reg##30_im;\
  a20_im += c##20_30_re * reg##30_im;\
  a20_im += c##20_30_im * reg##30_re;\
  a20_re += c##20_31_re * reg##31_re;\
  a20_re -= c##20_31_im * reg##31_im;\
  a20_im += c##20_31_re * reg##31_im;\
  a20_im += c##20_31_im * reg##31_re;\
  a20_re += c##20_32_re * reg##32_re;\
  a20_re -= c##20_32_im * reg##32_im;\
  a20_im += c##20_32_re * reg##32_im;\
  a20_im += c##20_32_im * reg##32_re;\
  \
  a21_re += c##21_20_re * reg##20_re;\
  a21_re -= c##21_20_im * reg##20_im;\
  a21_im += c##21_20_re * reg##20_im;\
  a21_im += c##21_20_im * reg##20_re;\
  a21_re += c##21_21_re * reg##21_re;\
  a21_im += c##21_21_re * reg##21_im;\
  a21_re += c##21_22_re * reg##22_re;\
  a21_re -= c##21_22_im * reg##22_im;\
  a21_im += c##21_22_re * reg##22_im;\
  a21_im += c##21_22_im * reg##22_re;\
  a21_re += c##21_30_re * reg##30_re;\
  a21_re -= c##21_30_im * reg##30_im;\
  a21_im += c##21_30_re * reg##30_im;\
  a21_im += c##21_30_im * reg##30_re;\
  a21_re += c##21_31_re * reg##31_re;\
  a21_re -= c##21_31_im * reg##31_im;\
  a21_im += c##21_31_re * reg##31_im;\
  a21_im += c##21_31_im * reg##31_re;\
  a21_re += c##21_32_re * reg##32_re;\
  a21_re -= c##21_32_im * reg##32_im;\
  a21_im += c##21_32_re * reg##32_im;\
  a21_im += c##21_32_im * reg##32_re;\
  \
  a22_re += c##22_20_re * reg##20_re;\
  a22_re -= c##22_20_im * reg##20_im;\
  a22_im += c##22_20_re * reg##20_im;\
  a22_im += c##22_20_im * reg##20_re;\
  a22_re += c##22_21_re * reg##21_re;\
  a22_re -= c##22_21_im * reg##21_im;\
  a22_im += c##22_21_re * reg##21_im;\
  a22_im += c##22_21_im * reg##21_re;\
  a22_re += c##22_22_re * reg##22_re;\
  a22_im += c##22_22_re * reg##22_im;\
  a22_re += c##22_30_re * reg##30_re;\
  a22_re -= c##22_30_im * reg##30_im;\
  a22_im += c##22_30_re * reg##30_im;\
  a22_im += c##22_30_im * reg##30_re;\
  a22_re += c##22_31_re * reg##31_re;\
  a22_re -= c##22_31_im * reg##31_im;\
  a22_im += c##22_31_re * reg##31_im;\
  a22_im += c##22_31_im * reg##31_re;\
  a22_re += c##22_32_re * reg##32_re;\
  a22_re -= c##22_32_im * reg##32_im;\
  a22_im += c##22_32_re * reg##32_im;\
  a22_im += c##22_32_im * reg##32_re;\
  \
  a30_re += c##30_20_re * reg##20_re;\
  a30_re -= c##30_20_im * reg##20_im;\
  a30_im += c##30_20_re * reg##20_im;\
  a30_im += c##30_20_im * reg##20_re;\
  a30_re += c##30_21_re * reg##21_re;\
  a30_re -= c##30_21_im * reg##21_im;\
  a30_im += c##30_21_re * reg##21_im;\
  a30_im += c##30_21_im * reg##21_re;\
  a30_re += c##30_22_re * reg##22_re;\
  a30_re -= c##30_22_im * reg##22_im;\
  a30_im += c##30_22_re * reg##22_im;\
  a30_im += c##30_22_im * reg##22_re;\
  a30_re += c##30_30_re * reg##30_re;\
  a30_im += c##30_30_re * reg##30_im;\
  a30_re += c##30_31_re * reg##31_re;\
  a30_re -= c##30_31_im * reg##31_im;\
  a30_im += c##30_31_re * reg##31_im;\
  a30_im += c##30_31_im * reg##31_re;\
  a30_re += c##30_32_re * reg##32_re;\
  a30_re -= c##30_32_im * reg##32_im;\
  a30_im += c##30_32_re * reg##32_im;\
  a30_im += c##30_32_im * reg##32_re;\
  \
  a31_re += c##31_20_re * reg##20_re;\
  a31_re -= c##31_20_im * reg##20_im;\
  a31_im += c##31_20_re * reg##20_im;\
  a31_im += c##31_20_im * reg##20_re;\
  a31_re += c##31_21_re * reg##21_re;\
  a31_re -= c##31_21_im * reg##21_im;\
  a31_im += c##31_21_re * reg##21_im;\
  a31_im += c##31_21_im * reg##21_re;\
  a31_re += c##31_22_re * reg##22_re;\
  a31_re -= c##31_22_im * reg##22_im;\
  a31_im += c##31_22_re * reg##22_im;\
  a31_im += c##31_22_im * reg##22_re;\
  a31_re += c##31_30_re * reg##30_re;\
  a31_re -= c##31_30_im * reg##30_im;\
  a31_im += c##31_30_re * reg##30_im;\
  a31_im += c##31_30_im * reg##30_re;\
  a31_re += c##31_31_re * reg##31_re;\
  a31_im += c##31_31_re * reg##31_im;\
  a31_re += c##31_32_re * reg##32_re;\
  a31_re -= c##31_32_im * reg##32_im;\
  a31_im += c##31_32_re * reg##32_im;\
  a31_im += c##31_32_im * reg##32_re;\
  \
  a32_re += c##32_20_re * reg##20_re;\
  a32_re -= c##32_20_im * reg##20_im;\
  a32_im += c##32_20_re * reg##20_im;\
  a32_im += c##32_20_im * reg##20_re;\
  a32_re += c##32_21_re * reg##21_re;\
  a32_re -= c##32_21_im * reg##21_im;\
  a32_im += c##32_21_re * reg##21_im;\
  a32_im += c##32_21_im * reg##21_re;\
  a32_re += c##32_22_re * reg##22_re;\
  a32_re -= c##32_22_im * reg##22_im;\
  a32_im += c##32_22_re * reg##22_im;\
  a32_im += c##32_22_im * reg##22_re;\
  a32_re += c##32_30_re * reg##30_re;\
  a32_re -= c##32_30_im * reg##30_im;\
  a32_im += c##32_30_re * reg##30_im;\
  a32_im += c##32_30_im * reg##30_re;\
  a32_re += c##32_31_re * reg##31_re;\
  a32_re -= c##32_31_im * reg##31_im;\
  a32_im += c##32_31_re * reg##31_im;\
  a32_im += c##32_31_im * reg##31_re;\
  a32_re += c##32_32_re * reg##32_re;\
  a32_im += c##32_32_re * reg##32_im;\
  \
  /*apply  i*(2*kappa*mu=a)*gamma5*/\
  a20_re = a20_re + .5*a* reg##20_im;  a20_im = a20_im - .5*a* reg##20_re;\
  a21_re = a21_re + .5*a* reg##21_im;  a21_im = a21_im - .5*a* reg##21_re;\
  a22_re = a22_re + .5*a* reg##22_im;  a22_im = a22_im - .5*a* reg##22_re;\
  a30_re = a30_re + .5*a* reg##30_im;  a30_im = a30_im - .5*a* reg##30_re;\
  a31_re = a31_re + .5*a* reg##31_im;  a31_im = a31_im - .5*a* reg##31_re;\
  a32_re = a32_re + .5*a* reg##32_im;  a32_im = a32_im - .5*a* reg##32_re;\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
}\
/*Apply inverse clover*/\
{\
  /*ASSN_CLOVER(TMCLOVERTEX, 1)*/ /* PUEDO EIMINARLO PORQUE YA LO HEMOS LEIDO */ \
  INVERT_CLOVER(c) \
  spinorFloat a20_re = 0; spinorFloat a20_im = 0;\
  spinorFloat a21_re = 0; spinorFloat a21_im = 0;\
  spinorFloat a22_re = 0; spinorFloat a22_im = 0;\
  spinorFloat a30_re = 0; spinorFloat a30_im = 0;\
  spinorFloat a31_re = 0; spinorFloat a31_im = 0;\
  spinorFloat a32_re = 0; spinorFloat a32_im = 0;\
  \
  a20_re += c##20_20_re * reg##20_re;\
  a20_im += c##20_20_re * reg##20_im;\
  a20_re += c##20_21_re * reg##21_re;\
  a20_re -= c##20_21_im * reg##21_im;\
  a20_im += c##20_21_re * reg##21_im;\
  a20_im += c##20_21_im * reg##21_re;\
  a20_re += c##20_22_re * reg##22_re;\
  a20_re -= c##20_22_im * reg##22_im;\
  a20_im += c##20_22_re * reg##22_im;\
  a20_im += c##20_22_im * reg##22_re;\
  a20_re += c##20_30_re * reg##30_re;\
  a20_re -= c##20_30_im * reg##30_im;\
  a20_im += c##20_30_re * reg##30_im;\
  a20_im += c##20_30_im * reg##30_re;\
  a20_re += c##20_31_re * reg##31_re;\
  a20_re -= c##20_31_im * reg##31_im;\
  a20_im += c##20_31_re * reg##31_im;\
  a20_im += c##20_31_im * reg##31_re;\
  a20_re += c##20_32_re * reg##32_re;\
  a20_re -= c##20_32_im * reg##32_im;\
  a20_im += c##20_32_re * reg##32_im;\
  a20_im += c##20_32_im * reg##32_re;\
  \
  a21_re += c##21_20_re * reg##20_re;\
  a21_re -= c##21_20_im * reg##20_im;\
  a21_im += c##21_20_re * reg##20_im;\
  a21_im += c##21_20_im * reg##20_re;\
  a21_re += c##21_21_re * reg##21_re;\
  a21_im += c##21_21_re * reg##21_im;\
  a21_re += c##21_22_re * reg##22_re;\
  a21_re -= c##21_22_im * reg##22_im;\
  a21_im += c##21_22_re * reg##22_im;\
  a21_im += c##21_22_im * reg##22_re;\
  a21_re += c##21_30_re * reg##30_re;\
  a21_re -= c##21_30_im * reg##30_im;\
  a21_im += c##21_30_re * reg##30_im;\
  a21_im += c##21_30_im * reg##30_re;\
  a21_re += c##21_31_re * reg##31_re;\
  a21_re -= c##21_31_im * reg##31_im;\
  a21_im += c##21_31_re * reg##31_im;\
  a21_im += c##21_31_im * reg##31_re;\
  a21_re += c##21_32_re * reg##32_re;\
  a21_re -= c##21_32_im * reg##32_im;\
  a21_im += c##21_32_re * reg##32_im;\
  a21_im += c##21_32_im * reg##32_re;\
  \
  a22_re += c##22_20_re * reg##20_re;\
  a22_re -= c##22_20_im * reg##20_im;\
  a22_im += c##22_20_re * reg##20_im;\
  a22_im += c##22_20_im * reg##20_re;\
  a22_re += c##22_21_re * reg##21_re;\
  a22_re -= c##22_21_im * reg##21_im;\
  a22_im += c##22_21_re * reg##21_im;\
  a22_im += c##22_21_im * reg##21_re;\
  a22_re += c##22_22_re * reg##22_re;\
  a22_im += c##22_22_re * reg##22_im;\
  a22_re += c##22_30_re * reg##30_re;\
  a22_re -= c##22_30_im * reg##30_im;\
  a22_im += c##22_30_re * reg##30_im;\
  a22_im += c##22_30_im * reg##30_re;\
  a22_re += c##22_31_re * reg##31_re;\
  a22_re -= c##22_31_im * reg##31_im;\
  a22_im += c##22_31_re * reg##31_im;\
  a22_im += c##22_31_im * reg##31_re;\
  a22_re += c##22_32_re * reg##32_re;\
  a22_re -= c##22_32_im * reg##32_im;\
  a22_im += c##22_32_re * reg##32_im;\
  a22_im += c##22_32_im * reg##32_re;\
  \
  a30_re += c##30_20_re * reg##20_re;\
  a30_re -= c##30_20_im * reg##20_im;\
  a30_im += c##30_20_re * reg##20_im;\
  a30_im += c##30_20_im * reg##20_re;\
  a30_re += c##30_21_re * reg##21_re;\
  a30_re -= c##30_21_im * reg##21_im;\
  a30_im += c##30_21_re * reg##21_im;\
  a30_im += c##30_21_im * reg##21_re;\
  a30_re += c##30_22_re * reg##22_re;\
  a30_re -= c##30_22_im * reg##22_im;\
  a30_im += c##30_22_re * reg##22_im;\
  a30_im += c##30_22_im * reg##22_re;\
  a30_re += c##30_30_re * reg##30_re;\
  a30_im += c##30_30_re * reg##30_im;\
  a30_re += c##30_31_re * reg##31_re;\
  a30_re -= c##30_31_im * reg##31_im;\
  a30_im += c##30_31_re * reg##31_im;\
  a30_im += c##30_31_im * reg##31_re;\
  a30_re += c##30_32_re * reg##32_re;\
  a30_re -= c##30_32_im * reg##32_im;\
  a30_im += c##30_32_re * reg##32_im;\
  a30_im += c##30_32_im * reg##32_re;\
  \
  a31_re += c##31_20_re * reg##20_re;\
  a31_re -= c##31_20_im * reg##20_im;\
  a31_im += c##31_20_re * reg##20_im;\
  a31_im += c##31_20_im * reg##20_re;\
  a31_re += c##31_21_re * reg##21_re;\
  a31_re -= c##31_21_im * reg##21_im;\
  a31_im += c##31_21_re * reg##21_im;\
  a31_im += c##31_21_im * reg##21_re;\
  a31_re += c##31_22_re * reg##22_re;\
  a31_re -= c##31_22_im * reg##22_im;\
  a31_im += c##31_22_re * reg##22_im;\
  a31_im += c##31_22_im * reg##22_re;\
  a31_re += c##31_30_re * reg##30_re;\
  a31_re -= c##31_30_im * reg##30_im;\
  a31_im += c##31_30_re * reg##30_im;\
  a31_im += c##31_30_im * reg##30_re;\
  a31_re += c##31_31_re * reg##31_re;\
  a31_im += c##31_31_re * reg##31_im;\
  a31_re += c##31_32_re * reg##32_re;\
  a31_re -= c##31_32_im * reg##32_im;\
  a31_im += c##31_32_re * reg##32_im;\
  a31_im += c##31_32_im * reg##32_re;\
  \
  a32_re += c##32_20_re * reg##20_re;\
  a32_re -= c##32_20_im * reg##20_im;\
  a32_im += c##32_20_re * reg##20_im;\
  a32_im += c##32_20_im * reg##20_re;\
  a32_re += c##32_21_re * reg##21_re;\
  a32_re -= c##32_21_im * reg##21_im;\
  a32_im += c##32_21_re * reg##21_im;\
  a32_im += c##32_21_im * reg##21_re;\
  a32_re += c##32_22_re * reg##22_re;\
  a32_re -= c##32_22_im * reg##22_im;\
  a32_im += c##32_22_re * reg##22_im;\
  a32_im += c##32_22_im * reg##22_re;\
  a32_re += c##32_30_re * reg##30_re;\
  a32_re -= c##32_30_im * reg##30_im;\
  a32_im += c##32_30_re * reg##30_im;\
  a32_im += c##32_30_im * reg##30_re;\
  a32_re += c##32_31_re * reg##31_re;\
  a32_re -= c##32_31_im * reg##31_im;\
  a32_im += c##32_31_re * reg##31_im;\
  a32_im += c##32_31_im * reg##31_re;\
  a32_re += c##32_32_re * reg##32_re;\
  a32_im += c##32_32_re * reg##32_im;\
  \
  /*store  the result*/\
  reg##20_re = a20_re;  reg##20_im = a20_im;\
  reg##21_re = a21_re;  reg##21_im = a21_im;\
  reg##22_re = a22_re;  reg##22_im = a22_im;\
  reg##30_re = a30_re;  reg##30_im = a30_im;\
  reg##31_re = a31_re;  reg##31_im = a31_im;\
  reg##32_re = a32_re;  reg##32_im = a32_im;\
  \
}\
\
/* change back from chiral basis*/\
/* (note: required factor of 1/2 is included in clover term normalization)*/\
{\
  spinorFloat a00_re =  reg##10_re + reg##30_re;\
  spinorFloat a00_im =  reg##10_im + reg##30_im;\
  spinorFloat a10_re = -reg##00_re - reg##20_re;\
  spinorFloat a10_im = -reg##00_im - reg##20_im;\
  spinorFloat a20_re =  reg##10_re - reg##30_re;\
  spinorFloat a20_im =  reg##10_im - reg##30_im;\
  spinorFloat a30_re = -reg##00_re + reg##20_re;\
  spinorFloat a30_im = -reg##00_im + reg##20_im;\
  \
  reg##00_re = a00_re*2.;  reg##00_im = a00_im*2.;\
  reg##10_re = a10_re*2.;  reg##10_im = a10_im*2.;\
  reg##20_re = a20_re*2.;  reg##20_im = a20_im*2.;\
  reg##30_re = a30_re*2.;  reg##30_im = a30_im*2.;\
}\
\
{\
  spinorFloat a01_re =  reg##11_re + reg##31_re;\
  spinorFloat a01_im =  reg##11_im + reg##31_im;\
  spinorFloat a11_re = -reg##01_re - reg##21_re;\
  spinorFloat a11_im = -reg##01_im - reg##21_im;\
  spinorFloat a21_re =  reg##11_re - reg##31_re;\
  spinorFloat a21_im =  reg##11_im - reg##31_im;\
  spinorFloat a31_re = -reg##01_re + reg##21_re;\
  spinorFloat a31_im = -reg##01_im + reg##21_im;\
  \
  reg##01_re = a01_re*2.;  reg##01_im = a01_im*2.;\
  reg##11_re = a11_re*2.;  reg##11_im = a11_im*2.;\
  reg##21_re = a21_re*2.;  reg##21_im = a21_im*2.;\
  reg##31_re = a31_re*2.;  reg##31_im = a31_im*2.;\
}\
\
{\
  spinorFloat a02_re =  reg##12_re + reg##32_re;\
  spinorFloat a02_im =  reg##12_im + reg##32_im;\
  spinorFloat a12_re = -reg##02_re - reg##22_re;\
  spinorFloat a12_im = -reg##02_im - reg##22_im;\
  spinorFloat a22_re =  reg##12_re - reg##32_re;\
  spinorFloat a22_im =  reg##12_im - reg##32_im;\
  spinorFloat a32_re = -reg##02_re + reg##22_re;\
  spinorFloat a32_im = -reg##02_im + reg##22_im;\
  \
  reg##02_re = a02_re*2.;  reg##02_im = a02_im*2.;\
  reg##12_re = a12_re*2.;  reg##12_im = a12_im*2.;\
  reg##22_re = a22_re*2.;  reg##22_im = a22_im*2.;\
  reg##32_re = a32_re*2.;  reg##32_im = a32_im*2.;\
}\
\

