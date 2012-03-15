
#define xcomm kparam.ghostDim[0]
#define ycomm kparam.ghostDim[1]
#define zcomm kparam.ghostDim[2]
#define tcomm kparam.ghostDim[3]

#if (N_IN_FLOATN == 4)
#define linka00_re LINKA0.x
#define linka00_im LINKA0.y
#define linka01_re LINKA0.z
#define linka01_im LINKA0.w
#define linka02_re LINKA1.x
#define linka02_im LINKA1.y
#define linka10_re LINKA1.z
#define linka10_im LINKA1.w
#define linka11_re LINKA2.x
#define linka11_im LINKA2.y
#define linka12_re LINKA2.z
#define linka12_im LINKA2.w
#define linka20_re LINKA3.x
#define linka20_im LINKA3.y
#define linka21_re LINKA3.z
#define linka21_im LINKA3.w
#define linka22_re LINKA4.x
#define linka22_im LINKA4.y


#define linkb00_re LINKB0.x
#define linkb00_im LINKB0.y
#define linkb01_re LINKB0.z
#define linkb01_im LINKB0.w
#define linkb02_re LINKB1.x
#define linkb02_im LINKB1.y
#define linkb10_re LINKB1.z
#define linkb10_im LINKB1.w
#define linkb11_re LINKB2.x
#define linkb11_im LINKB2.y
#define linkb12_re LINKB2.z
#define linkb12_im LINKB2.w
#define linkb20_re LINKB3.x
#define linkb20_im LINKB3.y
#define linkb21_re LINKB3.z
#define linkb21_im LINKB3.w
#define linkb22_re LINKB4.x
#define linkb22_im LINKB4.y

#else
#define linka00_re LINKA0.x
#define linka00_im LINKA0.y
#define linka01_re LINKA1.x
#define linka01_im LINKA1.y
#define linka02_re LINKA2.x
#define linka02_im LINKA2.y
#define linka10_re LINKA3.x
#define linka10_im LINKA3.y
#define linka11_re LINKA4.x
#define linka11_im LINKA4.y
#define linka12_re LINKA5.x
#define linka12_im LINKA5.y
#define linka20_re LINKA6.x
#define linka20_im LINKA6.y
#define linka21_re LINKA7.x
#define linka21_im LINKA7.y
#define linka22_re LINKA8.x
#define linka22_im LINKA8.y

#define linkb00_re LINKB0.x
#define linkb00_im LINKB0.y
#define linkb01_re LINKB1.x
#define linkb01_im LINKB1.y
#define linkb02_re LINKB2.x
#define linkb02_im LINKB2.y
#define linkb10_re LINKB3.x
#define linkb10_im LINKB3.y
#define linkb11_re LINKB4.x
#define linkb11_im LINKB4.y
#define linkb12_re LINKB5.x
#define linkb12_im LINKB5.y
#define linkb20_re LINKB6.x
#define linkb20_im LINKB6.y
#define linkb21_re LINKB7.x
#define linkb21_im LINKB7.y
#define linkb22_re LINKB8.x
#define linkb22_im LINKB8.y

#endif


#ifdef MULTI_GPU

#define COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mydir, idx) do {		\
    switch(mydir){							\
    case 0:								\
      new_mem_idx = ((!xcomm) && (new_x1 == (X1+1)))?(idx - X1m1): idx+1; \
      new_x1 = ((!xcomm)&& (new_x1 == (X1+1)))? (new_x1 - X1m1):(new_x1+1); \
      break;								\
    case 1:								\
      new_mem_idx = ((!ycomm) && (new_x2 == (X2+1)))?(idx - X2m1*E1): idx+E1; \
      new_x2 = ((!ycomm)&& (new_x2 == (X2+1)))? (new_x2 - X2m1):(new_x2+1); \
      break;								\
    case 2:								\
      new_mem_idx = ((!zcomm) && (new_x3 == (X3+1)))?(idx - X3m1*E2E1): idx+E2E1; \
      new_x3 = ((!zcomm)&& (new_x3 == (X3+1)))? (new_x3 - X3m1):(new_x3+1); \
      break;								\
    case 3:								\
      new_mem_idx = ((!tcomm) && (new_x4 == (X4+1)))?(idx - X4m1*E3E2E1): idx+E3E2E1; \
      new_x4 = ((!tcomm)&& (new_x4 == (X4+1)))? (new_x4 - X4m1):(new_x4+1); \
      break;								\
    }									\
  }while(0)

#define COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mydir, idx) do {		\
    switch(mydir){							\
    case 0:								\
      new_mem_idx = ((!xcomm) && new_x1 == 2)?(idx+X1m1):(idx-1);	\
      new_x1 = ((!xcomm) && new_x1 == 2)? (new_x1+X1m1): (new_x1-1);	\
      break;								\
    case 1:								\
      new_mem_idx = ((!ycomm) && new_x2 == 2)?(idx+X2m1*E1):(idx-E1);	\
      new_x2 = ((!ycomm) && new_x2 == 2)? (new_x2+X2m1): (new_x2-1);	\
      break;								\
    case 2:								\
      new_mem_idx = ((!zcomm) && new_x3 == 2)?(idx+X3m1*E2E1):(idx-E2E1); \
      new_x3 = ((!zcomm) && new_x3 == 2)? (new_x3+X3m1): (new_x3-1);	\
      break;								\
    case 3:								\
      new_mem_idx = ((!tcomm) && new_x4 == 2)?(idx+X4m1*E3E2E1):(idx-E3E2E1); \
      new_x4 = ((!tcomm) && new_x4 == 2)? (new_x4+X4m1): (new_x4-1);	\
      break;								\
    }									\
  }while(0)

#else
#define COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mydir, idx) do {		\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (new_x1==X1m1)?idx-X1m1:idx+1);		\
	    new_x1 = (new_x1==X1m1)?0:new_x1+1;				\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (new_x2==X2m1)?idx-X2X1mX1:idx+X1);		\
	    new_x2 = (new_x2==X2m1)?0:new_x2+1;				\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (new_x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
	    new_x3 = (new_x3==X3m1)?0:new_x3+1;				\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (new_x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
	    new_x4 = (new_x4==X4m1)?0:new_x4+1;				\
            break;                                                      \
        }                                                               \
    }while(0)

#define COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mydir, idx) do {		\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_mem_idx = ( (new_x1==0)?idx+X1m1:idx-1);		\
	    new_x1 = (new_x1==0)?X1m1:new_x1 - 1;			\
            break;                                                      \
        case 1:                                                         \
            new_mem_idx = ( (new_x2==0)?idx+X2X1mX1:idx-X1);		\
	    new_x2 = (new_x2==0)?X2m1:new_x2 - 1;			\
            break;                                                      \
        case 2:                                                         \
            new_mem_idx = ( (new_x3==0)?idx+X3X2X1mX2X1:idx-X2X1);	\
	    new_x3 = (new_x3==0)?X3m1:new_x3 - 1;			\
            break;                                                      \
        case 3:                                                         \
            new_mem_idx = ( (new_x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1); \
	    new_x4 = (new_x4==0)?X4m1:new_x4 - 1;			\
            break;                                                      \
        }                                                               \
    }while(0)

#endif


#define MULT_SU3_NN_TEST(ma, mb) do{				\
    Float fa_re,fa_im, fb_re, fb_im, fc_re, fc_im;		\
    fa_re =							\
      ma##00_re * mb##00_re - ma##00_im * mb##00_im +		\
	    ma##01_re * mb##10_re - ma##01_im * mb##10_im +	\
	    ma##02_re * mb##20_re - ma##02_im * mb##20_im;	\
	fa_im =							\
	    ma##00_re * mb##00_im + ma##00_im * mb##00_re +	\
	    ma##01_re * mb##10_im + ma##01_im * mb##10_re +	\
	    ma##02_re * mb##20_im + ma##02_im * mb##20_re;	\
	fb_re =							\
	    ma##00_re * mb##01_re - ma##00_im * mb##01_im +	\
	    ma##01_re * mb##11_re - ma##01_im * mb##11_im +	\
	    ma##02_re * mb##21_re - ma##02_im * mb##21_im;	\
	fb_im =							\
	    ma##00_re * mb##01_im + ma##00_im * mb##01_re +	\
	    ma##01_re * mb##11_im + ma##01_im * mb##11_re +	\
	    ma##02_re * mb##21_im + ma##02_im * mb##21_re;	\
	fc_re =							\
	    ma##00_re * mb##02_re - ma##00_im * mb##02_im +	\
	    ma##01_re * mb##12_re - ma##01_im * mb##12_im +	\
	    ma##02_re * mb##22_re - ma##02_im * mb##22_im;	\
	fc_im =							\
	    ma##00_re * mb##02_im + ma##00_im * mb##02_re +	\
	    ma##01_re * mb##12_im + ma##01_im * mb##12_re +	\
	    ma##02_re * mb##22_im + ma##02_im * mb##22_re;	\
	ma##00_re = fa_re;					\
	ma##00_im = fa_im;					\
	ma##01_re = fb_re;					\
	ma##01_im = fb_im;					\
	ma##02_re = fc_re;					\
	ma##02_im = fc_im;					\
	fa_re =							\
	    ma##10_re * mb##00_re - ma##10_im * mb##00_im +	\
	    ma##11_re * mb##10_re - ma##11_im * mb##10_im +	\
	    ma##12_re * mb##20_re - ma##12_im * mb##20_im;	\
	fa_im =							\
	    ma##10_re * mb##00_im + ma##10_im * mb##00_re +	\
	    ma##11_re * mb##10_im + ma##11_im * mb##10_re +	\
	    ma##12_re * mb##20_im + ma##12_im * mb##20_re;	\
	fb_re =							\
	    ma##10_re * mb##01_re - ma##10_im * mb##01_im +	\
	    ma##11_re * mb##11_re - ma##11_im * mb##11_im +	\
	    ma##12_re * mb##21_re - ma##12_im * mb##21_im;	\
	fb_im =							\
	    ma##10_re * mb##01_im + ma##10_im * mb##01_re +	\
	    ma##11_re * mb##11_im + ma##11_im * mb##11_re +	\
	    ma##12_re * mb##21_im + ma##12_im * mb##21_re;	\
	fc_re =							\
	    ma##10_re * mb##02_re - ma##10_im * mb##02_im +	\
	    ma##11_re * mb##12_re - ma##11_im * mb##12_im +	\
	    ma##12_re * mb##22_re - ma##12_im * mb##22_im;	\
	fc_im =							\
	    ma##10_re * mb##02_im + ma##10_im * mb##02_re +	\
	    ma##11_re * mb##12_im + ma##11_im * mb##12_re +	\
	    ma##12_re * mb##22_im + ma##12_im * mb##22_re;	\
	ma##10_re = fa_re;					\
	ma##10_im = fa_im;					\
	ma##11_re = fb_re;					\
	ma##11_im = fb_im;					\
	ma##12_re = fc_re;					\
	ma##12_im = fc_im;					\
	fa_re =							\
	    ma##20_re * mb##00_re - ma##20_im * mb##00_im +	\
	    ma##21_re * mb##10_re - ma##21_im * mb##10_im +	\
	    ma##22_re * mb##20_re - ma##22_im * mb##20_im;	\
	fa_im =							\
	    ma##20_re * mb##00_im + ma##20_im * mb##00_re +	\
	    ma##21_re * mb##10_im + ma##21_im * mb##10_re +	\
	    ma##22_re * mb##20_im + ma##22_im * mb##20_re;	\
	fb_re =							\
	    ma##20_re * mb##01_re - ma##20_im * mb##01_im +	\
	    ma##21_re * mb##11_re - ma##21_im * mb##11_im +	\
	    ma##22_re * mb##21_re - ma##22_im * mb##21_im;	\
	fb_im =							\
	    ma##20_re * mb##01_im + ma##20_im * mb##01_re +	\
	    ma##21_re * mb##11_im + ma##21_im * mb##11_re +	\
	    ma##22_re * mb##21_im + ma##22_im * mb##21_re;	\
	fc_re =							\
	    ma##20_re * mb##02_re - ma##20_im * mb##02_im +	\
	    ma##21_re * mb##12_re - ma##21_im * mb##12_im +	\
	    ma##22_re * mb##22_re - ma##22_im * mb##22_im;	\
	fc_im =							\
	    ma##20_re * mb##02_im + ma##20_im * mb##02_re +	\
	    ma##21_re * mb##12_im + ma##21_im * mb##12_re +	\
	    ma##22_re * mb##22_im + ma##22_im * mb##22_re;	\
	ma##20_re = fa_re;					\
	ma##20_im = fa_im;					\
	ma##21_re = fb_re;					\
	ma##21_im = fb_im;					\
	ma##22_re = fc_re;					\
	ma##22_im = fc_im;					\
    }while(0)


#define MULT_SU3_NA_TEST(ma, mb)	do{				\
	Float fa_re, fa_im, fb_re, fb_im, fc_re, fc_im;			\
	fa_re =								\
	    ma##00_re * mb##T00_re - ma##00_im * mb##T00_im +		\
	    ma##01_re * mb##T10_re - ma##01_im * mb##T10_im +		\
	    ma##02_re * mb##T20_re - ma##02_im * mb##T20_im;		\
	fa_im =								\
	    ma##00_re * mb##T00_im + ma##00_im * mb##T00_re +		\
	    ma##01_re * mb##T10_im + ma##01_im * mb##T10_re +		\
	    ma##02_re * mb##T20_im + ma##02_im * mb##T20_re;		\
	fb_re =								\
	    ma##00_re * mb##T01_re - ma##00_im * mb##T01_im +		\
	    ma##01_re * mb##T11_re - ma##01_im * mb##T11_im +		\
	    ma##02_re * mb##T21_re - ma##02_im * mb##T21_im;		\
	fb_im =								\
	    ma##00_re * mb##T01_im + ma##00_im * mb##T01_re +		\
	    ma##01_re * mb##T11_im + ma##01_im * mb##T11_re +		\
	    ma##02_re * mb##T21_im + ma##02_im * mb##T21_re;		\
	fc_re =								\
	    ma##00_re * mb##T02_re - ma##00_im * mb##T02_im +		\
	    ma##01_re * mb##T12_re - ma##01_im * mb##T12_im +		\
	    ma##02_re * mb##T22_re - ma##02_im * mb##T22_im;		\
	fc_im =								\
	    ma##00_re * mb##T02_im + ma##00_im * mb##T02_re +		\
	    ma##01_re * mb##T12_im + ma##01_im * mb##T12_re +		\
	    ma##02_re * mb##T22_im + ma##02_im * mb##T22_re;		\
	ma##00_re = fa_re;						\
	ma##00_im = fa_im;						\
	ma##01_re = fb_re;						\
	ma##01_im = fb_im;						\
	ma##02_re = fc_re;						\
	ma##02_im = fc_im;						\
	fa_re =								\
	    ma##10_re * mb##T00_re - ma##10_im * mb##T00_im +		\
	    ma##11_re * mb##T10_re - ma##11_im * mb##T10_im +		\
	    ma##12_re * mb##T20_re - ma##12_im * mb##T20_im;		\
	fa_im =								\
	    ma##10_re * mb##T00_im + ma##10_im * mb##T00_re +		\
	    ma##11_re * mb##T10_im + ma##11_im * mb##T10_re +		\
	    ma##12_re * mb##T20_im + ma##12_im * mb##T20_re;		\
	fb_re =								\
	    ma##10_re * mb##T01_re - ma##10_im * mb##T01_im +		\
	    ma##11_re * mb##T11_re - ma##11_im * mb##T11_im +		\
	    ma##12_re * mb##T21_re - ma##12_im * mb##T21_im;		\
	fb_im =								\
	    ma##10_re * mb##T01_im + ma##10_im * mb##T01_re +		\
	    ma##11_re * mb##T11_im + ma##11_im * mb##T11_re +		\
	    ma##12_re * mb##T21_im + ma##12_im * mb##T21_re;		\
	fc_re =								\
	    ma##10_re * mb##T02_re - ma##10_im * mb##T02_im +		\
	    ma##11_re * mb##T12_re - ma##11_im * mb##T12_im +		\
	    ma##12_re * mb##T22_re - ma##12_im * mb##T22_im;		\
	fc_im =								\
	    ma##10_re * mb##T02_im + ma##10_im * mb##T02_re +		\
	    ma##11_re * mb##T12_im + ma##11_im * mb##T12_re +		\
	    ma##12_re * mb##T22_im + ma##12_im * mb##T22_re;		\
	ma##10_re = fa_re;						\
	ma##10_im = fa_im;						\
	ma##11_re = fb_re;						\
	ma##11_im = fb_im;						\
	ma##12_re = fc_re;						\
	ma##12_im = fc_im;						\
	fa_re =								\
	    ma##20_re * mb##T00_re - ma##20_im * mb##T00_im +		\
	    ma##21_re * mb##T10_re - ma##21_im * mb##T10_im +		\
	    ma##22_re * mb##T20_re - ma##22_im * mb##T20_im;		\
	fa_im =								\
	    ma##20_re * mb##T00_im + ma##20_im * mb##T00_re +		\
	    ma##21_re * mb##T10_im + ma##21_im * mb##T10_re +		\
	    ma##22_re * mb##T20_im + ma##22_im * mb##T20_re;		\
	fb_re =								\
	    ma##20_re * mb##T01_re - ma##20_im * mb##T01_im +		\
	    ma##21_re * mb##T11_re - ma##21_im * mb##T11_im +		\
	    ma##22_re * mb##T21_re - ma##22_im * mb##T21_im;		\
	fb_im =								\
	    ma##20_re * mb##T01_im + ma##20_im * mb##T01_re +		\
	    ma##21_re * mb##T11_im + ma##21_im * mb##T11_re +		\
	    ma##22_re * mb##T21_im + ma##22_im * mb##T21_re;		\
	fc_re =								\
	    ma##20_re * mb##T02_re - ma##20_im * mb##T02_im +		\
	    ma##21_re * mb##T12_re - ma##21_im * mb##T12_im +		\
	    ma##22_re * mb##T22_re - ma##22_im * mb##T22_im;		\
	fc_im =								\
	    ma##20_re * mb##T02_im + ma##20_im * mb##T02_re +		\
	    ma##21_re * mb##T12_im + ma##21_im * mb##T12_re +		\
	    ma##22_re * mb##T22_im + ma##22_im * mb##T22_re;		\
	ma##20_re = fa_re;						\
	ma##20_im = fa_im;						\
	ma##21_re = fb_re;						\
	ma##21_im = fb_im;						\
	ma##22_re = fc_re;						\
	ma##22_im = fc_im;						\
    }while(0)



#define MULT_SU3_AN_TEST(ma, mb)	do{				\
	Float fa_re, fa_im, fb_re, fb_im, fc_re, fc_im;			\
	fa_re =								\
	    ma##T00_re * mb##00_re - ma##T00_im * mb##00_im +		\
	    ma##T01_re * mb##10_re - ma##T01_im * mb##10_im +		\
	    ma##T02_re * mb##20_re - ma##T02_im * mb##20_im;		\
	fa_im =								\
	    ma##T00_re * mb##00_im + ma##T00_im * mb##00_re +		\
	    ma##T01_re * mb##10_im + ma##T01_im * mb##10_re +		\
	    ma##T02_re * mb##20_im + ma##T02_im * mb##20_re;		\
	fb_re =								\
	    ma##T10_re * mb##00_re - ma##T10_im * mb##00_im +		\
	    ma##T11_re * mb##10_re - ma##T11_im * mb##10_im +		\
	    ma##T12_re * mb##20_re - ma##T12_im * mb##20_im;		\
	fb_im =								\
	    ma##T10_re * mb##00_im + ma##T10_im * mb##00_re +		\
	    ma##T11_re * mb##10_im + ma##T11_im * mb##10_re +		\
	    ma##T12_re * mb##20_im + ma##T12_im * mb##20_re;		\
	fc_re =								\
	    ma##T20_re * mb##00_re - ma##T20_im * mb##00_im +		\
	    ma##T21_re * mb##10_re - ma##T21_im * mb##10_im +		\
	    ma##T22_re * mb##20_re - ma##T22_im * mb##20_im;		\
	fc_im =								\
	    ma##T20_re * mb##00_im + ma##T20_im * mb##00_re +		\
	    ma##T21_re * mb##10_im + ma##T21_im * mb##10_re +		\
	    ma##T22_re * mb##20_im + ma##T22_im * mb##20_re;		\
	mb##00_re = fa_re;						\
	mb##00_im = fa_im;						\
	mb##10_re = fb_re;						\
	mb##10_im = fb_im;						\
	mb##20_re = fc_re;						\
	mb##20_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##01_re - ma##T00_im * mb##01_im +		\
	    ma##T01_re * mb##11_re - ma##T01_im * mb##11_im +		\
	    ma##T02_re * mb##21_re - ma##T02_im * mb##21_im;		\
	fa_im =								\
	    ma##T00_re * mb##01_im + ma##T00_im * mb##01_re +		\
	    ma##T01_re * mb##11_im + ma##T01_im * mb##11_re +		\
	    ma##T02_re * mb##21_im + ma##T02_im * mb##21_re;		\
	fb_re =								\
	    ma##T10_re * mb##01_re - ma##T10_im * mb##01_im +		\
	    ma##T11_re * mb##11_re - ma##T11_im * mb##11_im +		\
	    ma##T12_re * mb##21_re - ma##T12_im * mb##21_im;		\
	fb_im =								\
	    ma##T10_re * mb##01_im + ma##T10_im * mb##01_re +		\
	    ma##T11_re * mb##11_im + ma##T11_im * mb##11_re +		\
	    ma##T12_re * mb##21_im + ma##T12_im * mb##21_re;		\
	fc_re =								\
	    ma##T20_re * mb##01_re - ma##T20_im * mb##01_im +		\
	    ma##T21_re * mb##11_re - ma##T21_im * mb##11_im +		\
	    ma##T22_re * mb##21_re - ma##T22_im * mb##21_im;		\
	fc_im =								\
	    ma##T20_re * mb##01_im + ma##T20_im * mb##01_re +		\
	    ma##T21_re * mb##11_im + ma##T21_im * mb##11_re +		\
	    ma##T22_re * mb##21_im + ma##T22_im * mb##21_re;		\
	mb##01_re = fa_re;						\
	mb##01_im = fa_im;						\
	mb##11_re = fb_re;						\
	mb##11_im = fb_im;						\
	mb##21_re = fc_re;						\
	mb##21_im = fc_im;						\
	fa_re =								\
	    ma##T00_re * mb##02_re - ma##T00_im * mb##02_im +		\
	    ma##T01_re * mb##12_re - ma##T01_im * mb##12_im +		\
	    ma##T02_re * mb##22_re - ma##T02_im * mb##22_im;		\
	fa_im =								\
	    ma##T00_re * mb##02_im + ma##T00_im * mb##02_re +		\
	    ma##T01_re * mb##12_im + ma##T01_im * mb##12_re +		\
	    ma##T02_re * mb##22_im + ma##T02_im * mb##22_re;		\
	fb_re =								\
	    ma##T10_re * mb##02_re - ma##T10_im * mb##02_im +		\
	    ma##T11_re * mb##12_re - ma##T11_im * mb##12_im +		\
	    ma##T12_re * mb##22_re - ma##T12_im * mb##22_im;		\
	fb_im =								\
	    ma##T10_re * mb##02_im + ma##T10_im * mb##02_re +		\
	    ma##T11_re * mb##12_im + ma##T11_im * mb##12_re +		\
	    ma##T12_re * mb##22_im + ma##T12_im * mb##22_re;		\
	fc_re =								\
	    ma##T20_re * mb##02_re - ma##T20_im * mb##02_im +		\
	    ma##T21_re * mb##12_re - ma##T21_im * mb##12_im +		\
	    ma##T22_re * mb##22_re - ma##T22_im * mb##22_im;		\
	fc_im =								\
	    ma##T20_re * mb##02_im + ma##T20_im * mb##02_re +		\
	    ma##T21_re * mb##12_im + ma##T21_im * mb##12_re +		\
	    ma##T22_re * mb##22_im + ma##T22_im * mb##22_re;		\
	mb##02_re = fa_re;						\
	mb##02_im = fa_im;						\
	mb##12_re = fb_re;						\
	mb##12_im = fb_im;						\
	mb##22_re = fc_re;						\
	mb##22_im = fc_im;						\
    }while(0)



#define print_matrix(mul)						\
  printf(" (%f %f) (%f %f) (%f %f)\n", mul##00_re, mul##00_im, mul##01_re, mul##01_im, mul##02_re, mul##02_im);	\
  printf(" (%f %f) (%f %f) (%f %f)\n", mul##10_re, mul##10_im, mul##11_re, mul##11_im, mul##12_re, mul##12_im);	\
  printf(" (%f %f) (%f %f) (%f %f)\n", mul##20_re, mul##20_im, mul##21_re, mul##21_im, mul##22_re, mul##22_im);


//FloatN can be float2/float4/double2
//Float2 can be float2/double2
template<int oddBit, typename Float2, typename FloatN, typename Float>
  __global__ void
  GAUGE_FORCE_KERN_NAME(Float2* momEven, Float2* momOdd,
			int dir, double eb3,
			FloatN* linkEven, FloatN* linkOdd,
			int* input_path, 
			int* length, Float* path_coeff, int num_paths, kernel_param_t kparam)
{
  int i,j=0;
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  
  int z1 = sid / X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1 / X2;
  int x2 = z1 - z2*X2;
  int x4 = z2 / X3;
  int x3 = z2 - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2*x1h + x1odd;  

#ifdef MULTI_GPU
  x4 += 2; x3 += 2; x2 += 2; x1 += 2;
  int X = x4*E3E2E1 + x3*E2E1 + x2*E1 + x1;
#else
  int X = 2*sid + x1odd;  
#endif
    
  Float2* mymom=momEven;
  if (oddBit){
    mymom = momOdd;
  }

  DECLARE_LINK_VARS(LINKA);
  DECLARE_LINK_VARS(LINKB);
  Float2 STAPLE0, STAPLE1, STAPLE2, STAPLE3,STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
  Float2 AH0, AH1, AH2, AH3, AH4;

  int new_mem_idx;
    
    
  SET_SU3_MATRIX(staple, 0);
  for(i=0;i < num_paths; i++){
    int nbr_oddbit = (oddBit^1 );
	
    int new_x1 =x1;
    int new_x2 =x2;
    int new_x3 =x3;
    int new_x4 =x4;
    COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(dir, X);
	
    //linka: current matrix
    //linkb: the loaded matrix in this round	
    SET_UNIT_SU3_MATRIX(linka);	
    int* path = input_path + i*path_max_length;
	
    int lnkdir;
    int path0 = path[0];
    if (GOES_FORWARDS(path0)){
      lnkdir=path0;
    }else{
      lnkdir=OPP_DIR(path0);
      COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(path0), new_mem_idx);
      nbr_oddbit = nbr_oddbit^1;
	    
    }
	
    int nbr_idx = new_mem_idx >>1;
    if (nbr_oddbit){
      LOAD_ODD_MATRIX( lnkdir, nbr_idx, LINKB);
    }else{
      LOAD_EVEN_MATRIX( lnkdir, nbr_idx, LINKB);
    }
    RECONSTRUCT_MATRIX(1, linkb);
    
    if (GOES_FORWARDS(path0)){
      COPY_SU3_MATRIX(linkb, linka);
      COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(path0, new_mem_idx);
      nbr_oddbit = nbr_oddbit^1;
    }else{
      SU3_ADJOINT(linkb, linka);
    }	
	
    for(j=1; j < length[i]; j++){
	    
      int lnkdir;
      int pathj = path[j];
      if (GOES_FORWARDS(pathj)){
	lnkdir=pathj;
      }else{
	lnkdir=OPP_DIR(pathj);
	COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(pathj), new_mem_idx);
	nbr_oddbit = nbr_oddbit^1;

      }
	    
      int nbr_idx = new_mem_idx >>1;
      if (nbr_oddbit){
	LOAD_ODD_MATRIX(lnkdir, nbr_idx, LINKB);
      }else{
	LOAD_EVEN_MATRIX(lnkdir, nbr_idx, LINKB);
      }
      RECONSTRUCT_MATRIX(1, linkb);
      if (GOES_FORWARDS(pathj)){
	MULT_SU3_NN_TEST(linka, linkb);
		
	COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(pathj, new_mem_idx);
	nbr_oddbit = nbr_oddbit^1;
		
		
      }else{
	MULT_SU3_NA_TEST(linka, linkb);		
      }
	    
    }//j
    SCALAR_MULT_ADD_SU3_MATRIX(staple, linka, path_coeff[i], staple);
  }//i
    

  //update mom 
  if (oddBit){
    LOAD_ODD_MATRIX(dir, (X>>1), LINKA);
  }else{
    LOAD_EVEN_MATRIX(dir, (X>>1), LINKA);
  }
  RECONSTRUCT_MATRIX(1, linka);
  MULT_SU3_NN_TEST(linka, staple);
  LOAD_ANTI_HERMITIAN(mymom, dir, sid, AH);
  UNCOMPRESS_ANTI_HERMITIAN(ah, linkb);
  SCALAR_MULT_SUB_SU3_MATRIX(linkb, linka, eb3, linka);
  MAKE_ANTI_HERMITIAN(linka, ah);
    
  WRITE_ANTI_HERMITIAN(mymom, dir, sid, AH, mom_ga_stride);

  return;
}



#undef COMPUTE_NEW_FULL_IDX_PLUS_UPDATE
#undef COMPUTE_NEW_FULL_IDX_MINUS_UPDATE
#undef MULT_SU3_NN_TEST
#undef MULT_SU3_NA_TEST
#undef MULT_SU3_AN_TEST

#undef linka00_re 
#undef linka00_im 
#undef linka01_re 
#undef linka01_im 
#undef linka02_re 
#undef linka02_im 
#undef linka10_re 
#undef linka10_im 
#undef linka11_re 
#undef linka11_im 
#undef linka12_re 
#undef linka12_im 
#undef linka20_re 
#undef linka20_im 
#undef linka21_re 
#undef linka21_im 
#undef linka22_re 
#undef linka22_im 

#undef linkb00_re 
#undef linkb00_im 
#undef linkb01_re 
#undef linkb01_im 
#undef linkb02_re 
#undef linkb02_im 
#undef linkb10_re 
#undef linkb10_im 
#undef linkb11_re 
#undef linkb11_im 
#undef linkb12_re 
#undef linkb12_im 
#undef linkb20_re 
#undef linkb20_im 
#undef linkb21_re 
#undef linkb21_im 
#undef linkb22_re 
#undef linkb22_im 
