
#ifndef __KERNEL_COMMOM_MACRO_H__
#define __KERNEL_COMMOM_MACRO_H__

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir)	(7-(dir))
#define GOES_FORWARDS(dir) (dir<=3)
#define GOES_BACKWARDS(dir) (dir>3)


#define linkaT00_re (+linka00_re)
#define linkaT00_im (-linka00_im)
#define linkaT01_re (+linka10_re)
#define linkaT01_im (-linka10_im)
#define linkaT02_re (+linka20_re)
#define linkaT02_im (-linka20_im)
#define linkaT10_re (+linka01_re)
#define linkaT10_im (-linka01_im)
#define linkaT11_re (+linka11_re)
#define linkaT11_im (-linka11_im)
#define linkaT12_re (+linka21_re)
#define linkaT12_im (-linka21_im)
#define linkaT20_re (+linka02_re)
#define linkaT20_im (-linka02_im)
#define linkaT21_re (+linka12_re)
#define linkaT21_im (-linka12_im)
#define linkaT22_re (+linka22_re)
#define linkaT22_im (-linka22_im)


#define linkbT00_re (+linkb00_re)
#define linkbT00_im (-linkb00_im)
#define linkbT01_re (+linkb10_re)
#define linkbT01_im (-linkb10_im)
#define linkbT02_re (+linkb20_re)
#define linkbT02_im (-linkb20_im)
#define linkbT10_re (+linkb01_re)
#define linkbT10_im (-linkb01_im)
#define linkbT11_re (+linkb11_re)
#define linkbT11_im (-linkb11_im)
#define linkbT12_re (+linkb21_re)
#define linkbT12_im (-linkb21_im)
#define linkbT20_re (+linkb02_re)
#define linkbT20_im (-linkb02_im)
#define linkbT21_re (+linkb12_re)
#define linkbT21_im (-linkb12_im)
#define linkbT22_re (+linkb22_re)
#define linkbT22_im (-linkb22_im)




#define linkc00_re LINKC0.x
#define linkc00_im LINKC0.y
#define linkc01_re LINKC0.z
#define linkc01_im LINKC0.w
#define linkc02_re LINKC1.x
#define linkc02_im LINKC1.y
#define linkc10_re LINKC1.z
#define linkc10_im LINKC1.w
#define linkc11_re LINKC2.x
#define linkc11_im LINKC2.y
#define linkc12_re LINKC2.z
#define linkc12_im LINKC2.w
#define linkc20_re LINKC3.x
#define linkc20_im LINKC3.y
#define linkc21_re LINKC3.z
#define linkc21_im LINKC3.w
#define linkc22_re LINKC4.x
#define linkc22_im LINKC4.y

#define linkcT00_re (+linkc00_re)
#define linkcT00_im (-linkc00_im)
#define linkcT01_re (+linkc10_re)
#define linkcT01_im (-linkc10_im)
#define linkcT02_re (+linkc20_re)
#define linkcT02_im (-linkc20_im)
#define linkcT10_re (+linkc01_re)
#define linkcT10_im (-linkc01_im)
#define linkcT11_re (+linkc11_re)
#define linkcT11_im (-linkc11_im)
#define linkcT12_re (+linkc21_re)
#define linkcT12_im (-linkc21_im)
#define linkcT20_re (+linkc02_re)
#define linkcT20_im (-linkc02_im)
#define linkcT21_re (+linkc12_re)
#define linkcT21_im (-linkc12_im)
#define linkcT22_re (+linkc22_re)
#define linkcT22_im (-linkc22_im)


#define staple00_re STAPLE0.x
#define staple00_im STAPLE0.y
#define staple01_re STAPLE1.x
#define staple01_im STAPLE1.y
#define staple02_re STAPLE2.x
#define staple02_im STAPLE2.y
#define staple10_re STAPLE3.x
#define staple10_im STAPLE3.y
#define staple11_re STAPLE4.x
#define staple11_im STAPLE4.y
#define staple12_re STAPLE5.x
#define staple12_im STAPLE5.y
#define staple20_re STAPLE6.x
#define staple20_im STAPLE6.y
#define staple21_re STAPLE7.x
#define staple21_im STAPLE7.y
#define staple22_re STAPLE8.x
#define staple22_im STAPLE8.y

#define stapleT00_re (+staple00_re)
#define stapleT00_im (-staple00_im)
#define stapleT01_re (+staple10_re)
#define stapleT01_im (-staple10_im)
#define stapleT02_re (+staple20_re)
#define stapleT02_im (-staple20_im)
#define stapleT10_re (+staple01_re)
#define stapleT10_im (-staple01_im)
#define stapleT11_re (+staple11_re)
#define stapleT11_im (-staple11_im)
#define stapleT12_re (+staple21_re)
#define stapleT12_im (-staple21_im)
#define stapleT20_re (+staple02_re)
#define stapleT20_im (-staple02_im)
#define stapleT21_re (+staple12_re)
#define stapleT21_im (-staple12_im)
#define stapleT22_re (+staple22_re)
#define stapleT22_im (-staple22_im)

#ifdef FERMI_NO_DBLE_TEX
#define READ_DOUBLE2_TEXTURE(x_tex, x, i)      (x)[i]
#else
#define READ_DOUBLE2_TEXTURE(x_tex, x, i)  fetch_double2(x_tex, i)
#endif


#define LOAD_MATRIX_12_SINGLE(gauge, dir, idx, var, stride)do{		\
    var##0 = gauge[idx + dir*stride*3];					\
    var##1 = gauge[idx + dir*stride*3 + stride];			\
    var##2 = gauge[idx + dir*stride*3 + stride*2];			\
  }while(0)

#define LOAD_MATRIX_12_SINGLE_TEX(gauge, dir, idx, var, stride)do{	\
    var##0 = tex1Dfetch(gauge, idx + dir*stride*3);			\
    var##1 = tex1Dfetch(gauge, idx + dir*stride*3 + stride);		\
    var##2 = tex1Dfetch(gauge, idx + dir*stride*3 + stride*2);		\
  }while(0)

#define LOAD_MATRIX_12_DOUBLE(gauge, dir, idx, var, stride)do{		\
    var##0 = gauge[idx + dir*stride*6];					\
    var##1 = gauge[idx + dir*stride*6 + stride];			\
    var##2 = gauge[idx + dir*stride*6 + stride*2];			\
    var##3 = gauge[idx + dir*stride*6 + stride*3];			\
    var##4 = gauge[idx + dir*stride*6 + stride*4];			\
    var##5 = gauge[idx + dir*stride*6 + stride*5];			\
  }while(0)

#define LOAD_MATRIX_12_DOUBLE_TEX(gauge_tex, gauge, dir, idx, var, stride)do{ \
    var##0 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6); \
    var##1 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6 + stride); \
    var##2 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6 + stride*2); \
    var##3 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6 + stride*3); \
    var##4 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6 + stride*4); \
    var##5 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*6 + stride*5); \
  }while(0)

#define LOAD_MATRIX_18(gauge, dir, idx, var, stride)do{			\
    var##0 = gauge[idx + dir*stride*9];					\
    var##1 = gauge[idx + dir*stride*9 + stride];			\
    var##2 = gauge[idx + dir*stride*9 + stride*2];			\
    var##3 = gauge[idx + dir*stride*9 + stride*3];			\
    var##4 = gauge[idx + dir*stride*9 + stride*4];			\
    var##5 = gauge[idx + dir*stride*9 + stride*5];			\
    var##6 = gauge[idx + dir*stride*9 + stride*6];			\
    var##7 = gauge[idx + dir*stride*9 + stride*7];			\
    var##8 = gauge[idx + dir*stride*9 + stride*8];			\
  }while(0)

#define LOAD_MATRIX_18_SINGLE_TEX(gauge, dir, idx, var, stride)do{	\
    var##0 = tex1Dfetch(gauge, idx + dir*stride*9);			\
    var##1 = tex1Dfetch(gauge, idx + dir*stride*9 + stride);		\
    var##2 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*2);		\
    var##3 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*3);		\
    var##4 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*4);		\
    var##5 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*5);		\
    var##6 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*6);		\
    var##7 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*7);		\
    var##8 = tex1Dfetch(gauge, idx + dir*stride*9 + stride*8);		\
  }while(0)

#define LOAD_MATRIX_18_DOUBLE_TEX(gauge_tex, gauge, dir, idx, var, stride)do{ \
    var##0 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9); \
    var##1 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride); \
    var##2 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*2); \
    var##3 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*3); \
    var##4 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*4); \
    var##5 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*5); \
    var##6 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*6); \
    var##7 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*7); \
    var##8 = READ_DOUBLE2_TEXTURE(gauge_tex, gauge, idx + dir*stride*9 + stride*8); \
  }while(0)

#define MULT_SU3_NN(ma, mb, mc)					\
    mc##00_re =							\
	ma##00_re * mb##00_re - ma##00_im * mb##00_im +		\
	ma##01_re * mb##10_re - ma##01_im * mb##10_im +		\
	ma##02_re * mb##20_re - ma##02_im * mb##20_im;		\
    mc##00_im =							\
	ma##00_re * mb##00_im + ma##00_im * mb##00_re +		\
	ma##01_re * mb##10_im + ma##01_im * mb##10_re +		\
	ma##02_re * mb##20_im + ma##02_im * mb##20_re;		\
    mc##10_re =							\
	ma##10_re * mb##00_re - ma##10_im * mb##00_im +		\
	ma##11_re * mb##10_re - ma##11_im * mb##10_im +		\
	ma##12_re * mb##20_re - ma##12_im * mb##20_im;		\
    mc##10_im =							\
	ma##10_re * mb##00_im + ma##10_im * mb##00_re +		\
	ma##11_re * mb##10_im + ma##11_im * mb##10_re +		\
	ma##12_re * mb##20_im + ma##12_im * mb##20_re;		\
    mc##20_re =							\
	ma##20_re * mb##00_re - ma##20_im * mb##00_im +		\
	ma##21_re * mb##10_re - ma##21_im * mb##10_im +		\
	ma##22_re * mb##20_re - ma##22_im * mb##20_im;		\
    mc##20_im =							\
	ma##20_re * mb##00_im + ma##20_im * mb##00_re +		\
	ma##21_re * mb##10_im + ma##21_im * mb##10_re +		\
	ma##22_re * mb##20_im + ma##22_im * mb##20_re;		\
    mc##01_re =							\
	ma##00_re * mb##01_re - ma##00_im * mb##01_im +		\
	ma##01_re * mb##11_re - ma##01_im * mb##11_im +		\
	ma##02_re * mb##21_re - ma##02_im * mb##21_im;		\
    mc##01_im =							\
	ma##00_re * mb##01_im + ma##00_im * mb##01_re +		\
	ma##01_re * mb##11_im + ma##01_im * mb##11_re +		\
	ma##02_re * mb##21_im + ma##02_im * mb##21_re;		\
    mc##11_re =							\
	ma##10_re * mb##01_re - ma##10_im * mb##01_im +		\
	ma##11_re * mb##11_re - ma##11_im * mb##11_im +		\
	ma##12_re * mb##21_re - ma##12_im * mb##21_im;		\
    mc##11_im =							\
	ma##10_re * mb##01_im + ma##10_im * mb##01_re +		\
	ma##11_re * mb##11_im + ma##11_im * mb##11_re +		\
	ma##12_re * mb##21_im + ma##12_im * mb##21_re;		\
    mc##21_re =							\
	ma##20_re * mb##01_re - ma##20_im * mb##01_im +		\
	ma##21_re * mb##11_re - ma##21_im * mb##11_im +		\
	ma##22_re * mb##21_re - ma##22_im * mb##21_im;		\
    mc##21_im =							\
	ma##20_re * mb##01_im + ma##20_im * mb##01_re +		\
	ma##21_re * mb##11_im + ma##21_im * mb##11_re +		\
	ma##22_re * mb##21_im + ma##22_im * mb##21_re;		\
    mc##02_re =							\
	ma##00_re * mb##02_re - ma##00_im * mb##02_im +		\
	ma##01_re * mb##12_re - ma##01_im * mb##12_im +		\
	ma##02_re * mb##22_re - ma##02_im * mb##22_im;		\
    mc##02_im =							\
	ma##00_re * mb##02_im + ma##00_im * mb##02_re +		\
	ma##01_re * mb##12_im + ma##01_im * mb##12_re +		\
	ma##02_re * mb##22_im + ma##02_im * mb##22_re;		\
    mc##12_re =							\
	ma##10_re * mb##02_re - ma##10_im * mb##02_im +		\
	ma##11_re * mb##12_re - ma##11_im * mb##12_im +		\
	ma##12_re * mb##22_re - ma##12_im * mb##22_im;		\
    mc##12_im =							\
	ma##10_re * mb##02_im + ma##10_im * mb##02_re +		\
	ma##11_re * mb##12_im + ma##11_im * mb##12_re +		\
	ma##12_re * mb##22_im + ma##12_im * mb##22_re;		\
    mc##22_re =							\
	ma##20_re * mb##02_re - ma##20_im * mb##02_im +		\
	ma##21_re * mb##12_re - ma##21_im * mb##12_im +		\
	ma##22_re * mb##22_re - ma##22_im * mb##22_im;		\
    mc##22_im =							\
	ma##20_re * mb##02_im + ma##20_im * mb##02_re +		\
	ma##21_re * mb##12_im + ma##21_im * mb##12_re +		\
	ma##22_re * mb##22_im + ma##22_im * mb##22_re;



#define MULT_SU3_NA(ma, mb, mc)						\
    mc##00_re =								\
	ma##00_re * mb##T00_re - ma##00_im * mb##T00_im +		\
	ma##01_re * mb##T10_re - ma##01_im * mb##T10_im +		\
	ma##02_re * mb##T20_re - ma##02_im * mb##T20_im;		\
    mc##00_im =								\
	ma##00_re * mb##T00_im + ma##00_im * mb##T00_re +		\
	ma##01_re * mb##T10_im + ma##01_im * mb##T10_re +		\
	ma##02_re * mb##T20_im + ma##02_im * mb##T20_re;		\
    mc##10_re =								\
	ma##10_re * mb##T00_re - ma##10_im * mb##T00_im +		\
	ma##11_re * mb##T10_re - ma##11_im * mb##T10_im +		\
	ma##12_re * mb##T20_re - ma##12_im * mb##T20_im;		\
    mc##10_im =								\
	ma##10_re * mb##T00_im + ma##10_im * mb##T00_re +		\
	ma##11_re * mb##T10_im + ma##11_im * mb##T10_re +		\
	ma##12_re * mb##T20_im + ma##12_im * mb##T20_re;		\
    mc##20_re =								\
	ma##20_re * mb##T00_re - ma##20_im * mb##T00_im +		\
	ma##21_re * mb##T10_re - ma##21_im * mb##T10_im +		\
	ma##22_re * mb##T20_re - ma##22_im * mb##T20_im;		\
    mc##20_im =								\
	ma##20_re * mb##T00_im + ma##20_im * mb##T00_re +		\
	ma##21_re * mb##T10_im + ma##21_im * mb##T10_re +		\
	ma##22_re * mb##T20_im + ma##22_im * mb##T20_re;		\
    mc##01_re =								\
	ma##00_re * mb##T01_re - ma##00_im * mb##T01_im +		\
	ma##01_re * mb##T11_re - ma##01_im * mb##T11_im +		\
	ma##02_re * mb##T21_re - ma##02_im * mb##T21_im;		\
    mc##01_im =								\
	ma##00_re * mb##T01_im + ma##00_im * mb##T01_re +		\
	ma##01_re * mb##T11_im + ma##01_im * mb##T11_re +		\
	ma##02_re * mb##T21_im + ma##02_im * mb##T21_re;		\
    mc##11_re =								\
	ma##10_re * mb##T01_re - ma##10_im * mb##T01_im +		\
	ma##11_re * mb##T11_re - ma##11_im * mb##T11_im +		\
	ma##12_re * mb##T21_re - ma##12_im * mb##T21_im;		\
    mc##11_im =								\
	ma##10_re * mb##T01_im + ma##10_im * mb##T01_re +		\
	ma##11_re * mb##T11_im + ma##11_im * mb##T11_re +		\
	ma##12_re * mb##T21_im + ma##12_im * mb##T21_re;		\
    mc##21_re =								\
	ma##20_re * mb##T01_re - ma##20_im * mb##T01_im +		\
	ma##21_re * mb##T11_re - ma##21_im * mb##T11_im +		\
	ma##22_re * mb##T21_re - ma##22_im * mb##T21_im;		\
    mc##21_im =								\
	ma##20_re * mb##T01_im + ma##20_im * mb##T01_re +		\
	ma##21_re * mb##T11_im + ma##21_im * mb##T11_re +		\
	ma##22_re * mb##T21_im + ma##22_im * mb##T21_re;		\
    mc##02_re =								\
	ma##00_re * mb##T02_re - ma##00_im * mb##T02_im +		\
	ma##01_re * mb##T12_re - ma##01_im * mb##T12_im +		\
	ma##02_re * mb##T22_re - ma##02_im * mb##T22_im;		\
    mc##02_im =								\
	ma##00_re * mb##T02_im + ma##00_im * mb##T02_re +		\
	ma##01_re * mb##T12_im + ma##01_im * mb##T12_re +		\
	ma##02_re * mb##T22_im + ma##02_im * mb##T22_re;		\
    mc##12_re =								\
	ma##10_re * mb##T02_re - ma##10_im * mb##T02_im +		\
	ma##11_re * mb##T12_re - ma##11_im * mb##T12_im +		\
	ma##12_re * mb##T22_re - ma##12_im * mb##T22_im;		\
    mc##12_im =								\
	ma##10_re * mb##T02_im + ma##10_im * mb##T02_re +		\
	ma##11_re * mb##T12_im + ma##11_im * mb##T12_re +		\
	ma##12_re * mb##T22_im + ma##12_im * mb##T22_re;		\
    mc##22_re =								\
	ma##20_re * mb##T02_re - ma##20_im * mb##T02_im +		\
	ma##21_re * mb##T12_re - ma##21_im * mb##T12_im +		\
	ma##22_re * mb##T22_re - ma##22_im * mb##T22_im;		\
    mc##22_im =								\
	ma##20_re * mb##T02_im + ma##20_im * mb##T02_re +		\
	ma##21_re * mb##T12_im + ma##21_im * mb##T12_re +		\
	ma##22_re * mb##T22_im + ma##22_im * mb##T22_re;



#define MULT_SU3_AN(ma, mb, mc)						\
    mc##00_re =								\
	ma##T00_re * mb##00_re - ma##T00_im * mb##00_im +		\
	ma##T01_re * mb##10_re - ma##T01_im * mb##10_im +		\
	ma##T02_re * mb##20_re - ma##T02_im * mb##20_im;		\
    mc##00_im =								\
	ma##T00_re * mb##00_im + ma##T00_im * mb##00_re +		\
	ma##T01_re * mb##10_im + ma##T01_im * mb##10_re +		\
	ma##T02_re * mb##20_im + ma##T02_im * mb##20_re;		\
    mc##10_re =								\
	ma##T10_re * mb##00_re - ma##T10_im * mb##00_im +		\
	ma##T11_re * mb##10_re - ma##T11_im * mb##10_im +		\
	ma##T12_re * mb##20_re - ma##T12_im * mb##20_im;		\
    mc##10_im =								\
	ma##T10_re * mb##00_im + ma##T10_im * mb##00_re +		\
	ma##T11_re * mb##10_im + ma##T11_im * mb##10_re +		\
	ma##T12_re * mb##20_im + ma##T12_im * mb##20_re;		\
    mc##20_re =								\
	ma##T20_re * mb##00_re - ma##T20_im * mb##00_im +		\
	ma##T21_re * mb##10_re - ma##T21_im * mb##10_im +		\
	ma##T22_re * mb##20_re - ma##T22_im * mb##20_im;		\
    mc##20_im =								\
	ma##T20_re * mb##00_im + ma##T20_im * mb##00_re +		\
	ma##T21_re * mb##10_im + ma##T21_im * mb##10_re +		\
	ma##T22_re * mb##20_im + ma##T22_im * mb##20_re;		\
    mc##01_re =								\
	ma##T00_re * mb##01_re - ma##T00_im * mb##01_im +		\
	ma##T01_re * mb##11_re - ma##T01_im * mb##11_im +		\
	ma##T02_re * mb##21_re - ma##T02_im * mb##21_im;		\
    mc##01_im =								\
	ma##T00_re * mb##01_im + ma##T00_im * mb##01_re +		\
	ma##T01_re * mb##11_im + ma##T01_im * mb##11_re +		\
	ma##T02_re * mb##21_im + ma##T02_im * mb##21_re;		\
    mc##11_re =								\
	ma##T10_re * mb##01_re - ma##T10_im * mb##01_im +		\
	ma##T11_re * mb##11_re - ma##T11_im * mb##11_im +		\
	ma##T12_re * mb##21_re - ma##T12_im * mb##21_im;		\
    mc##11_im =								\
	ma##T10_re * mb##01_im + ma##T10_im * mb##01_re +		\
	ma##T11_re * mb##11_im + ma##T11_im * mb##11_re +		\
	ma##T12_re * mb##21_im + ma##T12_im * mb##21_re;		\
    mc##21_re =								\
	ma##T20_re * mb##01_re - ma##T20_im * mb##01_im +		\
	ma##T21_re * mb##11_re - ma##T21_im * mb##11_im +		\
	ma##T22_re * mb##21_re - ma##T22_im * mb##21_im;		\
    mc##21_im =								\
	ma##T20_re * mb##01_im + ma##T20_im * mb##01_re +		\
	ma##T21_re * mb##11_im + ma##T21_im * mb##11_re +		\
	ma##T22_re * mb##21_im + ma##T22_im * mb##21_re;		\
    mc##02_re =								\
	ma##T00_re * mb##02_re - ma##T00_im * mb##02_im +		\
	ma##T01_re * mb##12_re - ma##T01_im * mb##12_im +		\
	ma##T02_re * mb##22_re - ma##T02_im * mb##22_im;		\
    mc##02_im =								\
	ma##T00_re * mb##02_im + ma##T00_im * mb##02_re +		\
	ma##T01_re * mb##12_im + ma##T01_im * mb##12_re +		\
	ma##T02_re * mb##22_im + ma##T02_im * mb##22_re;		\
    mc##12_re =								\
	ma##T10_re * mb##02_re - ma##T10_im * mb##02_im +		\
	ma##T11_re * mb##12_re - ma##T11_im * mb##12_im +		\
	ma##T12_re * mb##22_re - ma##T12_im * mb##22_im;		\
    mc##12_im =								\
	ma##T10_re * mb##02_im + ma##T10_im * mb##02_re +		\
	ma##T11_re * mb##12_im + ma##T11_im * mb##12_re +		\
	ma##T12_re * mb##22_im + ma##T12_im * mb##22_re;		\
    mc##22_re =								\
	ma##T20_re * mb##02_re - ma##T20_im * mb##02_im +		\
	ma##T21_re * mb##12_re - ma##T21_im * mb##12_im +		\
	ma##T22_re * mb##22_re - ma##T22_im * mb##22_im;		\
    mc##22_im =								\
	ma##T20_re * mb##02_im + ma##T20_im * mb##02_re +		\
	ma##T21_re * mb##12_im + ma##T21_im * mb##12_re +		\
	ma##T22_re * mb##22_im + ma##T22_im * mb##22_re;

#define SET_SU3_MATRIX(a, value)		\
    a##00_re = value;				\
    a##00_im = value;				\
    a##01_re = value;				\
    a##01_im = value;				\
    a##02_re = value;				\
    a##02_im = value;				\
    a##10_re = value;				\
    a##10_im = value;				\
    a##11_re = value;				\
    a##11_im = value;				\
    a##12_re = value;				\
    a##12_im = value;				\
    a##20_re = value;				\
    a##20_im = value;				\
    a##21_re = value;				\
    a##21_im = value;				\
    a##22_re = value;				\
    a##22_im = value;				\

#define SCALAR_MULT_ADD_SU3_MATRIX(ma, mb, s, mc)	\
    mc##00_re = ma##00_re + mb##00_re * s;		\
    mc##00_im = ma##00_im + mb##00_im * s;		\
    mc##01_re = ma##01_re + mb##01_re * s;		\
    mc##01_im = ma##01_im + mb##01_im * s;		\
    mc##02_re = ma##02_re + mb##02_re * s;		\
    mc##02_im = ma##02_im + mb##02_im * s;		\
    mc##10_re = ma##10_re + mb##10_re * s;		\
    mc##10_im = ma##10_im + mb##10_im * s;		\
    mc##11_re = ma##11_re + mb##11_re * s;		\
    mc##11_im = ma##11_im + mb##11_im * s;		\
    mc##12_re = ma##12_re + mb##12_re * s;		\
    mc##12_im = ma##12_im + mb##12_im * s;		\
    mc##20_re = ma##20_re + mb##20_re * s;		\
    mc##20_im = ma##20_im + mb##20_im * s;		\
    mc##21_re = ma##21_re + mb##21_re * s;		\
    mc##21_im = ma##21_im + mb##21_im * s;		\
    mc##22_re = ma##22_re + mb##22_re * s;		\
    mc##22_im = ma##22_im + mb##22_im * s;		

#define SCALAR_MULT_SUB_SU3_MATRIX(ma, mb, s, mc)	\
    mc##00_re = ma##00_re - mb##00_re * s;		\
    mc##00_im = ma##00_im - mb##00_im * s;		\
    mc##01_re = ma##01_re - mb##01_re * s;		\
    mc##01_im = ma##01_im - mb##01_im * s;		\
    mc##02_re = ma##02_re - mb##02_re * s;		\
    mc##02_im = ma##02_im - mb##02_im * s;		\
    mc##10_re = ma##10_re - mb##10_re * s;		\
    mc##10_im = ma##10_im - mb##10_im * s;		\
    mc##11_re = ma##11_re - mb##11_re * s;		\
    mc##11_im = ma##11_im - mb##11_im * s;		\
    mc##12_re = ma##12_re - mb##12_re * s;		\
    mc##12_im = ma##12_im - mb##12_im * s;		\
    mc##20_re = ma##20_re - mb##20_re * s;		\
    mc##20_im = ma##20_im - mb##20_im * s;		\
    mc##21_re = ma##21_re - mb##21_re * s;		\
    mc##21_im = ma##21_im - mb##21_im * s;		\
    mc##22_re = ma##22_re - mb##22_re * s;		\
    mc##22_im = ma##22_im - mb##22_im * s;		


#define ah01_re AH0.x
#define ah01_im AH0.y
#define ah02_re AH1.x
#define ah02_im AH1.y
#define ah12_re AH2.x
#define ah12_im AH2.y
#define ah00_im AH3.x
#define ah11_im AH3.y
#define ah22_im AH4.x
#define ahspace AH4.y

#define UNCOMPRESS_ANTI_HERMITIAN(ah, m)	\
    m##00_re = 0;				\
    m##00_im = ah##00_im;			\
    m##11_re = 0;				\
    m##11_im = ah##11_im;			\
    m##22_re = 0;				\
    m##22_im = ah##22_im;			\
    m##01_re = ah##01_re;			\
    m##01_im = ah##01_im;			\
    m##10_re = -ah##01_re;			\
    m##10_im = ah##01_im;			\
    m##02_re = ah##02_re;			\
    m##02_im = ah##02_im;			\
    m##20_re = -ah##02_re;			\
    m##20_im = ah##02_im;			\
    m##12_re = ah##12_re;			\
    m##12_im = ah##12_im;			\
    m##21_re = -ah##12_re;			\
    m##21_im = ah##12_im;


#define MAKE_ANTI_HERMITIAN(m, ah) do {					\
	typeof(ah##space) temp;						\
	temp = (m##00_im + m##11_im + m##22_im)*0.33333333333333333;	\
	ah##00_im  = (m##00_im - temp);					\
	ah##11_im  = (m##11_im - temp);					\
	ah##22_im  = (m##22_im - temp);					\
	ah##01_re = (m##01_re - m##10_re)*0.5;				\
	ah##02_re = (m##02_re - m##20_re)*0.5;				\
	ah##12_re = (m##12_re - m##21_re)*0.5;				\
	ah##01_im = (m##01_im + m##10_im)*0.5;				\
	ah##02_im = (m##02_im + m##20_im)*0.5;				\
	ah##12_im = (m##12_im + m##21_im)*0.5;				\
	ah##space = 0;							\
    }while(0)						


#define LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, stride) do{	\
    int start_pos = idx + dir*stride*5;					\
    var##0 = src[start_pos];						\
    var##1 = src[start_pos + stride];					\
    var##2 = src[start_pos + stride*2];					\
    var##3 = src[start_pos + stride*3];					\
    var##4 = src[start_pos + stride*4];					\
  }while(0)

#define LOAD_ANTI_HERMITIAN_SINGLE_TEX(src, dir, idx, var) do{		\
    int start_pos = idx + dir*Vh*5;					\
    var##0 = tex1Dfetch(src, start_pos);				\
    var##1 = tex1Dfetch(src, start_pos + Vh);				\
    var##2 = tex1Dfetch(src, start_pos + Vh*2);				\
    var##3 = tex1Dfetch(src, start_pos + Vh*3);				\
    var##4 = tex1Dfetch(src, start_pos + Vh*4);				\
  }while(0)

#define WRITE_ANTI_HERMITIAN(mem, dir, idx, var, stride) do{		\
    int start_ps = idx + dir*stride*5;					\
    mem[start_ps] = var##0;						\
    mem[start_ps + stride] = var##1;					\
    mem[start_ps + stride*2] = var##2;					\
    mem[start_ps + stride*3] = var##3;					\
    mem[start_ps + stride*4] = var##4;					\
  }while(0)

#define COPY_SU3_MATRIX(a, b)		\
    b##00_re = a##00_re;		\
    b##00_im = a##00_im;		\
    b##01_re = a##01_re;		\
    b##01_im = a##01_im;		\
    b##02_re = a##02_re;		\
    b##02_im = a##02_im;		\
    b##10_re = a##10_re;		\
    b##10_im = a##10_im;		\
    b##11_re = a##11_re;		\
    b##11_im = a##11_im;		\
    b##12_re = a##12_re;		\
    b##12_im = a##12_im;		\
    b##20_re = a##20_re;		\
    b##20_im = a##20_im;		\
    b##21_re = a##21_re;		\
    b##21_im = a##21_im;		\
    b##22_re = a##22_re;		\
    b##22_im = a##22_im;		

#define SU3_ADJOINT(a, b)		\
    b##00_re = a##00_re;		\
    b##00_im = - a##00_im;		\
    b##01_re = a##10_re;		\
    b##01_im = - a##10_im;		\
    b##02_re = a##20_re;		\
    b##02_im = - a##20_im;		\
    b##10_re = a##01_re;		\
    b##10_im = - a##01_im;		\
    b##11_re = a##11_re;		\
    b##11_im = - a##11_im;		\
    b##12_re = a##21_re;		\
    b##12_im = - a##21_im;		\
    b##20_re = a##02_re;		\
    b##20_im = - a##02_im;		\
    b##21_re = a##12_re;		\
    b##21_im = - a##12_im;		\
    b##22_re = a##22_re;		\
    b##22_im = - a##22_im;		

#define SET_UNIT_SU3_MATRIX(a)			\
    a##00_re = 1.0;				\
    a##00_im = 0;				\
    a##01_re = 0;				\
    a##01_im = 0;				\
    a##02_re = 0;				\
    a##02_im = 0;				\
    a##10_re = 0;				\
    a##10_im = 0;				\
    a##11_re = 1.0;				\
    a##11_im = 0;				\
    a##12_re = 0;				\
    a##12_im = 0;				\
    a##20_re = 0;				\
    a##20_im = 0;				\
    a##21_re = 0;				\
    a##21_im = 0;				\
    a##22_re = 1.0;				\
    a##22_im = 0;				

// Performs the complex conjugated accumulation: a = b* c*
#define ACC_CONJ_PROD_ASSIGN(a, b, c)		\
  a##_re = b##_re * c##_re;			\
  a##_re -= b##_im * c##_im;			\
  a##_im = - b##_re * c##_im;			\
  a##_im -= b##_im * c##_re


#define RECONSTRUCT_LINK_12(sign, var)					\
    ACC_CONJ_PROD_ASSIGN(var##20, +var##01, +var##12);			\
    ACC_CONJ_PROD(var##20, -var##02, +var##11);				\
    ACC_CONJ_PROD_ASSIGN(var##21, +var##02, +var##10);			\
    ACC_CONJ_PROD(var##21, -var##00, +var##12);				\
    ACC_CONJ_PROD_ASSIGN(var##22, +var##00, +var##11);			\
    ACC_CONJ_PROD(var##22, -var##01, +var##10);				\
    var##20_re *=sign;var##20_im *=sign; var##21_re *=sign; var##21_im *=sign; \
    var##22_re *=sign;var##22_im *=sign;

#define COMPUTE_NEW_IDX_PLUS(mydir, idx) do {				\
	switch(mydir){							\
	case 0:								\
	    new_mem_idx = ( (x1==X1m1)?idx-X1m1:idx+1)>> 1;		\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==X2m1)?idx-X2X1mX1:idx+X1) >> 1;	\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1) >> 1;	\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1) >> 1; \
	    break;							\
	}								\
    }while(0)

#define COMPUTE_NEW_IDX_MINUS(mydir, idx) do {				\
	switch(mydir){							\
	case 0:								\
	    new_mem_idx = ( (x1==0)?idx+X1m1:X-1);			\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==0)?idx+X2X1mX1:X-X1);			\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==0)?idx+X3X2X1mX2X1:X-X2X1);		\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==0)?idx+X4X3X2X1mX3X2X1:X-X3X2X1);	\
	    break;							\
	}								\
    }while(0)


#define COMPUTE_NEW_FULL_IDX_PLUS(mydir, idx) do {			\
	switch(mydir){							\
	case 0:								\
	    new_mem_idx = ( (x1==X1m1)?idx-X1m1:idx+1);			\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==X2m1)?idx-X2X1mX1:idx+X1);		\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
	    break;							\
	}								\
    }while(0)
    
#define COMPUTE_NEW_FULL_IDX_MINUS(mydir, idx) do {			\
	switch(mydir){							\
	case 0:								\
	    new_mem_idx = ( (x1==0)?idx+X1m1:X-1);			\
	    break;							\
	case 1:								\
	    new_mem_idx = ( (x2==0)?idx+X2X1mX1:X-X1);			\
	    break;							\
	case 2:								\
	    new_mem_idx = ( (x3==0)?idx+X3X2X1mX2X1:X-X2X1);		\
	    break;							\
	case 3:								\
	    new_mem_idx = ( (x4==0)?idx+X4X3X2X1mX3X2X1:X-X3X2X1);	\
	    break;							\
	}								\
    }while(0)


#endif
