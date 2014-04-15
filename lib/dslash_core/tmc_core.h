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
  ASSN_CLOVER(TM_INV_CLOVERTEX, 0)\
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
  ASSN_CLOVER(TM_INV_CLOVERTEX, 1)\
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
\

