#define Vsh_x ghostFace[0]
#define Vsh_y ghostFace[1]
#define Vsh_z ghostFace[2]
#define Vsh_t ghostFace[3]
#define xcomm kparam.ghostDim[0]
#define ycomm kparam.ghostDim[1]
#define zcomm kparam.ghostDim[2]
#define tcomm kparam.ghostDim[3]
#define dimcomm kparam.ghostDim


#define D1 kparam.D1
#define D2 kparam.D2
#define D3 kparam.D3
#define D4 kparam.D4
#define D1h kparam.D1h

#if (RECONSTRUCT == 18)
#define DECLARE_VAR_SIGN 
#define DECLARE_NEW_X 
#define DECLARE_X_ARRAY int x[4] = {x1,x2,x3, x4};
#else //RECONSTRUCT == 12
#define DECLARE_VAR_SIGN short sign=1
#define DECLARE_NEW_X short new_x1=x1; short new_x2=x2; short new_x4=x4;
#define DECLARE_X_ARRAY int x[4] = {x1,x2,x3, x4};

#endif

#if (PRECISION == 1 && RECONSTRUCT == 12)

#define a00_re A0.x
#define a00_im A0.y
#define a01_re A0.z
#define a01_im A0.w
#define a02_re A1.x
#define a02_im A1.y
#define a10_re A1.z
#define a10_im A1.w
#define a11_re A2.x
#define a11_im A2.y
#define a12_re A2.z
#define a12_im A2.w
#define a20_re A3.x
#define a20_im A3.y
#define a21_re A3.z
#define a21_im A3.w
#define a22_re A4.x
#define a22_im A4.y

#define b00_re B0.x
#define b00_im B0.y
#define b01_re B0.z
#define b01_im B0.w
#define b02_re B1.x
#define b02_im B1.y
#define b10_re B1.z
#define b10_im B1.w
#define b11_re B2.x
#define b11_im B2.y
#define b12_re B2.z
#define b12_im B2.w
#define b20_re B3.x
#define b20_im B3.y
#define b21_re B3.z
#define b21_im B3.w
#define b22_re B4.x
#define b22_im B4.y

#define c00_re C0.x
#define c00_im C0.y
#define c01_re C0.z
#define c01_im C0.w
#define c02_re C1.x
#define c02_im C1.y
#define c10_re C1.z
#define c10_im C1.w
#define c11_re C2.x
#define c11_im C2.y
#define c12_re C2.z
#define c12_im C2.w
#define c20_re C3.x
#define c20_im C3.y
#define c21_re C3.z
#define c21_im C3.w
#define c22_re C4.x
#define c22_im C4.y

#define f00_re F0.x
#define f00_im F0.y
#define f01_re F1.x
#define f01_im F1.y
#define f02_re F2.x
#define f02_im F2.y
#define f10_re F3.x
#define f10_im F3.y
#define f11_re F4.x
#define f11_im F4.y
#define f12_re F5.x
#define f12_im F5.y
#define f20_re F6.x
#define f20_im F6.y
#define f21_re F7.x
#define f21_im F7.y
#define f22_re F8.x
#define f22_im F8.y

#define WRITE_LONG_MATRIX WRITE_GAUGE_MATRIX_FLOAT2

#else
#define a00_re A0.x
#define a00_im A0.y
#define a01_re A1.x
#define a01_im A1.y
#define a02_re A2.x
#define a02_im A2.y
#define a10_re A3.x
#define a10_im A3.y
#define a11_re A4.x
#define a11_im A4.y
#define a12_re A5.x
#define a12_im A5.y
#define a20_re A6.x
#define a20_im A6.y
#define a21_re A7.x
#define a21_im A7.y
#define a22_re A8.x
#define a22_im A8.y

#define b00_re B0.x
#define b00_im B0.y
#define b01_re B1.x
#define b01_im B1.y
#define b02_re B2.x
#define b02_im B2.y
#define b10_re B3.x
#define b10_im B3.y
#define b11_re B4.x
#define b11_im B4.y
#define b12_re B5.x
#define b12_im B5.y
#define b20_re B6.x
#define b20_im B6.y
#define b21_re B7.x
#define b21_im B7.y
#define b22_re B8.x
#define b22_im B8.y

#define c00_re C0.x
#define c00_im C0.y
#define c01_re C1.x
#define c01_im C1.y
#define c02_re C2.x
#define c02_im C2.y
#define c10_re C3.x
#define c10_im C3.y
#define c11_re C4.x
#define c11_im C4.y
#define c12_re C5.x
#define c12_im C5.y
#define c20_re C6.x
#define c20_im C6.y
#define c21_re C7.x
#define c21_im C7.y
#define c22_re C8.x
#define c22_im C8.y

#define f00_re F0.x
#define f00_im F0.y
#define f01_re F1.x
#define f01_im F1.y
#define f02_re F2.x
#define f02_im F2.y
#define f10_re F3.x
#define f10_im F3.y
#define f11_re F4.x
#define f11_im F4.y
#define f12_re F5.x
#define f12_im F5.y
#define f20_re F6.x
#define f20_im F6.y
#define f21_re F7.x
#define f21_im F7.y
#define f22_re F8.x
#define f22_im F8.y

#define WRITE_LONG_MATRIX WRITE_GAUGE_MATRIX_FLOAT2

#endif


#define bb00_re BB0.x
#define bb00_im BB0.y
#define bb01_re BB1.x
#define bb01_im BB1.y
#define bb02_re BB2.x
#define bb02_im BB2.y
#define bb10_re BB3.x
#define bb10_im BB3.y
#define bb11_re BB4.x
#define bb11_im BB4.y
#define bb12_re BB5.x
#define bb12_im BB5.y
#define bb20_re BB6.x
#define bb20_im BB6.y
#define bb21_re BB7.x
#define bb21_im BB7.y
#define bb22_re BB8.x
#define bb22_im BB8.y



#define aT00_re (+a00_re)
#define aT00_im (-a00_im)
#define aT01_re (+a10_re)
#define aT01_im (-a10_im)
#define aT02_re (+a20_re)
#define aT02_im (-a20_im)
#define aT10_re (+a01_re)
#define aT10_im (-a01_im)
#define aT11_re (+a11_re)
#define aT11_im (-a11_im)
#define aT12_re (+a21_re)
#define aT12_im (-a21_im)
#define aT20_re (+a02_re)
#define aT20_im (-a02_im)
#define aT21_re (+a12_re)
#define aT21_im (-a12_im)
#define aT22_re (+a22_re)
#define aT22_im (-a22_im)

#define bT00_re (+b00_re)
#define bT00_im (-b00_im)
#define bT01_re (+b10_re)
#define bT01_im (-b10_im)
#define bT02_re (+b20_re)
#define bT02_im (-b20_im)
#define bT10_re (+b01_re)
#define bT10_im (-b01_im)
#define bT11_re (+b11_re)
#define bT11_im (-b11_im)
#define bT12_re (+b21_re)
#define bT12_im (-b21_im)
#define bT20_re (+b02_re)
#define bT20_im (-b02_im)
#define bT21_re (+b12_re)
#define bT21_im (-b12_im)
#define bT22_re (+b22_re)
#define bT22_im (-b22_im)

#define cT00_re (+c00_re)
#define cT00_im (-c00_im)
#define cT01_re (+c10_re)
#define cT01_im (-c10_im)
#define cT02_re (+c20_re)
#define cT02_im (-c20_im)
#define cT10_re (+c01_re)
#define cT10_im (-c01_im)
#define cT11_re (+c11_re)
#define cT11_im (-c11_im)
#define cT12_re (+c21_re)
#define cT12_im (-c21_im)
#define cT20_re (+c02_re)
#define cT20_im (-c02_im)
#define cT21_re (+c12_re)
#define cT21_im (-c12_im)
#define cT22_re (+c22_re)
#define cT22_im (-c22_im)


#define tempa00_re TEMPA0.x
#define tempa00_im TEMPA0.y
#define tempa01_re TEMPA1.x
#define tempa01_im TEMPA1.y
#define tempa02_re TEMPA2.x
#define tempa02_im TEMPA2.y
#define tempa10_re TEMPA3.x
#define tempa10_im TEMPA3.y
#define tempa11_re TEMPA4.x
#define tempa11_im TEMPA4.y
#define tempa12_re TEMPA5.x
#define tempa12_im TEMPA5.y
#define tempa20_re TEMPA6.x
#define tempa20_im TEMPA6.y
#define tempa21_re TEMPA7.x
#define tempa21_im TEMPA7.y
#define tempa22_re TEMPA8.x
#define tempa22_im TEMPA8.y

#define tempb00_re TEMPB0.x
#define tempb00_im TEMPB0.y
#define tempb01_re TEMPB1.x
#define tempb01_im TEMPB1.y
#define tempb02_re TEMPB2.x
#define tempb02_im TEMPB2.y
#define tempb10_re TEMPB3.x
#define tempb10_im TEMPB3.y
#define tempb11_re TEMPB4.x
#define tempb11_im TEMPB4.y
#define tempb12_re TEMPB5.x
#define tempb12_im TEMPB5.y
#define tempb20_re TEMPB6.x
#define tempb20_im TEMPB6.y
#define tempb21_re TEMPB7.x
#define tempb21_im TEMPB7.y
#define tempb22_re TEMPB8.x
#define tempb22_im TEMPB8.y

#define fat00_re FAT0.x
#define fat00_im FAT0.y
#define fat01_re FAT1.x
#define fat01_im FAT1.y
#define fat02_re FAT2.x
#define fat02_im FAT2.y
#define fat10_re FAT3.x
#define fat10_im FAT3.y
#define fat11_re FAT4.x
#define fat11_im FAT4.y
#define fat12_re FAT5.x
#define fat12_im FAT5.y
#define fat20_re FAT6.x
#define fat20_im FAT6.y
#define fat21_re FAT7.x
#define fat21_im FAT7.y
#define fat22_re FAT8.x
#define fat22_im FAT8.y

#define NUM_FLOATS 5
#define TEMPA0 sd_data[threadIdx.x + 0*blockDim.x]
#define TEMPA1 sd_data[threadIdx.x + 1*blockDim.x ]
#define TEMPA2 sd_data[threadIdx.x + 2*blockDim.x ]
#define TEMPA3 sd_data[threadIdx.x + 3*blockDim.x ]
#define TEMPA4 sd_data[threadIdx.x + 4*blockDim.x ]
    

#undef UPDATE_COOR_PLUS
#undef UPDATE_COOR_MINUS
#undef UPDATE_COOR_LOWER_STAPLE
#undef UPDATE_COOR_LOWER_STAPLE_DIAG
#undef UPDATE_COOR_LOWER_STAPLE_EX
#undef COMPUTE_RECONSTRUCT_SIGN
#if (RECONSTRUCT  != 18)
#define UPDATE_COOR_PLUS(mydir, n, idx) do {				\
    new_x1 = x1; new_x2 = x2; new_x4 = x4;				\
    switch(mydir){                                                      \
    case 0:                                                             \
      new_x1 = x1+n;							\
      break;                                                            \
    case 1:                                                             \
      new_x2 = x2+n;							\
      break;                                                            \
    case 2:                                                             \
      break;                                                            \
    case 3:								\
      new_x4 = x4+n;							\
      break;                                                            \
    }                                                                   \
  }while(0)

#define UPDATE_COOR_MINUS(mydir, idx) do {				\
    new_x1 = x1; new_x2 = x2; new_x4 = x4;				\
    switch(mydir){                                                      \
    case 0:                                                             \
      new_x1 = x1-1;							\
      break;                                                            \
    case 1:                                                             \
      new_x2 = x2-1;							\
      break;                                                            \
    case 2:                                                             \
      break;                                                            \
    case 3:                                                             \
      new_x4 = x4-1;							\
      break;                                                            \
    }                                                                   \
  }while(0)

#define UPDATE_COOR_LOWER_STAPLE(mydir1, mydir2) do {	\
    new_x1 = x1; new_x2 = x2;  new_x4 = x4;				\
    if(dimcomm[mydir1] == 0 || x[mydir1] > 0){				\
      switch(mydir1){							\
      case 0:								\
	new_x1 = x1 - 1;						\
	break;								\
      case 1:								\
	new_x2 = x2 - 1;						\
	break;								\
      case 2:								\
	break;								\
      case 3:								\
	new_x4 = x4 - 1;						\
	break;								\
      }									\
      switch(mydir2){							\
      case 0:								\
	new_x1 = x1+1;							\
	break;								\
      case 1:								\
	new_x2 = x2+1;							\
	break;								\
      case 2:								\
	break;								\
      case 3:								\
	new_x4 = x4+1;							\
	break;								\
      }									\
    }else{								\
      /*the case where both dir1/dir2 are out of boundary are dealed with a different macro (_DIAG)*/ \
      switch(mydir2){							\
      case 0:								\
	new_x1 = x1+1;							\
	break;								\
      case 1:								\
	new_x2 = x2+1;							\
	break;								\
      case 2:								\
	break;								\
      case 3:								\
	new_x4 = x4+1;							\
	break;								\
      }									\
      switch(mydir1){/*mydir1 is 0 here */				\
      case 0:								\
	new_x1 = x1-1;							\
	break;								\
      case 1:								\
	new_x2 = x2-1;							\
	break;								\
      case 2:								\
	break;								\
      case 3:								\
	new_x4 = x4-1;							\
	break;								\
      }									\
    }									\
  }while(0)



#define UPDATE_COOR_LOWER_STAPLE_DIAG(nu, mu, dir1, dir2) do {		\
    int	new_x[4]; 							\
    new_x[3] = x4; new_x[1] = x2; new_x[0] = x1;			\
    new_x[nu] =  -1;							\
    new_x[mu] = 0;							\
    new_x1 = new_x[0]; 							\
    new_x2 = new_x[1]; 							\
    new_x4 = new_x[3]; 							\
  }while(0)

#define UPDATE_COOR_LOWER_STAPLE_EX(mydir1, mydir2) do {                \
    new_x1 = x1; new_x2 = x2;  new_x4 = x4;                             \
    switch(mydir1){                                                     \
    case 0:                                                             \
      new_x1 = x1 - 1;                                                  \
      break;                                                            \
    case 1:                                                             \
      new_x2 = x2 - 1;                                                  \
      break;                                                            \
    case 2:                                                             \
      break;                                                            \
    case 3:                                                             \
      new_x4 = x4 - 1;                                                  \
      break;                                                            \
    }                                                                   \
    switch(mydir2){                                                     \
    case 0:                                                             \
      new_x1 = x1+1;                                                    \
      break;                                                            \
    case 1:                                                             \
      new_x2 = x2+1;                                                    \
      break;                                                            \
    case 2:                                                             \
      break;                                                            \
    case 3:                                                             \
      new_x4 = x4+1;                                                    \
      break;                                                            \
    }                                                                   \
  }while(0)

#define COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4) do {	\
    sign =1;							\
    switch(dir){						\
    case XUP:							\
      if ( (i4 & 1) != 0){					\
	sign = -1;						\
      }								\
      break;							\
    case YUP:							\
      if ( ((i4+i1) & 1) != 0){					\
	sign = -1;						\
      }								\
      break;							\
    case ZUP:							\
      if ( ((i4+i1+i2) & 1) != 0){				\
	sign = -1;						\
      }								\
      break;							\
    case TUP:							\
      if (i4 == X4m1 && PtNm1){					\
	sign = -1;						\
      }else if(i4 == -1 && Pt0){				\
	sign = -1;						\
      }								\
      break;							\
    }								\
  }while (0)


#else

#define UPDATE_COOR_PLUS(mydir, n, idx)
#define UPDATE_COOR_MINUS(mydir, idx)
#define UPDATE_COOR_LOWER_STAPLE(mydir1, mydir2)
#define UPDATE_COOR_LOWER_STAPLE_DIAG(nu, mu, dir1, dir2)
#define COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4)
#define UPDATE_COOR_LOWER_STAPLE_EX(mydir1, mydir2)
#endif

#define LLFAT_COMPUTE_NEW_IDX_PLUS(mydir, n, idx) do {                  \
    switch(mydir){                                                      \
    case 0:                                                             \
      new_mem_idx = (x1>=(X1-n))? ((Vh+Vsh_x+ spacecon_x)*xcomm+(idx-(X1-n))/2*(1-xcomm)):((idx+n)>>1); \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = (x2>=(X2-n))? ((Vh+2*(Vsh_x)+Vsh_y+ spacecon_y)*ycomm+(idx-(X2-n)*X1)/2*(1-ycomm)):((idx+n*X1)>>1); \
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = (x3>=(X3-n))? ((Vh+2*(Vsh_x+Vsh_y)+Vsh_z+ spacecon_z))*zcomm+(idx-(X3-n)*X2X1)/2*(1-zcomm):((idx+n*X2X1)>>1); \
      break;                                                            \
    case 3:								\
      new_mem_idx = ( (x4>=(X4-n))? ((Vh+2*(Vsh_x+Vsh_y+Vsh_z)+Vsh_t+spacecon_t))*tcomm+(idx-(X4-n)*X3X2X1)/2*(1-tcomm): (idx+n*X3X2X1)>>1); \
      break;                                                            \
    }                                                                   \
    UPDATE_COOR_PLUS(mydir, n, idx);					\
  }while(0)

#define LLFAT_COMPUTE_NEW_IDX_MINUS(mydir, idx) do {			\
    switch(mydir){                                                      \
    case 0:                                                             \
      new_mem_idx = (x1==0)?( (Vh+spacecon_x)*xcomm+(idx+X1m1)/2*(1-xcomm)):((idx-1) >> 1); \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = (x2==0)?( (Vh+2*Vsh_x+spacecon_y)*ycomm+(idx+X2X1mX1)/2*(1-ycomm)):((idx-X1) >> 1);	\
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = (x3==0)?((Vh+2*(Vsh_x+Vsh_y)+spacecon_z)*zcomm+(idx+X3X2X1mX2X1)/2*(1-zcomm)):((idx-X2X1) >> 1); \
      break;                                                            \
    case 3:                                                             \
      new_mem_idx = (x4==0)?((Vh+2*(Vsh_x+Vsh_y+Vsh_z)+ spacecon_t)*tcomm + (idx+X4X3X2X1mX3X2X1)/2*(1-tcomm)):((idx-X3X2X1) >> 1); \
      break;                                                            \
    }                                                                   \
    UPDATE_COOR_MINUS(mydir, idx);					\
  }while(0)

#define LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE(mydir1, mydir2) do {		\
    int local_new_x1=x1;						\
    int local_new_x2=x2;						\
    int local_new_x3=x3;						\
    int local_new_x4=x4;						\
    new_mem_idx=X;							\
    if(dimcomm[mydir1] == 0 || x[mydir1] > 0){				\
      switch(mydir1){/*mydir1 is not partitioned or x[mydir1]!=  0*/	\
      case 0:								\
	new_mem_idx = (x1==0)?(new_mem_idx+X1m1):(new_mem_idx-1);	\
	local_new_x1 = (x1==0)?X1m1:(x1 - 1);				\
	break;								\
      case 1:								\
	new_mem_idx = (x2==0)?(new_mem_idx+X2X1mX1):(new_mem_idx-X1);	\
	local_new_x2 = (x2==0)?X2m1:(x2 - 1);				\
	break;								\
      case 2:								\
	new_mem_idx = (x3==0)?(new_mem_idx+X3X2X1mX2X1):(new_mem_idx-X2X1); \
	local_new_x3 = (x3==0)?X3m1:(x3 -1);				\
	break;								\
      case 3:								\
	new_mem_idx = (x4==0)?(new_mem_idx+X4X3X2X1mX3X2X1):(new_mem_idx-X3X2X1); \
	local_new_x4 = (x4==0)?X4m1:(x4 - 1);				\
	break;								\
      }									\
      switch(mydir2){							\
      case 0:								\
	new_mem_idx = (x1==X1m1)?(2*(Vh+Vsh_x)+((local_new_x4*X3X2+local_new_x3*X2+local_new_x2)))*xcomm+(new_mem_idx-X1m1)*(1-xcomm):(new_mem_idx+1); \
	break;								\
      case 1:								\
	new_mem_idx = (x2==X2m1)?(2*(Vh+2*(Vsh_x)+Vsh_y)+((local_new_x4*X3X1+local_new_x3*X1+local_new_x1)))*ycomm+(new_mem_idx-X2X1mX1)*(1-ycomm):(new_mem_idx+X1); \
	break;								\
      case 2:								\
	new_mem_idx = (x3==X3m1)?(2*(Vh+2*(Vsh_x+Vsh_y)+Vsh_z)+((local_new_x4*X2X1+local_new_x2*X1+local_new_x1)))*zcomm+(new_mem_idx-X3X2X1mX2X1)*(1-zcomm):(new_mem_idx+X2X1); \
	break;								\
      case 3:								\
	new_mem_idx = (x4==X4m1)?(2*(Vh+2*(Vsh_x+Vsh_y+Vsh_z)+Vsh_t)+((local_new_x3*X2X1+local_new_x2*X1+local_new_x1)))*tcomm+(new_mem_idx-X4X3X2X1mX3X2X1)*(1-tcomm):(new_mem_idx+X3X2X1); \
	break;								\
      }									\
    }else{								\
      /*the case where both dir1/dir2 are out of boundary are dealed with a different macro (_DIAG)*/ \
      switch(mydir2){	/*mydir2 is not partitioned or x[mydir2]!=  0*/	\
      case 0:								\
	new_mem_idx = (x1==X1m1)?(new_mem_idx-X1m1):(new_mem_idx+1);	\
	local_new_x1 = (x1==X1m1)?0:(x1+1);				\
	break;								\
      case 1:								\
	new_mem_idx = (x2==X2m1)?(new_mem_idx-X2X1mX1):(new_mem_idx+X1); \
	local_new_x2 = (x2==X2m1)?0:(x2+1);				\
	break;								\
      case 2:								\
	new_mem_idx = (x3==X3m1)?(new_mem_idx-X3X2X1mX2X1):(new_mem_idx+X2X1); \
	local_new_x3 = (x3==X3m1)?0:(x3+1);				\
	break;								\
      case 3:								\
	new_mem_idx = (x4==X4m1)?(new_mem_idx-X4X3X2X1mX3X2X1):(new_mem_idx+X3X2X1); \
	local_new_x4 = (x4==X4m1)?0:(x4+1);				\
	break;								\
      }									\
      switch(mydir1){/*mydir1 is 0 here */				\
      case 0:								\
	new_mem_idx = (x1==0)?(2*(Vh)+(local_new_x4*X3X2+local_new_x3*X2+local_new_x2))*xcomm+(new_mem_idx+X1m1)*(1-xcomm):(new_mem_idx -1); \
	break;								\
      case 1:								\
	new_mem_idx = (x2==0)?(2*(Vh+2*Vsh_x)+(local_new_x4*X3X1+local_new_x3*X1+local_new_x1))*ycomm+(new_mem_idx+X2X1mX1)*(1-ycomm):(new_mem_idx-X1); \
	break;								\
      case 2:								\
	new_mem_idx = (x3==0)?(2*(Vh+2*(Vsh_x+Vsh_y))+(local_new_x4*X2X1+local_new_x2*X1+local_new_x1))*zcomm+(new_mem_idx+X3X2X1mX2X1)*(1-zcomm):(new_mem_idx-X2X1); \
	break;								\
      case 3:								\
	new_mem_idx = (x4==0)?(2*(Vh+2*(Vsh_x+Vsh_y+Vsh_z))+(local_new_x3*X2X1+local_new_x2*X1+local_new_x1))*tcomm+(new_mem_idx+X4X3X2X1mX3X2X1)*(1-tcomm):(new_mem_idx-X3X2X1); \
	break;								\
      }									\
    }									\
    new_mem_idx = new_mem_idx >> 1;					\
    UPDATE_COOR_LOWER_STAPLE(mydir1, mydir2);				\
  }while(0)


#define LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE_DIAG(nu, mu, dir1, dir2) do { \
    new_mem_idx = Vh+2*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + mu*Vh_2d_max + ((x[dir2]*Z[dir1] + x[dir1])>>1); \
    UPDATE_COOR_LOWER_STAPLE_DIAG(nu, mu, dir1, dir2);			\
  }while(0)

#define LLFAT_COMPUTE_NEW_IDX_PLUS_EX(mydir, n, idx) do {               \
    switch(mydir){                                                      \
    case 0:                                                             \
      new_mem_idx = (idx+n)>>1;                                         \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = (idx+n*E1)>>1;                                      \
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = (idx+n*E2E1)>>1;                                    \
      break;                                                            \
    case 3:                                                             \
      new_mem_idx = (idx+n*E3E2E1)>>1;                                  \
      break;                                                            \
    }                                                                   \
    UPDATE_COOR_PLUS(mydir, n, idx);                                    \
  }while(0)

#define LLFAT_COMPUTE_NEW_IDX_MINUS_EX(mydir, idx) do {                 \
    switch(mydir){                                                      \
    case 0:                                                             \
      new_mem_idx = (idx-1) >> 1;                                       \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = (idx-E1) >> 1;                                      \
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = (idx-E2E1) >> 1;                                    \
      break;                                                            \
    case 3:                                                             \
      new_mem_idx = (idx-E3E2E1) >> 1;                                  \
      break;                                                            \
    }                                                                   \
    UPDATE_COOR_MINUS(mydir, idx);                                      \
  }while(0)


#define LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE_EX(mydir1, mydir2) do {      \
    switch(mydir1){                                                     \
    case 0:                                                             \
      new_mem_idx = X-1;                                                \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = X-E1;                                               \
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = X-E2E1;                                             \
      break;                                                            \
    case 3:                                                             \
      new_mem_idx = X-E3E2E1;                                           \
      break;                                                            \
    }                                                                   \
    switch(mydir2){                                                     \
    case 0:                                                             \
      new_mem_idx = (new_mem_idx+1)>> 1;                                \
      break;                                                            \
    case 1:                                                             \
      new_mem_idx = (new_mem_idx+E1) >> 1;                              \
      break;                                                            \
    case 2:                                                             \
      new_mem_idx = (new_mem_idx+E2E1) >> 1;                            \
      break;                                                            \
    case 3:                                                             \
      new_mem_idx = (new_mem_idx+E3E2E1) >> 1;                          \
      break;                                                            \
    }                                                                   \
    UPDATE_COOR_LOWER_STAPLE_EX(mydir1, mydir2);                        \
  }while(0)


template<int mu, int nu, int odd_bit>
  __global__ void
  LLFAT_KERNEL_EX(do_siteComputeGenStapleParity, RECONSTRUCT)(FloatM* staple_even, FloatM* staple_odd, 
							      const FloatN* sitelink_even, const FloatN* sitelink_odd, 
							      FloatM* fatlink_even, FloatM* fatlink_odd,	
							      Float mycoeff, llfat_kernel_param_t kparam)
{
  extern __shared__ FloatM sd_data[]; //sd_data is a macro name defined in llfat_quda.cu

  
  //FloatM TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4, TEMPA5, TEMPA6, TEMPA7, TEMPA8;
  FloatM TEMPA5, TEMPA6, TEMPA7, TEMPA8;
  FloatM STAPLE0, STAPLE1, STAPLE2, STAPLE3, STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
    
  int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(mem_idx >= kparam.threads) return;

  int z1 = mem_idx/D1h;
  short x1h = mem_idx - z1*D1h;
  int z2 = z1/D2;
  short x2 = z1 - z2*D2;
  short x4 = z2/D3;
  short x3 = z2 - x4*D3;
  
  short x1odd = (x2 + x3 + x4 + odd_bit) & 1;
  short x1 = 2*x1h + x1odd;
  
  x1 += kparam.base_idx;
  x2 += kparam.base_idx;
  x3 += kparam.base_idx;
  x4 += kparam.base_idx;
  int X = x4*E3E2E1 + x3*E2E1 + x2*E1 + x1;
  mem_idx = X/2;
 
  int new_mem_idx;
  DECLARE_VAR_SIGN;
  DECLARE_NEW_X;

  /* Upper staple */
  /* Computes the staple :
   *                 mu (B)
   *               +-------+
   *       nu	   |	   | 
   *	     (A)   |	   |(C)
   *		   X	   X
   *
   */
    
  {
    /* load matrix A*/
    LOAD_EVEN_SITE_MATRIX(nu, mem_idx, A);   
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (x1-2), (x2-2), (x3-2), (x4-2));
    RECONSTRUCT_SITE_LINK(sign, a);


    
    /* load matrix B*/  
    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(nu, 1, X);    
    LOAD_ODD_SITE_MATRIX(mu, new_mem_idx, B);
    COMPUTE_RECONSTRUCT_SIGN(sign, mu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, b);
    

    MULT_SU3_NN(a, b, tempa);    
    
    /* load matrix C*/
        
    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(mu, 1, X);    
    LOAD_ODD_SITE_MATRIX(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, c);

    MULT_SU3_NA(tempa, c, staple);		   
 
  }

  /***************lower staple****************
   *
   *                   X       X
   *             nu    |       | 
   *	         (A)   |       | (C)
   *		       +-------+
   *                  mu (B)
   *
   *********************************************/
    {
    /* load matrix A*/
    LLFAT_COMPUTE_NEW_IDX_MINUS_EX(nu,X);    
    
    LOAD_ODD_SITE_MATRIX(nu, (new_mem_idx), A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, a);
    
    /* load matrix B*/				
    LOAD_ODD_SITE_MATRIX(mu, (new_mem_idx), B);
    COMPUTE_RECONSTRUCT_SIGN(sign, mu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, b);
    
    MULT_SU3_AN(a, b, tempa);



    /* load matrix C*/
    LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE_EX(nu, mu);
    LOAD_EVEN_SITE_MATRIX(nu, new_mem_idx, C);
   
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, c);
    
    
    MULT_SU3_NN(tempa, c, b);		
    LLFAT_ADD_SU3_MATRIX(b, staple, staple);

  }
   
    
    if( !(x1 == 1 || x1 == X1 + 2 || x2 == 1 || x2 == X2 + 2
	|| x3 == 1 || x3 == X3 + 2 || x4 == 1 || x4 == X4 + 2)){
    int orig_idx = ((x4-2)* X3X2X1 + (x3-2)*X2X1 + (x2-2)*X1 + (x1-2))>>1;
    
    LOAD_EVEN_FAT_MATRIX(mu, orig_idx);
    SCALAR_MULT_ADD_SU3_MATRIX(fat, staple, mycoeff, fat);
    WRITE_FAT_MATRIX(fatlink_even,mu,  orig_idx);	
  }
  WRITE_STAPLE_MATRIX(staple_even, mem_idx);	

  return;
}

template<int mu, int nu, int odd_bit, int save_staple>
  __global__ void
  LLFAT_KERNEL_EX(do_computeGenStapleFieldParity,RECONSTRUCT)(FloatM* staple_even, FloatM* staple_odd, 
							      const FloatN* sitelink_even, const FloatN* sitelink_odd,
							      FloatM* fatlink_even, FloatM* fatlink_odd,			    
							      const FloatM* mulink_even, const FloatM* mulink_odd, 
							      Float mycoeff, llfat_kernel_param_t kparam)
{
  //__shared__ FloatM sd_data[NUM_FLOATS*64];
  extern __shared__ FloatM sd_data[]; //sd_data is a macro name defined in llfat_quda.cu
  //FloatM TEMPA0, TEMPA1, TEMPA2, TEMPA3, TEMPA4, TEMPA5, TEMPA6, TEMPA7, TEMPA8;  
  FloatM  TEMPA5, TEMPA6, TEMPA7, TEMPA8;  
  FloatM STAPLE0, STAPLE1, STAPLE2, STAPLE3, STAPLE4, STAPLE5, STAPLE6, STAPLE7, STAPLE8;
  
  int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(mem_idx >= kparam.threads) return;
  
  int z1 = mem_idx/D1h;
  short x1h = mem_idx - z1*D1h;
  int z2 = z1/D2;
  short x2 = z1 - z2*D2;
  short x4 = z2/D3;
  short x3 = z2 - x4*D3;

  short x1odd = (x2 + x3 + x4 + odd_bit) & 1;
  short x1 = 2*x1h + x1odd;

  x1 += kparam.base_idx;
  x2 += kparam.base_idx;
  x3 += kparam.base_idx;
  x4 += kparam.base_idx;
  int X = x4*E3E2E1 + x3*E2E1 + x2*E1 + x1;
  mem_idx = X/2;

  int new_mem_idx;
  DECLARE_VAR_SIGN;
  DECLARE_NEW_X;

  /* Upper staple */
  /* Computes the staple :
   *                mu (BB)
   *               +-------+
   *       nu	   |	   | 
   *	     (A)   |	   |(C)
   *		   X	   X
   *
   */
  {		
    /* load matrix A*/
    LOAD_EVEN_SITE_MATRIX(nu, mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (x1-2), (x2-2), (x3-2), (x4-2));
    RECONSTRUCT_SITE_LINK(sign, a);
    
    /* load matrix BB*/
    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(nu, 1, X);    
    LOAD_ODD_MULINK_MATRIX(0, new_mem_idx, BB);
    MULT_SU3_NN(a, bb, tempa);    
    
    /* load matrix C*/
    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(mu, 1, X);    
    LOAD_ODD_SITE_MATRIX(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, c);
    
    MULT_SU3_NA(tempa, c, staple);
  }
  
  /***************lower staple****************
   *
   *                   X       X
   *             nu    |       | 
   *	         (A)   |       | (C)
   *		       +-------+
   *                  mu (B)
   *
   *********************************************/
    
  {
    /* load matrix A*/
    LLFAT_COMPUTE_NEW_IDX_MINUS_EX(nu, X);
    
    LOAD_ODD_SITE_MATRIX(nu, new_mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, a);
    
    /* load matrix B*/
    LLFAT_COMPUTE_NEW_IDX_MINUS_EX(nu, X);				
    LOAD_ODD_MULINK_MATRIX(0, new_mem_idx, BB);
    
    MULT_SU3_AN(a, bb, tempa);
    
    /* load matrix C*/
 
    LLFAT_COMPUTE_NEW_IDX_LOWER_STAPLE_EX(nu, mu);
    
    LOAD_EVEN_SITE_MATRIX(nu, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, nu, (new_x1-2), (new_x2-2), 0, (new_x4-2));
    RECONSTRUCT_SITE_LINK(sign, c);				
    
    MULT_SU3_NN(tempa, c, a);	

    LLFAT_ADD_SU3_MATRIX(a, staple, staple);
  }

    
  if( !(x1 == 1 || x1 == X1 + 2 || x2 == 1 || x2 == X2 + 2
	|| x3 == 1 || x3 == X3 + 2 || x4 == 1 || x4 == X4 + 2)){
    int orig_idx = ((x4-2)* X3X2X1 + (x3-2)*X2X1 + (x2-2)*X1 + (x1-2))>>1;
    LOAD_EVEN_FAT_MATRIX(mu, orig_idx);    
    SCALAR_MULT_ADD_SU3_MATRIX(fat, staple, mycoeff, fat);    
    WRITE_FAT_MATRIX(fatlink_even, mu,  orig_idx);	
  }

  if(save_staple){
    WRITE_STAPLE_MATRIX(staple_even, mem_idx);		    
  }
  
  return;
}


__global__ void 
LLFAT_KERNEL_EX(llfatOneLink, RECONSTRUCT)(const FloatN* sitelink_even, const FloatN* sitelink_odd,
					   FloatM* fatlink_even, FloatM* fatlink_odd,
					   Float coeff0, Float coeff5, llfat_kernel_param_t kparam)
{

  const FloatN* my_sitelink;
  FloatM* my_fatlink;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int idx = sid;
  
  if(sid >= 2*kparam.threads) return;
  
  short odd_bit= 0;
  
  my_sitelink = sitelink_even;
  my_fatlink = fatlink_even;
  if (idx >= kparam.threads){
    odd_bit=1;
    idx = idx - kparam.threads;
    my_sitelink = sitelink_odd;
    my_fatlink = fatlink_odd;
  }
  
  int z1 = idx/D1h;
  short x1h = idx - z1*D1h;
  int z2 = z1/D2;
  short x2 = z1 - z2*D2;
  int x4 = z2/D3;
  short x3 = z2 - x4*D3;
  short x1odd = (x2 + x3 + x4 + odd_bit) & 1;
  short x1 = 2*x1h + x1odd; 
  DECLARE_VAR_SIGN;

  x1 += kparam.base_idx;
  x2 += kparam.base_idx;
  x3 += kparam.base_idx;
  x4 += kparam.base_idx;
  int X = x4*E3E2E1 + x3*E2E1 + x2*E1 + x1;
  int mem_idx = X/2;

  for(int dir=0;dir < 4; dir++){
    LOAD_SITE_MATRIX(my_sitelink, dir, mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, dir, (x1-2), (x2-2), (x3-2), (x4-2));
    RECONSTRUCT_SITE_LINK(sign, a);
  
    LOAD_FAT_MATRIX(my_fatlink, dir, idx);
    
    SCALAR_MULT_SU3_MATRIX((coeff0 - 6.0*coeff5), a, fat); 
    
    WRITE_FAT_MATRIX(my_fatlink,dir, idx);	 
  }
    
  return;
}


template<int odd_bit>
__global__ void LLFAT_KERNEL(computeLongLinkParity,RECONSTRUCT)
    (FloatM* const outField,
    const FloatN* const sitelink_even, const FloatN* const sitelink_odd,
    Float coeff,
    const llfat_kernel_param_t kparam)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int mem_idx = idx;
  if(mem_idx >= kparam.threads) return;

  int z1 = mem_idx/D1h;
  short x1h = mem_idx - z1*D1h;
  int z2 = z1/D2;
  short x2 = z1 - z2*D2;
  short x4 = z2/D3;
  short x3 = z2 - x4*D3;

  short x1odd = (x2 + x3 + x4 + odd_bit) & 1;
  short x1 = 2*x1h + x1odd;

  x1 += 2;  
  x2 += 2;  
  x3 += 2;  
  x4 += 2;
  int X = x4*E3E2E1 + x3*E2E1 + x2*E1 + x1;
  mem_idx = X/2;

  int new_mem_idx;
  DECLARE_VAR_SIGN;
  DECLARE_NEW_X;
  
  FloatM F0, F1, F2, F3, F4, F5, F6, F7, F8;

  for(int dir=0; dir<4; ++dir){
    LOAD_EVEN_SITE_MATRIX(dir, mem_idx, A);
    COMPUTE_RECONSTRUCT_SIGN(sign, dir, x1-2, x2-2, x3-2, x4-2);
    RECONSTRUCT_SITE_LINK(sign, a);

    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(dir, 1, X);
    LOAD_ODD_SITE_MATRIX(dir, new_mem_idx, B);
    COMPUTE_RECONSTRUCT_SIGN(sign, dir, new_x1-2, new_x2-2, 0, new_x4-2);
    RECONSTRUCT_SITE_LINK(sign, b);

    LLFAT_COMPUTE_NEW_IDX_PLUS_EX(dir, 2, X);
    LOAD_EVEN_SITE_MATRIX(dir, new_mem_idx, C);
    COMPUTE_RECONSTRUCT_SIGN(sign, dir, new_x1-2, new_x2-2, 0, new_x4-2);
    RECONSTRUCT_SITE_LINK(sign, c);

    SCALAR_MULT_SU3_MATRIX(coeff, a, f); 
    MULT_SU3_NN(f,b,a);
    MULT_SU3_NN(a,c,f);

  
    WRITE_LONG_MATRIX(outField, F, dir, idx, fl.fat_ga_stride);
  }
  return; 
}

#undef D1
#undef D2
#undef D3
#undef D4
#undef D1h

#undef DECLARE_VAR_SIGN 
#undef DECLARE_NEW_X 
#undef DECLARE_X_ARRAY

#undef a00_re 
#undef a00_im 
#undef a01_re 
#undef a01_im 
#undef a02_re 
#undef a02_im 
#undef a10_re 
#undef a10_im 
#undef a11_re 
#undef a11_im 
#undef a12_re 
#undef a12_im 
#undef a20_re 
#undef a20_im 
#undef a21_re 
#undef a21_im 
#undef a22_re 
#undef a22_im 

#undef b00_re 
#undef b00_im 
#undef b01_re 
#undef b01_im 
#undef b02_re 
#undef b02_im 
#undef b10_re 
#undef b10_im 
#undef b11_re 
#undef b11_im 
#undef b12_re 
#undef b12_im 
#undef b20_re 
#undef b20_im 
#undef b21_re 
#undef b21_im 
#undef b22_re 
#undef b22_im 

#undef bb00_re 
#undef bb00_im 
#undef bb01_re 
#undef bb01_im 
#undef bb02_re 
#undef bb02_im 
#undef bb10_re 
#undef bb10_im 
#undef bb11_re 
#undef bb11_im 
#undef bb12_re 
#undef bb12_im 
#undef bb20_re 
#undef bb20_im 
#undef bb21_re 
#undef bb21_im 
#undef bb22_re 
#undef bb22_im 

#undef c00_re 
#undef c00_im 
#undef c01_re 
#undef c01_im 
#undef c02_re 
#undef c02_im 
#undef c10_re 
#undef c10_im 
#undef c11_re 
#undef c11_im 
#undef c12_re 
#undef c12_im 
#undef c20_re 
#undef c20_im 
#undef c21_re 
#undef c21_im 
#undef c22_re 
#undef c22_im 

#undef f00_re 
#undef f00_im 
#undef f01_re 
#undef f01_im 
#undef f02_re 
#undef f02_im 
#undef f10_re 
#undef f10_im 
#undef f11_re 
#undef f11_im 
#undef f12_re 
#undef f12_im 
#undef f20_re 
#undef f20_im 
#undef f21_re 
#undef f21_im 
#undef f22_re 
#undef f22_im 

#undef aT00_re 
#undef aT00_im 
#undef aT01_re 
#undef aT00_re 
#undef aT00_im 
#undef aT01_re 
#undef aT01_im 
#undef aT02_re 
#undef aT02_im 
#undef aT10_re 
#undef aT10_im 
#undef aT11_re 
#undef aT11_im 
#undef aT12_re 
#undef aT12_im 
#undef aT20_re 
#undef aT20_im 
#undef aT21_re 
#undef aT21_im 
#undef aT22_re 
#undef aT22_im 

#undef bT00_re 
#undef bT00_im 
#undef bT01_re 
#undef bT01_im 
#undef bT02_re 
#undef bT02_im 
#undef bT10_re 
#undef bT10_im 
#undef bT11_re 
#undef bT11_im 
#undef bT12_re 
#undef bT12_im 
#undef bT20_re 
#undef bT20_im 
#undef bT21_re 
#undef bT21_im 
#undef bT22_re 
#undef bT22_im 

#undef cT00_re 
#undef cT00_im 
#undef cT01_re 
#undef cT01_im 
#undef cT02_re 
#undef cT02_im 
#undef cT10_re 
#undef cT10_im 
#undef cT11_re 
#undef cT11_im 
#undef cT12_re 
#undef cT12_im 
#undef cT20_re 
#undef cT20_im 
#undef cT21_re 
#undef cT21_im 
#undef cT22_re 
#undef cT22_im 


#undef tempa00_re 
#undef tempa00_im 
#undef tempa01_re 
#undef tempa01_im 
#undef tempa02_re 
#undef tempa02_im 
#undef tempa10_re 
#undef tempa10_im 
#undef tempa11_re 
#undef tempa11_im 
#undef tempa12_re 
#undef tempa12_im 
#undef tempa20_re 
#undef tempa20_im 
#undef tempa21_re 
#undef tempa21_im 
#undef tempa22_re 
#undef tempa22_im 

#undef tempb00_re 
#undef tempb00_im 
#undef tempb01_re 
#undef tempb01_im 
#undef tempb02_re 
#undef tempb02_im 
#undef tempb10_re 
#undef tempb10_im 
#undef tempb11_re 
#undef tempb11_im 
#undef tempb12_re 
#undef tempb12_im 
#undef tempb20_re 
#undef tempb20_im 
#undef tempb21_re 
#undef tempb21_im 
#undef tempb22_re 
#undef tempb22_im 

#undef fat00_re 
#undef fat00_im 
#undef fat01_re 
#undef fat01_im 
#undef fat02_re 
#undef fat02_im 
#undef fat10_re 
#undef fat10_im 
#undef fat11_re 
#undef fat11_im 
#undef fat12_re 
#undef fat12_im 
#undef fat20_re 
#undef fat20_im 
#undef fat21_re 
#undef fat21_im 
#undef fat22_re 
#undef fat22_im 

#undef WRITE_LONG_MATRIX

