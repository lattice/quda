// *** CUDA DSLASH ***
#undef SHARED_FLOATS_PER_THREAD 
#define SHARED_FLOATS_PER_THREAD 6


// input spinor
#if (DD_PREC==0)

#define gaugeFloat double
#define gaugeFloat2 double2
#define spinorFloat double
#define spinorFloat2 double2
#define time_boundary t_boundary

#else

#define gaugeFloat float
#define gaugeFloat2 float2
#define spinorFloat float
#define spinorFloat2 float2
#define time_boundary t_boundary_f

#endif

// gauge link
#define fat00_re FAT[0].x
#define fat00_im FAT[0].y
#define fat01_re FAT[1].x
#define fat01_im FAT[1].y
#define fat02_re FAT[2].x
#define fat02_im FAT[2].y
#define fat10_re FAT[3].x
#define fat10_im FAT[3].y
#define fat11_re FAT[4].x
#define fat11_im FAT[4].y
#define fat12_re FAT[5].x
#define fat12_im FAT[5].y
#define fat20_re FAT[6].x
#define fat20_im FAT[6].y
#define fat21_re FAT[7].x
#define fat21_im FAT[7].y
#define fat22_re FAT[8].x
#define fat22_im FAT[8].y

#define long00_re LONG[0].x
#define long00_im LONG[0].y
#define long01_re LONG[1].x
#define long01_im LONG[1].y
#define long02_re LONG[2].x
#define long02_im LONG[2].y
#define long10_re LONG[3].x
#define long10_im LONG[3].y
#define long11_re LONG[4].x
#define long11_im LONG[4].y
#define long12_re LONG[5].x
#define long12_im LONG[5].y
#define long20_re LONG[6].x
#define long20_im LONG[6].y
#define long21_re LONG[7].x
#define long21_im LONG[7].y
#define long22_re LONG[8].x
#define long22_im LONG[8].y

#ifdef PARALLEL_DIR

extern __shared__ spinorFloat s_data[];

// output spinor
#if (DD_PREC == 0)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi+
#else
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi+
#endif

spinorFloat *s = s_data + 
  SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*((threadIdx.x+blockDim.x*threadIdx.y)/SHARED_STRIDE)
  + ((threadIdx.x+blockDim.x*threadIdx.y) % SHARED_STRIDE);

#endif // PARALLEL_DIR


complex<spinorFloat> I[3];
#if (DD_IMPROVED==1)
complex<spinorFloat> T[3];
#endif

complex<gaugeFloat> FAT[9];
#if (DD_IMPROVED==1)
complex<gaugeFloat> LONG[9];
#endif

// output spinor
complex<spinorFloat> O[3];

#include "read_gauge.h"
#include "io_spinor.h"

#define MAT_MUL_V(VOUT, M, V)			\
  complex<spinorFloat> VOUT[6];			\
  for (int i=0; i<3; i++) {			\
    VOUT[i] = M[i*3+0]*V[0];			\
    for (int j=1; j<3; j++) {			\
      VOUT[i] += M[i*3+j]*V[j];			\
    }						\
  }

#define ADJ_MAT_MUL_V(VOUT, M, V)		\
  complex<spinorFloat> VOUT[6];			\
  for (int i=0; i<3; i++) {			\
    VOUT[i] = conj(M[0*3+i])*V[0];		\
    for (int j=1; j<3; j++) {			\
      VOUT[i] += conj(M[j*3+i])*V[j];		\
    }						\
  }

#if (DD_IMPROVED==1)
#define NFACE 3
#else
#define NFACE 1
#endif

  const auto *X = param.X;
  const int& fat_stride = param.gauge_stride;
#if (DD_IMPROVED == 1)
  const int& long_stride = param.long_gauge_stride;
#endif
#if (DD_PREC == 2) // half precision
  const float& fat_link_max = param.fat_link_max;
#endif

#if ((DD_LONG_RECON==9 || DD_LONG_RECON==13) && DD_IMPROVED==1)
#if (DD_PREC==0) // double precision
  double PHASE = 0.;
#else
  float PHASE = 0.f;
#endif
#endif

  int idx = block_idx(param.swizzle)*blockDim.x + threadIdx.x;
  if(idx >= param.threads) return;
 
  int src_idx = blockIdx.y*blockDim.y + threadIdx.y;
  if (src_idx >= param.Ls) return;

  const int X1X0 = X[1]*X[0];
  const int X2X1X0 = X[2]*X1X0;
#if (DD_IMPROVED == 1 && defined(MULTI_GPU))
  const int X3X1X0 = X[3]*X1X0;
#endif

int y[4] = {};

int full_idx=0;
#ifdef MULTI_GPU
  bool active = false;
  int dim = dimFromFaceIndex (idx, param);
    
  if(dim == 0){
    coordsFromFaceIndexStaggered<EXTERIOR_KERNEL_X,NFACE,2>(y, idx, param);
  }else if(dim == 1){
    coordsFromFaceIndexStaggered<EXTERIOR_KERNEL_Y,NFACE,2>(y, idx, param);
  }else if(dim == 2){
    coordsFromFaceIndexStaggered<EXTERIOR_KERNEL_Z,NFACE,2>(y, idx, param);
  }else if(dim == 3){
    coordsFromFaceIndexStaggered<EXTERIOR_KERNEL_T,NFACE,2>(y, idx, param);
  }

  const int half_volume = (X[0]*X[1]*X[2]*X[3] >> 1);
  full_idx = ((y[3]*X[2] +y[2])*X[1] +y[1])*X[0]+y[0];
#endif // MULTI_GPU
  int half_idx = full_idx>>1;


  for (int i=0; i<3; i++) O[i] = static_cast<spinorFloat>(0.0);

#if(DD_FAT_RECON == 13 || DD_FAT_RECON == 9)
int fat_sign = 1;
#endif
#if((DD_LONG_RECON == 13 || DD_LONG_RECON == 9) && DD_IMPROVED==1)
int long_sign = 1;
#endif


#ifdef PARALLEL_DIR
if (threadId.z & 1)
#endif // PARALLEL_DIR
{
  //direction: +X

#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = (y[3]%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = (y[3]%2 == 1) ? -1 : 1;
#endif

  int ga_idx = half_idx;

#ifdef MULTI_GPU
  if ( isActive(dim,0,NFACE,y,param.commDim,X) && y[0] >= (X[0]-1) )
#endif
  {
    int sp_idx_1st_nbr = ((y[0]==(X[0]-1)) ? full_idx-(X[0]-1) : full_idx+1) >> 1;
    READ_FAT_MATRIX(FATLINK0TEX, 0, ga_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	    
#ifdef MULTI_GPU
    active = true;
    int space_con = ((y[3]*X[2]+y[2])*X[1]+y[1])/2;	
    nbr_idx1 = param.ghostOffset[0][1] + src_idx*NFACE*ghostFace[0] + (y[0]-(X[0]-1))*ghostFace[0] + space_con;
    stride1 = NFACE*ghostFace[0]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[0][1] + src_idx*NFACE*ghostFace[0] + (y[0]-(X[0]-1))*ghostFace[0] + space_con;
#endif		    
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(0, fat, ga_idx, fat_sign);
    MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] += A[i];
  }

#if (DD_IMPROVED==1)
#ifdef MULTI_GPU
  if (isActive(dim,0,NFACE,y,param.commDim,X) && y[0] >= (X[0]-3))
#endif
  {
    int sp_idx_3rd_nbr = ((y[0] >= (X[0]-3)) ? full_idx-(X[0]-3) : full_idx+3) >> 1;
    READ_LONG_MATRIX(LONGLINK0TEX, 0, ga_idx, long_stride);        
    READ_LONG_PHASE(LONGPHASE0TEX, 0, ga_idx, long_stride);
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;    
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
    active = true;
    int space_con = ((y[3]*X[2]+y[2])*X[1] + y[1])/2;		
    nbr_idx3 = param.ghostOffset[0][1] + src_idx*NFACE*ghostFace[0] + (y[0]-(X[0]-3))*ghostFace[0] + space_con;
    stride3 = NFACE*ghostFace[0]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[0][1] + src_idx*NFACE*ghostFace[0] + (y[0]-(X[0]-3))*ghostFace[0] + space_con;
#endif	
#endif
    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);

    RECONSTRUCT_LONG_GAUGE_MATRIX(0, long, ga_idx, long_sign);
    MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] += B[i];
  }

#endif

} // direction: +X


#ifdef PARALLEL_DIR
if (!(threadIdx.z & 1))
#endif // PARALLEL_DIR
{
  // direction: -X
#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = (y[3]%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = (y[3]%2 == 1) ? -1 : 1;
#endif
  int dir =1;

#ifdef MULTI_GPU
  int space_con = ((y[3]*X[2] + y[2])*X[1] + y[1]) >>1;
  if (isActive(dim,0,NFACE,y,param.commDim,X) && y[0] < 1)
#endif
  {
    int sp_idx_1st_nbr = ((y[0]==0) ? full_idx+(X[0]-1) : full_idx-1) >> 1;
    int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if ((y[0] -1) < 0){
      fat_idx = half_volume + space_con;
    }
#endif
    READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[0][0] + src_idx*NFACE*ghostFace[0] + (y[0]+NFACE-1)*ghostFace[0] + space_con;
    stride1 = NFACE*ghostFace[0]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[0][0] + src_idx*NFACE*ghostFace[0] + (y[0]+NFACE-1)*ghostFace[0] + space_con;
#endif	
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(1, fat, ga_idx, fat_sign);
    ADJ_MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] -= A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU    
  if (isActive(dim,0,NFACE,y,param.commDim,X) && y[0] < 3)
#endif
  {
    int sp_idx_3rd_nbr = ((y[0]<3) ? full_idx+(X[0]-3): full_idx-3)>>1; 
    int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
    if ((y[0] -3) < 0){
      long_idx =half_volume + y[0]*(X[3]*X[2]*(X[1]>>1)) + space_con;
    }    
#endif
    READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx, long_stride); 		
    READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx, long_stride); 		
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	     
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[0][0] + src_idx*NFACE*ghostFace[0] + y[0]*ghostFace[0] + space_con;
    stride3 = NFACE*ghostFace[0]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[0][0] + src_idx*NFACE*ghostFace[0] + y[0]*ghostFace[0] + space_con;
#endif
#endif

    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);  
    RECONSTRUCT_LONG_GAUGE_MATRIX(1, long, sp_idx_3rd_nbr, long_sign);
    ADJ_MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] -= B[i];
  }
#endif

}


#ifdef PARALLEL_DIR
if (threadId.z & 1)
#endif // PARALLEL_DIR
{
  //direction: +Y
#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = ((y[3]+y[0])%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = ((y[3]+y[0])%2 == 1) ? -1 : 1;
#endif

  int ga_idx = half_idx;

#ifdef MULTI_GPU
  int space_con = ((y[3]*X[2]+y[2])*X[0]+y[0])/2;
  if (isActive(dim,1,NFACE,y,param.commDim,X) && y[1] >= (X[1]-1))
#endif
  {
    int sp_idx_1st_nbr = ((y[1]==(X[1]-1)) ? full_idx-(X1X0-X[0]) : full_idx+X[0]) >> 1;
    READ_FAT_MATRIX(FATLINK0TEX, 2, ga_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[1][1] + src_idx*NFACE*ghostFace[1] + (y[1]-(X[1]-1))*ghostFace[1] + space_con;
    stride1 = NFACE*ghostFace[1]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[1][1] + src_idx*NFACE*ghostFace[1] + (y[1]-(X[1]-1))*ghostFace[1] + space_con;
#endif		    
#endif 
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(2, fat, ga_idx, fat_sign);
    MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] += A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if (isActive(dim,1,NFACE,y,param.commDim,X) && y[1] >= (X[1]-3))    
#endif
  {
    int sp_idx_3rd_nbr = ((y[1] >= (X[1]-3) ) ? full_idx-(X[1]-3)*X[0] : full_idx+3*X[0]) >> 1;    
    READ_LONG_MATRIX(LONGLINK0TEX, 2, ga_idx, long_stride);
    READ_LONG_PHASE(LONGPHASE0TEX, 2, ga_idx, long_stride);
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;        
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[1][1] + src_idx*NFACE*ghostFace[1] + (y[1]-(X[1]-3))*ghostFace[1] + space_con;
    stride3 = NFACE*ghostFace[1]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[1][1] + src_idx*NFACE*ghostFace[1] + (y[1]-(X[1]-3))*ghostFace[1] + space_con;
#endif		    
#endif    
    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);

    RECONSTRUCT_LONG_GAUGE_MATRIX(2, long, ga_idx, long_sign);
    MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] += B[i];
  }
#endif
}


#ifdef PARALLEL_DIR
if (!(threadIdx.z & 1))
#endif // PARALLEL_DIR
{
  //direction: -Y

#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = ((y[3]+y[0])%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = ((y[3]+y[0])%2 == 1) ? -1 : 1;
#endif

  int dir=3;
#ifdef MULTI_GPU
  int space_con = (y[3]*X[2]*X[0] + y[2]*X[0] + y[0]) >>1;    
  if (isActive(dim,1,NFACE,y,param.commDim,X) && y[1] < 1)
#endif
  {
    int sp_idx_1st_nbr = ((y[1]==0)    ? full_idx+(X1X0-X[0]) : full_idx-X[0]) >> 1;
    int fat_idx=sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if ((y[1] -1) < 0){
      fat_idx = half_volume + space_con;
    }    
#endif
    READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[1][0] + src_idx*NFACE*ghostFace[1] + (y[1]+NFACE-1)*ghostFace[1] + space_con;
    stride1 = NFACE*ghostFace[1]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[1][0] + src_idx*NFACE*ghostFace[1] + (y[1]+NFACE-1)*ghostFace[1] + space_con;
#endif	
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(3, fat, ga_idx, fat_sign);
    ADJ_MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] -= A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if (isActive(dim,1,NFACE,y,param.commDim,X) && y[1] < 3)
#endif
  {
    int sp_idx_3rd_nbr = ((y[1] < 3) ? full_idx + (X[1]-3)*X[0]: full_idx -3*X[0] )>> 1; 
    int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
    if ((y[1]-3) < 0){
      long_idx = half_volume+ y[1]*(X[3]*X[2]*X[0] >> 1) + space_con;
    }    
#endif
    READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx, long_stride); 
    READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx, long_stride); 
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;    
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[1][0] + src_idx*NFACE*ghostFace[1] + y[1]*ghostFace[1] + space_con;
    stride3 = NFACE*ghostFace[1]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[1][0] + src_idx*NFACE*ghostFace[1] + y[1]*ghostFace[1] + space_con;
#endif
#endif
    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);

    RECONSTRUCT_LONG_GAUGE_MATRIX(3, long, sp_idx_3rd_nbr,long_sign);
    ADJ_MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] -= B[i];
  }    
#endif

}


#ifdef PARALLEL_DIR
if (threadId.z & 1)
#endif // PARALLEL_DIR
{
  //direction: +Z

#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = ((y[3]+y[0]+y[1])%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = ((y[3]+y[0]+y[1])%2 == 1) ? -1 : 1;
#endif

  int ga_idx = half_idx;

#ifdef MULTI_GPU
  int space_con = ((y[3]*X[1]+y[1])*X[0]+y[0])/2;
  if (isActive(dim,2,NFACE,y,param.commDim,X) && y[2] >= (X[2]-1))
#endif
  {
    int sp_idx_1st_nbr = ((y[2]==(X[2]-1)) ? full_idx-(X[2]-1)*X1X0 : full_idx+X1X0) >> 1;
    READ_FAT_MATRIX(FATLINK0TEX, 4, ga_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[2][1] + src_idx*NFACE*ghostFace[2] + (y[2]-(X[2]-1))*ghostFace[2] + space_con;
    stride1 = NFACE*ghostFace[2]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[2][1] + src_idx*NFACE*ghostFace[2] + (y[2]-(X[2]-1))*ghostFace[2] + space_con;
#endif		
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(4, fat, ga_idx, fat_sign);
    MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] += A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if (isActive(dim,2,NFACE,y,param.commDim,X)  && y[2] >= (X[2]-3))
#endif
  {
    int sp_idx_3rd_nbr = ((y[2]>= (X[2]-3))? full_idx -(X[2]-3)*X1X0: full_idx + 3*X1X0)>> 1;    
    READ_LONG_MATRIX(LONGLINK0TEX, 4, ga_idx, long_stride);
    READ_LONG_PHASE(LONGPHASE0TEX, 4, ga_idx, long_stride);
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[2][1] + src_idx*NFACE*ghostFace[2] + (y[2]-(X[2]-3))*ghostFace[2] + space_con;
    stride3 = NFACE*ghostFace[2]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[2][1] + src_idx*NFACE*ghostFace[2] + (y[2]-(X[2]-3))*ghostFace[2] + space_con;
#endif
#endif
    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);

    RECONSTRUCT_LONG_GAUGE_MATRIX(4, long, ga_idx, long_sign);
    MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] += B[i];
  }
#endif

}


#ifdef PARALLEL_DIR
if (!(threadIdx.z & 1))
#endif // PARALLEL_DIR
{
  //direction: -Z

#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = ((y[3]+y[0]+y[1])%2 == 1) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = ((y[3]+y[0]+y[1])%2 == 1) ? -1 : 1;
#endif

  int dir = 5;

#ifdef MULTI_GPU
  int space_con = ((y[3]*X[1] + y[1])*X[0] + y[0]) >>1;    
  if (isActive(dim,2,NFACE,y,param.commDim,X) && y[2] < 1)
#endif
  {
    int sp_idx_1st_nbr = ((y[2]==0)    ? full_idx+(X[2]-1)*X1X0 : full_idx-X1X0) >> 1;
    int fat_idx = sp_idx_1st_nbr;
#ifdef MULTI_GPU
    if ((y[2] -1) < 0){
      fat_idx = half_volume + space_con;
    }    
#endif
    READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[2][0] + src_idx*NFACE*ghostFace[2] + (y[2]+NFACE-1)*ghostFace[2] + space_con;
    stride1 = NFACE*ghostFace[2]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[2][0] + src_idx*NFACE*ghostFace[2] + (y[2]+NFACE-1)*ghostFace[2] + space_con;
#endif			    
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(5, fat, ga_idx, fat_sign);
    ADJ_MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] -= A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if (isActive(dim,2,NFACE,y,param.commDim,X) && y[2] < 3)
#endif
  {
    int sp_idx_3rd_nbr = ((y[2] <3) ? full_idx + (X[2]-3)*X1X0: full_idx - 3*X1X0)>>1;
    int long_idx = sp_idx_3rd_nbr;
#ifdef MULTI_GPU
    if ((y[2] -3) < 0){
      long_idx = half_volume + y[2]*(X3X1X0 >> 1) + space_con;
    }    
#endif
    READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx, long_stride);         
    READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx, long_stride);         
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;    
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	 
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[2][0] + src_idx*NFACE*ghostFace[2] + y[2]*ghostFace[2] + space_con;
    stride3 = NFACE*ghostFace[2]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[2][0] + src_idx*NFACE*ghostFace[2] + y[2]*ghostFace[2] + space_con;
#endif			    
#endif
    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);

    RECONSTRUCT_LONG_GAUGE_MATRIX(5, long, sp_idx_3rd_nbr,long_sign);
    ADJ_MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] -= B[i];
  }
#endif

}


#ifdef PARALLEL_DIR
if (threadId.z & 1)
#endif // PARALLEL_DIR
{
  //direction: +T
#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = (y[3] >= (X4-1)) ? time_boundary : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = (y[3] >= (X4-3)) ? time_boundary : 1;
#endif

  int ga_idx = half_idx;

#ifdef MULTI_GPU
  int space_con = (y[2]*X1X0+y[1]*X[0]+y[0])/2;
  if ((y[3] >= (X[3]-1)) && isActive(dim,3,NFACE,y,param.commDim,X))
#endif
  {    
    int sp_idx_1st_nbr = ((y[3]==(X[3]-1)) ? full_idx-(X[3]-1)*X2X1X0 : full_idx+X2X1X0) >> 1;
    READ_FAT_MATRIX(FATLINK0TEX, 6, ga_idx, fat_stride);
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif
#ifdef MULTI_GPU
    active = true;
    nbr_idx1 = param.ghostOffset[3][1] + src_idx*NFACE*ghostFace[3] + (y[3]-(X[3]-1))*ghostFace[3] + space_con;
    stride1 = NFACE*ghostFace[3]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[3][1] + src_idx*NFACE*ghostFace[3] + (y[3]-(X[3]-1))*ghostFace[3] + space_con;
#endif
#endif
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);    
    RECONSTRUCT_FAT_GAUGE_MATRIX(6, fat, ga_idx, fat_sign);
    MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] += A[i];
  }


#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ((isActive(dim,3,NFACE,y,param.commDim,X) &&  y[3] >= (X[3]-3)))
#endif
  {
    int sp_idx_3rd_nbr = ((y[3]>=(X[3]-3))? full_idx -(X[3]-3)*X2X1X0 : full_idx + 3*X2X1X0)>> 1;     
    READ_LONG_MATRIX(LONGLINK0TEX, 6, ga_idx, long_stride);    
    READ_LONG_PHASE(LONGPHASE0TEX, 6, ga_idx, long_stride);    
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;    
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif
#ifdef MULTI_GPU
    active = true;
    nbr_idx3 = param.ghostOffset[3][1] + src_idx*NFACE*ghostFace[3] + (y[3]-(X[3]-3))*ghostFace[3] + space_con;
    stride3 = NFACE*ghostFace[3]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[3][1] + src_idx*NFACE*ghostFace[3] + (y[3]-(X[3]-3))*ghostFace[3] + space_con;
#endif
#endif

    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3); 

    RECONSTRUCT_LONG_GAUGE_MATRIX(6, long, ga_idx, long_sign);
    MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] += B[i];
  }
#endif
}


#ifdef PARALLEL_DIR
if (!(threadIdx.z & 1))
#endif // PARALLEL_DIR
{
  //direction: -T
#if (DD_FAT_RECON == 12 || DD_FAT_RECON == 8)
  int fat_sign = ( ((y[3]+(X[3]-1))%X[3])>= (X[3]-1) ) ? -1 : 1;
#endif
#if ((DD_LONG_RECON == 12 || DD_LONG_RECON == 8) && DD_IMPROVED==1)
  int long_sign = ( ((y[3]+(X[3]-3))%X[3])>= (X[3]-3) ) ? -1 : 1;
#endif

  int dir = 7;

#ifdef MULTI_GPU
  int space_con = (y[2]*X1X0+y[1]*X[0]+y[0])/2;
  if ((y[3] < 1) && isActive(dim,3,NFACE,y,param.commDim,X))
#endif
  {
    int sp_idx_1st_nbr = ((y[3]==0)    ? full_idx+(X[3]-1)*X2X1X0 : full_idx-X2X1X0) >> 1;
    int fat_idx = sp_idx_1st_nbr;    
    int nbr_idx1 = sp_idx_1st_nbr + src_idx*Vh;
    int stride1 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx1 = nbr_idx1;
#endif
#ifdef MULTI_GPU
    active = true;
    fat_idx = half_volume + space_con;

    nbr_idx1 = param.ghostOffset[3][0] + src_idx*NFACE*ghostFace[3] + (y[3]+NFACE-1)*ghostFace[3] + space_con;
    stride1 = NFACE*ghostFace[3]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx1 = param.ghostNormOffset[3][0] + src_idx*NFACE*ghostFace[3] + (y[3]+NFACE-1)*ghostFace[3] + space_con;
#endif		    
#endif
    READ_FAT_MATRIX(FATLINK1TEX, dir, fat_idx, fat_stride);
    READ_1ST_NBR_SPINOR( GHOSTSPINORTEX, nbr_idx1, stride1);
    RECONSTRUCT_FAT_GAUGE_MATRIX(7, fat, ga_idx, fat_sign);
    ADJ_MAT_MUL_V(A, FAT, I);
    for (int i=0; i<3; i++) O[i] -= A[i];
  }

#if (DD_IMPROVED==1)

#ifdef MULTI_GPU
  if ((y[3]<3) && isActive(dim,3,NFACE,y,param.commDim,X))
#endif
  {
    int sp_idx_3rd_nbr = ((y[3]<3) ? full_idx + (X[3]-3)*X2X1X0: full_idx - 3*X2X1X0) >> 1;
    int long_idx = sp_idx_3rd_nbr;
    int nbr_idx3 = sp_idx_3rd_nbr + src_idx*Vh;
    int stride3 = param.sp_stride;
#if (DD_PREC == 2) //half precision
    int norm_idx3 = nbr_idx3;
#endif	    
#ifdef MULTI_GPU
    active = true;
    long_idx = half_volume + y[3]*ghostFace[3]+ space_con;
    nbr_idx3 = param.ghostOffset[3][0] + src_idx*NFACE*ghostFace[3] + y[3]*ghostFace[3] + space_con;
    stride3 = NFACE*ghostFace[3]*param.Ls;
#if (DD_PREC == 2) //half precision
    norm_idx3 = param.ghostNormOffset[3][0] + src_idx*NFACE*ghostFace[3] + y[3]*ghostFace[3] + space_con;
#endif		    
#endif	    
    READ_LONG_MATRIX(LONGLINK1TEX, dir, long_idx, long_stride);
    READ_LONG_PHASE(LONGPHASE1TEX, dir, long_idx, long_stride);

    READ_3RD_NBR_SPINOR(T, GHOSTSPINORTEX, nbr_idx3, stride3);       

    RECONSTRUCT_LONG_GAUGE_MATRIX(7, long, sp_idx_3rd_nbr, long_sign);
    ADJ_MAT_MUL_V(B, LONG, T);
    for (int i=0; i<3; i++) O[i] -= B[i];
  }        
#endif

}


#if (DD_DAG == 1)
{
  for (int i=0; i<3; i++) O[i] = -O[i];
}
#endif

#ifdef DSLASH_AXPY
#ifdef MULTI_GPU
for (int i=0; i<3; i++) O[i] = -O[i];
#else
READ_ACCUM(ACCUMTEX, half_idx+src_idx*Vh);
for (int i=0; i<3; i++) O[i]  = -O[i] + a*T[i];
#endif //MULTI_GPU
#endif // DSLASH_AXPY

#ifdef MULTI_GPU
if (active){
  READ_AND_SUM_SPINOR(INTERTEX, half_idx+src_idx*Vh);
  WRITE_SPINOR(param.out, O, half_idx+src_idx*Vh, param.sp_stride);
}
#endif




// undefine to prevent warning when precision is changed
#undef time_boundary

#undef gaugeFloat
#undef gaugeFloat2
#undef spinorFloat
#undef spinorFloat2
#undef SHARED_STRIDE

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

#undef long00_re
#undef long00_im
#undef long01_re
#undef long01_im
#undef long02_re
#undef long02_im
#undef long10_re
#undef long10_im
#undef long11_re
#undef long11_im
#undef long12_re
#undef long12_im
#undef long20_re
#undef long20_im
#undef long21_re
#undef long21_im
#undef long22_re
#undef long22_im

#undef long_re
#undef long_im

#undef SHARED_FLOATS_PER_THREAD
#undef kernel_type

#undef NFACE
