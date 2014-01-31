#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <sys/time.h>

#include <quda_internal.h>
#include <comm_quda.h>
#include <fat_force_quda.h>
#include <face_quda.h>

using namespace quda;

extern cudaStream_t *stream;
  
/**************************************************************
 * Staple exchange routine
 * used in fat link computation
 ***************************************************************/
//#ifndef CLOVER_FORCE
//#define CLOVER_FORCE
//#endif

#if defined(MULTI_GPU) && (defined(GPU_FATLINK) || defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE) || defined(GPU_HISQ_FORCE) || defined(CLOVER_FORCE)) || defined(GPU_CLOVER_DIRAC)

enum {
  XUP = 0,
  YUP = 1,
  ZUP = 2,
  TUP = 3,
  TDOWN = 4,
  ZDOWN = 5,
  YDOWN = 6,
  XDOWN = 7
};

#define gaugeSiteSize 18

#ifndef GPU_DIRECT
static void* fwd_nbr_staple_cpu[4];
static void* back_nbr_staple_cpu[4];
static void* fwd_nbr_staple_sendbuf_cpu[4];
static void* back_nbr_staple_sendbuf_cpu[4];
#endif

static void* fwd_nbr_staple_gpu[4];
static void* back_nbr_staple_gpu[4];

static void* fwd_nbr_staple[4];
static void* back_nbr_staple[4];
static void* fwd_nbr_staple_sendbuf[4];
static void* back_nbr_staple_sendbuf[4];

static int dims[4];
static int X1,X2,X3,X4;
static int V;
static int volumeCB;
static int Vs[4], Vsh[4];
static int Vs_x, Vs_y, Vs_z, Vs_t;
static int Vsh_x, Vsh_y, Vsh_z, Vsh_t;

static struct {
  MsgHandle *fwd[4];
  MsgHandle *back[4];
} llfat_recv, llfat_send;

#include "gauge_field.h"
extern void setup_dims_in_gauge(int *XX);

static void
setup_dims(int* X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    dims[d] = X[d];
  }
  volumeCB = V/2;
  
  X1=X[0];
  X2=X[1];
  X3=X[2];
  X4=X[3];

  Vs[0] = Vs_x = X[1]*X[2]*X[3];
  Vs[1] = Vs_y = X[0]*X[2]*X[3];
  Vs[2] = Vs_z = X[0]*X[1]*X[3];
  Vs[3] = Vs_t = X[0]*X[1]*X[2];

  Vsh[0] = Vsh_x = Vs_x/2;
  Vsh[1] = Vsh_y = Vs_y/2;
  Vsh[2] = Vsh_z = Vs_z/2;
  Vsh[3] = Vsh_t = Vs_t/2;
}


void exchange_llfat_init(QudaPrecision prec)
{
  static bool initialized = false;

  if (initialized) return;
  initialized = true;
  
  for (int i=0; i < 4; i++) {

    size_t packet_size = Vs[i]*gaugeSiteSize*prec;

    fwd_nbr_staple_gpu[i] = device_malloc(packet_size);
    back_nbr_staple_gpu[i] = device_malloc(packet_size);

    fwd_nbr_staple[i] = pinned_malloc(packet_size);
    back_nbr_staple[i] = pinned_malloc(packet_size);
    fwd_nbr_staple_sendbuf[i] = pinned_malloc(packet_size);
    back_nbr_staple_sendbuf[i] = pinned_malloc(packet_size);

#ifndef GPU_DIRECT
    fwd_nbr_staple_cpu[i] = safe_malloc(packet_size);
    back_nbr_staple_cpu[i] = safe_malloc(packet_size);
    fwd_nbr_staple_sendbuf_cpu[i] = safe_malloc(packet_size);
    back_nbr_staple_sendbuf_cpu[i] = safe_malloc(packet_size);
#endif  

  }
}


template<typename Float>
void exchange_sitelink_diag(int* X, Float** sitelink,  Float** ghost_sitelink_diag, int optflag)
{
  /*
    nu |          |
       |__________|
           mu 

  * There are total 12 different combinations for (nu,mu)
  * since nu/mu = X,Y,Z,T and nu != mu
  * For each combination, we need to communicate with the corresponding
  * neighbor and get the diag ghost data
  * The neighbor we need to get data from is dx[nu]=-1, dx[mu]= +1
  * and we need to send our data to neighbor with dx[nu]=+1, dx[mu]=-1
  */
  
  for(int nu = XUP; nu <=TUP; nu++){
    for(int mu = XUP; mu <= TUP; mu++){
      if(nu == mu){
	continue;
      }
      if(optflag && (!commDimPartitioned(mu) || !commDimPartitioned(nu))){
	continue;
      }

      int dir1, dir2; //other two dimensions
      for(dir1=0; dir1 < 4; dir1 ++){
	if(dir1 != nu && dir1 != mu){
	  break;
	}
      }
      for(dir2=0; dir2 < 4; dir2 ++){
	if(dir2 != nu && dir2 != mu && dir2 != dir1){
	  break;
	}
      }  
      
      if(dir1 == 4 || dir2 == 4){
	errorQuda("Invalid dir1/dir2");
      }
      int len = X[dir1]*X[dir2]*gaugeSiteSize*sizeof(Float);
      void *sendbuf = safe_malloc(len);
      
      pack_gauge_diag(sendbuf, X, (void**)sitelink, nu, mu, dir1, dir2, (QudaPrecision)sizeof(Float));
  
      int dx[4] = {0};
      dx[nu] = -1;
      dx[mu] = +1;
      MsgHandle *mh_recv = comm_declare_receive_displaced(ghost_sitelink_diag[nu*4+mu], dx, len);
      comm_start(mh_recv);

      dx[nu] = +1;
      dx[mu] = -1;
      MsgHandle *mh_send = comm_declare_send_displaced(sendbuf, dx, len);
      comm_start(mh_send);
      
      comm_wait(mh_send);
      comm_wait(mh_recv);
            
      comm_free(mh_send);
      comm_free(mh_recv);
            
      host_free(sendbuf);
    }
  }
}


template<typename Float>
void
exchange_sitelink(int*X, Float** sitelink, Float** ghost_sitelink, Float** ghost_sitelink_diag, 
		  Float** sitelink_fwd_sendbuf, Float** sitelink_back_sendbuf, int optflag)
{


#if 0
  int i;
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);
  for(i=0;i < 4;i++){
    Float* even_sitelink_back_src = sitelink[i];
    Float* odd_sitelink_back_src = sitelink[i] + volumeCB*gaugeSiteSize;
    Float* sitelink_back_dst = sitelink_back_sendbuf[3] + 2*i*Vsh_t*gaugeSiteSize;

    if(dims[3] % 2 == 0){    
      memcpy(sitelink_back_dst, even_sitelink_back_src, len);
      memcpy(sitelink_back_dst + Vsh_t*gaugeSiteSize, odd_sitelink_back_src, len);
    }else{
      //switching odd and even ghost sitelink
      memcpy(sitelink_back_dst, odd_sitelink_back_src, len);
      memcpy(sitelink_back_dst + Vsh_t*gaugeSiteSize, even_sitelink_back_src, len);
    }
  }

  for(i=0;i < 4;i++){
    Float* even_sitelink_fwd_src = sitelink[i] + (volumeCB - Vsh_t)*gaugeSiteSize;
    Float* odd_sitelink_fwd_src = sitelink[i] + volumeCB*gaugeSiteSize + (volumeCB - Vsh_t)*gaugeSiteSize;
    Float* sitelink_fwd_dst = sitelink_fwd_sendbuf[3] + 2*i*Vsh_t*gaugeSiteSize;
    if(dims[3] % 2 == 0){    
      memcpy(sitelink_fwd_dst, even_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, odd_sitelink_fwd_src, len);
    }else{
      //switching odd and even ghost sitelink
      memcpy(sitelink_fwd_dst, odd_sitelink_fwd_src, len);
      memcpy(sitelink_fwd_dst + Vsh_t*gaugeSiteSize, even_sitelink_fwd_src, len);
    }
    
  }
#else
  int nFace =1;
  for(int dir=0; dir < 4; dir++){
    if(optflag && !commDimPartitioned(dir)) continue;
    pack_ghost_all_links((void**)sitelink, (void**)sitelink_back_sendbuf, (void**)sitelink_fwd_sendbuf, dir, nFace, (QudaPrecision)(sizeof(Float)), X);
  }
#endif

  for (int dir = 0; dir < 4; dir++) {
    if(optflag && !commDimPartitioned(dir)) continue;
    int len = Vsh[dir]*gaugeSiteSize*sizeof(Float);
    Float* ghost_sitelink_back = ghost_sitelink[dir];
    Float* ghost_sitelink_fwd = ghost_sitelink[dir] + 8*Vsh[dir]*gaugeSiteSize;

    MsgHandle *mh_recv_back;
    MsgHandle *mh_recv_fwd;
    MsgHandle *mh_send_fwd;
    MsgHandle *mh_send_back;

    mh_recv_back = comm_declare_receive_relative(ghost_sitelink_back, dir, -1, 8*len);
    mh_recv_fwd = comm_declare_receive_relative(ghost_sitelink_fwd, dir, +1, 8*len);
    mh_send_fwd = comm_declare_send_relative(sitelink_fwd_sendbuf[dir], dir, +1, 8*len);
    mh_send_back = comm_declare_send_relative(sitelink_back_sendbuf[dir], dir, -1, 8*len);

    comm_start(mh_recv_back);
    comm_start(mh_recv_fwd);
    comm_start(mh_send_fwd);
    comm_start(mh_send_back);

    comm_wait(mh_send_fwd);
    comm_wait(mh_send_back);
    comm_wait(mh_recv_back);
    comm_wait(mh_recv_fwd);

    comm_free(mh_send_fwd);
    comm_free(mh_send_back);
    comm_free(mh_recv_back);
    comm_free(mh_recv_fwd);
  }

  exchange_sitelink_diag(X, sitelink, ghost_sitelink_diag, optflag);
}


//this function is used for link fattening computation
//@optflag: if this flag is set, we only communicate in directions that are partitioned
//          if not set, then we communicate in all directions regradless of partitions
void exchange_cpu_sitelink(int* X,
			   void** sitelink, void** ghost_sitelink,
			   void** ghost_sitelink_diag,
			   QudaPrecision gPrecision, QudaGaugeParam* param, int optflag)
{  
  setup_dims(X);
  static void*  sitelink_fwd_sendbuf[4];
  static void*  sitelink_back_sendbuf[4];
  static bool allocated = false;

  if (!allocated) {
    for (int i=0; i<4; i++) {
      int nbytes = 4*Vs[i]*gaugeSiteSize*gPrecision;
      sitelink_fwd_sendbuf[i] = safe_malloc(nbytes);
      sitelink_back_sendbuf[i] = safe_malloc(nbytes);
      memset(sitelink_fwd_sendbuf[i], 0, nbytes);
      memset(sitelink_back_sendbuf[i], 0, nbytes);
    }
    allocated = true;
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_sitelink(X, (double**)sitelink, (double**)(ghost_sitelink), (double**)ghost_sitelink_diag, 
		      (double**)sitelink_fwd_sendbuf, (double**)sitelink_back_sendbuf, optflag);
  }else{ //single
    exchange_sitelink(X, (float**)sitelink, (float**)(ghost_sitelink), (float**)ghost_sitelink_diag, 
		      (float**)sitelink_fwd_sendbuf, (float**)sitelink_back_sendbuf, optflag);
  }
  
  if(!(param->preserve_gauge & QUDA_FAT_PRESERVE_COMM_MEM)){
    for(int i=0;i < 4;i++){
      host_free(sitelink_fwd_sendbuf[i]);
      host_free(sitelink_back_sendbuf[i]);
    }
    allocated = false;
  }
}


#define MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_buf, dst_idx, sitelink, src_idx, num, dir, geom) \
  if(src_oddness) src_idx += Vh_ex;					\
  if(dst_oddness) dst_idx += R[dir]*slice_3d[dir]/2;			\
  if(cpu_order == QUDA_QDP_GAUGE_ORDER) {				\
    for(int linkdir=0; linkdir < 4; linkdir++){				\
      char* src = (char*) sitelink[linkdir] + (src_idx)*gaugebytes;	\
      char* dst = ((char*)ghost_buf[dir])+ linkdir*R[dir]*slice_3d[dir]*gaugebytes + (dst_idx)*gaugebytes; \
      memcpy(dst, src, gaugebytes*(num));				\
    }									\
  } else if (cpu_order == QUDA_MILC_GAUGE_ORDER) {			\
    char* src = ((char*)sitelink)+ (geom)*(src_idx)*gaugebytes;		\
    char* dst = ((char*)ghost_buf[dir]) + (geom)*(dst_idx)*gaugebytes;	\
    memcpy(dst, src, (geom)*gaugebytes*(num));				\
  } else {								\
    errorQuda("Unsupported gauge order");				\
  }									\

#define MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_buf, src_idx, num, dir, geom) \
  if(oddness){								\
    if(commDimPartitioned(dir)){					\
      src_idx += R[dir]*slice_3d[dir]/2;				\
    }else{								\
      src_idx += Vh_ex;							\
    }									\
    dst_idx += Vh_ex;							\
  }									\
  if(cpu_order == QUDA_QDP_GAUGE_ORDER){				\
    for(int linkdir=0; linkdir < 4; linkdir++){				\
      char* src;							\
      if(commDimPartitioned(dir)){					\
	src = ((char*)ghost_buf[dir])+ linkdir*R[dir]*slice_3d[dir]*gaugebytes + (src_idx)*gaugebytes; \
      }else{								\
	src = ((char*)sitelink[linkdir])+ (src_idx)*gaugebytes;		\
      }									\
      char* dst = (char*) sitelink[linkdir] + (dst_idx)*gaugebytes;	\
      memcpy(dst, src, gaugebytes*(num));				\
    }									\
  } else if (cpu_order == QUDA_MILC_GAUGE_ORDER) {			\
    char* src;								\
    if(commDimPartitioned(dir)){					\
      src=((char*)ghost_buf[dir]) + (geom)*(src_idx)*gaugebytes;	\
    }else{								\
      src = ((char*)sitelink)+ (geom)*(src_idx)*gaugebytes;		\
    }									\
    char* dst = ((char*)sitelink) + (geom)*(dst_idx)*gaugebytes;	\
    memcpy(dst, src, (geom)*gaugebytes*(num));				\
  } else {								\
    errorQuda("Unsupported gauge order");				\
  }

#define MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_buf, dst_face, src_face, dir, geom) \
  /*even*/								\
  int even_dst_idx = (dst_face*E[2]*E[1]*E[0])/2;				\
  int even_src_idx;							\
  if(commDimPartitioned(dir)){						\
    even_src_idx = 0;							\
  }else{								\
    even_src_idx = (src_face*E[2]*E[1]*E[0])/2;					\
  }									\
  /*odd*/								\
  int odd_dst_idx = even_dst_idx+Vh_ex;					\
  int odd_src_idx;							\
  if(commDimPartitioned(dir)){						\
    odd_src_idx = R[dir]*slice_3d[dir]/2;				\
  }else{								\
    odd_src_idx = even_src_idx+Vh_ex;					\
  }									\
  if(cpu_order == QUDA_QDP_GAUGE_ORDER){				\
    for(int linkdir=0; linkdir < 4; linkdir ++){			\
      char* dst = (char*)sitelink[linkdir];				\
      char* src;							\
      if(commDimPartitioned(dir)){					\
	src = ((char*)ghost_buf[dir]) + linkdir*R[dir]*slice_3d[dir]*gaugebytes; \
      }else{								\
	src = (char*)sitelink[linkdir];					\
      }									\
      memcpy(dst + even_dst_idx * gaugebytes, src + even_src_idx*gaugebytes, R[dir]*slice_3d[dir]*gaugebytes/2); \
      memcpy(dst + odd_dst_idx * gaugebytes, src + odd_src_idx*gaugebytes, R[dir]*slice_3d[dir]*gaugebytes/2); \
    }									\
  } else if (cpu_order == QUDA_MILC_GAUGE_ORDER) {			\
    char* dst = (char*)sitelink;					\
    char* src;								\
    if(commDimPartitioned(dir)){					\
      src = (char*)ghost_buf[dir];					\
    }else{								\
      src = (char*)sitelink;						\
    }									\
    memcpy(dst+(geom)*even_dst_idx*gaugebytes, src+(geom)*even_src_idx*gaugebytes, (geom)*R[dir]*slice_3d[dir]*gaugebytes/2); \
    memcpy(dst+(geom)*odd_dst_idx*gaugebytes, src+(geom)*odd_src_idx*gaugebytes, (geom)*R[dir]*slice_3d[dir]*gaugebytes/2); \
  } else {								\
    errorQuda("Unsupported gauge order\n");				\
  }

/* This function exchange the sitelink and store them in the correspoinding portion of 
 * the extended sitelink memory region
 * @sitelink: this is stored according to dimension size  (X4+R4) * (X1+R1) * (X2+R2) * (X3+R3)
 */

// gaugeSiteSize

void exchange_cpu_sitelink_ex(int* X, int *R, void** sitelink, QudaGaugeFieldOrder cpu_order,
			      QudaPrecision gPrecision, int optflag, int geometry)
{
  int E[4];
  for (int i=0; i<4; i++) E[i] = X[i] + 2*R[i];
  int Vh_ex = E[3]*E[2]*E[1]*E[0]/2;
  
  //...............x.........y.....z......t
  int starta[] = {R[3],      R[3],       R[3],     0};
  int enda[]   = {X[3]+R[3], X[3]+R[3],  X[3]+R[3], X[2]+2*R[2]};
  
  int startb[] = {R[2],     R[2],       0,     0};
  int endb[]   = {X[2]+R[2], X[2]+R[2], X[1]+2*R[1], X[1]+2*R[1]};
  
  int startc[] = {R[1],     0,       0,     0};
  int endc[]   = {X[1]+R[1], X[0]+2*R[0],  X[0]+2*R[0],  X[0]+2*R[0]};
  
  int f_main[4][4] = {
    {E[2]*E[1]*E[0], E[1]*E[0], E[0],              1},
    {E[2]*E[1]*E[0], E[1]*E[0],    1,           E[0]},
    {E[2]*E[1]*E[0],      E[0],    1,      E[1]*E[0]},
    {E[1]*E[0],           E[0],    1, E[2]*E[1]*E[0]}
  };  
  
  int f_bound[4][4]={
    {E[2]*E[1], E[1], 1, E[3]*E[2]*E[1]},
    {E[2]*E[0], E[0], 1, E[3]*E[2]*E[0]}, 
    {E[1]*E[0], E[0], 1, E[3]*E[1]*E[0]},
    {E[1]*E[0], E[0], 1, E[2]*E[1]*E[0]}
  };
  
  int slice_3d[] = { E[3]*E[2]*E[1], E[3]*E[2]*E[0], E[3]*E[1]*E[0], E[2]*E[1]*E[0]};  
  int len[4];
  for(int i=0; i<4;i++){
    len[i] = slice_3d[i] * R[i] * geometry*gaugeSiteSize*gPrecision; //2 slices, 4 directions' links
  }

  void* ghost_sitelink_fwd_sendbuf[4];
  void* ghost_sitelink_back_sendbuf[4];
  void* ghost_sitelink_fwd[4];
  void* ghost_sitelink_back[4];  

  for(int i=0; i<4; i++) {
    if(!commDimPartitioned(i)) continue;
    ghost_sitelink_fwd_sendbuf[i] = safe_malloc(len[i]);
    ghost_sitelink_back_sendbuf[i] = safe_malloc(len[i]);
    ghost_sitelink_fwd[i] = safe_malloc(len[i]);
    ghost_sitelink_back[i] = safe_malloc(len[i]);
  }

  int gaugebytes = gaugeSiteSize*gPrecision;
  int a, b, c,d;
  for(int dir =0;dir < 4;dir++){
    if( (!commDimPartitioned(dir)) && optflag) continue;
    if(commDimPartitioned(dir)){
      //fill the sendbuf here
      //back
      for(d=R[dir]; d < 2*R[dir]; d++)
	for(a=starta[dir];a < enda[dir]; a++)
	  for(b=startb[dir]; b < endb[dir]; b++)

	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-R[dir])*f_bound[dir][3])>> 1;	      
		
		int src_oddness = oddness;
		int dst_oddness = oddness;
		if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		  dst_oddness = 1-oddness;
		}

		MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_back_sendbuf, dst_idx, sitelink, src_idx, 1, dir, geometry);		

	      }//c
	    }else{
	      for(int loop=0; loop < 2; loop++){
		c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-R[dir])*f_bound[dir][3])>> 1;	      
		  
		  int src_oddness = oddness;
		  int dst_oddness = oddness;
		  if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		    dst_oddness = 1-oddness;
		  }
		  MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_back_sendbuf, dst_idx, sitelink, src_idx, (endc[dir]-c+1)/2, dir, geometry);	

		}//if c
	      }//for loop
	    }//if
      
      
      //fwd
      for(d=X[dir]; d < X[dir]+R[dir]; d++) {
	for(a=starta[dir];a < enda[dir]; a++) {
	  for(b=startb[dir]; b < endb[dir]; b++) {
	    
	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir])*f_bound[dir][3])>> 1;
		
		int src_oddness = oddness;
		int dst_oddness = oddness;
		if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		  dst_oddness = 1-oddness;
		}
		
		MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_fwd_sendbuf, dst_idx, sitelink, src_idx, 1,dir, geometry);
	      }//c
	    }else{
	      for(int loop=0; loop < 2; loop++){
		c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int dst_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir])*f_bound[dir][3])>> 1;
		  
		  int src_oddness = oddness;
		  int dst_oddness = oddness;
		  if((X[dir] % 2 ==1) && (commDim(dir) > 1)){ //switch even/odd position
		    dst_oddness = 1-oddness;
		  }
		  MEMCOPY_GAUGE_FIELDS_GRID_TO_BUF(ghost_sitelink_fwd_sendbuf, dst_idx, sitelink, src_idx, (endc[dir]-c+1)/2,dir, geometry);
		}
	      }//for loop
	    }//if

	  }
	}
      }
      
      MsgHandle *mh_recv_back;
      MsgHandle *mh_recv_fwd;
      MsgHandle *mh_send_fwd;
      MsgHandle *mh_send_back;
  
      mh_recv_back = comm_declare_receive_relative(ghost_sitelink_back[dir], dir, -1, len[dir]);
      mh_recv_fwd = comm_declare_receive_relative(ghost_sitelink_fwd[dir], dir, +1, len[dir]);
      mh_send_fwd = comm_declare_send_relative(ghost_sitelink_fwd_sendbuf[dir], dir, +1, len[dir]);
      mh_send_back = comm_declare_send_relative(ghost_sitelink_back_sendbuf[dir], dir, -1, len[dir]);

      comm_start(mh_recv_back);
      comm_start(mh_recv_fwd);
      comm_start(mh_send_fwd);
      comm_start(mh_send_back);

      comm_wait(mh_send_fwd);
      comm_wait(mh_send_back);
      comm_wait(mh_recv_back);
      comm_wait(mh_recv_fwd);

      comm_free(mh_send_fwd);
      comm_free(mh_send_back);
      comm_free(mh_recv_back);
      comm_free(mh_recv_fwd);
      
    }//if

    //use the messages to fill the sitelink data
    //back
    if (dir < 3 ) {

      for(d=0; d < R[dir]; d++) {
	for(a=starta[dir];a < enda[dir]; a++) {
	  for(b=startb[dir]; b < endb[dir]; b++) {

	    if(f_main[dir][2] != 1 || f_bound[dir][2] !=1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int src_idx;
		if(commDimPartitioned(dir)){
		  src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + d*f_bound[dir][3])>> 1;
		}else{
		  src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d+X[dir])*f_main[dir][3])>> 1;
		}

		MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_back, src_idx, 1, dir, geometry);
		
	      }//c    
	    }else{
	      //optimized copy
	      //first half:   startc[dir] -> end[dir] with step=2

	      for(int loop =0;loop <2;loop++){
		int c=startc[dir]+loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int src_idx;
		  if(commDimPartitioned(dir)){
		    src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + d*f_bound[dir][3])>> 1;
		  }else{
		    src_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d+X[dir])*f_main[dir][3])>> 1;
		  }

		  MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_back, src_idx, (endc[dir]-c+1)/2, dir, geometry);

		}//if c  		
	      }//for loop	      
	    }//if

	  }
	}
      }

    }else{
      //when dir == 3 (T direction), the data layout format in sitelink and the message is the same, we can do large copys  

      MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_sitelink_back, 0, X[3], dir, geometry)
    }//if
    
    //fwd
    if( dir < 3 ){

      for(d=X[dir]+R[dir]; d < X[dir]+2*R[dir]; d++) {
	for(a=starta[dir];a < enda[dir]; a++) {
	  for(b=startb[dir]; b < endb[dir]; b++) {

	    if(f_main[dir][2] != 1 || f_bound[dir][2] != 1){
	      for (c=startc[dir]; c < endc[dir]; c++){
		int oddness = (a+b+c+d)%2;
		int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		int src_idx;
		if(commDimPartitioned(dir)){
		  src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir]-R[dir])*f_bound[dir][3])>> 1;
		}else{
		  src_idx =  ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d-X[dir])*f_main[dir][3])>> 1;
		}

		MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_fwd, src_idx, 1, dir, geometry);

	      }//c
	    }else{
	      for(int loop =0; loop < 2; loop++){
		//for (c=startc[dir]; c < endc[dir]; c++){
		c=startc[dir] + loop;
		if(c < endc[dir]){
		  int oddness = (a+b+c+d)%2;
		  int dst_idx = ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + d*f_main[dir][3])>> 1;
		  int src_idx;
		  if(commDimPartitioned(dir)){
		    src_idx = ( a*f_bound[dir][0] + b*f_bound[dir][1]+ c*f_bound[dir][2] + (d-X[dir]-R[dir])*f_bound[dir][3])>> 1;
		  }else{
		    src_idx =  ( a*f_main[dir][0] + b*f_main[dir][1]+ c*f_main[dir][2] + (d-X[dir])*f_main[dir][3])>> 1;
		  }
		  MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID(sitelink, dst_idx, ghost_sitelink_fwd, src_idx, (endc[dir]-c+1)/2, dir, geometry);
		}//if		
	      }//for loop
	    }//if 

	  }
	}
      }
      

    } else {

      //when dir == 3 (T direction), the data layout format in sitelink and the message is the same, we can do large copys
      MEMCOPY_GAUGE_FIELDS_BUF_TO_GRID_T(sitelink, ghost_sitelink_fwd, (X[3]+R[3]), 2, dir, geometry) // TESTME 2

    }//if    

  }//dir for loop
    
  
  for(int dir=0;dir < 4;dir++){
    if(!commDimPartitioned(dir)) continue;
    host_free(ghost_sitelink_fwd_sendbuf[dir]);
    host_free(ghost_sitelink_back_sendbuf[dir]);    
    host_free(ghost_sitelink_fwd[dir]);
    host_free(ghost_sitelink_back[dir]);    
  }
    
}



template<typename Float>
void
do_exchange_cpu_staple(Float* staple, Float** ghost_staple, Float** staple_fwd_sendbuf, Float** staple_back_sendbuf, int* X)
{


#if 0  
  int len = Vsh_t*gaugeSiteSize*sizeof(Float);
  Float* even_staple_back_src = staple;
  Float* odd_staple_back_src = staple + volumeCB*gaugeSiteSize;
  Float* staple_back_dst = staple_back_sendbuf[3];
  
  if(dims[3] % 2 == 0){    
    memcpy(staple_back_dst, even_staple_back_src, len);
    memcpy(staple_back_dst + Vsh_t*gaugeSiteSize, odd_staple_back_src, len);
  }else{
    //switching odd and even ghost staple
    memcpy(staple_back_dst, odd_staple_back_src, len);
    memcpy(staple_back_dst + Vsh_t*gaugeSiteSize, even_staple_back_src, len);
  }
  
  
  Float* even_staple_fwd_src = staple + (volumeCB - Vsh_t)*gaugeSiteSize;
  Float* odd_staple_fwd_src = staple + volumeCB*gaugeSiteSize + (volumeCB - Vsh_t)*gaugeSiteSize;
  Float* staple_fwd_dst = staple_fwd_sendbuf[3];
  if(dims[3] % 2 == 0){    
    memcpy(staple_fwd_dst, even_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, odd_staple_fwd_src, len);
  }else{
    //switching odd and even ghost staple
    memcpy(staple_fwd_dst, odd_staple_fwd_src, len);
    memcpy(staple_fwd_dst + Vsh_t*gaugeSiteSize, even_staple_fwd_src, len);
  }
#else
  int nFace =1;
  pack_ghost_all_staples_cpu(staple, (void**)staple_back_sendbuf, 
			     (void**)staple_fwd_sendbuf,  nFace, (QudaPrecision)(sizeof(Float)), X);

#endif  
  
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};
  int len[4] = {
    Vsh_x*gaugeSiteSize*sizeof(Float),
    Vsh_y*gaugeSiteSize*sizeof(Float),
    Vsh_z*gaugeSiteSize*sizeof(Float),
    Vsh_t*gaugeSiteSize*sizeof(Float)
  };
  
  for (int dir=0;dir < 4; dir++) {

    Float *ghost_staple_back = ghost_staple[dir];
    Float *ghost_staple_fwd = ghost_staple[dir] + 2*Vsh[dir]*gaugeSiteSize;

    MsgHandle *mh_recv_back;
    MsgHandle *mh_recv_fwd;
    MsgHandle *mh_send_fwd;
    MsgHandle *mh_send_back;

    mh_recv_back = comm_declare_receive_relative(ghost_staple_back, dir, -1, 2*len[dir]);
    mh_recv_fwd = comm_declare_receive_relative(ghost_staple_fwd, dir, +1, 2*len[dir]);
    mh_send_fwd = comm_declare_send_relative(staple_fwd_sendbuf[dir], dir, +1, 2*len[dir]);
    mh_send_back = comm_declare_send_relative(staple_back_sendbuf[dir], dir, -1, 2*len[dir]);
    
    comm_start(mh_recv_back);
    comm_start(mh_recv_fwd);
    comm_start(mh_send_fwd);
    comm_start(mh_send_back);

    comm_wait(mh_send_fwd);
    comm_wait(mh_send_back);
    comm_wait(mh_recv_back);
    comm_wait(mh_recv_fwd);

    comm_free(mh_send_fwd);
    comm_free(mh_send_back);
    comm_free(mh_recv_back);
    comm_free(mh_recv_fwd);
  }
}


//this function is used for link fattening computation
void exchange_cpu_staple(int* X, void* staple, void** ghost_staple, QudaPrecision gPrecision)
{  
  setup_dims(X);

  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  void *staple_fwd_sendbuf[4];
  void *staple_back_sendbuf[4];

  for(int i=0;i < 4; i++){
    staple_fwd_sendbuf[i] = safe_malloc(Vs[i]*gaugeSiteSize*gPrecision);
    staple_back_sendbuf[i] = safe_malloc(Vs[i]*gaugeSiteSize*gPrecision);
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION) {
    do_exchange_cpu_staple((double*)staple, (double**)ghost_staple, 
			   (double**)staple_fwd_sendbuf, (double**)staple_back_sendbuf, X);
  } else { //single
    do_exchange_cpu_staple((float*)staple, (float**)ghost_staple, 
			   (float**)staple_fwd_sendbuf, (float**)staple_back_sendbuf, X);
  }
  
  for (int i=0;i < 4;i++) {
    host_free(staple_fwd_sendbuf[i]);
    host_free(staple_back_sendbuf[i]);
  }
}

//@whichway indicates send direction
void
exchange_gpu_staple_start(int* X, void* _cudaStaple, int dir, int whichway, cudaStream_t * stream)
{
  setup_dims(X);
  
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;
  exchange_llfat_init(cudaStaple->Precision());
  

  void* even = cudaStaple->Even_p();
  void* odd = cudaStaple->Odd_p();
  int volume = cudaStaple->VolumeCB();
  QudaPrecision prec = cudaStaple->Precision();
  int stride = cudaStaple->Stride();
  
  packGhostStaple(X, even, odd, volume, prec, stride, 
		  dir, whichway, fwd_nbr_staple_gpu, back_nbr_staple_gpu,
		  fwd_nbr_staple_sendbuf, back_nbr_staple_sendbuf, stream);
}


void exchange_gpu_staple_comms(int* X, void* _cudaStaple, int dim, int send_dir, cudaStream_t *stream)
{
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;  
  QudaPrecision prec = cudaStaple->Precision();
  
  cudaStreamSynchronize(*stream);  

  int recv_dir = (send_dir == QUDA_BACKWARDS) ? QUDA_FORWARDS : QUDA_BACKWARDS;

  int len = Vs[dim]*gaugeSiteSize*prec;

  if (recv_dir == QUDA_BACKWARDS) {

#ifdef GPU_DIRECT
    llfat_recv.back[dim] = comm_declare_receive_relative(back_nbr_staple[dim], dim, -1, len);
    llfat_send.fwd[dim] = comm_declare_send_relative(fwd_nbr_staple_sendbuf[dim], dim, +1, len);
#else
    llfat_recv.back[dim] = comm_declare_receive_relative(back_nbr_staple_cpu[dim], dim, -1, len);
    memcpy(fwd_nbr_staple_sendbuf_cpu[dim], fwd_nbr_staple_sendbuf[dim], len);
    llfat_send.fwd[dim] = comm_declare_send_relative(fwd_nbr_staple_sendbuf_cpu[dim], dim, +1, len);
#endif

    comm_start(llfat_recv.back[dim]);
    comm_start(llfat_send.fwd[dim]);

  } else { // QUDA_FORWARDS

#ifdef GPU_DIRECT
    llfat_recv.fwd[dim] = comm_declare_receive_relative(fwd_nbr_staple[dim], dim, +1, len);
    llfat_send.back[dim] = comm_declare_send_relative(back_nbr_staple_sendbuf[dim], dim, -1, len);
#else
    llfat_recv.fwd[dim] = comm_declare_receive_relative(fwd_nbr_staple_cpu[dim], dim, +1, len);
    memcpy(back_nbr_staple_sendbuf_cpu[dim], back_nbr_staple_sendbuf[dim], len);
    llfat_send.back[dim] = comm_declare_send_relative(back_nbr_staple_sendbuf_cpu[dim], dim, -1, len);
#endif

    comm_start(llfat_recv.fwd[dim]);
    comm_start(llfat_send.back[dim]);

  }
}


//@whichway indicates send direction
//we use recv_whichway to indicate recv direction
void
exchange_gpu_staple_wait(int* X, void* _cudaStaple, int dim, int send_dir, cudaStream_t * stream)
{
  cudaGaugeField* cudaStaple = (cudaGaugeField*) _cudaStaple;  

  void* even = cudaStaple->Even_p();
  void* odd = cudaStaple->Odd_p();
  int volume = cudaStaple->VolumeCB();
  QudaPrecision prec = cudaStaple->Precision();
  int stride = cudaStaple->Stride();

  int recv_dir = (send_dir == QUDA_BACKWARDS) ? QUDA_FORWARDS : QUDA_BACKWARDS;

#ifndef GPU_DIRECT
  int len = Vs[dim]*gaugeSiteSize*prec;
#endif  

  if (recv_dir == QUDA_BACKWARDS) {   

    comm_wait(llfat_send.fwd[dim]);
    comm_wait(llfat_recv.back[dim]);

    comm_free(llfat_send.fwd[dim]);
    comm_free(llfat_recv.back[dim]);

#ifdef GPU_DIRECT
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      dim, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else   
    memcpy(back_nbr_staple[dim], back_nbr_staple_cpu[dim], len);
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      dim, QUDA_BACKWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#endif

  } else { // QUDA_FORWARDS

    comm_wait(llfat_send.back[dim]);
    comm_wait(llfat_recv.fwd[dim]);

    comm_free(llfat_send.back[dim]);
    comm_free(llfat_recv.fwd[dim]);

#ifdef GPU_DIRECT
    unpackGhostStaple(X, even, odd, volume, prec, stride, 
		      dim, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#else        
    memcpy(fwd_nbr_staple[dim], fwd_nbr_staple_cpu[dim], len);
    unpackGhostStaple(X, even, odd, volume, prec, stride,
		      dim, QUDA_FORWARDS, fwd_nbr_staple, back_nbr_staple, stream);
#endif

  }
}


void exchange_llfat_cleanup(void)
{
  for (int i=0; i<4; i++) {

    if(fwd_nbr_staple_gpu[i]){
      device_free(fwd_nbr_staple_gpu[i]); fwd_nbr_staple_gpu[i] = NULL;
    }      
    if(back_nbr_staple_gpu[i]){
      device_free(back_nbr_staple_gpu[i]); back_nbr_staple_gpu[i] = NULL;
    }

#ifndef GPU_DIRECT
    if(fwd_nbr_staple_cpu[i]){
      host_free(fwd_nbr_staple_cpu[i]); fwd_nbr_staple_cpu[i] = NULL;
    }      
    if(back_nbr_staple_cpu[i]){
      host_free(back_nbr_staple_cpu[i]);back_nbr_staple_cpu[i] = NULL;
    }
    if(fwd_nbr_staple_sendbuf_cpu[i]){
      host_free(fwd_nbr_staple_sendbuf_cpu[i]); fwd_nbr_staple_sendbuf_cpu[i] = NULL;
    }
    if(back_nbr_staple_sendbuf_cpu[i]){
      host_free(back_nbr_staple_sendbuf_cpu[i]); back_nbr_staple_sendbuf_cpu[i] = NULL;
    }    
#endif

    if(fwd_nbr_staple[i]){
      host_free(fwd_nbr_staple[i]); fwd_nbr_staple[i] = NULL;
    }
    if(back_nbr_staple[i]){
      host_free(back_nbr_staple[i]); back_nbr_staple[i] = NULL;
    }
    if(fwd_nbr_staple_sendbuf[i]){
      host_free(fwd_nbr_staple_sendbuf[i]); fwd_nbr_staple_sendbuf[i] = NULL;
    }
    if(back_nbr_staple_sendbuf[i]){
      host_free(back_nbr_staple_sendbuf[i]); back_nbr_staple_sendbuf[i] = NULL;
    }

  }
  checkCudaError();
}

#endif
