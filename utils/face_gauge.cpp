#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <quda_internal.h>
#include <comm_quda.h>

#include <host_utils.h>

using namespace quda;

extern qudaStream_t *stream;

/**************************************************************
 * Staple exchange routine
 * used in fat link computation
 ***************************************************************/

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


//FIXME remove this legacy macro
#define gauge_site_size 18 // real numbers per gauge field

static void* fwd_nbr_staple[4];
static void* back_nbr_staple[4];
static void* fwd_nbr_staple_sendbuf[4];
static void* back_nbr_staple_sendbuf[4];

static int dims[4];
static int X1,X2,X3,X4;
static int volumeCB;
static int Vs[4], Vsh[4];

#include "gauge_field.h"
// extern void setup_dims_in_gauge(int *XX);

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

template <typename Float>
void packGhostAllStaples(Float *cpuStaple, Float **cpuGhostBack,Float**cpuGhostFwd, int nFace, int* X) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  int volumeCB = X[0]*X[1]*X[2]*X[3]/2;
  int faceVolumeCB[4]={
    X[1]*X[2]*X[3]/2,
    X[0]*X[2]*X[3]/2,
    X[0]*X[1]*X[3]/2,
    X[0]*X[1]*X[2]/2
  };

  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];

  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];

  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];

  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };

  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }

    //collect back ghost staple
    for(int dir =0; dir < 4; dir++){
      int d;
      int a,b,c;
      //ther is only one staple in the same location
      for(int linkdir=0; linkdir < 1; linkdir ++){
	Float* even_src = cpuStaple;
        Float *odd_src = cpuStaple + volumeCB * gauge_site_size;

        Float *even_dst;
        Float *odd_dst;

        // switching odd and even ghost cpuLink when that dimension size is odd
        // only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
        if ((X[dir] % 2 == 0) || (comm_dim(dir) == 1)) {
          even_dst = dst[dir];
          odd_dst = even_dst + nFace * faceVolumeCB[dir] * gauge_site_size;
        } else {
          odd_dst = dst[dir];
          even_dst = dst[dir] + nFace * faceVolumeCB[dir] * gauge_site_size;
        }

        int even_dst_index = 0;
        int odd_dst_index = 0;
        int startd;
        int endd;
        if (ite == 0) { // back
          startd = 0;
          endd = nFace;
        } else { // fwd
          startd = X[dir] - nFace;
          endd = X[dir];
        }
        for (d = startd; d < endd; d++) {
          for (a = 0; a < A[dir]; a++) {
            for (b = 0; b < B[dir]; b++) {
              for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  for(int i=0;i < 18;i++){
		    even_dst[18*even_dst_index+i] = even_src[18*index + i];
		  }
		  even_dst_index++;
		}else{ //odd
		  for(int i=0;i < 18;i++){
		    odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		  }
		  odd_dst_index++;
		}
	      }//c
            }  // b
          }    // a
        }      // d
        assert(even_dst_index == nFace * faceVolumeCB[dir]);
        assert(odd_dst_index == nFace * faceVolumeCB[dir]);
      }//linkdir
    }//dir
  }//ite
}


void pack_ghost_all_staples_cpu(void *staple, void **cpuGhostStapleBack, void** cpuGhostStapleFwd,
				int nFace, QudaPrecision precision, int* X) {
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllStaples((double*)staple, (double**)cpuGhostStapleBack, (double**) cpuGhostStapleFwd, nFace, X);
  } else {
    packGhostAllStaples((float*)staple, (float**)cpuGhostStapleBack, (float**)cpuGhostStapleFwd, nFace, X);
  }
}

void pack_gauge_diag(void* buf, int* X, void** sitelink, int nu, int mu, int dir1, int dir2, QudaPrecision prec)
{
  /*
    nu |          |
    |__________|
    mu
    *
    * nu, mu are the directions we are working on
    * Since we are packing our own data, we need to go to the north-west corner in the diagram
    * i.e. x[nu] = X[nu]-1, x[mu]=0, and looop throught x[dir1],x[dir2]
    * in the remaining two directions (dir1/dir2), dir2 is the slowest changing dim when computing
    * index
    */

  int mul_factor[4]={1, X[0], X[1]*X[0], X[2]*X[1]*X[0]};

  int even_dst_idx = 0;
  int odd_dst_idx = 0;
  char* dst_even =(char*)buf;
  char *dst_odd = dst_even + (X[dir1] * X[dir2] / 2) * gauge_site_size * prec;
  char* src_even = (char*)sitelink[nu];
  char *src_odd = src_even + (X[0] * X[1] * X[2] * X[3] / 2) * gauge_site_size * prec;

  if( (X[nu]+X[mu]) % 2 == 1){
    //oddness will change between me and the diagonal neighbor
    //switch it now
    char* tmp = dst_odd;
    dst_odd = dst_even;
    dst_even = tmp;
  }

  for(int i=0;i < X[dir2]; i++){
    for(int j=0; j < X[dir1]; j++){
      int src_idx = ((X[nu]-1)*mul_factor[nu]+ 0*mul_factor[mu]+i*mul_factor[dir2]+j*mul_factor[dir1])>>1;
      int oddness = ( (X[nu]-1) + 0 + i + j) %2;
      if(oddness==0){
        for (int tmpidx = 0; tmpidx < gauge_site_size; tmpidx++) {
          memcpy(&dst_even[(18 * even_dst_idx + tmpidx) * prec], &src_even[(18 * src_idx + tmpidx) * prec], prec);
        }
        even_dst_idx++;
      }else{
        for (int tmpidx = 0; tmpidx < gauge_site_size; tmpidx++) {
          memcpy(&dst_odd[(18 * odd_dst_idx + tmpidx) * prec], &src_odd[(18 * src_idx + tmpidx) * prec], prec);
        }
        odd_dst_idx++;
      }//if
    }//for j
  }//for i
  if( (even_dst_idx != X[dir1]*X[dir2]/2)|| (odd_dst_idx != X[dir1]*X[dir2]/2)){
    errorQuda("even_dst_idx/odd_dst_idx(%d/%d) does not match the value of X[dir1]*X[dir2]/2 (%d)\n",
	      even_dst_idx, odd_dst_idx, X[dir1]*X[dir2]/2);
  }
  return ;
}

/*
  This is the packing kernel for the multi-dimensional ghost zone in
  the padded region.  This is called by cpuexchangesitelink in
  FaceBuffer (MPI only), which was called by loadLinkToGPU (defined at
  the bottom).

  Not currently included since it will be replaced by Guochun's new
  routine which uses an enlarged domain instead of a ghost zone.
*/
template <typename Float>
void packGhostAllLinks(Float **cpuLink, Float **cpuGhostBack,Float**cpuGhostFwd, int dir, int nFace, int* X) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  int volumeCB = X[0]*X[1]*X[2]*X[3]/2;
  int faceVolumeCB[4]={
    X[1]*X[2]*X[3]/2,
    X[0]*X[2]*X[3]/2,
    X[0]*X[1]*X[3]/2,
    X[0]*X[1]*X[2]/2
  };

  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];

  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];

  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];

  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };

  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }
    //collect back ghost gauge field
    //for(int dir =0; dir < 4; dir++){
    int d;
    int a,b,c;

    //we need copy all 4 links in the same location
    for(int linkdir=0; linkdir < 4; linkdir ++){
      Float* even_src = cpuLink[linkdir];
      Float *odd_src = cpuLink[linkdir] + volumeCB * gauge_site_size;
      Float* even_dst;
      Float* odd_dst;

      //switching odd and even ghost cpuLink when that dimension size is odd
      //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (comm_dim(dir) == 1)){
        even_dst = dst[dir] + 2 * linkdir * nFace * faceVolumeCB[dir] * gauge_site_size;
        odd_dst = even_dst + nFace * faceVolumeCB[dir] * gauge_site_size;
      }else{
        odd_dst = dst[dir] + 2 * linkdir * nFace * faceVolumeCB[dir] * gauge_site_size;
        even_dst = odd_dst + nFace * faceVolumeCB[dir] * gauge_site_size;
      }

      int even_dst_index = 0;
      int odd_dst_index = 0;
      int startd;
      int endd;
      if(ite == 0){ //back
	startd = 0;
	endd= nFace;
      }else{//fwd
	startd = X[dir] - nFace;
	endd =X[dir];
      }
      for(d = startd; d < endd; d++){
	for(a = 0; a < A[dir]; a++){
	  for(b = 0; b < B[dir]; b++){
	    for(c = 0; c < C[dir]; c++){
	      int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
	      int oddness = (a+b+c+d)%2;
	      if (oddness == 0){ //even
		for(int i=0;i < 18;i++){
		  even_dst[18*even_dst_index+i] = even_src[18*index + i];
		}
		even_dst_index++;
	      }else{ //odd
		for(int i=0;i < 18;i++){
		  odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		}
		odd_dst_index++;
	      }
	    }//c
	  }//b
	}//a
      }//d
      assert( even_dst_index == nFace*faceVolumeCB[dir]);
      assert( odd_dst_index == nFace*faceVolumeCB[dir]);
    }//linkdir
  }//ite
}


void pack_ghost_all_links(void **cpuLink, void **cpuGhostBack, void** cpuGhostFwd,
			  int dir, int nFace, QudaPrecision precision, int *X) {
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllLinks((double**)cpuLink, (double**)cpuGhostBack, (double**) cpuGhostFwd, dir,  nFace, X);
  } else {
    packGhostAllLinks((float**)cpuLink, (float**)cpuGhostBack, (float**)cpuGhostFwd, dir, nFace, X);
  }
}

void exchange_llfat_init(QudaPrecision prec)
{
  static bool initialized = false;

  if (initialized) return;
  initialized = true;
  
  for (int i=0; i < 4; i++) {

    size_t packet_size = Vs[i] * gauge_site_size * prec;

    fwd_nbr_staple[i] = pinned_malloc(packet_size);
    back_nbr_staple[i] = pinned_malloc(packet_size);
    fwd_nbr_staple_sendbuf[i] = pinned_malloc(packet_size);
    back_nbr_staple_sendbuf[i] = pinned_malloc(packet_size);

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
      int len = X[dir1] * X[dir2] * gauge_site_size * sizeof(Float);
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

  int nFace =1;
  for(int dir=0; dir < 4; dir++){
    if(optflag && !commDimPartitioned(dir)) continue;
    pack_ghost_all_links((void**)sitelink, (void**)sitelink_back_sendbuf, (void**)sitelink_fwd_sendbuf, dir, nFace, (QudaPrecision)(sizeof(Float)), X);
  }

  for (int dir = 0; dir < 4; dir++) {
    if(optflag && !commDimPartitioned(dir)) continue;
    int len = Vsh[dir] * gauge_site_size * sizeof(Float);
    Float* ghost_sitelink_back = ghost_sitelink[dir];
    Float *ghost_sitelink_fwd = ghost_sitelink[dir] + 8 * Vsh[dir] * gauge_site_size;

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

  for (int i=0; i<4; i++) {
    int nbytes = 4 * Vs[i] * gauge_site_size * gPrecision;
    sitelink_fwd_sendbuf[i] = safe_malloc(nbytes);
    sitelink_back_sendbuf[i] = safe_malloc(nbytes);
    memset(sitelink_fwd_sendbuf[i], 0, nbytes);
    memset(sitelink_back_sendbuf[i], 0, nbytes);
  }
  
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_sitelink(X, (double**)sitelink, (double**)(ghost_sitelink), (double**)ghost_sitelink_diag, 
		      (double**)sitelink_fwd_sendbuf, (double**)sitelink_back_sendbuf, optflag);
  }else{ //single
    exchange_sitelink(X, (float**)sitelink, (float**)(ghost_sitelink), (float**)ghost_sitelink_diag, 
		      (float**)sitelink_fwd_sendbuf, (float**)sitelink_back_sendbuf, optflag);
  }
  
  for(int i=0;i < 4;i++){
    host_free(sitelink_fwd_sendbuf[i]);
    host_free(sitelink_back_sendbuf[i]);
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

// gauge_site_size

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
    len[i] = slice_3d[i] * R[i] * geometry * gauge_site_size * gPrecision; // 2 slices, 4 directions' links
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

  int gaugebytes = gauge_site_size * gPrecision;
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

  int nFace =1;
  pack_ghost_all_staples_cpu(staple, (void**)staple_back_sendbuf, 
			     (void**)staple_fwd_sendbuf,  nFace, (QudaPrecision)(sizeof(Float)), X);

  
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};
  size_t len[4] = {Vsh_x * gauge_site_size * sizeof(Float), Vsh_y * gauge_site_size * sizeof(Float),
                   Vsh_z * gauge_site_size * sizeof(Float), Vsh_t * gauge_site_size * sizeof(Float)};

  for (int dir=0;dir < 4; dir++) {

    Float *ghost_staple_back = ghost_staple[dir];
    Float *ghost_staple_fwd = ghost_staple[dir] + 2 * Vsh[dir] * gauge_site_size;

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
    staple_fwd_sendbuf[i] = safe_malloc(Vs[i] * gauge_site_size * gPrecision);
    staple_back_sendbuf[i] = safe_malloc(Vs[i] * gauge_site_size * gPrecision);
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

void exchange_llfat_cleanup(void)
{
  for (int i=0; i<4; i++) {

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

#undef gauge_site_size
