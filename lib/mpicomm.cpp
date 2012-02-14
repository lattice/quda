#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include "mpicomm.h"

char hostname[128];
static int fwd_nbr=-1;
static int back_nbr=-1;
static int rank = 0;
static int size = -1;
extern int verbose;
static int num_nodes;
extern int getGpuCount();
static int which_gpu =-1;

static int x_fwd_nbr=-1;
static int y_fwd_nbr=-1;
static int z_fwd_nbr=-1;
static int t_fwd_nbr=-1;
static int x_back_nbr=-1;
static int y_back_nbr=-1;
static int z_back_nbr=-1;
static int t_back_nbr=-1;

static int xgridsize=1;
static int ygridsize=1;
static int zgridsize=1;
static int tgridsize=1;
static int xgridid = -1;
static int ygridid = -1;
static int zgridid = -1;
static int tgridid = -1;

static int manual_set_partition[4] ={0, 0, 0, 0};

void
comm_set_gridsize(int x, int y, int z, int t)
{
  xgridsize = x;
  ygridsize = y;
  zgridsize = z;
  tgridsize = t;

  return;
}

/* This function is for and testing debugging purpose only
 * The partitioning schedume should be generated automically 
 * in production runs. Don't use this function if you don't know
 * what you are doing
 */
void
comm_dim_partitioned_set(int dir)
{
  manual_set_partition[dir] = 1;
  return;
}


int 
comm_dim_partitioned(int dir)
{
  int ret = 0;
  
  switch(dir){
  case 0: 
    ret = (xgridsize > 1);    
    break;
  case 1: 
    ret = (ygridsize > 1);
    break;
  case 2: 
    ret = (zgridsize > 1);
    break;
  case 3: 
    ret = (tgridsize > 1);
    break;    
  default:
    printf("ERROR: invalid direction\n");
    comm_exit(1);
  }
  
  if( manual_set_partition[dir]){
    ret = manual_set_partition[dir];
  }
  
  return ret;
}

static void
comm_partition(void)
{
  /*
  printf("xgridsize=%d\n", xgridsize);
  printf("ygridsize=%d\n", ygridsize);
  printf("zgridsize=%d\n", zgridsize);
  printf("tgridsize=%d\n", tgridsize);
  */
  if(xgridsize*ygridsize*zgridsize*tgridsize != size){
    if (rank ==0){
      printf("ERROR: Invalid configuration (t,z,y,x gridsize=%d %d %d %d) "
             "but # of MPI processes is %d\n", tgridsize, zgridsize, ygridsize, xgridsize, size);
    }
    comm_exit(1);
  }

  int leftover;

#if 0
  tgridid  = rank/(zgridsize*ygridsize*xgridsize);
  leftover = rank%(zgridsize*ygridsize*xgridsize);
  zgridid  = leftover/(ygridsize*xgridsize);
  leftover = leftover%(ygridsize*xgridsize);
  ygridid  = leftover/xgridsize;
  xgridid  = leftover%xgridsize;
  #define GRID_ID(xid,yid,zid,tid) (tid*zgridsize*ygridsize*xgridsize+zid*ygridsize*xgridsize+yid*xgridsize+xid)
#else

  xgridid  = rank/(ygridsize*zgridsize*tgridsize);
  leftover = rank%(ygridsize*zgridsize*tgridsize);
  ygridid  = leftover/(zgridsize*tgridsize);
  leftover = leftover%(zgridsize*tgridsize);
  zgridid  = leftover/tgridsize;
  tgridid  = leftover%tgridsize;  
#define GRID_ID(xid,yid,zid,tid) (xid*ygridsize*zgridsize*tgridsize+yid*zgridsize*tgridsize+zid*tgridsize+tid)
#endif
  printf("My rank: %d, gridid(t,z,y,x): %d %d %d %d\n", rank, tgridid, zgridid, ygridid, xgridid);


  int xid, yid, zid, tid;
  //X direction neighbors
  yid =ygridid;
  zid =zgridid;
  tid =tgridid;
  xid=(xgridid +1)%xgridsize;
  x_fwd_nbr = GRID_ID(xid,yid,zid,tid);
  xid=(xgridid -1+xgridsize)%xgridsize;
  x_back_nbr = GRID_ID(xid,yid,zid,tid);

  //Y direction neighbors
  xid =xgridid;
  zid =zgridid;
  tid =tgridid;
  yid =(ygridid+1)%ygridsize;
  y_fwd_nbr = GRID_ID(xid,yid,zid,tid);
  yid=(ygridid -1+ygridsize)%ygridsize;
  y_back_nbr = GRID_ID(xid,yid,zid,tid);

  //Z direction neighbors
  xid =xgridid;
  yid =ygridid;
  tid =tgridid;
  zid =(zgridid+1)%zgridsize;
  z_fwd_nbr = GRID_ID(xid,yid,zid,tid);
  zid=(zgridid -1+zgridsize)%zgridsize;
  z_back_nbr = GRID_ID(xid,yid,zid,tid);

  //T direction neighbors
  xid =xgridid;
  yid =ygridid;
  zid =zgridid;
  tid =(tgridid+1)%tgridsize;
  t_fwd_nbr = GRID_ID(xid,yid,zid,tid);
  tid=(tgridid -1+tgridsize)%tgridsize;
  t_back_nbr = GRID_ID(xid,yid,zid,tid);

  printf("MPI rank: rank=%d, hostname=%s, x_fwd_nbr=%d, x_back_nbr=%d\n", rank, hostname, x_fwd_nbr, x_back_nbr);
  printf("MPI rank: rank=%d, hostname=%s, y_fwd_nbr=%d, y_back_nbr=%d\n", rank, hostname, y_fwd_nbr, y_back_nbr);
  printf("MPI rank: rank=%d, hostname=%s, z_fwd_nbr=%d, z_back_nbr=%d\n", rank, hostname, z_fwd_nbr, z_back_nbr);
  printf("MPI rank: rank=%d, hostname=%s, t_fwd_nbr=%d, t_back_nbr=%d\n", rank, hostname, t_fwd_nbr, t_back_nbr);

  
}

int 
comm_get_neighbor_rank(int dx, int dy, int dz, int dt)
{
  int ret;
#if 0
#define GRID_ID(xid,yid,zid,tid) (tid*zgridsize*ygridsize*xgridsize+zid*ygridsize*xgridsize+yid*xgridsize+xid)
#else
#define GRID_ID(xid,yid,zid,tid) (xid*ygridsize*zgridsize*tgridsize+yid*zgridsize*tgridsize+zid*tgridsize+tid)
#endif

  
  int xid, yid, zid, tid;
  xid=(xgridid + dx + xgridsize)%xgridsize;
  yid=(ygridid + dy + ygridsize)%ygridsize;
  zid=(zgridid + dz + zgridsize)%zgridsize;
  tid=(tgridid + dt + tgridsize)%tgridsize;

  ret = GRID_ID(xid,yid,zid,tid);

  return ret;
}


void 
comm_init()
{
  int i;
  
  static int firsttime=1;
  if (!firsttime){ 
    return;
  }
  firsttime = 0;

  gethostname(hostname, 128);  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int gpus_per_node = getGpuCount();  

  comm_partition();

  back_nbr = (rank -1 + size)%size;
  fwd_nbr = (rank +1)%size;
  num_nodes=size / getGpuCount();
  if(num_nodes ==0) {
	num_nodes=1;
  }

  //determine which gpu this MPI process is going to use
  char* hostname_recv_buf = (char*)malloc(128*size);
  if(hostname_recv_buf == NULL){
    printf("ERROR: malloc failed for host_recv_buf\n");
    comm_exit(1);
  }
  
  gethostname(hostname, 128);
  int rc = MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Allgather failed for hostname\n");
    comm_exit(1);
  }

  which_gpu=0;
  for(i=0;i < size; i++){
    if (i == rank){
      break;
    }
    if (strncmp(hostname, hostname_recv_buf + 128*i, 128) == 0){
      which_gpu ++;
    }
  }
  
  if (which_gpu >= gpus_per_node){
    printf("ERROR: invalid gpu(%d) to use in rank=%d mpi process\n", which_gpu, rank);
    comm_exit(1);
  }
  
  srand(rank*999);
  
  free(hostname_recv_buf);
  return;
}

int comm_gpuid()
{
  //int gpu = rank%getGpuCount();

  return which_gpu;
}
int
comm_rank(void)
{
  return rank;
}

int
comm_size(void)
{
  return size;
}

int
comm_dim(int dir) {

  int i;
  switch(dir) {
  case 0:
    i = xgridsize;
    break;
  case 1:
    i = ygridsize;
    break;
  case 2:
    i = zgridsize;
    break;
  case 3:
    i = tgridsize;
    break;
  default:
    printf("Cannot get direction %d", dir);
    comm_exit(1);
  }

  return i;
}

int
comm_coords(int dir) {

  int i;
  switch(dir) {
  case 0:
    i = xgridid;
    break;
  case 1:
    i = ygridid;
    break;
  case 2:
    i = zgridid;
    break;
  case 3:
    i = tgridid;
    break;
  default:
    printf("Cannot get direction %d", dir);
    comm_exit(1);
  }

  return i;
}

unsigned long
comm_send(void* buf, int len, int dst, void* _request)
{
  
  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }

  int dstproc;
  int sendtag=99;
  if (dst == BACK_NBR){
    dstproc = back_nbr;
    sendtag = BACK_NBR;
  }else if (dst == FWD_NBR){
    dstproc = fwd_nbr;
    sendtag = FWD_NBR;
  }else{
    printf("ERROR: invalid dest\n");
    comm_exit(1);
  }

  MPI_Isend(buf, len, MPI_BYTE, dstproc, sendtag, MPI_COMM_WORLD, request);  
  return (unsigned long)request;  
}

unsigned long
comm_send_to_rank(void* buf, int len, int dst_rank, void* _request)
{
  
  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  if(dst_rank < 0 || dst_rank >= comm_size()){
    printf("ERROR: Invalid dst rank(%d)\n", dst_rank);
    comm_exit(1);
  }
  int sendtag = 99;
  MPI_Isend(buf, len, MPI_BYTE, dst_rank, sendtag, MPI_COMM_WORLD, request);  
  return (unsigned long)request;  
}

unsigned long
comm_send_with_tag(void* buf, int len, int dst, int tag, void*_request)
{

  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }

  int dstproc = -1;
  switch(dst){
  case X_BACK_NBR:
    dstproc = x_back_nbr;
    break;
  case X_FWD_NBR:
    dstproc = x_fwd_nbr;
    break;
  case Y_BACK_NBR:
    dstproc = y_back_nbr;
    break;
  case Y_FWD_NBR:
    dstproc = y_fwd_nbr;
    break;
  case Z_BACK_NBR:
    dstproc = z_back_nbr;
    break;
  case Z_FWD_NBR:
    dstproc = z_fwd_nbr;
    break;
  case T_BACK_NBR:
    dstproc = t_back_nbr;
    break;
  case T_FWD_NBR:
    dstproc = t_fwd_nbr;
    break;
  default:
    printf("ERROR: invalid dest, line %d, file %s\n", __LINE__, __FILE__);
    comm_exit(1);
  }

  MPI_Isend(buf, len, MPI_BYTE, dstproc, tag, MPI_COMM_WORLD, request);
  return (unsigned long)request;
}



unsigned long
comm_recv(void* buf, int len, int src, void*_request)
{
  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  int srcproc=-1;
  int recvtag=99; //recvtag is opposite to the sendtag
  if (src == BACK_NBR){
    srcproc = back_nbr;
    recvtag = FWD_NBR;
  }else if (src == FWD_NBR){
    srcproc = fwd_nbr;
    recvtag = BACK_NBR;
  }else{
    printf("ERROR: invalid source\n");
    comm_exit(1);
  }
  
  MPI_Irecv(buf, len, MPI_BYTE, srcproc, recvtag, MPI_COMM_WORLD, request);
  
  return (unsigned long)request;
}

unsigned long
comm_recv_from_rank(void* buf, int len, int src_rank, void* _request)
{
  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  if(src_rank < 0 || src_rank >= comm_size()){
    printf("ERROR: Invalid src rank(%d)\n", src_rank);
    comm_exit(1);
  }
  
  int recvtag = 99;
  MPI_Irecv(buf, len, MPI_BYTE, src_rank, recvtag, MPI_COMM_WORLD, request);
  
  return (unsigned long)request;
}

unsigned long
comm_recv_with_tag(void* buf, int len, int src, int tag, void* _request)
{ 
  MPI_Request* request = (MPI_Request*)_request;
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  int srcproc=-1;
  switch (src){
  case X_BACK_NBR:
    srcproc = x_back_nbr;
    break;
  case X_FWD_NBR:
    srcproc = x_fwd_nbr;
    break;
  case Y_BACK_NBR:
    srcproc = y_back_nbr;
    break;
  case Y_FWD_NBR:
    srcproc = y_fwd_nbr;
    break;
  case Z_BACK_NBR:
    srcproc = z_back_nbr;
    break;
  case Z_FWD_NBR:
    srcproc = z_fwd_nbr;
    break;
  case T_BACK_NBR:
    srcproc = t_back_nbr;
    break;
  case T_FWD_NBR:
    srcproc = t_fwd_nbr;
    break;
  default:
    printf("ERROR: invalid source, line %d, file %s\n", __LINE__, __FILE__);
    comm_exit(1);
  }
  MPI_Irecv(buf, len, MPI_BYTE, srcproc, tag, MPI_COMM_WORLD, request);
  
  return (unsigned long)request;
}

int comm_query(void* request) 
{
  MPI_Status status;
  int query;
  int rc = MPI_Test( (MPI_Request*)request, &query, &status);
  if (rc != MPI_SUCCESS) {
    printf("ERROR: MPI_Test failed\n");
    comm_exit(1);
  }

  return query;
}

void comm_free(void* request) {
  free((void*)request);
  return;
}

//this request should be some return value from comm_recv
void 
comm_wait(void* request)
{
  
  MPI_Status status;
  int rc = MPI_Wait( (MPI_Request*)request, &status);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Wait failed\n");
    comm_exit(1);
  }
  
  return;
}

//we always reduce one double value
void
comm_allreduce(double* data)
{
  double recvbuf;
  int rc = MPI_Allreduce ( data, &recvbuf,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Allreduce failed\n");
    comm_exit(1);
  }
  
  *data = recvbuf;
  
  return;
} 

//reduce n double value
void
comm_allreduce_array(double* data, size_t size)
{
  double recvbuf[size];
  int rc = MPI_Allreduce ( data, &recvbuf,size,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Allreduce failed\n");
    comm_exit(1);
  }
  
  memcpy(data, recvbuf, sizeof(recvbuf));
  
  return;
}

//we always reduce one double value
void
comm_allreduce_max(double* data)
{
  double recvbuf;
  int rc = MPI_Allreduce ( data, &recvbuf,1,MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Allreduce failed\n");
    comm_exit(1);
  }
  
  *data = recvbuf;
  
  return;
} 

void
comm_barrier(void)
{
  MPI_Barrier(MPI_COMM_WORLD);  
}
void 
comm_cleanup()
{
  MPI_Finalize();
}

void
comm_exit(int ret)
{
  MPI_Finalize();
  exit(ret);
}

