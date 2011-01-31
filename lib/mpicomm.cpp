#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include "mpicomm.h"

static int fwd_nbr=-1;
static int back_nbr=-1;
static int rank = -1;
static int size = -1;
extern int verbose;
static int num_nodes;
extern int getGpuCount();
static int which_gpu =-1;

void 
comm_init()
{
  int i;
  
  static int firsttime=1;
  if (!firsttime){ 
    return;
  }
  firsttime = 0;
  
  int gpus_per_node = getGpuCount();
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  back_nbr = (rank -1 + size)%size;
  fwd_nbr = (rank +1)%size;
  num_nodes=size / getGpuCount();
  if(num_nodes ==0) {
	num_nodes=1;
  }

  //determine which gpu this MPI process is going to use
  char hostname[128];
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
  
  printf("rank=%d, back_neighbor=%d, fwd_nbr=%d, host=%s, use gpu=%d\n", 
	 rank, back_nbr, fwd_nbr, hostname, which_gpu);
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

unsigned long
comm_send(void* buf, int len, int dst)
{
  
  MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request));
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }

  int dstproc;
  int sendtag;
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
comm_recv(void* buf, int len, int src)
{
  MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request));
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  int srcproc;
  int recvtag; //recvtag is opposite to the sendtag
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


//this request should be some return value from comm_recv
void 
comm_wait(unsigned long request)
{
  
  MPI_Status status;
  int rc = MPI_Wait( (MPI_Request*)request, &status);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Wait failed\n");
    comm_exit(1);
  }
  
  free((void*)request);
  
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

