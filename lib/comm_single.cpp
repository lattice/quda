/**
   Dummy communications layer for single GPU backend.
 */

#include <stdlib.h>
#include <comm_quda.h>

static char hostname[128] = "undetermined";

void comm_create(int argc, char **argv);

void comm_init() { ; }

void comm_cleanup() { ; }

void comm_exit() { ; }

char *comm_hostname(void) { return hostname; }

int comm_rank() { return 0; }

int comm_size() { return 1; }

void* comm_declare_send_relative(void *buffer, int i, int dir, size_t bytes) 
{ return (void*)0; }
void* comm_declare_receive_relative(void *buffer, int i, int dir, size_t bytes) 
{ return (void*)0; }

void comm_free(void *comm) { ; }

void comm_start(void *comm) { ; }

void comm_wait(void *comm) { ; }

int comm_query(void *comm) { return 1; }

int comm_dim_partitioned(int dir) { return 0;} 

void comm_dim_partitioned_set(int dir) { ; }

int comm_dim(int dir) { return 1; }

int comm_coords(int dir) { return 0; }

void comm_allreduce(double* data) { ; } 

void comm_allreduce_int(int* data) { ; }

void comm_allreduce_array(double* data, size_t size) { ; }

void comm_allreduce_max(double* data) { ; } 

void comm_barrier(void) { ; }

void comm_broadcast(void *data, size_t nbytes) { ; }

void comm_set_gridsize(const int *X, int nDim) { ; }
