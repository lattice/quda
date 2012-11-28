#include <unistd.h>
#include <qmp.h>
#include <comm_quda.h>
#include <quda_internal.h>
#include <util_quda.h>

struct CommQMP {
  QMP_msgmem_t mem;
  QMP_msghandle_t handle;
};

extern int getGpuCount();

#define QMP_CHECK(a)							\
  {QMP_status_t status;							\
  if ((status = a) != QMP_SUCCESS)					\
    errorQuda("QMP returned with error %s", QMP_error_string(status) );	\
  }

static char hostname[128] = "undetermined";

void comm_create(int argc, char **argv)
{
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
}

void comm_init(void)
{
  if ( QMP_is_initialized() != QMP_TRUE ) {
    errorQuda("QMP is not initialized");
  }

  static int firsttime = 1;
  if (!firsttime) return;
  firsttime = 0;

  gethostname(hostname, 128);
  hostname[127] = '\0';
}

void comm_cleanup()
{
  QMP_finalize_msg_passing();
}

void comm_exit(int ret)
{
  if (ret) QMP_abort(ret);
}

char *comm_hostname(void)
{
  return hostname;
}

int comm_gpuid()
{
  return comm_rank() % getGpuCount();
}

int comm_rank(void)
{
  return QMP_get_node_number();
}

int comm_size(void)
{
  return QMP_get_number_of_nodes();
}


void* comm_declare_send_relative(void *buffer, int i, int dir, size_t bytes)
{
  CommQMP* comm = (CommQMP*)safe_malloc(sizeof(CommQMP));
  comm->mem = QMP_declare_msgmem(buffer, bytes);
  if( comm->mem == NULL ) errorQuda("Unable to allocate send message mem");
  comm->handle = QMP_declare_send_relative(comm->mem, i, dir, 0);
  if( comm->handle == NULL ) errorQuda("Unable to allocate send message handle");
  return (void*)comm;
}

void* comm_declare_receive_relative(void *buffer, int i, int dir, size_t bytes)
{
  CommQMP* comm = (CommQMP*)safe_malloc(sizeof(CommQMP));
  comm->mem = QMP_declare_msgmem(buffer, bytes);
  if( comm->mem == NULL ) errorQuda("Unable to receive receive message mem");
  comm->handle = QMP_declare_receive_relative(comm->mem, i, dir, 0);
  if( comm->handle == NULL ) errorQuda("Unable to allocate receive message handle");
  return (void*)comm;
}


void comm_free(void *comm) {
  CommQMP *qmp = (CommQMP*)comm;
  QMP_free_msghandle(qmp->handle);
  QMP_free_msgmem(qmp->mem);
  host_free(qmp);
}

void comm_barrier(void)
{
  QMP_CHECK(QMP_barrier());  
}

//we always reduce one double value
void comm_allreduce(double* data)
{
  QMP_CHECK(QMP_sum_double(data));
} 

void comm_allreduce_int(int* data)
{
  QMP_CHECK(QMP_sum_int(data));
}

//reduce n double value
void comm_allreduce_array(double* data, size_t len)
{
  QMP_CHECK(QMP_sum_double_array(data,len));
}

//we always reduce one double value
void comm_allreduce_max(double* data)
{
  QMP_CHECK(QMP_max_double(data));
} 

void comm_broadcast(void *data, size_t nbytes)
{
  QMP_CHECK(QMP_broadcast(data, nbytes));
}

void comm_set_gridsize(const int *X, int nDim)
{
  if (nDim != 4) errorQuda("Comms dimensions %d != 4", nDim);

  QMP_CHECK(QMP_declare_logical_topology(X, nDim));
}

void comm_start(void *request)
{
  CommQMP *qmp = (CommQMP*)request;
  QMP_CHECK(QMP_start(qmp->handle));
}

int comm_query(void* request) 
{
  CommQMP *qmp = (CommQMP*)request;
  if (QMP_is_complete(qmp->handle) == QMP_TRUE) return 1;
  return 0;
}

void comm_wait(void *request)
{
  CommQMP *qmp = (CommQMP*)request;
  QMP_CHECK(QMP_wait(qmp->handle)); 
}

static int manual_set_partition[4] ={0, 0, 0, 0};
int comm_dim(int dir) { return QMP_get_logical_dimensions()[dir]; }
int comm_coords(int dir) { return QMP_get_logical_coordinates()[dir]; }
int comm_dim_partitioned(int dir){ return (manual_set_partition[dir] || ((comm_dim(dir) > 1)));}
void comm_dim_partitioned_set(int dir){ manual_set_partition[dir] = 1; }

