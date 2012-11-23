#include <unistd.h>
#include <qmp.h>
#include <comm_quda.h>
#include <util_quda.h>

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
  QMP_msghandle_t handle = QMP_declare_send_relative(buffer, i, dir, 0);
  if( handle == NULL ) errorQuda("Unable to allocate send message handle");
  return (void*)handle;
}

void* comm_declare_receive_relative(void *buffer, int i, int dir, size_t bytes)
{
  QMP_msghandle_t handle = QMP_declare_receive_relative(buffer, i, dir, 0);
  if( handle == NULL ) errorQuda("Unable to allocate receive message handle");
  return (void*)handle;
}


void comm_free(void *handle) {
  QMP_free_msghandle((QMP_msghandle_t)handle);
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
  QMP_CHECK(QMP_start((QMP_msghandle_t)request));
}

int comm_query(void* request) 
{
  if (QMP_is_complete((QMP_msghandle_t)request) == QMP_TRUE) return 1;
  return 0;
}
