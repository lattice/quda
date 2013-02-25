#include <unistd.h>
#include <qmp.h>
#include <comm_quda.h>
#include <quda_internal.h>
#include <util_quda.h>


#define QMP_CHECK(qmp_call) do {                     \
  QMP_status_t status = qmp_call;                    \
  if (status != QMP_SUCCESS)                         \
    errorQuda("(QMP) %s", QMP_error_string(status)); \
} while (0)


extern int getGpuCount();


struct MsgHandle_s {
  QMP_msgmem_t mem;
  QMP_msghandle_t handle;
};


static char hostname[128] = "undetermined";


void comm_create(int argc, char **argv)
{
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
}


void comm_init(void)
{
  static bool initialized = false;

  if (!initialized) {

    if ( QMP_is_initialized() != QMP_TRUE ) {
      errorQuda("QMP is not initialized");
    }

    gethostname(hostname, 128);
    hostname[127] = '\0';

    initialized = true;
  }
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
  return (comm_rank() % getGpuCount());
}


int comm_rank(void)
{
  return QMP_get_node_number();
}


int comm_size(void)
{
  return QMP_get_number_of_nodes();
}


MsgHandle *comm_declare_send_relative(void *buffer, int i, int dir, size_t bytes)
{
  MsgHandle *h = (MsgHandle *) safe_malloc(sizeof(MsgHandle));

  h->mem = QMP_declare_msgmem(buffer, bytes);
  if (h->mem == NULL) errorQuda("Unable to allocate send message mem");

  h->handle = QMP_declare_send_relative(h->mem, i, dir, 0);
  if (h->handle == NULL) errorQuda("Unable to allocate send message handle");

  return h;
}


MsgHandle *comm_declare_receive_relative(void *buffer, int i, int dir, size_t bytes)
{
  MsgHandle *h = (MsgHandle *) safe_malloc(sizeof(MsgHandle));

  h->mem = QMP_declare_msgmem(buffer, bytes);
  if (h->mem == NULL) errorQuda("Unable to receive receive message mem");

  h->handle = QMP_declare_receive_relative(h->mem, i, dir, 0);
  if (h->handle == NULL) errorQuda("Unable to allocate receive message handle");

  return h;
}


void comm_free(MsgHandle *handle) {
  QMP_free_msghandle(handle->handle);
  QMP_free_msgmem(handle->mem);
  host_free(handle);
}


void comm_start(MsgHandle *handle)
{
  QMP_CHECK(QMP_start(handle->handle));
}


void comm_wait(MsgHandle *handle)
{
  QMP_CHECK(QMP_wait(handle->handle)); 
}


int comm_query(MsgHandle *handle) 
{
  return (QMP_is_complete(handle->handle) == QMP_TRUE);
}


void comm_barrier(void)
{
  QMP_CHECK(QMP_barrier());  
}


void comm_allreduce(double* data)
{
  QMP_CHECK(QMP_sum_double(data));
} 


void comm_allreduce_int(int* data)
{
  QMP_CHECK(QMP_sum_int(data));
}


void comm_allreduce_array(double* data, size_t len)
{
  QMP_CHECK(QMP_sum_double_array(data,len));
}


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


int comm_dim(int dir)
{
  return QMP_get_logical_dimensions()[dir];
}


int comm_coords(int dir)
{
  return QMP_get_logical_coordinates()[dir];
}


static int manual_set_partition[4] = {0, 0, 0, 0};

int comm_dim_partitioned(int dir)
{
  return (manual_set_partition[dir] || (comm_dim(dir) > 1));
}

void comm_dim_partitioned_set(int dir)
{ 
  manual_set_partition[dir] = 1;
}

