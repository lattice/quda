#include <unistd.h>
#include <qmp.h>
#include <comm_quda.h>
#include <util_quda.h>

#define QMP_CHECK(a)							\
  {QMP_status_t status;							\
  if ((status = a) != QMP_SUCCESS)					\
    errorQuda("QMP returned with error %s", QMP_error_string(status) );	\
  }

static char hostname[128] = "undetermined";

void comm_init(void)
{
  gethostname(hostname, 128);
  hostname[127] = '\0';
}

void comm_exit(int ret)
{
  if (ret) QMP_abort(ret);
}

char *comm_hostname(void)
{
  return hostname;
}

int comm_rank(void)
{
  return QMP_get_node_number();
}

int comm_size(void)
{
  return QMP_get_number_of_nodes();
}

void comm_broadcast(void *data, size_t nbytes)
{
  QMP_CHECK(QMP_broadcast(data, nbytes));
}
