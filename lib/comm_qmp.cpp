#include <unistd.h>
#include <qmp.h>
#include <comm_quda.h>

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

void comm_broadcast(void *data, size_t nbytes)
{
  QMP_broadcast(data, nbytes);
}
