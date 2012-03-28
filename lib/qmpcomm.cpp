#include <qmp.h>
#include <comm_quda.h>

int comm_rank()
{
  return QMP_get_node_number();
}

void comm_broadcast(void *data, size_t nbytes)
{
  QMP_broadcast(data, nbytes);
}
