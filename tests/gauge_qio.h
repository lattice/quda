#ifndef _GAUGE_QIO_H
#define _GAUGE_QIO_H

#ifdef HAVE_QIO
void read_gauge_field(char *filename, void *gauge[], QudaPrecision prec, int *X, int argc, char *argv[]);
#else
void read_gauge_field(char *filename, void *gauge[], QudaPrecision prec, int *X, int argc, char *argv[]) {
  printf("QIO support has not been enabled\n");
  exit(-1);
}
#endif

#endif // _GAUGE_QIO_H
