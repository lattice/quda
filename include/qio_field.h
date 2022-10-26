#pragma once

#ifdef HAVE_QIO
void read_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X,
		      int argc, char *argv[]);
void write_gauge_field(const char *filename, void *gauge[], QudaPrecision prec, const int *X, int argc, char *argv[]);
void read_spinor_field(const char *filename, void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                       QudaParity parity, int nColor, int nSpin, int Nvec, int argc, char *argv[]);
void write_spinor_field(const char *filename, const void *V[], QudaPrecision precision, const int *X, QudaSiteSubset subset,
                        QudaParity parity, int nColor, int nSpin, int Nvec, int argc, char *argv[]);
#else
inline void read_gauge_field(const char *, void *[], QudaPrecision, const int *, int, char *[])
{
  printf("QIO support has not been enabled\n");
  exit(-1);
}
inline void write_gauge_field(const char *, void *[], QudaPrecision, const int *, int, char *[])
{
  printf("QIO support has not been enabled\n");
  exit(-1);
}
inline void read_spinor_field(const char *, void *[], QudaPrecision, const int *, QudaSiteSubset, QudaParity, int, int,
                              int, int, char *[])
{
  printf("QIO support has not been enabled\n");
  exit(-1);
}
inline void write_spinor_field(const char *, const void *[], QudaPrecision, const int *, QudaSiteSubset, QudaParity, int, int,
                               int, int, char *[])
{
  printf("QIO support has not been enabled\n");
  exit(-1);
}

#endif
