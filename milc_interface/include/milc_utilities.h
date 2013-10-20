#ifndef _QUDA_MILC_UTILITIES_H
#define _QUDA_MILC_UTILITIES_H

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <enum_quda.h>
#include <quda.h>
#include <util_quda.h> // for printfQuda
// General layout
// stores the local lattice dimensions 
// and the MPI grid dimensions

namespace milc_interface {

// This class is a bit of a holdall for 
// persistent data in the interface. 
// At the moment, it just contains the verbosity
class PersistentData
{
  private:
    static QudaVerbosity verbosity;
  public:
    void setVerbosity(QudaVerbosity verb){ verbosity = verb; }
    QudaVerbosity  getVerbosity() const { return verbosity; }	
};


class Layout
{

  typedef int array4[4];

  public:
    void setLocalDim(const int X[4]);
    void setGridDim(const int X[4]);

    const array4 &getLocalDim() const
    {
      return local_dim;
    }

    const array4 &getGridDim() const
    {
      return grid_dim;
    }

/*
    const int (&getLocalDim() const)[4]
    {
      return local_dim;
    }
    const int (&getGridDim() const)[4]
    {
      return grid_dim;
    }

*/
  private:
    static int local_dim[4];
    static int grid_dim[4];
};


class GridInfo
{
  public:
    GridInfo(){};
    GridInfo(const int dim[4]){ setDim(dim); }
    void setDim(const int dim[4]);
	  // be careful here, not to get a dangling reference when GridInfo's destructor is 
    // called. I should use a smart pointer.
    const int (&getDim() const)[4]
	  {
	    return dim;
    }
    int getVolume() const;
    int getSliceVolume(int i) const;
    int getArea(int i, int j) const;
    int getMaxArea() const;

  private:
    int dim[4];
    int volume;
};


int getVolume(const int dim[4]);

int getRealSize(QudaPrecision precision);

class  MilcFieldLoader
{
  const QudaPrecision milc_precision;
  const QudaPrecision quda_precision;
  int volume;
  bool exchange_parity; 
    
  public:
    MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam);
    MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam, bool exchange);
    void loadGaugeField(const void* const milc_field, void* quda_field[4]) const;
};


// Allocate a one-dimensional color field
void allocateColorField(int volume, QudaPrecision prec, bool usePinnedMemory, void*& field);

void assignExtendedQDPGaugeField(const int dim[4], QudaPrecision precision, const void* const src, void** const dst);

void updateExtendedQDPBorders(const int dim[4], QudaPrecision precision, void** const qdp_field);

} // namespace milc_interface



#endif // _QUDA_MILC_UTILITIES_H
