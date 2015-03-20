#ifndef _MONTE_H
#define _MONTE_H



#include <random.h>
#include <quda.h>



namespace quda {
  
  double2 Plaquette( cudaGaugeField& data) ;
  void Monte( cudaGaugeField& data, cuRNGState *rngstate, double Beta, unsigned int nhb, unsigned int nover);
  void InitGaugeField( cudaGaugeField& data);
  void InitGaugeField( cudaGaugeField& data, cuRNGState *rngstate);

  /*Exchange "borders" between nodes
  int R[4] = {0,0,0,0};
  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
  Although the radius border is 2, it only updates the interior radius border, i.e., at 1 and X[d-2]
  where X[d] already includes the Radius border, and don't update at 0 and X[d-1] faces  */
  void PGaugeExchange( cudaGaugeField& data, const int dir, const int parity);
  void PGaugeExchangeFree();

}

#endif 
