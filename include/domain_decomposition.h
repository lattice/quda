#ifndef _DOMAIN_DECOMPOSITION_H
#define _DOMAIN_DECOMPOSITION_H

namespace quda {

  struct DecompParams
  {
    // dimensions of the smaller lattice
    unsigned int X1, X2, X3, X4;
    unsigned int X1h, X2h, X3h, X4h;
    unsigned int X2X1, X2X1h;
    unsigned int X3X2X1, X3X2X1h;

    // new 
    unsigned int X4X3X2, X4X3X2h;
    unsigned int X3X2h;
    
    unsigned int X4X3X1, X4X3X1h;
    unsigned int X3X1h;

    unsigned int X4X2X1, X4X2X1h;
    //

    // dimensions of the larger lattice
    unsigned int Y1, Y2, Y3, Y4;
    unsigned int Y1h, Y2h, Y3h, Y4h;
    unsigned int Y2Y1, Y2Y1h;
    unsigned int Y3Y2Y1, Y3Y2Y1h;

    // dimensions of the "border" regions
    unsigned int B1, B2, B3, B4;    
  };

  void initDecompParams(DecompParams* const params, const int X[], const int Y[]);

}
#endif
