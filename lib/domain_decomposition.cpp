#include <domain_decomposition.h>
#include <cassert>

namespace quda {

 void initDecompParams(DecompParams* const params, const int X[], const int Y[])
 {
   for(int dir=0; dir<4; ++dir){
    const int border = Y[dir] - X[dir];
    assert( border >= 0 && !(border & 1) ); // !(border & 1) 
   }

   params->X1 = X[0];
   params->X2 = X[1];
   params->X3 = X[2];
   params->X4 = X[3];
   
   params->X1h = X[0]/2;
   params->X2h = X[1]/2;
   params->X3h = X[2]/2;
   params->X4h = X[3]/2;
 
   params->X2X1 = params->X2*params->X1; 
   params->X2X1h = params->X2*params->X1h;
   params->X3X2X1  = params->X3*params->X2*params->X1;
   params->X3X2X1h = params->X3*params->X2*params->X1h;



   params->Y1 = Y[0];
   params->Y2 = Y[1];
   params->Y3 = Y[2];
   params->Y4 = Y[3];

   params->Y1h = Y[0]/2;
   params->Y2h = Y[1]/2;
   params->Y3h = Y[2]/2;
   params->Y4h = Y[3]/2;


   params->Y3Y2Y1 = params->Y3*params->Y2*params->Y1;
   params->Y3Y2Y1h = params->Y3*params->Y2*params->Y1h; 



   params->B1 = (Y[0] - X[0])/2;
   params->B2 = (Y[1] - X[1])/2;
   params->B3 = (Y[2] - X[2])/2;
   params->B4 = (Y[3] - X[3])/2;







   return;
 }
  

} // namespace quda


