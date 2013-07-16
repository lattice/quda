// !! to be completed
//===================================================//
//        stout smear	  			     //
//	  coded by lj			   	     //
//	  2012 12			     	     //
//	  for qcdsf confs 			     //
//	  PRD 84 054509 			     //
//===================================================//
#include "ihep_utility.h"

template <typename Float>
void sum_of_staples(ColorMatrix& V, int site, int mu, Float* gauge[4], int* NeighbourSitesTable){

  /*===============================================================
	  nu			
  (n+nu)  |---------1---------|
	  |                   |
	  0                   2	
	  |                   |(n+mu)
	  n ----------------------->mu
	  |		      |
	  3	   	      5
	  |		      |
	  -----------4----------- (site+mu-nu)
    (site-nu)          

  V_mu(n) = SUM_nu(link0*link1*link2^ + link3^*link4*link5)
  ==============================================================*/
  int munu[4][4];
  int count = 0;
  for(int i=0; i<4; i++)
  for(int j=0; j<4; j++)
    if(i==j){
      munu[i][j] = 0;}
    else{
      munu[i][j]=count;
      count++;}

  ColorMatrix link[6]; 
  V = V - V;

  for(int nu=0; nu<4; nu++){
    if(nu == mu)	continue; 
    link[0] = ColorMatrix( gauge[nu] + site*18);
    link[1] = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu[mu][nu]* 7 + 1]*18);
    link[2] = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu[mu][nu]* 7 + 0]*18);
    link[3] = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu[mu][nu]* 7 + 5]*18);
    link[4] = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu[mu][nu]* 7 + 5]*18);
    link[5] = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu[mu][nu]* 7 + 6]*18);
    V = V + link[0]*link[1]*link[2].dagger() + link[3].dagger()*link[4]*link[5];
  }
}

template <typename Float>
void Q_mu(ColorMatrix& Q, int site, int mu, Float** gauge, int* NeighbourSitesTable, double alpha){
  /*===============================================================
  Q_mu(n) = (alpha/2i)(V*U^ - U*V^ - (1/3)Tr(V*U^ - U*V^))
  ==============================================================*/
  ColorMatrix V; 
  sum_of_staples(V, site, mu, gauge, NeighbourSitesTable);

  ColorMatrix U; 
  U = ColorMatrix( gauge[mu] + site*18);
  Q = V*U.dagger() - U*V.dagger();
  Q = Q - (1.0/3.0)*Q.trace();
  Q = alpha /cdouble(0,2) * Q;
}

ColorMatrix ExpMatrix(ColorMatrix Q){
  // PRD 69 054501i, return exp(iQ)
  // Q hermitian, trace is real 
  double c0 = real((Q*Q*Q).trace())/3.0; 
  double c1 = real((Q*Q).trace())/2.0; 
  double c0max = 2.0*pow(c1/3.0,1.5); 

  double theta = acos(c0/c0max);
  double u = sqrt(c1/3.0)*cos(theta/3.0); 
  double w = sqrt(c1)*sin(theta/3.0); 

  double xi = sin(w)/w; 

  cdouble h0 = (u*u-w*w)*exp(cdouble(0,2*u)) + exp(cdouble(0,-u))*
		       (8*u*u*cos(w) + cdouble(0,2*u)*(3*u*u+w*w)*xi);

  cdouble h1 = 2*u*exp(cdouble(0,2*u)) - exp(cdouble(0,-u))*
		       (2*u*cos(w) - cdouble(0,1)*(3*u*u-w*w)*xi);

  cdouble h2 = exp(cdouble(0,2*u)) - exp(cdouble(0,-u))*
		       (cos(w) + cdouble(0,3*u)*xi);

  double factor = 9*u*u - w*w;

  cdouble f0 = h0/factor;
  cdouble f1 = h1/factor;
  cdouble f2 = h2/factor;

  ColorMatrix expQ;
  expQ = f0 + f1*Q +f2*Q*Q;

  return expQ;
}


template <typename Float>
void stout(Float** gauge, int* NeighbourSitesTable, double alpha){

  /*===============================================================
  U_mu'(n) = exp(Q) * U_mu
  ==============================================================*/
  ColorMatrix Q; 
  ColorMatrix U; 
  for(int mu=0; mu<4; mu++)
  for(int i=0; i<V; i++){
    Q_mu(Q, i, mu, gauge, NeighbourSitesTable, alpha);
    U = ColorMatrix( gauge[mu] + i*18);
    U = ExpMatrix(Q)*U;
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++){
      *(gauge[mu] + i*18 + j*6 +k*2 + 0) = real(U.GetU(j,k));		
      *(gauge[mu] + i*18 + j*6 +k*2 + 1) = imag(U.GetU(j,k));	
    }
  }
}

// update the gauge links to stout smeared links
void stout_smear(void** gauge, QudaPrecision GaugePrecision, double alpha, int *NeighbourSitesTable){
  cout<<"begin to smear the links..."<<endl;	
  if(GaugePrecision == QUDA_DOUBLE_PRECISION)
    stout((double**)gauge, NeighbourSitesTable, alpha);

  else if(GaugePrecision == QUDA_SINGLE_PRECISION)
    stout((float**)gauge, NeighbourSitesTable, alpha);
  cout<<"all the links smeared!"<<endl;	
}
