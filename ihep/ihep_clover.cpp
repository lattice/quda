//===================================================//
//        calc anisotropiuc clover term  	     //
//        and its inversion	     		     //
//	  by Jian Liang		   	   	     //
//	  2012 Nov. ~ Dec.			     //
//===================================================//
#include "ihep_utility.h"

template <typename Float>
void Cholesky_decomposition(Float* in, Float* out, const int dim){
//====================================================================================
// inverse a hermitian & positive defined matrix using Cholesky decomposition method
// here the clover matrix is hermitian and its diag elements are far larger the 
// off-diag ones(positive-defined??).
// see http://en.wikipedia.org/wiki/Cholesky_decomposition for details
// 2012 12
//====================================================================================
	
  int size = (dim*dim+dim)/2;
  cdouble* A = new cdouble [dim*dim];
  cdouble* T = new cdouble [dim*dim];
  cdouble* L = new cdouble [dim*dim];
  cdouble* L_dag = new cdouble [size];
  if(A==NULL||T==NULL||L==NULL||L_dag==NULL) {
    cout<<"Allocate mem wrong!"<<endl;
  }
  cdouble* b_star = NULL;
  cdouble* B = NULL;
  double aii = 0;
  int i, j, k;

  // initialize temp output pointer T to a unit matrix
  for(i=0; i<dim; i++)
  for(j=0; j<dim; j++)
    T[ i*dim+j ] = 0;
  for(i=0; i<dim; i++)
    T[ i*dim+i ] = 1;

  // initialize A using pointer in for the first time
  // the input in's order is
  //      | 0
  // in   | 3    1
  //      | 4    5    2 ...

  for(i=0; i<dim; i++) A[ i*dim+i ] = in[i];
  k = dim;
  for(i=0; i<dim-1; i++)
  for(j=i+1; j<dim; j++){
    A [ j*dim+i ] = cdouble(in[k],in[k+1]);
    k+=2;
  }
  for(i=0; i<dim-1; i++)
  for(j=i+1; j<dim; j++)
    A[ i*dim+j ] = conj(A[ j*dim+i ]);
  /*
  // the input in's order is
  //      | 0
  // in   | 1    3
  //      | 2    4    5 ...
  k = 0;
  for(i=0; i<dim; i++)
  for(j=i; j<dim; j++){
    A [ j*dim+i ] = in[k];
    k++;
  }
  for(i=0; i<dim; i++)
  for(j=i+1; j<dim; j++)
    A[ i*dim+j ] = conj(A[ j*dim+i ]);
  */

  // main loop to do the decomposition, dim times
  for(i=0; i<dim; i++){
    if( i>0 ){ // re calc A 
      // here borrow L for temp use because B & b_star are both pointer
      for(j=0; j<dim*dim; j++) L[ j ] = 0;
      for(j=0; j<i; j++) L[ j*dim+ j ] = 1;
      for(j=i; j<dim; j++)
      for(k=i; k<dim; k++) 
	L[ j*dim+ k ] = B[ (j-i)*dim+(k-i)] - conj(b_star[j-i])*b_star[k-i]/aii;
      // refresh A when L is done
      for(j=0; j<dim*dim; j++) A[j] = L[j];
    }
    if(i != dim-1){
      aii = real(A[ i*dim+ i ]);
      b_star = A + i*dim + i + 1;
      B = A + (i+1)*dim + (i+1);
      for(j=0; j<dim*dim; j++) L[ j ] = 0;
      for(j=0; j<dim; j++) L[ j*dim+ j ] = 1;
      L[ i*dim+ i ] = sqrt(aii);
      for(j=i+1; j<dim; j++) L[ j*dim+ i ] = conj( b_star[ j-i-1 ] )/sqrt(aii);
    }else{// this branch is just to keep the pointer not to slop over   
      aii = real(A[ i*dim+ i ]);
      for(j=0; j<dim*dim; j++) L[ j ] = 0;
      for(j=0; j<dim-1; j++) L[ j*dim+ j ] = 1;
      L[ i*dim+ i ] = sqrt(aii);
    }
    // L = L1*L2*...*Ln
    ComplexMatrixProduct(T, L, dim);
  }

  //output L & L_dag, order is
  //     | 0
  // L   | 1    2
  //     | 3    4    5 ...
  k=0;
  for(i=0; i<dim; i++)
  for(j=0; j<i+1; j++){
    L[ k ] = T[ i*dim+j ];
    k++;
  }

  //     | 0    1    2
  // L^  |      3    4
  //     |           5 ...
  k=0;
  for(i=0; i<dim; i++)
  for(j=i; j<dim; j++){
    L_dag[ k ] =conj( T[ j*dim+i] );
    k++;
  }

  delete[] A;
  delete[] T;
  A = T= b_star  = B = NULL;

  // begin to do inverse
  // A G = S  =>  (L L^) G= S  =>  L G' = S while L^ G = G' 
  // need be sloved twice
  double* S = new double [dim];
  cdouble* G_p = new cdouble [dim];
  cdouble* G     = new cdouble [dim];
  cdouble* Tout  = new cdouble [dim*dim];
  if(S==NULL||G_p==NULL||G==NULL||Tout==NULL){
    cout<<"Allocate mem wrong!"<<endl;
  }
  cdouble sum = 0;

  for(i=0; i<dim; i++) {// source loop
    for(j=0; j<dim; j++) S[j]=0;
    S[i] = 1;

    // solve  L G' = S
    G_p[0] = S[0]/L[0]; 
    for(j=1; j<dim; j++)  // G' index
    {
      sum = 0;  //dim=4 size=20 j=3 k=0-2
      for(k=0; k<j; k++) sum += L[ j*(j+1)/2 + k ]*G_p[k];
      G_p[j] = (S[j] - sum)/L[  j*(j+1)/2 + j ];
    }

    // solve  L^ G = G'
    G[dim-1] = G_p[dim-1]/L_dag[size-1]; 
    int newj;
    for(j=dim-2; j>-1; j--)  // G index
    {
      newj = dim - j - 1;   //dim=3 size=6 j=1-0 newj= 1-2 k=0 
      sum = 0;
      for(k=0; k<newj; k++) sum += L_dag[ size- (newj*(newj+1)/2) - k - 1 ]*G[dim - k - 1];
      G[j] = (G_p[j] - sum)/L_dag[ size- (newj*(newj+1)/2) - newj -1 ];
    }
	  
    //for each source S[i] = 1; G is the i_th column of the final output inversed matrix
    for(j=0; j<dim; j++) Tout[j*dim + i] = G[j];
  }

  // the output out's order is
  //      | 0
  // out  | 3    1
  //      | 4    5    2 ...

  for(i=0; i<dim; i++) out[i] = real(Tout[ i*dim+i ]);
  k = dim;
  for(i=0; i<dim-1; i++)
  for(j=i+1; j<dim; j++){
    out[k  ] = real(Tout[ j*dim+i ]);
    out[k+1] = imag(Tout[ j*dim+i ]);
    k+=2;
  }

  delete[] L;
  delete[] L_dag;
  delete[] G_p;
  delete[] G;
  delete[] S;
  delete[] Tout;
  G = G_p = L = L_dag = Tout = NULL;
  S = NULL;
}

template <typename Float>
void Fmunu(cdouble* F, int site, int mu, int nu, int munu, Float* gauge[4], int * NeighbourSitesTable){
/*===============================================================
// calc F_{\mu\nu} in a given site at a given plane \mu\nu
// F = (1/8i)SUM(U-U^)  U =U1+U2+U3+U4
/*===============================================================

					 (site+nu)
    |-------4-------nu-------2---------|
    |		    |                  |
    5	U2          3        U1        1	
    |		    |                  |
    |-------6----(site)------0-------------->mu
    |		    |	               |
    7	U3	    9	      U4       11
    |		    |	               |
    |-------8----------------10--------| (site+mu-nu)
  (site-mu-nu)          (site-nu)

  U1->0	1	2^+	3^+
  U2->3	4^+	5^+	6
  U3->6^+	7^+	8	9
  U4->9^+	10 	11	0^+
==============================================================*/

  ColorMatrix U1, U2, U3, U4;
  ColorMatrix link[12]; 
  link[0]  = ColorMatrix( gauge[mu] + site*18);
  link[1]  = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu*7 + 0]*18);
  link[2]  = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu*7 + 1]*18);
  link[3]  = ColorMatrix( gauge[nu] + site*18);
  link[4]  = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu*7 + 2]*18);
  link[5]  = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu*7 + 3]*18);
  link[6]  = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu*7 + 3]*18);
  link[7]  = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu*7 + 4]*18);
  link[8]  = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu*7 + 4]*18);
  link[9]  = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu*7 + 5]*18);
  link[10] = ColorMatrix( gauge[mu] + NeighbourSitesTable[site*84 + munu*7 + 5]*18);
  link[11] = ColorMatrix( gauge[nu] + NeighbourSitesTable[site*84 + munu*7 + 6]*18);

  U1 = link[0]* link[1]* link[2].dagger()* link[3].dagger();
  U2 = link[3]* link[4].dagger()* link[5].dagger()* link[6];
  U3 = link[6].dagger()* link[7].dagger()* link[8]* link[9];
  U4 = link[9].dagger()* link[10]* link[11]* link[0].dagger();

  ColorMatrix fmunu1, fmunu2;
  fmunu1 = U1+U2+U3+U4;
  fmunu2 = fmunu1.dagger();
  fmunu1 = fmunu1 - fmunu2;

  // for F is hermitian, here we need only 6 complex numbers, the order is
  //   |0 
  // F |1 3
  //   |2 4 5
  int Count = 0;
  for(int i=0; i<3; i++)
  for(int j=i; j<3; j++){
    F[Count] = fmunu1.GetU(j,i)*cdouble(0, -0.125);
    Count ++;
  }
}

template <typename Float1, typename Float2>
void SigmaMultiF(Float1* clover, cdouble *sigma, Float2** gauge, double kappa,
		 double Csw, int *NeighbourSitesTable, QudaGaugeParam *param){
/*============================================================================
// calc clover = factor * SUM(sigma_{ij} * F^{ij}) and then do I + clover
// action is 1 + kappa*D_wilson - (1/2)*Csw*kappa*sigma_{\mu\nu}*F^{\mu\nu} 
// sigma_{\mu\nu} = (i/2)[\gamma_{\mu}, \gamma_{\nu}] F^{\mu\nu} = (1/8i)(U-U^)
=============================================================================*/

  int SigmaOffset;

  // the last 2 comes from sigma12F12 = sigma21F21 
  // - comes from the action
  double factor1 = -kappa*Csw*0.5*2;
  double factor2 = -kappa*Csw*0.5*2;
  // anisotropic here
  factor1 = -kappa*0.5*(1+param->Xsi)/param->Us/param->Us;
  factor2 = -kappa/param->Us/param->Us/param->Us/param->Us;
  cout<<"factor1 = "<<factor1<<"   ";
  cout<<"factor2 = "<<factor2<<endl;

  cdouble* F = new cdouble[6];
  int munu[4][4];
  int count = 0;
  for(int i=0; i<4; i++)
  for(int j=0; j<4; j++)
    if(i==j){
      munu[i][j] = 0;}
    else{
      munu[i][j]=count;
      count++;}

  for(int site=0; site<V; site++){
    SigmaOffset = 0;
    for(int mu=0; mu<3; mu++)
    for(int nu=mu+1; nu<4; nu++){
      Fmunu(F, site, mu, nu, munu[mu][nu], gauge, NeighbourSitesTable);
      if(nu == 3) // t directory
	 for(int i=0;i<6;i++) F[i] *= factor1;
      else 
	 for(int i=0;i<6;i++) F[i] *= factor2;

      // for clover is divided into 2 parts(left_upper & right_lower)
      // each part is a 6*6 hermitian matrix(F & sigma are both hermitian), the diag components is real 
      // so each part need 6(diag) + (5+4+3+2+1)*2(off_diag) = 6+30 real numbers 
      // so the whole clover term need V*72 real numbers
      // the order is (0~5 are real while 6~20 are complex)
      //           | 0 
      // Clover_lu | 6   1
      //           | 7  11  2
      //           | 8  12  15   3
      //           | 9  13  16  18  4
      //           |10  14  17  19  20  5
      
      //           | 0 
      // Clover_rl | 6   1
      //           | 7  11  2
      //           | 8  12  15   3
      //           | 9  13  16  18  4
      //           |10  14  17  19  20  5
      // The index can not be expressed well in a simple loop
      // so I write them term by term as follows 
      // a little too long
      // but at least quite clear...
      
      clover[site*72 + 0]  += real(sigma[SigmaOffset*6 + 0]*F[0]);
      clover[site*72 + 1]  += real(sigma[SigmaOffset*6 + 0]*F[3]);
      clover[site*72 + 2]  += real(sigma[SigmaOffset*6 + 0]*F[5]);
      clover[site*72 + 3]  += real(sigma[SigmaOffset*6 + 2]*F[0]);
      clover[site*72 + 4]  += real(sigma[SigmaOffset*6 + 2]*F[3]);
      clover[site*72 + 5]  += real(sigma[SigmaOffset*6 + 2]*F[5]);

      clover[site*72 + 6]  += real(sigma[SigmaOffset*6 + 0]*F[1]);
      clover[site*72 + 7]  += imag(sigma[SigmaOffset*6 + 0]*F[1]);
      clover[site*72 + 8]  += real(sigma[SigmaOffset*6 + 0]*F[2]);
      clover[site*72 + 9]  += imag(sigma[SigmaOffset*6 + 0]*F[2]);
      clover[site*72 + 10] += real(sigma[SigmaOffset*6 + 1]*F[0]);
      clover[site*72 + 11] += imag(sigma[SigmaOffset*6 + 1]*F[0]);
      clover[site*72 + 12] += real(sigma[SigmaOffset*6 + 1]*F[1]);
      clover[site*72 + 13] += imag(sigma[SigmaOffset*6 + 1]*F[1]);
      clover[site*72 + 14] += real(sigma[SigmaOffset*6 + 1]*F[2]);
      clover[site*72 + 15] += imag(sigma[SigmaOffset*6 + 1]*F[2]);

      clover[site*72 + 16] += real(sigma[SigmaOffset*6 + 0]*F[4]);
      clover[site*72 + 17] += imag(sigma[SigmaOffset*6 + 0]*F[4]);
      clover[site*72 + 18] += real(sigma[SigmaOffset*6 + 1]*conj(F[1]));
      clover[site*72 + 19] += imag(sigma[SigmaOffset*6 + 1]*conj(F[1]));
      clover[site*72 + 20] += real(sigma[SigmaOffset*6 + 1]*F[3]);
      clover[site*72 + 21] += imag(sigma[SigmaOffset*6 + 1]*F[3]);
      clover[site*72 + 22] += real(sigma[SigmaOffset*6 + 1]*F[4]);
      clover[site*72 + 23] += imag(sigma[SigmaOffset*6 + 1]*F[4]);

      clover[site*72 + 24] += real(sigma[SigmaOffset*6 + 1]*conj(F[2]));
      clover[site*72 + 25] += imag(sigma[SigmaOffset*6 + 1]*conj(F[2]));
      clover[site*72 + 26] += real(sigma[SigmaOffset*6 + 1]*conj(F[4]));
      clover[site*72 + 27] += imag(sigma[SigmaOffset*6 + 1]*conj(F[4]));
      clover[site*72 + 28] += real(sigma[SigmaOffset*6 + 1]*F[5]);
      clover[site*72 + 29] += imag(sigma[SigmaOffset*6 + 1]*F[5]);

      clover[site*72 + 30] += real(sigma[SigmaOffset*6 + 2]*F[1]);
      clover[site*72 + 31] += imag(sigma[SigmaOffset*6 + 2]*F[1]);
      clover[site*72 + 32] += real(sigma[SigmaOffset*6 + 2]*F[2]);
      clover[site*72 + 33] += imag(sigma[SigmaOffset*6 + 2]*F[2]);

      clover[site*72 + 34] += real(sigma[SigmaOffset*6 + 2]*F[4]);
      clover[site*72 + 35] += imag(sigma[SigmaOffset*6 + 2]*F[4]);

      clover[site*72 + 36] += real(sigma[SigmaOffset*6 + 3]*F[0]);
      clover[site*72 + 37] += real(sigma[SigmaOffset*6 + 3]*F[3]);
      clover[site*72 + 38] += real(sigma[SigmaOffset*6 + 3]*F[5]);
      clover[site*72 + 39] += real(sigma[SigmaOffset*6 + 5]*F[0]);
      clover[site*72 + 40] += real(sigma[SigmaOffset*6 + 5]*F[3]);
      clover[site*72 + 41] += real(sigma[SigmaOffset*6 + 5]*F[5]);

      clover[site*72 + 42] += real(sigma[SigmaOffset*6 + 3]*F[1]);
      clover[site*72 + 43] += imag(sigma[SigmaOffset*6 + 3]*F[1]);
      clover[site*72 + 44] += real(sigma[SigmaOffset*6 + 3]*F[2]);
      clover[site*72 + 45] += imag(sigma[SigmaOffset*6 + 3]*F[2]);
      clover[site*72 + 46] += real(sigma[SigmaOffset*6 + 4]*F[0]);
      clover[site*72 + 47] += imag(sigma[SigmaOffset*6 + 4]*F[0]);
      clover[site*72 + 48] += real(sigma[SigmaOffset*6 + 4]*F[1]);
      clover[site*72 + 49] += imag(sigma[SigmaOffset*6 + 4]*F[1]);
      clover[site*72 + 50] += real(sigma[SigmaOffset*6 + 4]*F[2]);
      clover[site*72 + 51] += imag(sigma[SigmaOffset*6 + 4]*F[2]);

      clover[site*72 + 52] += real(sigma[SigmaOffset*6 + 3]*F[4]);
      clover[site*72 + 53] += imag(sigma[SigmaOffset*6 + 3]*F[4]);
      clover[site*72 + 54] += real(sigma[SigmaOffset*6 + 4]*conj(F[1]));
      clover[site*72 + 55] += imag(sigma[SigmaOffset*6 + 4]*conj(F[1]));
      clover[site*72 + 56] += real(sigma[SigmaOffset*6 + 4]*F[3]);
      clover[site*72 + 57] += imag(sigma[SigmaOffset*6 + 4]*F[3]);
      clover[site*72 + 58] += real(sigma[SigmaOffset*6 + 4]*F[4]);
      clover[site*72 + 59] += imag(sigma[SigmaOffset*6 + 4]*F[4]);

      clover[site*72 + 60] += real(sigma[SigmaOffset*6 + 4]*conj(F[2]));
      clover[site*72 + 61] += imag(sigma[SigmaOffset*6 + 4]*conj(F[2]));
      clover[site*72 + 62] += real(sigma[SigmaOffset*6 + 4]*conj(F[4]));
      clover[site*72 + 63] += imag(sigma[SigmaOffset*6 + 4]*conj(F[4]));
      clover[site*72 + 64] += real(sigma[SigmaOffset*6 + 4]*F[5]);
      clover[site*72 + 65] += imag(sigma[SigmaOffset*6 + 4]*F[5]);

      clover[site*72 + 66] += real(sigma[SigmaOffset*6 + 5]*F[1]);
      clover[site*72 + 67] += imag(sigma[SigmaOffset*6 + 5]*F[1]);
      clover[site*72 + 68] += real(sigma[SigmaOffset*6 + 5]*F[2]);
      clover[site*72 + 69] += imag(sigma[SigmaOffset*6 + 5]*F[2]);

      clover[site*72 + 70] += real(sigma[SigmaOffset*6 + 5]*F[4]);
      clover[site*72 + 71] += imag(sigma[SigmaOffset*6 + 5]*F[4]);
      SigmaOffset ++;
    }
		  
    //for(int i=0; i<72; i++){
    // clover[site*72 +  i] *= factor;}
	    
    for(int i=0; i<6; i++){
      clover[site*72 +  i] += 1;
      clover[site*72 +  36 + i] += 1;}
  }
  if(F != NULL){
    delete [] F; 
    F = NULL;}
}

// construct the clover term using gauge field
void construct_clover_term_from_gauge(void* clover_inv, void** gauge, QudaPrecision GaugePrecision, QudaPrecision CloverPrecision,
				      double kappa, double Csw, int* NeighbourSitesTable, QudaGaugeParam *param){
  /*
  Gamma matrices are on Chiral Basis (DeGrand - Rossi Basis) here 0->t 1->x 2->y 3->z
  the 6 independent sigma components: sigma01, sigma02, sigma03, sigma12, sigma13, sigma23
  -------------------------------------------------------------------------------
  |  0  1  0 0  |  0  i 0  0 |  1  0  0 0  |  1  0 0  0 |  0 -i 0  0 |  0 1 0 0 |  	
  |  1  0  0 0  |  -i 0 0  0 |  0 -1  0 0  |  0 -1 0  0 |  i  0 0  0 |  1 0 0 0 |  
  |  0  0  0 -1 |  0  0 0 -i |  0  0 -1 0  |  0  0 1  0 |  0  0 0 -i |  0 0 0 1 |  
  |  0  0 -1 0  |  0  0 i  0 |  0  0  0 1  |  0  0 0 -1 |  0  0 i  0 |  0 0 1 0 |  
  -------------------------------------------------------------------------------
  Each component's upper left part & lower right part are 4*4 hermitian matrices.

  because the \mu\nu order in LDG format is 0->x 1->y 2->z 3->t
  new01->12, new02->13, new03->-01, new12->23, new13->-02, new23->-03.
  new 6 independent sigma components: sigma01, sigma02, sigma03, sigma12, sigma13, sigma23
  -------------------------------------------------------------------------------
  |  1  0 0  0 |  0 -i 0  0 |  0 -1  0  0 |  0 1 0 0 |  0 -i  0  0 | -1 0  0  0 |
  |  0 -1 0  0 |  i  0 0  0 | -1  0  0  0 |  1 0 0 0 |  i  0  0  0 |  0 1  0  0 |
  |  0  0 1  0 |  0  0 0 -i |  0  0  0  1 |  0 0 0 1 |  0  0  0  i |  0 0  1  0 |
  |  0  0 0 -1 |  0  0 i  0 |  0  0  1  0 |  0 0 1 0 |  0  0 -i  0 |  0 0  0 -1 |
  -------------------------------------------------------------------------------

  each sigma 6 has complex numbers, the order is 	
		  -------------			
		  | 0  1^     |		
  sigma：         | 1  2      |		
		  |      3  4^|		
		  |      4  5 |
		  -------------
  */
  printf("Begin to construct clover term ...\n");
  fflush(stdout);
	
  cdouble  sigma[36] = 
    {  
      cdouble(1),cdouble(0),cdouble(-1),  cdouble(1),cdouble(0),cdouble(-1),
      cdouble(0),cdouble(0,1),cdouble(0), cdouble(0),cdouble(0,1),cdouble(0),
      cdouble(0),cdouble(-1),cdouble(0),  cdouble(0),cdouble(1),cdouble(0),
      cdouble(0),cdouble(1),cdouble(0),   cdouble(0),cdouble(1),cdouble(0),
      cdouble(0),cdouble(0,1),cdouble(0), cdouble(0),cdouble(0,-1),cdouble(0),
      cdouble(-1),cdouble(0),cdouble(1),  cdouble(1),cdouble(0),cdouble(-1)
    };

  //note here clover precision should be the same with gauge precision
  if(CloverPrecision != GaugePrecision)
    errorQuda("clover & gauge precision should match");


  if(CloverPrecision == QUDA_DOUBLE_PRECISION && GaugePrecision == QUDA_DOUBLE_PRECISION){
    for(int i=0; i<V*72; i++)
      *((double*)clover_inv + i) = 0;  
    SigmaMultiF((double*) clover_inv,  sigma, (double**) gauge,  kappa, Csw, NeighbourSitesTable, param);
    for(int i=0; i<V; i++){
      Cholesky_decomposition((double*)clover_inv + i*72,     (double*)clover_inv + i*72,    6);
      Cholesky_decomposition((double*)clover_inv + i*72 +36, (double*)clover_inv + i*72+36, 6);
    }
  }
  else if(CloverPrecision  == QUDA_SINGLE_PRECISION && GaugePrecision == QUDA_SINGLE_PRECISION){
    for(int i=0; i<V*72; i++)
      *((float*)clover_inv + i) = 0;  
    SigmaMultiF((float*) clover_inv,  sigma, (float**) gauge, kappa, Csw, NeighbourSitesTable, param);
    for(int i=0; i<V; i++){
      Cholesky_decomposition((float*)clover_inv + i*72,     (float*)clover_inv + i*72,    6);
      Cholesky_decomposition((float*)clover_inv + i*72 +36, (float*)clover_inv + i*72+36, 6);
    }
  }

  printf("Clover term constructed! \n");
  fflush(stdout);
}
