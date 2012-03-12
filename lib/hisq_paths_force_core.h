/**************************do_middle_link_kernel*****************************
 *
 *
 * Generally we need
 * READ
 *    3 LINKS:         ab_link,     bc_link,    ad_link
 *    3 COLOR MATRIX:  newOprod_at_A, oprod_at_C,  Qprod_at_D
 * WRITE
 *    4 COLOR MATRIX:  newOprod_at_A, P3_at_A, Pmu_at_B, Qmu_at_A
 *
 * Three call variations:
 *   1. when Qprev == NULL:   Qprod_at_D does not exit and is not read in
 *   2. full read/write
 *   3. when Pmu/Qmu == NULL,   Pmu_at_B and Qmu_at_A are not written out
 *
 *   In all three above case, if the direction sig is negative, newOprod_at_A is 
 *   not read in or written out.
 *
 * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
 *   Call 1: 
 *             if (sig is positive):    (3, 6) 
 *             else               :     (3, 4) 
 *   Call 2: 
 *             if (sig is positive):    (3, 7) 
 *             else               :     (3, 5)  
 *   Call 3: 
 *             if (sig is positive):    (3, 5) 
 *             else               :     (3, 3)  
 * 
 * note: oprod_at_C could actually be read in from D when it is the fresh outer product
 *       and we call it oprod_at_C to simply naming. This does not affect our data traffic analysis
 *
 ****************************************************************************/

template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit> 
  __global__ void
  HISQ_KERNEL_NAME(do_middle_link, EXT)(const RealA* const oprodEven, const RealA* const oprodOdd,
					const RealA* const QprevEven, const RealA* const QprevOdd,  
					const RealB* const linkEven,  const RealB* const linkOdd,
					int sig, int mu, 
					typename RealTypeId<RealA>::Type coeff,
					RealA* const PmuEven, RealA* const PmuOdd, 
					RealA* const P3Even, RealA* const P3Odd,
					RealA* const QmuEven, RealA* const QmuOdd, 
					RealA* const newOprodEven, RealA* const newOprodOdd) 
{		
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int new_x[4];
  int new_mem_idx;
  int ad_link_sign=1;
  int ab_link_sign=1;
  int bc_link_sign=1;

  RealB ab_link[ArrayLength<RealB>::result];
  RealB bc_link[ArrayLength<RealB>::result];
  RealB ad_link[ArrayLength<RealB>::result];

  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
  RealA COLOR_MAT_X[ArrayLength<RealA>::result];
  
  /*        A________B
   *   mu   |        |
   *  	   D|        |C
   *	  
   *	  A is the current point (sid)
   *
   */
  
  int point_b, point_c, point_d;
  int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
  int mymu;

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu = mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
  }
  point_d = (new_mem_idx >> 1);
  if (mu_positive){
    ad_link_nbr_idx = point_d;
    reconstructSign(&ad_link_sign, mymu, new_x);
  }else{
    ad_link_nbr_idx = sid;
    reconstructSign(&ad_link_sign, mymu, x);	
  }

  int mysig; 
  if(sig_positive){
    mysig = sig;
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
  }else{
    mysig = OPP_DIR(sig);
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
  }
  point_c = (new_mem_idx >> 1);
  if (mu_positive){
    bc_link_nbr_idx = point_c;	
    reconstructSign(&bc_link_sign, mymu, new_x);
  }

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(sig_positive){
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
  }else{
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
  }
  point_b = (new_mem_idx >> 1); 

  if (!mu_positive){
    bc_link_nbr_idx = point_b;
    reconstructSign(&bc_link_sign, mymu, new_x);
  }   

  if(sig_positive){
    ab_link_nbr_idx = sid;
    reconstructSign(&ab_link_sign, mysig, x);	
  }else{	
    ab_link_nbr_idx = point_b;
    reconstructSign(&ab_link_sign, mysig, new_x);
  }
  // now we have ab_link_nbr_idx


  // load the link variable connecting a and b 
  // Store in ab_link 
  if(sig_positive){
    HISQ_LOAD_LINK(linkEven, linkOdd, mysig, ab_link_nbr_idx, ab_link, oddBit);
  }else{
    HISQ_LOAD_LINK(linkEven, linkOdd, mysig, ab_link_nbr_idx, ab_link, 1-oddBit);
  }
  RECONSTRUCT_SITE_LINK(ab_link, ab_link_sign)

  // load the link variable connecting b and c 
  // Store in bc_link
  if(mu_positive){
    HISQ_LOAD_LINK(linkEven, linkOdd, mymu, bc_link_nbr_idx, bc_link, oddBit);
  }else{ 
    HISQ_LOAD_LINK(linkEven, linkOdd, mymu, bc_link_nbr_idx, bc_link, 1-oddBit);
  }
  RECONSTRUCT_SITE_LINK(bc_link, bc_link_sign)
  
  if(QprevOdd == NULL){
    if(sig_positive){
      loadMatrixFromField(oprodEven, oprodOdd, sig, point_d, COLOR_MAT_Y, 1-oddBit);
    }else{
      loadMatrixFromField(oprodEven, oprodOdd, OPP_DIR(sig), point_c, COLOR_MAT_Y, oddBit);
      adjointMatrix(COLOR_MAT_Y);
    }
  }else{ // QprevOdd != NULL
    loadMatrixFromField(oprodEven, oprodOdd, point_c, COLOR_MAT_Y, oddBit);
  }
  
  
  MATRIX_PRODUCT(bc_link, COLOR_MAT_Y, !mu_positive, COLOR_MAT_W);
  if(PmuOdd){
    storeMatrixToField(COLOR_MAT_W, point_b, PmuEven, PmuOdd, 1-oddBit);
  }
  MATRIX_PRODUCT(ab_link, COLOR_MAT_W, sig_positive,COLOR_MAT_Y);
  storeMatrixToField(COLOR_MAT_Y, sid, P3Even, P3Odd, oddBit);
  
  
  if(mu_positive){
    HISQ_LOAD_LINK(linkEven, linkOdd, mymu, ad_link_nbr_idx, ad_link, 1-oddBit);
    RECONSTRUCT_SITE_LINK(ad_link, ad_link_sign)    
  }else{
    HISQ_LOAD_LINK(linkEven, linkOdd, mymu, ad_link_nbr_idx, ad_link, oddBit);
    RECONSTRUCT_SITE_LINK(ad_link, ad_link_sign)
    adjointMatrix(ad_link);
    
  }
  
  
  if(QprevOdd == NULL){
    if(sig_positive){
      MAT_MUL_MAT(COLOR_MAT_W, ad_link, COLOR_MAT_Y);
    }
    if(QmuEven){
      ASSIGN_MAT(ad_link, COLOR_MAT_X); 
      storeMatrixToField(COLOR_MAT_X, sid, QmuEven, QmuOdd, oddBit);
    }
  }else{ 
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_Y, 1-oddBit);
    MAT_MUL_MAT(COLOR_MAT_Y, ad_link, COLOR_MAT_X);
    if(QmuEven){
      storeMatrixToField(COLOR_MAT_X, sid, QmuEven, QmuOdd, oddBit);
    }
    if(sig_positive){
      MAT_MUL_MAT(COLOR_MAT_W, COLOR_MAT_X, COLOR_MAT_Y);
    }	
  }
    
  if(sig_positive){
    addMatrixToField(COLOR_MAT_Y, sig, sid, coeff, newOprodEven, newOprodOdd, oddBit);
  }

  return;
}

/***********************************do_side_link_kernel***************************
 *
 * In general we need
 * READ
 *    1  LINK:          ad_link
 *    4  COLOR MATRIX:  shortP_at_D, newOprod, P3_at_A, Qprod_at_D, 
 * WRITE
 *    2  COLOR MATRIX:  shortP_at_D, newOprod,
 *
 * Two call variations:
 *   1. full read/write
 *   2. when shortP == NULL && Qprod == NULL:  
 *          no need to read ad_link/shortP_at_D or write shortP_at_D
 *          Qprod_at_D does not exit and is not read in                                     
 *
 *
 * Therefore the data traffic, in two-number pair (num_of_links, num_of_color_matrix)
 *   Call 1:                (1, 6) 
 *             
 *   Call 2:                (0, 3)
 *
 * note: newOprod can be at point D or A, depending on if mu is postive or negative
 *
 *
 *********************************************************************************/

template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
  __global__ void
  HISQ_KERNEL_NAME(do_side_link, EXT)(const RealA* const P3Even, const RealA* const P3Odd,
				      const RealA* const QprodEven, const RealA* const QprodOdd,
				      const RealB* const linkEven,  const RealB* const linkOdd,
				      int sig, int mu, 
				      typename RealTypeId<RealA>::Type coeff, 
				      typename RealTypeId<RealA>::Type accumu_coeff,
				      RealA* const shortPEven, RealA* const shortPOdd,
				      RealA* const newOprodEven, RealA* const newOprodOdd)
{

  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int ad_link_sign = 1;

  RealB ad_link[ArrayLength<RealB>::result];

  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
  // The compiler probably knows to reorder so that loads are done early on
  loadMatrixFromField(P3Even, P3Odd, sid, COLOR_MAT_Y, oddBit);

  /*      compute the side link contribution to the momentum
   *
   *             sig
   *          A________B
   *           |       |   mu
   *         D |       |C
   *
   *      A is the current point (sid)
   *
   */

  typename RealTypeId<RealA>::Type mycoeff;
  int point_d;
  int ad_link_nbr_idx;
  int mymu;
  int new_mem_idx;

  int new_x[4];
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu=mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mymu,X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mymu, X, new_mem_idx);
  }
  point_d = (new_mem_idx >> 1);


  // Should all be inside if (shortPOdd)
  if (shortPOdd){
    if (mu_positive){
      ad_link_nbr_idx = point_d;
      reconstructSign(&ad_link_sign, mymu, new_x);
    }else{
      ad_link_nbr_idx = sid;
      reconstructSign(&ad_link_sign, mymu, x);	
    }

    
    if(mu_positive){
      HISQ_LOAD_LINK(linkEven, linkOdd, mymu, ad_link_nbr_idx, ad_link, 1-oddBit);
    }else{
      HISQ_LOAD_LINK(linkEven, linkOdd, mymu, ad_link_nbr_idx, ad_link, oddBit);
    }
    RECONSTRUCT_SITE_LINK(ad_link, ad_link_sign)
   
    MATRIX_PRODUCT(ad_link, COLOR_MAT_Y, mu_positive, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  }


  mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;

  if(QprodOdd){
    loadMatrixFromField(QprodEven, QprodOdd, point_d, COLOR_MAT_X, 1-oddBit);
    if(mu_positive){
      MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_W);

      // Added by J.F.
      if(!oddBit){ mycoeff = -mycoeff; }
      addMatrixToField(COLOR_MAT_W, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit);
    }else{
      ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_W);
      if(oddBit){ mycoeff = -mycoeff; }
      addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven, newOprodOdd, oddBit);
    } 
  }

  if(!QprodOdd){
    if(mu_positive){
      if(!oddBit){ mycoeff = -mycoeff;}
      addMatrixToField(COLOR_MAT_Y, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit);
    }else{
      if(oddBit){ mycoeff = -mycoeff; }
      ADJ_MAT(COLOR_MAT_Y, COLOR_MAT_W);
      addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven, newOprodOdd,  oddBit);
    }
  }

  return;
}

/********************************do_all_link_kernel*********************************************
*
* In this function we need
*   READ
*     3 LINKS:         ad_link, ab_link, bc_link
*     5 COLOR MATRIX:  Qprev_at_D, oprod_at_C, newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D
*   WRITE: 
*     3 COLOR MATRIX:  newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D,
*
* If sig is negative, then we don't need to read/write the color matrix newOprod_at_A(sig)
*
* Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
*
*             if (sig is positive):    (3, 8) 
*             else               :     (3, 6) 
*
*
************************************************************************************************/

template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
  __global__ void
  HISQ_KERNEL_NAME(do_all_link, EXT)(const RealA* const oprodEven, const RealA* const oprodOdd, 
				     const RealA* const QprevEven, const RealA* const QprevOdd,
				     const RealB* const linkEven, const RealB* const linkOdd,
				     int sig, int mu, 
				     typename RealTypeId<RealA>::Type coeff, 
				     typename RealTypeId<RealA>::Type accumu_coeff,
				     RealA* const shortPEven, RealA* const shortPOdd,
				     RealA* const newOprodEven, RealA* const newOprodOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int ad_link_sign=1;
  int ab_link_sign=1;
  int bc_link_sign=1;

  int new_x[4];

  RealB ab_link[ArrayLength<RealB>::result];
  RealB bc_link[ArrayLength<RealB>::result];
  RealB ad_link[ArrayLength<RealB>::result];

  RealA COLOR_MAT_X[ArrayLength<RealA>::result];  
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_Z[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_W[ArrayLength<RealA>::result]; 
 

  /*            sig
   *         A________B
   *      mu  |      |
   *        D |      |C
   *
   *   A is the current point (sid)
   *
   */
  
  int point_b, point_c, point_d;
  int ab_link_nbr_idx;
  int new_mem_idx;
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(sig_positive){
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
  }else{
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
  }
  point_b = (new_mem_idx >> 1);
  ab_link_nbr_idx = (sig_positive) ? sid : point_b;
  if(sig_positive){
    reconstructSign(&ab_link_sign, sig, x);
  }else{
    reconstructSign(&ab_link_sign, sig, new_x);    
  }
  if(!mu_positive){
    reconstructSign(&bc_link_sign, OPP_DIR(mu),  new_x);
  }
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  
  const typename RealTypeId<RealA>::Type & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
  if(mu_positive){ //positive mu
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
    point_d = (new_mem_idx >> 1);
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_X, 1-oddBit);	   // COLOR_MAT_X
    HISQ_LOAD_LINK(linkEven, linkOdd, mu, point_d, ad_link, 1-oddBit); 
    reconstructSign(&ad_link_sign, mu, new_x);   
    RECONSTRUCT_SITE_LINK(ad_link, ad_link_sign)
    
    if(sig_positive){
      FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
    }else{
      FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
    }
    point_c = (new_mem_idx >> 1);
    loadMatrixFromField(oprodEven,oprodOdd,  point_c, COLOR_MAT_Y, oddBit);		// COLOR_MAT_Y
    HISQ_LOAD_LINK(linkEven, linkOdd, mu, point_c, bc_link, oddBit);   
    reconstructSign(&bc_link_sign, mu, new_x);  
    RECONSTRUCT_SITE_LINK(bc_link, bc_link_sign)
    
    MATRIX_PRODUCT(bc_link, COLOR_MAT_Y, 0, COLOR_MAT_Z); // COMPUTE_LINK_X
    if (sig_positive)
      {
	MAT_MUL_MAT(COLOR_MAT_X, ad_link, COLOR_MAT_Y);
	MAT_MUL_MAT(COLOR_MAT_Z, COLOR_MAT_Y, COLOR_MAT_W);
	addMatrixToField(COLOR_MAT_W, sig, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);
      }

    if (sig_positive){
      HISQ_LOAD_LINK(linkEven, linkOdd, sig, ab_link_nbr_idx, ab_link, oddBit);
    }else{
      HISQ_LOAD_LINK(linkEven, linkOdd, OPP_DIR(sig), ab_link_nbr_idx, ab_link, 1-oddBit);
    }
    RECONSTRUCT_SITE_LINK(ab_link, ab_link_sign)

    MATRIX_PRODUCT(ab_link, COLOR_MAT_Z, sig_positive, COLOR_MAT_Y); // COLOR_MAT_Y is assigned here

    MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, mu, point_d, -Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, 1-oddBit);

    MAT_MUL_MAT(ad_link, COLOR_MAT_Y, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  } else{ //negative mu
    mu = OPP_DIR(mu);
    
    new_x[0] = x[0];
    new_x[1] = x[1];
    new_x[2] = x[2];
    new_x[3] = x[3];
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mu, X, new_mem_idx);	
    point_d = (new_mem_idx >> 1);
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_X, 1-oddBit);         // COLOR_MAT_X used!
    HISQ_LOAD_LINK(linkEven, linkOdd, mu, sid, ad_link, oddBit);  
    reconstructSign(&ad_link_sign, mu, x);
    RECONSTRUCT_SITE_LINK(ad_link, ad_link_sign)
    
    if(sig_positive){
      FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
    }else{
      FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
    }
    point_c = (new_mem_idx >> 1);
    loadMatrixFromField(oprodEven, oprodOdd, point_c, COLOR_MAT_Y, oddBit);	     // COLOR_MAT_Y used
    HISQ_LOAD_LINK(linkEven, linkOdd, mu, point_b, bc_link, 1-oddBit);    
    RECONSTRUCT_SITE_LINK(bc_link, bc_link_sign)  //bc_link_sign is computed earlier in the function
    
    if(sig_positive){
      MAT_MUL_ADJ_MAT(COLOR_MAT_X, ad_link, COLOR_MAT_W);
    }
    MAT_MUL_MAT(bc_link, COLOR_MAT_Y, COLOR_MAT_Z);
    if (sig_positive){	
      MAT_MUL_MAT(COLOR_MAT_Z, COLOR_MAT_W, COLOR_MAT_Y);
      addMatrixToField(COLOR_MAT_Y, sig, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);
      }

    if (sig_positive){
      HISQ_LOAD_LINK(linkEven, linkOdd, sig, ab_link_nbr_idx, ab_link, oddBit); 
    }else{
      HISQ_LOAD_LINK(linkEven, linkOdd, OPP_DIR(sig), ab_link_nbr_idx, ab_link, 1-oddBit);
    }
    RECONSTRUCT_SITE_LINK(ab_link, ab_link_sign)

    MATRIX_PRODUCT(ab_link, COLOR_MAT_Z, sig_positive, COLOR_MAT_Y);
    ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_W);	
    addMatrixToField(COLOR_MAT_W, mu, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);

    MATRIX_PRODUCT(ad_link, COLOR_MAT_Y, 0, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  } 

  return;
}

