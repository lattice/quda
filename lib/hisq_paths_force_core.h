
//macro KERNEL_ENABLED is used to control compile time, debug purpose only
#if (PRECISION == 0 && RECON == 18)
#define EXT _dp_18_
#elif (PRECISION == 0 && RECON == 12)
#define EXT _dp_12_
#elif (PRECISION == 1 && RECON == 18)
#define EXT _sp_18_
#else 
#define EXT _sp_12_
#endif


#define print_matrix(mul)                                               \
printf(" (%f %f) (%f %f) (%f %f)\n", mul##00_re, mul##00_im, mul##01_re, mul##01_im, mul##02_re, mul##02_im); \
printf(" (%f %f) (%f %f) (%f %f)\n", mul##10_re, mul##10_im, mul##11_re, mul##11_im, mul##12_re, mul##12_im); \
printf(" (%f %f) (%f %f) (%f %f)\n", mul##20_re, mul##20_im, mul##21_re, mul##21_im, mul##22_re, mul##22_im);


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
 *   Call 1:  (called 48 times, half positive sig, half negative sig)
 *             if (sig is positive):    (3, 6) 
 *             else               :     (3, 4) 
 *   Call 2:  (called 192 time, half positive sig, half negative sig) 
 *             if (sig is positive):    (3, 7) 
 *             else               :     (3, 5)  
 *   Call 3:  (called 48 times, half positive sig, half negative sig)
 *             if (sig is positive):    (3, 5) 
 *             else               :     (3, 2) no need to loadQprod_at_D in this case  
 * 
 * note: oprod_at_C could actually be read in from D when it is the fresh outer product
 *       and we call it oprod_at_C to simply naming. This does not affect our data traffic analysis
 * 
 * Flop count, in two-number pair (matrix_multi, matrix_add)
 *   call 1:     if (sig is positive)  (3, 1)
 *               else                  (2, 0)
 *   call 2:     if (sig is positive)  (4, 1) 
 *               else                  (3, 0)
 *   call 3:     if (sig is positive)  (4, 1) 
 *               else                  (2, 0) 
 *
 ****************************************************************************/
template<class RealA, class RealB, int sig_positive, int mu_positive, int _oddBit, int oddness_change> 
  __global__ void
                 HISQ_KERNEL_NAME(do_middle_link, EXT)(const RealA* const oprodEven, const RealA* const oprodOdd,
                     const RealA* const QprevEven, const RealA* const QprevOdd,  
                     const RealB* const linkEven,  const RealB* const linkOdd,
                     int sig, int mu, 
                     typename RealTypeId<RealA>::Type coeff,
                     RealA* const PmuEven, RealA* const PmuOdd, 
                     RealA* const P3Even, RealA* const P3Odd,
                     RealA* const QmuEven, RealA* const QmuOdd, 
                     RealA* const newOprodEven, RealA* const newOprodOdd,
                     hisq_kernel_param_t kparam) 
{


  int oddBit = _oddBit;
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if(sid >= kparam.threads) return;
  int dx[4] = {0,0,0,0};
  int x[4];

  getCoords(x, sid, kparam.D, oddBit);


  Matrix<RealA,3> Uab, Ubc, Uad;
  Matrix<RealA,3> Ow, Ox, Oy;


  /*        A________B
   *   mu   |        |
   *  	   D|        |C
   *	  
   *	  A is the current point (sid)
   *
   */

  int point_b, point_c, point_d;
  int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
  int mymu = posDir(mu);



#ifdef MULTI_GPU
  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};

  x[0] = x[0] + kparam.base_idx[0];
  x[1] = x[1] + kparam.base_idx[1];
  x[2] = x[2] + kparam.base_idx[2];
  x[3] = x[3] + kparam.base_idx[3];
  int new_sid = linkIndex(x,dx,E);
  oddBit = _oddBit ^ oddness_change;

#else
  int E[4] = {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif

  int y[4] = {x[0], x[1], x[2], x[3]};

  mymu = posDir(mu);

  updateCoords(y, mymu, (mu_positive ? -1 : 1), kparam.X, kparam.ghostDim[mymu]);

  point_d = linkIndex(y, dx, E);

  if (mu_positive){
    ad_link_nbr_idx = point_d;
  }else{
    ad_link_nbr_idx = new_sid;
  }

  int mysig = posDir(sig);
  updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
  point_c = linkIndex(y, dx, E);

  if (mu_positive){
    bc_link_nbr_idx = point_c;	
  }


  for(int dir=0; dir<4; ++dir) y[dir] = x[dir];
  updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
  point_b = linkIndex(y, dx, E);

  if (!mu_positive){
    bc_link_nbr_idx = point_b;
  }   

  if(sig_positive){
    ab_link_nbr_idx = new_sid;
  }else{	
    ab_link_nbr_idx = point_b;
  }
  // now we have ab_link_nbr_idx


  // load the link variable connecting a and b 
  // Store in ab_link 
  //loadLink<18>(linkEven, linkOdd, mysig, ab_link_nbr_idx, Uab.data, sig_positive^(1-oddBit), kparam.thin_link_stride);
  loadLink<18>(linkEven, linkOdd, mysig, ab_link_nbr_idx, Uab.data, sig_positive^(1-oddBit), kparam.thin_link_stride);


  // load the link variable connecting b and c 
  // Store in bc_link
  loadLink<18>(linkEven, linkOdd, mymu, bc_link_nbr_idx, Ubc.data, mu_positive^(1-oddBit), kparam.thin_link_stride);

  if(QprevOdd == NULL){
    loadMatrixFromField(oprodEven, oprodOdd, posDir(sig), (sig_positive ? point_d : point_c), Oy.data, sig_positive^oddBit, kparam.color_matrix_stride);
    if(!sig_positive) Oy = conj(Oy);
  }else{ // QprevOdd != NULL
    loadMatrixFromField(oprodEven, oprodOdd, point_c, Oy.data, oddBit, kparam.color_matrix_stride);
  }


  if(!mu_positive){
    Ow = Ubc*Oy;
  }else{
    Ow = conj(Ubc)*Oy;
  }

  if(PmuOdd){
    storeMatrixToField(Ow.data, point_b, PmuEven, PmuOdd, 1-oddBit, kparam.color_matrix_stride);
  }
  if(sig_positive){
    Oy = Uab*Ow;
  }else{
    Oy = conj(Uab)*Ow;
  }

  storeMatrixToField(Oy.data, new_sid, P3Even, P3Odd, oddBit, kparam.color_matrix_stride);

  loadLink<18>(linkEven, linkOdd, mymu, ad_link_nbr_idx, Uad.data, mu_positive^oddBit, kparam.thin_link_stride);
  if(!mu_positive)  Uad = conj(Uad);


  if(QprevOdd == NULL){
    if(sig_positive){
      Oy = Ow*Uad;
    }

    if(QmuEven){
      Ox = Uad;
      storeMatrixToField(Ox.data, new_sid, QmuEven, QmuOdd, oddBit, kparam.color_matrix_stride);
    }
  }else{ 
    if(QmuEven || sig_positive){
      loadMatrixFromField(QprevEven, QprevOdd, point_d, Oy.data, 1-oddBit, kparam.color_matrix_stride);
      Ox = Oy*Uad;
    }
    if(QmuEven){
      storeMatrixToField(Ox.data, new_sid, QmuEven, QmuOdd, oddBit, kparam.color_matrix_stride);
    }
    if(sig_positive){
      Oy = Ow*Ox;
    }	
  }

  if(sig_positive){
    addMatrixToNewOprod(Oy.data, sig, new_sid, coeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);
  }

  return;
}


template<class RealA, class RealB, int sig_positive, int mu_positive, int _oddBit, int oddness_change> 
  __global__ void
HISQ_KERNEL_NAME(do_lepage_middle_link, EXT)(const RealA* const oprodEven, const RealA* const oprodOdd,
    const RealA* const QprevEven, const RealA* const QprevOdd,  
    const RealB* const linkEven,  const RealB* const linkOdd,
    int sig, int mu, 
    typename RealTypeId<RealA>::Type coeff,
    RealA* const P3Even, RealA* const P3Odd,
    RealA* const newOprodEven, RealA* const newOprodOdd,
    hisq_kernel_param_t kparam) 
{

  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if(sid >= kparam.threads) return;
  int oddBit = _oddBit;


  Matrix<RealA,3> Uab, Ubc, Uad;
  Matrix<RealA,3> Ow, Ox, Oy;




  /*        A________B
   *   mu   |        |
   *       D|        |C
   *    
   *   A is the current point (sid)
   *
   */

  int point_b, point_c, point_d;
  int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
  int mymu;

  int x[4];
  int dx[4] = {0,0,0,0};
  getCoords(x, sid, kparam.D, oddBit);

#ifdef MULTI_GPU
  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  x[0] = x[0] + kparam.base_idx[0];
  x[1] = x[1] + kparam.base_idx[1];
  x[2] = x[2] + kparam.base_idx[2];
  x[3] = x[3] + kparam.base_idx[3];
  int new_sid = linkIndex(x,dx,E);
  oddBit = _oddBit ^ oddness_change;
#else
  int E[4]= {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif


  mymu = posDir(mu);
  int y[4] = {x[0], x[1], x[2], x[3]};
  updateCoords(y, mymu, (mu_positive ? -1 : 1), kparam.X, kparam.ghostDim[mymu]);
  point_d = linkIndex(y, dx, E);



  if (mu_positive){
    ad_link_nbr_idx = point_d;
  }else{
    ad_link_nbr_idx = new_sid;
  }

  int mysig = posDir(sig);
  updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
  point_c = linkIndex(y, dx, E);



  if (mu_positive){
    bc_link_nbr_idx = point_c;  
  }


  for(int dir=0; dir<4; ++dir) y[dir] = x[dir];
  updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
  point_b = linkIndex(y, dx, E);

  if (!mu_positive){
    bc_link_nbr_idx = point_b;
  }   

  if(sig_positive){
    ab_link_nbr_idx = new_sid;
  }else{  
    ab_link_nbr_idx = point_b;
  }
  // now we have ab_link_nbr_idx
  //
  //
  // load the link variable connecting a and b 
  // Store in ab_link 
  loadLink<18>(linkEven, linkOdd, mysig, ab_link_nbr_idx, Uab.data, sig_positive^(1-oddBit), kparam.thin_link_stride);

  // load the link variable connecting b and c 
  // Store in bc_link
  loadLink<18>(linkEven, linkOdd, mymu, bc_link_nbr_idx, Ubc.data, mu_positive^(1-oddBit), kparam.thin_link_stride);


  loadMatrixFromField(oprodEven, oprodOdd, point_c, Oy.data, oddBit, kparam.color_matrix_stride);

  if(!mu_positive){
    Ow = Ubc*Oy;
  }else{
    Ow = conj(Ubc)*Oy;
  }

  if(sig_positive){
    Oy = Uab*Ow;
  }else{
    Oy = conj(Uab)*Ow;
  }


  storeMatrixToField(Oy.data, new_sid, P3Even, P3Odd, oddBit, kparam.color_matrix_stride);
  if(sig_positive){
    loadLink<18>(linkEven, linkOdd, mymu, ad_link_nbr_idx, Uad.data, mu_positive^oddBit, kparam.thin_link_stride);
    if(!mu_positive) Uad = conj(Uad);

    loadMatrixFromField(QprevEven, QprevOdd, point_d, Oy.data, 1-oddBit, kparam.color_matrix_stride);
    
    Ox = Oy*Uad;
    Oy = Ow*Ox;

    addMatrixToNewOprod(Oy.data, sig, new_sid, coeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);
  }

//#endif  
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
 *   Call 1:   (called 192 times)        
 *                           (1, 6) 
 *             
 *   Call 2:   (called 48 times)             
 *                           (0, 3)
 *
 * note: newOprod can be at point D or A, depending on if mu is postive or negative
 *
 * Flop count, in two-number pair (matrix_multi, matrix_add)
 *   call 1:       (2, 2)
 *   call 2:       (0, 1) 
 *
 *********************************************************************************/

template<class RealA, class RealB, int sig_positive, int mu_positive, int _oddBit, int oddness_change>
  __global__ void
HISQ_KERNEL_NAME(do_side_link, EXT)(const RealA* const P3Even, const RealA* const P3Odd,
    const RealA* const QprodEven, const RealA* const QprodOdd,
    const RealB* const linkEven,  const RealB* const linkOdd,
    int sig, int mu, 
    typename RealTypeId<RealA>::Type coeff, 
    typename RealTypeId<RealA>::Type accumu_coeff,
    RealA* const shortPEven, RealA* const shortPOdd,
    RealA* const newOprodEven, RealA* const newOprodOdd,
    hisq_kernel_param_t kparam)
{
  int oddBit = _oddBit;
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if(sid >= kparam.threads) return;


  int x[4];
  int dx[4] = {0,0,0,0};
  getCoords(x, sid, kparam.D, oddBit);





#ifdef MULTI_GPU
  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  x[0] = x[0] + kparam.base_idx[0];
  x[1] = x[1] + kparam.base_idx[1];
  x[2] = x[2] + kparam.base_idx[2];
  x[3] = x[3] + kparam.base_idx[3];
  int new_sid = linkIndex(x,dx,E);
  oddBit = _oddBit ^ oddness_change;
#else
  int E[4]= {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif



  Matrix<RealA,3> Uad;
  Matrix<RealA,3> Ow, Ox, Oy;





  loadMatrixFromField(P3Even, P3Odd, new_sid, Oy.data, oddBit, kparam.color_matrix_stride);

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


  int y[4] = {x[0], x[1], x[2], x[3]}; 

  typename RealTypeId<RealA>::Type mycoeff;
  int point_d;
  int ad_link_nbr_idx;
  int mymu = posDir(mu);
  updateCoords(y, mymu, (mu_positive ? -1 : 1), kparam.X, kparam.ghostDim[mymu]);
  point_d = linkIndex(y,dx,E);



  if (mu_positive){
    ad_link_nbr_idx = point_d;
  }else{
    ad_link_nbr_idx = new_sid;
  }

  loadLink<18>(linkEven, linkOdd, mymu, ad_link_nbr_idx, Uad.data, mu_positive^oddBit, kparam.thin_link_stride);


  if(mu_positive){
    Ow = Uad*Oy;
  }else{
    Ow = conj(Uad)*Oy;
  }



  addMatrixToField(Ow.data, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit, kparam.color_matrix_stride);
  mycoeff = CoeffSign<sig_positive,_oddBit ^ oddness_change>::result*coeff;

  loadMatrixFromField(QprodEven, QprodOdd, point_d, Ox.data, 1-oddBit, kparam.color_matrix_stride);

  if(mu_positive){
    Ow = Oy*Ox;
    if(!oddBit){ mycoeff = -mycoeff; }
    addMatrixToNewOprod(Ow.data, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit, kparam.color_matrix_stride);
  }else{
    Ow = conj(Ox)*conj(Oy);
    if(oddBit){ mycoeff = -mycoeff; }
    addMatrixToNewOprod(Ow.data, OPP_DIR(mu), new_sid, mycoeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);
  } 
//#endif
  return;
}




template<class RealA, class RealB, int sig_positive, int mu_positive, int _oddBit, int oddness_change>
  __global__ void
HISQ_KERNEL_NAME(do_side_link_short, EXT)(const RealA* const P3Even, const RealA* const P3Odd,
    const RealB* const linkEven,  const RealB* const linkOdd,
    int sig, int mu, 
    typename RealTypeId<RealA>::Type coeff, 
    RealA* const newOprodEven, RealA* const newOprodOdd,
    hisq_kernel_param_t kparam)
{
  int oddBit = _oddBit;
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if(sid >= kparam.threads) return;


  int x[4];
  int dx[4] = {0,0,0,0};
  getCoords(x, sid, kparam.D, oddBit);

#ifdef MULTI_GPU
  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  x[0] = x[0] + kparam.base_idx[0];
  x[1] = x[1] + kparam.base_idx[1];
  x[2] = x[2] + kparam.base_idx[2];
  x[3] = x[3] + kparam.base_idx[3];
  int new_sid = linkIndex(x,dx,E);
  oddBit = _oddBit ^ oddness_change;
#else
  int E[4]= {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif

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

  Matrix<RealA,3> Ow, Oy;


  loadMatrixFromField(P3Even, P3Odd, new_sid, Oy.data, oddBit, kparam.color_matrix_stride);

  typename RealTypeId<RealA>::Type mycoeff;
  int point_d;
  int mymu = posDir(mu);
  int y[4] = {x[0], x[1], x[2], x[3]};  

  updateCoords(y, mymu, (mu_positive ? -1 : 1), kparam.X, kparam.ghostDim[mymu]);
  point_d = linkIndex(y,dx,E);
  mycoeff = CoeffSign<sig_positive,_oddBit ^ oddness_change>::result*coeff;

  if(mu_positive){
    if(!oddBit){ mycoeff = -mycoeff;} // need to change this to get away from oddBit
    addMatrixToNewOprod(Oy.data, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit, kparam.color_matrix_stride);
  }else{
    if(oddBit){ mycoeff = -mycoeff; }
    Ow = conj(Oy);
    addMatrixToNewOprod(Ow.data, OPP_DIR(mu), new_sid, mycoeff, newOprodEven, newOprodOdd,  oddBit, kparam.color_matrix_stride);
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
 * This function is called 384 times, half positive sig, half negative sig
 *
 * Flop count, in two-number pair (matrix_multi, matrix_add)
 *             if(sig is positive)      (6,3)
 *             else                     (4,2)
 *
 ************************************************************************************************/


template<class RealA, class RealB, int sig_positive, int mu_positive, int _oddBit, int oddness_change>
  __global__ void
HISQ_KERNEL_NAME(do_all_link, EXT)(const RealA* const oprodEven, const RealA* const oprodOdd, 
    const RealA* const QprevEven, const RealA* const QprevOdd,
    const RealB* const linkEven, const RealB* const linkOdd,
    int sig, int mu, 
    typename RealTypeId<RealA>::Type coeff, 
    typename RealTypeId<RealA>::Type accumu_coeff,
    RealA* const shortPEven, RealA* const shortPOdd,
    RealA* const newOprodEven, RealA* const newOprodOdd,
    hisq_kernel_param_t kparam)
{
  int oddBit = _oddBit;
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if(sid >= kparam.threads) return;


  int x[4];
  int dx[4] = {0,0,0,0};
  getCoords(x, sid, kparam.D, oddBit);



  Matrix<RealA,3> Uab, Ubc, Uad;
  Matrix<RealA,3> Ow, Ox, Oy, Oz;


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

#ifdef MULTI_GPU
  x[0] = x[0] + kparam.base_idx[0];
  x[1] = x[1] + kparam.base_idx[1];
  x[2] = x[2] + kparam.base_idx[2];
  x[3] = x[3] + kparam.base_idx[3];

  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  int new_sid = linkIndex(x,dx,E);
  oddBit = _oddBit ^ oddness_change;
#else
  int E[4]= {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif

  int y[4] = {x[0], x[1], x[2], x[3]};
  int mysig = posDir(sig);
  updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
  point_b = linkIndex(y,dx,E);

  ab_link_nbr_idx = (sig_positive) ? new_sid : point_b;

  for(int dir=0; dir<4; ++dir) y[dir] = x[dir];


  const typename RealTypeId<RealA>::Type & mycoeff = CoeffSign<sig_positive,_oddBit ^ oddness_change>::result*coeff;
  if(mu_positive){ //positive mu

    updateCoords(y, mu, -1, kparam.X, kparam.ghostDim[mu]);
    point_d = linkIndex(y,dx,E);

    updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
    point_c = linkIndex(y,dx,E);

    loadMatrixFromField(QprevEven, QprevOdd, point_d, Ox.data, 1-oddBit, kparam.color_matrix_stride);	   // COLOR_MAT_X
    loadLink<18>(linkEven, linkOdd, mu, point_d, Uad.data, 1-oddBit, kparam.thin_link_stride); 

    loadMatrixFromField(oprodEven,oprodOdd,  point_c, Oy.data, oddBit, kparam.color_matrix_stride);		// COLOR_MAT_Y
    loadLink<18>(linkEven, linkOdd, mu, point_c, Ubc.data, oddBit, kparam.thin_link_stride);   

    Oz = conj(Ubc)*Oy;

    if (sig_positive)
    {
      Ow = Oz*Ox*Uad;
      addMatrixToNewOprod(Ow.data, sig, new_sid, Sign<_oddBit ^ oddness_change>::result*mycoeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);
    }

    
    loadLink<18>(linkEven, linkOdd, posDir(sig), ab_link_nbr_idx, Uab.data, sig_positive^(1-oddBit), kparam.thin_link_stride);

    if(sig_positive){
      Oy = Uab*Oz;
    }else{
      Oy = conj(Uab)*Oz;
    }


    Ow = Oy*Ox;
    addMatrixToNewOprod(Ow.data, mu, point_d, -Sign<_oddBit ^ oddness_change>::result*mycoeff, newOprodEven, newOprodOdd, 1-oddBit, kparam.color_matrix_stride);
    Ow = Uad*Oy;
    addMatrixToField(Ow.data, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit, kparam.color_matrix_stride);

  } else{ //negative mu

    mu = OPP_DIR(mu);
    updateCoords(y, mu, 1, kparam.X, kparam.ghostDim[mu]);
    point_d = linkIndex(y,dx,E);
    updateCoords(y, mysig, (sig_positive ? 1 : -1), kparam.X, kparam.ghostDim[mysig]);
    point_c = linkIndex(y,dx,E);
  
    loadMatrixFromField(QprevEven, QprevOdd, point_d, Ox.data, 1-oddBit, kparam.color_matrix_stride);         // COLOR_MAT_X used!

    loadLink<18>(linkEven, linkOdd, mu, new_sid, Uad.data, oddBit, kparam.thin_link_stride);  

    loadMatrixFromField(oprodEven, oprodOdd, point_c, Oy.data, oddBit, kparam.color_matrix_stride);	     // COLOR_MAT_Y used
    loadLink<18>(linkEven, linkOdd, mu, point_b, Ubc.data, 1-oddBit, kparam.thin_link_stride);    


    if(sig_positive){
      Ow = Ox*conj(Uad);
    }
    Oz = Ubc*Oy;

    if (sig_positive){	
      Oy = Oz*Ow;
      addMatrixToNewOprod(Oy.data, sig, new_sid, Sign<_oddBit ^ oddness_change>::result*mycoeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);
    }
    loadLink<18>(linkEven, linkOdd, posDir(sig), ab_link_nbr_idx, Uab.data, sig_positive^(1-oddBit), kparam.thin_link_stride); 

 
    if(sig_positive){ 
      Oy = Uab*Oz;
    }else{
      Oy = conj(Uab)*Oz;
    }

    Ow = conj(Ox)*conj(Oy);

    addMatrixToNewOprod(Ow.data, mu, new_sid, Sign<_oddBit ^ oddness_change>::result*mycoeff, newOprodEven, newOprodOdd, oddBit, kparam.color_matrix_stride);

    Ow = conj(Uad)*Oy;

    addMatrixToField(Ow.data, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit, kparam.color_matrix_stride);

  } 
  return;
}





template<class RealA, class RealB,  int oddBit>
  __global__ void 
HISQ_KERNEL_NAME(do_longlink, EXT)(const RealB* const linkEven, const RealB* const linkOdd,
    const RealA* const naikOprodEven, const RealA* const naikOprodOdd,
    typename RealTypeId<RealA>::Type coeff,
    RealA* const outputEven, RealA* const outputOdd,
    hisq_kernel_param_t kparam)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= kparam.threads) return;


  int x[4];
  int dx[4] = {0,0,0,0};

  getCoords(x, sid, kparam.X, oddBit);
#ifdef MULTI_GPU
  int E[4]= {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  for(int i=0; i<4; ++i) x[i] += 2;
  int new_sid = linkIndex(x,dx,E);
#else
  int E[4] = {kparam.X[0], kparam.X[1], kparam.X[2], kparam.X[3]};
  int new_sid = sid;
#endif



  const int & point_c = new_sid;
  int point_a, point_b, point_d, point_e;


  /*
   * 
   *    A   B    C    D    E    
   *    ---- ---- ---- ----  
   *
   *   ---> sig direction
   *
   *   C is the current point (sid)
   *
   */


  Matrix<RealA,3> Uab, Ubc, Ude, Uef;
  Matrix<RealA,3> Ox, Oy, Oz;

  // compute the force for forward long links
  for(int sig=0; sig<4; ++sig){

    dx[sig]++;
    point_d = linkIndex(x,dx,E);

    dx[sig]++;
    point_e = linkIndex(x,dx,E);	  

    dx[sig] = -1;
    point_b = linkIndex(x,dx,E);	  

    dx[sig]--;
    point_a = linkIndex(x,dx,E);	  
    dx[sig]=0;


    loadLink<18>(linkEven, linkOdd, sig, point_a, Uab.data, oddBit, kparam.thin_link_stride); 
    loadLink<18>(linkEven, linkOdd, sig, point_b, Ubc.data, 1-oddBit, kparam.thin_link_stride);
    loadLink<18>(linkEven, linkOdd, sig, point_d, Ude.data, 1-oddBit, kparam.thin_link_stride);
    loadLink<18>(linkEven, linkOdd, sig, point_e, Uef.data, oddBit, kparam.thin_link_stride);

    loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_c, Oz.data, oddBit, kparam.color_matrix_stride);
    loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_b, Oy.data, 1-oddBit, kparam.color_matrix_stride);
    loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_a, Ox.data, oddBit, kparam.color_matrix_stride);

    Matrix<RealA,3> temp = Ude*Uef*Oz - Ude*Oy*Ubc + Ox*Uab*Ubc;

    addMatrixToField(temp.data, sig, new_sid,  coeff, outputEven, outputOdd, oddBit, kparam.color_matrix_stride);
  } // loop over sig

  return;
}


template<class RealA, class RealB, int oddBit>
  __global__ void 
HISQ_KERNEL_NAME(do_complete_force, EXT)(const RealB* const linkEven, const RealB* const linkOdd, 
    const RealA* const oprodEven, const RealA* const oprodOdd,
    int sig,
    RealA* const forceEven, RealA* const forceOdd,
    hisq_kernel_param_t kparam)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= kparam.threads) return;


  int x[4];
  int dx[4] = {0,0,0,0};
  getCoords(x, sid, kparam.X, oddBit);

  int new_sid=sid;
#ifdef MULTI_GPU
  x[0] = x[0]+2;
  x[1] = x[1]+2;
  x[2] = x[2]+2;
  x[3] = x[3]+2;
  int E[4] = {kparam.X[0]+4, kparam.X[1]+4, kparam.X[2]+4, kparam.X[3]+4};
  new_sid = linkIndex(x,dx,E);
#endif


  Matrix<RealA,3> Uw, Ow, Ox;


  loadLink<18>(linkEven, linkOdd, sig, new_sid, Uw.data, oddBit, kparam.thin_link_stride);  

  loadMatrixFromField(oprodEven, oprodOdd, sig, new_sid, Ox.data, oddBit, kparam.color_matrix_stride);
  typename RealTypeId<RealA>::Type coeff = (oddBit==1) ? -1 : 1;
  Ow = Uw*Ox;

  storeMatrixToMomentumField(Ow.data, sig, sid, coeff, forceEven, forceOdd, oddBit, kparam.momentum_stride); 
  return;
}

#undef EXT
