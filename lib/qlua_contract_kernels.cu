#include <qlua_contract_kernels.cuh>

namespace quda {


  // hard-coded DeGrand-Rossi basis
  // g{x,y,z,t} = g{1,2,3,4}
  // gamma matrices are specified as either single linear element in the algebra
  //    G(n) = g1^n0 . g2^n1 . g3^n2 . g4^n3, with n=n0*2^0+n1*2^1+n2*2^2+n3*2^3
  // or a linear combination of them
  //    G(v) = \sum_{n=0..15} v(n) G(n)

  // parameterization of gamma matrices for efficient unrolling
  // [G(n)]_{ij} = gamma_left_coeff[n][i] * (gamma_left_ind[n][i]==j ? 1 : 0)
  //             = C(n)_i * delta(j==J(n)_i)
  // Tr[A.G(n)] = \sum_j C(n)_j * A_{J(n)_j, j}
  // [G(n)   . A]_{i,      j}       = C(n)_i * A_{J(n)_i,j}
  // [G(n)^T . A]_{J(n)_k, j} = C(n)_k * A_{k,j}
  // [A . G(n)  ]_{i, J(n)_k} = A_{i,k} * C(n)_k
  // [A . G(n)^T]_{i,      j} = A_{i,J(n)_j} * C(n)_j
  inline __device__ __host__ QC_REAL gamma_left_coeff_Re(int m, int n, int c){
    CONSTVAR_ QC_REAL gamma_left_coeff_Re_[QC_LEN_G][QC_Ns][2] = {
      { {1,0}, {1,0}, {1,0}, {1,0} },             /* G0 = 1 */
      { {0,1}, {0,1},{0,-1},{0,-1} },             /* G1 = g1 */
      {{-1,0}, {1,0}, {1,0},{-1,0} },             /* G2 = g2 */
      {{0,-1}, {0,1},{0,-1}, {0,1} },             /* G3 = g1 g2 */
      { {0,1},{0,-1},{0,-1}, {0,1} },             /* G4 = g3 */
      {{-1,0}, {1,0},{-1,0}, {1,0} },             /* G5 = g1 g3 */
      {{0,-1},{0,-1},{0,-1},{0,-1} },             /* G6 = g2 g3 */
      { {1,0}, {1,0},{-1,0},{-1,0} },             /* G7 = g1 g2 g3 */
      { {1,0}, {1,0}, {1,0}, {1,0} },             /* G8 = g4 */
      { {0,1}, {0,1},{0,-1},{0,-1} },             /* G9 = g1 g4 */
      {{-1,0}, {1,0}, {1,0},{-1,0} },             /* G10= g2 g4 */
      {{0,-1}, {0,1},{0,-1}, {0,1} },             /* G11= g1 g2 g4 */
      { {0,1},{0,-1},{0,-1}, {0,1} },             /* G12= g3 g4 */
      {{-1,0}, {1,0},{-1,0}, {1,0} },             /* G13= g1 g3 g4 */
      {{0,-1},{0,-1},{0,-1},{0,-1} },             /* G14= g2 g3 g4 */
      { {1,0}, {1,0},{-1,0},{-1,0} },             /* G15= g1 g2 g3 g4 */
    };
    return gamma_left_coeff_Re_[m][n][c];
  }

  inline __device__ __host__ int gamma_left_ind(int m, int n){
    CONSTVAR_ int gamma_left_ind_[QC_LEN_G][QC_Ns] = {
      { 0, 1, 2, 3 },             /* G0 = 1 */
      { 3, 2, 1, 0 },             /* G1 = g1 */
      { 3, 2, 1, 0 },             /* G2 = g2 */
      { 0, 1, 2, 3 },             /* G3 = g1 g2 */
      { 2, 3, 0, 1 },             /* G4 = g3 */
      { 1, 0, 3, 2 },             /* G5 = g1 g3 */
      { 1, 0, 3, 2 },             /* G6 = g2 g3 */
      { 2, 3, 0, 1 },             /* G7 = g1 g2 g3 */
      { 2, 3, 0, 1 },             /* G8 = g4 */
      { 1, 0, 3, 2 },             /* G9 = g1 g4 */
      { 1, 0, 3, 2 },             /* G10= g2 g4 */
      { 2, 3, 0, 1 },             /* G11= g1 g2 g4 */
      { 0, 1, 2, 3 },             /* G12= g3 g4 */
      { 3, 2, 1, 0 },             /* G13= g1 g3 g4 */
      { 3, 2, 1, 0 },             /* G14= g2 g3 g4 */
      { 0, 1, 2, 3 },             /* G15= g1 g2 g3 g4 */
    };
    return gamma_left_ind_[m][n];
  }


  QC_REAL gamma_left_coeff_Re_cMem(int m, int n, int c){
    return gamma_left_coeff_Re(m,n,c);
  }

  int gamma_left_ind_cMem(int m, int n){
    return gamma_left_ind(m,n);
  }


  /* bits (gammas) in 0..15 (in G0..G15) */
  inline __device__ int qc_bitcount16(int n){
    CONSTVAR_ int qc_bitcount16_[QC_LEN_G] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
    return qc_bitcount16_[n];
  }


  /*  G(n)^\dag = (-1)**gamma_adj_parity[n] * G(n);
      G(n).G(n) = (-1)**gamma_adj_parity[n]  (since G(n)^\dag.G(n)=G(n).G(n)^\dag=1) */
  inline __device__ int qc_gamma_adj_parity(int n){
    CONSTVAR_ int qc_gamma_adj_parity_[QC_LEN_G] = { 0, 0, 0, 1, 0, 1, 1, 1,
						     0, 1, 1, 1, 1, 1, 1, 0};
    return qc_gamma_adj_parity_[n];
  }

  /*  G(n)^* = (-1)**gamma_conj_parity[n] * G(n) */
  inline __device__ int qc_gamma_conj_parity(int n){
    CONSTVAR_ int qc_gamma_conj_parity_[QC_LEN_G] = { 0, 1, 0, 1, 1, 0, 1, 0,
						      0, 1, 0, 1, 1, 0, 1, 0 };
    return qc_gamma_conj_parity_[n];
  }

  /*  G(n)^T = (-1)**gamma_transp_parity[n] * G(n) */
  inline __device__ int qc_gamma_transp_parity(int n){
    CONSTVAR_ int qc_gamma_transp_parity_[QC_LEN_G]={ 0, 1, 0, 0, 1, 1, 0, 1,
						      0, 0, 1, 0, 0, 1, 0, 0 };
    return qc_gamma_transp_parity_[n];
  }

  /* G(n)^\dag . G(m) . G(n) = (-1)**gamma_uni_parity(m,n) * G(m) */
  inline __device__ int qc_gamma_uni_parity(int m,int n){
    return ( qc_bitcount16(m) * qc_bitcount16(n) - qc_bitcount16(m&n) ) % 2;
  }

  /* G(n)^T . G(m) . G(n)    = (-1)**gamma_sim_parity(m,n) * G(m) */
  inline __device__ int qc_gamma_sim_parity(int m, int n){
    return ( qc_gamma_uni_parity(m,n) + qc_gamma_conj_parity(n) ) % 2;
  }


#define gamma_left_coeff(m,n) (complex<QC_REAL>{gamma_left_coeff_Re(m,n,0), gamma_left_coeff_Re(m,n,1)})


  
  /* elementary actions on gamma matrices */
  /* a <- a^\dag */
  inline __device__ void 
    QC(gvec_adj)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = conj(a_gvec[i]) * (QC_CPLX)(1 - 2 * qc_gamma_adj_parity(i));
  }
  /* a <- a^T */
  inline __device__ void 
    QC(gvec_transp)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_transp_parity(i));
  }
  /* a <- conj(a) */
  inline __device__ void 
    QC(gvec_conj)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = conj(a_gvec[i]) * (QC_CPLX)(1 - 2 * qc_gamma_conj_parity(i));
  }
  /* a <- G(ng)^dag . a . G(ng) */
  inline __device__ void 
    QC(gvec_uni_transf)(QC_CPLX *a_gvec, int ng)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_uni_parity(i, ng));
  }
  /* a <- G(ng)^T . a . G(ng) */
  inline __device__ void 
    QC(gvec_sim_transf)(QC_CPLX *a_gvec, int ng)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_sim_parity(i, ng));
  }
  // TODO Cliffort algegra mult: (a*b)[i^j] += a[i]*b[j] * mul_sign(i,j)


  /* general BLAS-style functions for multiplying by gamma 
     having general b is ok since extra MADD cost in negl. on GPU 
  */

  inline __device__ void 
    QC(agx_pby_gind_D)(
		       QC_CPLX a, int ng, const QC_CPLX *x, int x_stride,
		       QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int ic = 0 ; ic < QC_Nc ; ic++)
      for (int is = 0 ; is < QC_Ns ; is++) {
        int ks  = gamma_left_ind(ng,is);
        QC_CPLX gik = gamma_left_coeff(ng,is);
        int yi = y_stride * QC_LIDX_D(ic, is),
	  xi = x_stride * QC_LIDX_D(ic, ks);
        y[yi] = b * y[yi] + a * gik * x[xi];
      }
  }
  inline __device__ void 
    QC(agTx_pby_gind_D)(
			QC_CPLX a, int ng, const QC_CPLX *x, int x_stride,
			QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int ic = 0 ; ic < QC_Nc ; ic++)
      for (int is = 0 ; is < QC_Ns ; is++) {
        int ks  = gamma_left_ind(ng,is);
        QC_CPLX gik = gamma_left_coeff(ng,is);
        int yi = y_stride*QC_LIDX_D(ic, ks),
	  xi = x_stride*QC_LIDX_D(ic, is);
        y[yi] = b * y[yi] + a * gik * x[xi];
      }
  }
  inline __device__ void 
    QC(agx_pby_gind_P)(
		       QC_CPLX a, int ng, const QC_CPLX *x, int x_stride,
		       QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int ks  = gamma_left_ind(ng,is);
	    QC_CPLX gik = gamma_left_coeff(ng,is);
	    int yi = y_stride * QC_LIDX_P(ic, is, jc, js),
	      xi = x_stride * QC_LIDX_P(ic, ks, jc, js);
	    y[yi] = b * y[yi] + a * gik * x[xi];
	  }
  }
  inline __device__ void 
    QC(agTx_pby_gind_P)(
			QC_CPLX a, int ng, const QC_CPLX *x, int x_stride, 
			QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int ks  = gamma_left_ind(ng,is);
	    QC_CPLX gik = gamma_left_coeff(ng,is);
	    int yi = y_stride * QC_LIDX_P(ic, ks, jc, js),
	      xi = x_stride * QC_LIDX_P(ic, is, jc, js);
	    y[yi] = b * y[yi] + a * gik * x[xi];
	  }
  }
  inline __device__ void 
    QC(axg_pby_gind_P)(
		       QC_CPLX a, int ng, const QC_CPLX *x, int x_stride,
		       QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int ks  = gamma_left_ind(ng,js);
	    QC_CPLX gik = gamma_left_coeff(ng,js);
	    int yi = y_stride * QC_LIDX_P(ic, is, jc, ks),
	      xi = x_stride * QC_LIDX_P(ic, is, jc, js);
	    y[yi] = b * y[yi] + a * gik * x[xi];
	  }
  }
  inline __device__ void 
    QC(axgT_pby_gind_P)(
			QC_CPLX a, int ng, const QC_CPLX *x, int x_stride,
			QC_CPLX b, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int ks  = gamma_left_ind(ng,js);
	    QC_CPLX gik = gamma_left_coeff(ng,js);
	    int yi = y_stride * QC_LIDX_P(ic, is, jc, js),
	      xi = x_stride * QC_LIDX_P(ic, is, jc, ks);
	    y[yi] = b * y[yi] + a * gik * x[xi];
	  }
  }


  /* y <- 0 */
  inline __device__ void
    QC(cplx_vec_zero)(QC_CPLX *y, int y_stride, int len)
  {
    for (int i = len ; i-- ; y += y_stride) 
      *y = 0;
  }
  /* y <- x */
  inline __device__ void
    QC(cplx_vec_copy)(QC_CPLX *y, int y_stride, 
		      const QC_CPLX *x, int x_stride, int len)
  {
    for (int i = len ; i-- ; x += x_stride, y += y_stride)
      *y = *x;
  }
  /* x <- a * x */
  inline __device__ void 
    QC(cplx_vec_scal)(QC_CPLX *x, int x_stride, QC_CPLX a, int len)
  {
    for (int i = len ; i-- ; x += x_stride)
      *x *= a;
  }
  /* y <- a * x */
  inline __device__ void 
    QC(cplx_vec_scal_copy)(QC_CPLX *y, int y_stride, 
			   QC_CPLX a, const QC_CPLX *x, int x_stride, int len)
  {
    for (int i = len ; i-- ; x += x_stride, y += y_stride)
      *y = *x * a;
  }
  /* y <- conj(y) */
  inline __device__ void
    QC(cplx_vec_conj)(QC_CPLX *y, int y_stride, int len)
  {
    for (int i = len ; i-- ; y += y_stride) 
      *y = QC_CONJ(*y);
  }


#define def_gvec_func(gvec_func, gind_func, len)			\
  inline __device__ void gvec_func(					\
				  QC_CPLX a, const QC_CPLX *gvec, const QC_CPLX *x, int x_stride, \
				  QC_CPLX b, QC_CPLX *y, int y_stride) { \
    for (int ng = 0 ; ng < QC_LEN_G; ng++) {				\
      if ((QC_CPLX)0 != gvec[ng]) {					\
	gind_func(gvec[ng] * a, ng, x, x_stride, b, y, y_stride);	\
	b = 1;								\
      }									\
    }									\
    if ((QC_CPLX)1 != b) QC(cplx_vec_scal)(y, y_stride, b, len);	\
  }
  def_gvec_func( QC(agx_pby_gvec_D),  QC(agx_pby_gind_D), QC_LEN_D);
  def_gvec_func(QC(agTx_pby_gvec_D), QC(agTx_pby_gind_D), QC_LEN_D);
  def_gvec_func( QC(agx_pby_gvec_P),  QC(agx_pby_gind_P), QC_LEN_P);
  def_gvec_func(QC(agTx_pby_gvec_P), QC(agTx_pby_gind_P), QC_LEN_P);
  def_gvec_func( QC(axg_pby_gvec_P),  QC(axg_pby_gind_P), QC_LEN_P);
  def_gvec_func(QC(axgT_pby_gvec_P), QC(axgT_pby_gind_P), QC_LEN_P);
#undef def_gvec_func


  /* quark contraction functions */
#define def_quarkcontractMN(MN, A,B,C,D) inline __device__ void		\
    QC(quarkContract##MN)(QC_CPLX *r, int r_stride,			\
			  const QC_CPLX *q1, int q1_stride, const QC_CPLX *q2, int q2_stride) { \
    const int eps[3][3] = { { 0, 1, 2}, { 1, 2, 0}, { 2, 0, 1} };	\
    int p_a, p_b;							\
    for (p_a = 0; p_a < 3; p_a++) { /* Nc */				\
      int i1 = eps[p_a][0], j1 = eps[p_a][1], k1 = eps[p_a][2];		\
      for (p_b = 0; p_b < 3; p_b++) { /* Nc */				\
	int i2 = eps[p_b][0], j2 = eps[p_b][1], k2 = eps[p_b][2];	\
	int a, b, c;							\
	for (a = 0; a < QC_Ns; a++) {					\
	  for (b = 0; b < QC_Ns; b++) {					\
	    QC_CPLX s3 = 0.;						\
	    for (c = 0; c < QC_Ns; c++) {				\
	      s3 += q1[q1_stride*QC_LIDX_P(i1,(A),i2,(B))]		\
		* q2[q2_stride*QC_LIDX_P(j1,(C),j2,(D))];		\
	      s3 -= q1[q1_stride*QC_LIDX_P(i1,(A),j2,(B))]		\
		* q2[q2_stride*QC_LIDX_P(j1,(C),i2,(D))];		\
	      s3 -= q1[q1_stride*QC_LIDX_P(j1,(A),i2,(B))]		\
		* q2[q2_stride*QC_LIDX_P(i1,(C),j2,(D))];		\
	      s3 += q1[q1_stride*QC_LIDX_P(j1,(A),j2,(B))]		\
		* q2[q2_stride*QC_LIDX_P(i1,(C),i2,(D))];		\
	      r[r_stride*QC_LIDX_P(k2,a,k1,b)] = s3;			\
	    } } } } }							\
  }
  def_quarkcontractMN(12, c,c,a,b)
    def_quarkcontractMN(13, c,a,c,b)
    def_quarkcontractMN(14, c,a,b,c)
    def_quarkcontractMN(23, a,c,c,b)
    def_quarkcontractMN(24, a,c,b,c)
    def_quarkcontractMN(34, a,b,c,c)
#undef def_quarkcontractMN


    /* r[ic,is,jc,js] = \delta{is==js} * \sum_{ks} x[ic,ks,jc,ks] */
    inline __device__ void
    QC(P_eq_spintrace_P)(QC_CPLX *r, int r_stride, const QC_CPLX *x, int x_stride)
    {
      QC(cplx_vec_zero)(r, r_stride, QC_LEN_P);
      for (int jc = 0 ; jc < QC_Nc ; jc++)
	for (int ic = 0 ; ic < QC_Nc ; ic++) {
	  QC_CPLX tr = 0;
	  for (int is = 0 ; is < QC_Ns ; is++)
            tr += x[x_stride * QC_LIDX_P(ic, is, jc, is)];
	  for (int is = 0 ; is < QC_Ns ; is++)
            r[r_stride * QC_LIDX_P(ic, is, jc, is)] = tr;
	}
    }
  /* r[ic,is,jc,js] <- r[ic,js,jc,is] */
  inline __device__ void
    QC(spintranspose_P)(QC_CPLX *r, int r_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int ic = 0 ; ic < QC_Nc ; ic++)
	for (int is = 0 ; is < QC_Ns ; is++)
	  for (int js = is + 1 ; js < QC_Ns ; js++) {
            int i1 = r_stride * QC_LIDX_P(ic, is, jc, js);
            int i2 = r_stride * QC_LIDX_P(ic, js, jc, is);
            QC_CPLX aux = r[i1];  r[i1] = r[i2];  r[i2] = aux;
	  }
  }
  /* r[ic,is,jc,js] <- x[ic,js,jc,is] */
  inline __device__ void
    QC(P_eq_spintranspose_P)(QC_CPLX *r, int r_stride, QC_CPLX *x, int x_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++)
	    r[r_stride * QC_LIDX_P(ic, js, jc, is)] = 
	      x[x_stride * QC_LIDX_P(ic, is, jc, js)];  
  }
  /* r[ic,is,jc,js] <- x[jc,js,ic,is]^* */
  inline __device__ void
    QC(P_eq_aP)(QC_CPLX *r, int r_stride, const QC_CPLX *x, int x_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    r[r_stride*QC_LIDX_P(ic,is,jc,js)] = 
	      QC_CONJ(x[x_stride*QC_LIDX_P(jc,js,ic,is)]);
	  }
  }
  /* r[ic,is,jc,js] <- (g5.x^\dag.g5)[ic,is,jc,js] */
  inline __device__ void
    QC(P_eq_hP)(QC_CPLX *r, int r_stride, const QC_CPLX *x, int x_stride)
  {
    QC_CPLX tmpP1[QC_LEN_P];
    QC(cplx_vec_zero)(r,r_stride, QC_LEN_P);
    QC(P_eq_aP)(r,r_stride, x,x_stride);
    QC(cplx_vec_zero)(tmpP1,1, QC_LEN_P);
    QC(agx_pby_gind_P)(1., 15, r,r_stride, 0., tmpP1,1);
    QC(axg_pby_gind_P)(1., 15, tmpP1,1, 0., r,r_stride);
  }
  /* y <- a*x + y */
  inline __device__ void
    QC(ax_py_P)(QC_CPLX a, const QC_CPLX *x, int x_stride, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int i = QC_LIDX_P(ic, is, jc, js);
	    y[y_stride * i] += a * x[x_stride * i];
	  }
  }
  /* y <- a*spintranspose(x) + y */
  inline __device__ void
    QC(axTs_py_P)(QC_CPLX a, const QC_CPLX *x, int x_stride, QC_CPLX *y, int y_stride)
  {
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++)
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++) {
	    int ix  = x_stride * QC_LIDX_P(ic, is, jc, js);
	    int iy  = y_stride * QC_LIDX_P(ic, js, jc, is);
	    y[iy] += a * x[ix];
	  }
  }
  /* Tr[x . y] */
  inline __device__ QC_CPLX 
    QC(trace_P_dot_P)(
		      const QC_CPLX *x, int x_stride,
		      const QC_CPLX *y, int y_stride) 
  {
    QC_CPLX s = 0;
    for (int jc = 0 ; jc < QC_Nc ; jc++)
      for (int js = 0 ; js < QC_Ns ; js++) 
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int is = 0 ; is < QC_Ns ; is++)
	    s += x[x_stride*QC_LIDX_P(ic, is, jc, js)] 
	      *y[y_stride*QC_LIDX_P(jc, js, ic, is)];

    return s;
  }
  /* Tr[x . y^\dag] */
  inline __device__ QC_CPLX 
    QC(trace_P_dot_aP)(
		       const QC_CPLX *x, int x_stride,
		       const QC_CPLX *y, int y_stride) 
  {
    QC_CPLX s = 0;
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) 
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    s +=         x[x_stride*QC_LIDX_P(ic, is, jc, js)] 
	      *QC_CONJ(y[y_stride*QC_LIDX_P(ic, is, jc, js)]);

    return s;
  }
  /* Tr[x^\dag . y] ; FIXME reuse P_dot_aP instead */
  inline __device__ QC_CPLX 
    QC(trace_aP_dot_P)(
		       const QC_CPLX *x, int x_stride,
		       const QC_CPLX *y, int y_stride) 
  {
    QC_CPLX s = 0;
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) 
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    s += QC_CONJ(x[x_stride*QC_LIDX_P(jc, js, ic, is)])
	      *        y[y_stride*QC_LIDX_P(jc, js, ic, is)];

    return s;
  }
  /* Tr[x^\dag . y^\dag] ; FIXME reuse P_dot_P instead */
  inline __device__ QC_CPLX 
    QC(trace_aP_dot_aP)(
			const QC_CPLX *x, int x_stride,
			const QC_CPLX *y, int y_stride) 
  {
    QC_CPLX s = 0;
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) 
	for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    s += QC_CONJ(x[x_stride*QC_LIDX_P(jc, js, ic, is)]) 
	      *QC_CONJ(y[y_stride*QC_LIDX_P(ic, is, jc, js)]);

    return s;
  }

  /* gres[is,js] <- Tr_c[x . y] = \sum_{ic,jc,ks} x[ic,is,jc,ks] * y[jc,ks,ic,js] */
  inline __device__ void 
    QC(G_peqa_colortrace_P_dot_P)(
				  QC_CPLX *gres, int gres_stride, QC_CPLX a,
				  const QC_CPLX *x, int x_stride,
				  const QC_CPLX *y, int y_stride) 
  {
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) {
        QC_CPLX s = 0;
        for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    for (int ks = 0 ; ks < QC_Ns ; ks++) {
	      s += x[x_stride*QC_LIDX_P(ic, is, jc, ks)] 
		* y[y_stride*QC_LIDX_P(jc, ks, ic, js)];
	    }
        gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }
  /* gres[is,js] <- Tr_c[x . y^\dag] = \sum_{ic,jc,ks} x[ic,is,jc,ks] * conj(y[ic,js,jc,ks]) */
  inline __device__ void 
    QC(G_peqa_colortrace_P_dot_aP)(
				   QC_CPLX *gres, int gres_stride, QC_CPLX a,
				   const QC_CPLX *x, int x_stride,
				   const QC_CPLX *y, int y_stride) 
  {
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) {
        QC_CPLX s = 0;
        for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    for (int ks = 0 ; ks < QC_Ns ; ks++) {
	      s +=         x[x_stride*QC_LIDX_P(ic, is, jc, ks)]
		* QC_CONJ(y[y_stride*QC_LIDX_P(ic, js, jc, ks)]);
	    }
        gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }
  /* gres[is,js] <- Tr_c[x^\dag . y] = \sum_{ic,jc,ks} conj(x[jc,ks,ic,is]) * y[jc,ks,ic,js]
     FIXME reuse P_dot_aP + conj + transpose */
  inline __device__ void 
    QC(G_peqa_colortrace_aP_dot_P)(
				   QC_CPLX *gres, QC_CPLX a, int gres_stride,
				   const QC_CPLX *x, int x_stride,
				   const QC_CPLX *y, int y_stride) 
  {
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) {
        QC_CPLX s = 0;
        for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    for (int ks = 0 ; ks < QC_Ns ; ks++) {
	      s += QC_CONJ(x[x_stride*QC_LIDX_P(jc, ks, ic, is)])
		*         y[y_stride*QC_LIDX_P(jc, ks, ic, js)];
	    }
        gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }
  /* gres[is,js] <- Tr_c[x^\dag . y] = \sum_{ic,jc,ks} conj(x[jc,ks,ic,is]) * conj(y[ic,js,jc,ks])
     FIXME reuse P_dot_P + conj + transpose */
  inline __device__ void 
    QC(G_peqa_colortrace_aP_dot_aP)(
				    QC_CPLX *gres, int gres_stride, QC_CPLX a,
				    const QC_CPLX *x, int x_stride,
				    const QC_CPLX *y, int y_stride) 
  {
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++) {
        QC_CPLX s = 0;
        for (int ic = 0 ; ic < QC_Nc ; ic++)
	  for (int jc = 0 ; jc < QC_Nc ; jc++)
	    for (int ks = 0 ; ks < QC_Ns ; ks++) {
	      s += QC_CONJ(x[x_stride*QC_LIDX_P(jc, ks, ic, is)])
		* QC_CONJ(y[y_stride*QC_LIDX_P(ic, js, jc, ks)]);
	    }
        gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }

  /* Tr_s[Gamma[gn] . g1] */
  inline __device__ QC_CPLX
    QC(trace_gamma_dot_G)(int ng, const QC_CPLX *g1, int g1_stride)
  {
    QC_CPLX s = 0;
    for (int is = 0 ; is < QC_Ns ; is++) {
      int js = gamma_left_ind(ng,is);
      s += gamma_left_coeff(ng,is) * g1[g1_stride * QC_LIDX_G(js, is)];
    }
    return s;
  }
  /* res[ng] <- Tr_s[Gamma[ng].g1] */
  inline __device__ void
    QC(gvec_eq_trace_gamma_dot_G)(
				  QC_CPLX *gres, int gres_stride, 
				  const QC_CPLX *g1, int g1_stride)
  {
    for (int ng = 0 ; ng < QC_LEN_G ; ng++) 
      gres[gres_stride * ng] = QC(trace_gamma_dot_G)(ng, g1,g1_stride);
  }

  /* compute res[gn] = Tr [Gamma[gn] * F * B] */
  __device__ void 
    QC(contract_tr_g_P_P)(
			  QC_CPLX *gres, int gres_stride,
			  const QC_CPLX *F, int F_stride,
			  const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);
    QC(G_peqa_colortrace_P_dot_P)(tmpG1,1, 1., F,F_stride, B,B_stride);
    QC(gvec_eq_trace_gamma_dot_G)(gres,gres_stride, tmpG1,1);
  }
  /* compute res[gn] = Tr [Gamma[gn] * F * B^\dag] */
  __device__ void 
    QC(contract_tr_g_P_aP)(
			   QC_CPLX *gres, int gres_stride,
			   const QC_CPLX *F, int F_stride,
			   const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);
    QC(G_peqa_colortrace_P_dot_aP)(tmpG1,1, 1., F,F_stride, B,B_stride);
    QC(gvec_eq_trace_gamma_dot_G)(gres,gres_stride, tmpG1,1);
  }
  /* compute res[gn] = Tr [Gamma[gn] * F * g5.B^\dag.g5] */
  __device__ void 
    QC(contract_tr_g_P_hP)(
			   QC_CPLX *gres, int gres_stride,
			   const QC_CPLX *F, int F_stride,
			   const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpG1[QC_LEN_G];
    QC_CPLX tmpP1[QC_LEN_P];
    QC(P_eq_hP)(tmpP1,1, B,B_stride);
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);
    QC(G_peqa_colortrace_P_dot_P)(tmpG1,1, 1., F,F_stride, tmpP1,1);
    QC(gvec_eq_trace_gamma_dot_G)(gres,gres_stride, tmpG1,1);
  }

  inline __device__ void 
    QC(GG_eq_colortrace_P_P)(
			     QC_CPLX *gg, int gg_stride, 
			     const QC_CPLX *F, int F_stride, 
			     const QC_CPLX *B, int B_stride) {
#define QC_LIDX_GG(i1,i2,i3,i4) ((i4) + QC_Ns*((i3)+QC_Ns*((i2)+QC_Ns*(i1))))
    for (int i1 = 0 ; i1 < QC_Ns ; i1++)
      for (int i2 = 0 ; i2 < QC_Ns ; i2++)
	for (int i3 = 0 ; i3 < QC_Ns ; i3++)
	  for (int i4 = 0 ; i4 < QC_Ns ; i4++) {
	    QC_CPLX s = 0.;
	    for (int ic = 0 ; ic < QC_Nc ; ic++)
	      for (int jc = 0 ; jc < QC_Nc ; jc++)
		s +=  F[F_stride*QC_LIDX_P(ic,i1,jc,i2)] 
		  * B[B_stride*QC_LIDX_P(jc,i3,ic,i4)];
	    gg[gg_stride*QC_LIDX_GG(i1,i2,i3,i4)] = s;
	  }
  }
  inline __device__ void 
    QC(G_eq_diag_GG)(
		     QC_CPLX *g, int g_stride, 
		     const QC_CPLX *gg, int gg_stride) {
    for (int ng = 0 ; ng < QC_LEN_G ; ng++){
      QC_CPLX s = 0.;
      for (int is = 0 ; is < QC_Ns ; is++) {
	QC_CPLX ci = gamma_left_coeff(ng,is);
	int ks = gamma_left_ind(ng,is);
	QC_CPLX s1 = 0.;
	for (int js = 0 ; js < QC_Ns ; js++) {
	  QC_CPLX cj = gamma_left_coeff(ng,js);
	  int ls = gamma_left_ind(ng,js);
	  s1 += cj*gg[gg_stride*QC_LIDX_GG(ks,js,ls,is)];
	}
	s += ci * s1;
      }
      g[g_stride * ng] = s;
    }
  }

  /* compute res[gn] = Tr [Gamma[gn] * F * (-g4.Gamma[gn]^\dag.g4) * B] */
  __device__ void
    QC(contract_tr_g_P_mgbar_P)(
				QC_CPLX *gres, int gres_stride,
				const QC_CPLX *F, int F_stride,
				const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpGG[QC_LEN_GG];
    QC(GG_eq_colortrace_P_P)(tmpGG,1, F,F_stride, B,B_stride);
    QC(G_eq_diag_GG)(gres, gres_stride, tmpGG,1);
    for (int ng = 0 ; ng < QC_LEN_G ; ng++) {
      int mgbar_sign = -(1 - 2*qc_gamma_adj_parity(ng)) 
	* (1 - 2*qc_gamma_uni_parity(ng,8));
      gres[gres_stride * ng] *= mgbar_sign;
    }
  }
  /* compute res[gn] = Tr [Gamma[gn] * F * (-g4.Gamma[gn]^\dag.g4) * B^\dag] */
  __device__ void
    QC(contract_tr_g_P_mgbar_aP)(
				 QC_CPLX *gres, int gres_stride,
				 const QC_CPLX *F, int F_stride,
				 const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpP1[QC_LEN_P];
    QC(P_eq_aP)(tmpP1,1, B,B_stride);
    QC(contract_tr_g_P_mgbar_P)(gres, gres_stride, F, F_stride, tmpP1,1);
  }
  /* compute res[gn] = Tr [Gamma[gn] * F * (-g4.Gamma[gn]^\dag.g4) * g5.B^\dag.g5] */
  __device__ void
    QC(contract_tr_g_P_mgbar_hP)(
				 QC_CPLX *gres, int gres_stride,
				 const QC_CPLX *F, int F_stride,
				 const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpP1[QC_LEN_P];
    QC(P_eq_hP)(tmpP1,1, B,B_stride);
    QC(contract_tr_g_P_mgbar_P)(gres, gres_stride, F, F_stride, tmpP1,1);
  }


  __device__ void 
    QC(baryon_sigma_seqsource_u)(
				 QC_CPLX *r, int r_stride,
				 const QC_CPLX *Fu, int Fu_stride, 
				 const QC_CPLX *Fd, int Fd_stride,
				 const QC_CPLX *T_gvec)
  {
    //  Fu, Fd      u and d quark propagators (can be the same ptr)
    //  S2, S1      diquark spin structure in the nucleon operators 
    //              at the sink and the source, respectively
    //  Tpol        nucleon polarization matrix
    //  *** Qlua equivalent code:
    //  local ut      = Fu * T
    //  local s1bar   = gamma{mu=3} * S1:adjoin() * gamma{mu=3}:transpose()
    //  local s2ds1b  = S2 * Fd * s1bar
    //  return (  T * L:DiracPropagator(qcd.quarkContract24(Fu, s2ds1b):spintrace())
    //          + qcd.quarkContract12(ut, s2ds1b):spintranspose()
    //          + qcd.quarkContract24(T * Fu, s2ds1b)
    //          + qcd.quarkContract13(s2ds1b, ut)
    //         )
    QC_CPLX ut[QC_LEN_P], 
      tmpP1[QC_LEN_P],
      tmpP2[QC_LEN_P],
      s2ds1b[QC_LEN_P];
    QC_CPLX S1bar_gvec[QC_LEN_G];

    QC(cplx_vec_zero)(r,r_stride, QC_LEN_P);

    // ut <- Fu . T
    QC(cplx_vec_zero)(ut,1, QC_LEN_P);
    QC(axg_pby_gvec_P)(1., T_gvec, Fu,Fu_stride, 0., ut,1);
    // S1bar <- g4 . S1^dag . g4^T
    QC(cplx_vec_copy)(S1bar_gvec,1, cS1_gvec,1, QC_LEN_G);
    QC(gvec_adj)(S1bar_gvec);
    QC(gvec_sim_transf)(S1bar_gvec, 8);
    // s2ds1b <- S2 . Fd . S1bar
    QC(cplx_vec_zero)(tmpP1,1, QC_LEN_P);
    QC(agx_pby_gvec_P)(1., cS2_gvec,    Fd,Fd_stride, 0., tmpP1,1);
    QC(cplx_vec_zero)(s2ds1b,1, QC_LEN_P);
    QC(axg_pby_gvec_P)(1., S1bar_gvec, tmpP1,1,      0., s2ds1b,1);
    
    // (1)res += T*qC24(Fu, s2ds1b):spintrace
    QC(quarkContract24)(tmpP1,1, Fu,Fu_stride, s2ds1b,1);
    QC(P_eq_spintrace_P)(tmpP2,1, tmpP1,1);
    QC(agx_pby_gvec_P)(1., T_gvec, tmpP2,1,   0, r,r_stride);

    // (2)res += qC12(ut, s2ds1b):spintranspose
    QC(quarkContract12)(tmpP1,1, ut,1, s2ds1b,1);
    QC(axTs_py_P)(1., tmpP1,1, r,r_stride);

    // (3)res += qC24(T*Fu, s2ds1b)
    QC(cplx_vec_zero)(tmpP1,1, QC_LEN_P);
    QC(agx_pby_gvec_P)(1., T_gvec, Fu,Fu_stride, 0., tmpP1,1);
    QC(quarkContract24)(tmpP2,1, tmpP1,1, s2ds1b,1);
    QC(ax_py_P)(1., tmpP2,1, r,r_stride);

    // (4)res +=qC13(s2ds1b, ut)
    QC(quarkContract13)(tmpP1,1, s2ds1b,1, ut,1);
    QC(ax_py_P)(1., tmpP1,1, r,r_stride);
  }


  /* gres[is,js] <- Tr_c[u . x . y] = \sum_{ic,kc,jc,ks} u[ic,jc] * x[jc,is,kc,ks] * y[kc,ks,kc,is] */
  inline __device__ void 
  QC(G_peqa_colortrace_U_dot_P_dot_P)(
				      QC_CPLX *gres, int gres_stride, QC_CPLX a,
				      const QC_CPLX *u, int u_stride,
				      const QC_CPLX *x, int x_stride,
				      const QC_CPLX *y, int y_stride) 
  {
#pragma unroll
    for (int is = 0 ; is < QC_Ns ; is++)
#pragma unroll
      for (int js = 0 ; js < QC_Ns ; js++) {
	QC_CPLX s = 0;
#pragma unroll
	for (int ic = 0 ; ic < QC_Nc ; ic++)
#pragma unroll
	  for (int jc = 0 ; jc < QC_Nc ; jc++) {
	    QC_CPLX t = 0;
#pragma unroll
	    for (int kc = 0 ; kc < QC_Nc ; kc++)
#pragma unroll
	      for (int ks = 0 ; ks < QC_Ns ; ks++) {
		t += x[x_stride*QC_LIDX_P(jc, is, kc, ks)] 
		  * y[y_stride*QC_LIDX_P(kc, ks, ic, js)];
	      }
	    s += u[u_stride*QC_LIDX_M(ic, jc)] * t;
	  }
	gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }

  /* compute res[gn] = Tr [Gamma[gn] * F * B] */
  __device__ void 
  QC(contract_tr_g_U_P_P)(
			  QC_CPLX *gres, int gres_stride,
			  const QC_CPLX *U, int U_stride,
			  const QC_CPLX *F, int F_stride,
			  const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);
    QC(G_peqa_colortrace_U_dot_P_dot_P)(tmpG1,1, 1., U,U_stride, F,F_stride, B,B_stride);
    QC(gvec_eq_trace_gamma_dot_G)(gres,gres_stride, tmpG1,1);
  }
  

  __device__ void 
    QC(baryon_sigma_seqsource_d)(
				 QC_CPLX *r, int r_stride,
				 const QC_CPLX *Fu1, int Fu1_stride,
				 const QC_CPLX *Fu2, int Fu2_stride,
				 const QC_CPLX *T_gvec)
  {
    //  Fu1, Fu2    are u quark propagators (can be the same ptr)
    //              enumerated according to the source (ubar) quark 
    //              in the nucleon source
    //  S2, S1      diquark spin structure in the nucleon operators 
    //              at the sink and the source, respectively
    //  *** Qlua equivalent code:
    //  Tpol        nucleon polarization matrix
    //  local L       = Fu1.lattice
    //  local u1t     = Fu1 * T
    //  local s1bar   = gamma{mu=3} * S1:adjoin() * gamma{mu=3}:transpose()
    //  return s1bar * (  qcd.quarkContract12(u1t, Fu2)
    //           + qcd.quarkContract23(u1t, Fu2) ):spintranspose() * S2

    QC_CPLX u1t[QC_LEN_P], 
      tmp1[QC_LEN_P],
      tmp2[QC_LEN_P];
    QC_CPLX S1bar_gvec[QC_LEN_G];

    QC(cplx_vec_zero)(r,r_stride, QC_LEN_P);

    // u1t <- Fu1 . T
    QC(cplx_vec_zero)(u1t,1, QC_LEN_P);
    QC(axg_pby_gvec_P)(1., T_gvec, Fu1,Fu1_stride, 0., u1t,1);
    // S1bar <- g4 . S1^dag . g4^T
    QC(cplx_vec_copy)(S1bar_gvec,1, cS1_gvec,1, QC_LEN_G);
    QC(gvec_adj)(S1bar_gvec);
    QC(gvec_sim_transf)(S1bar_gvec, 8);

    // tmp1 <- qC12(u1t, Fu2) + qC23(u1t, Fu2)
    QC(quarkContract12)(tmp1,1, u1t,1, Fu2,Fu2_stride);
    QC(quarkContract23)(tmp2,1, u1t,1, Fu2,Fu2_stride);
    QC(ax_py_P)(1., tmp2,1, tmp1,1);
    QC(spintranspose_P)(tmp1,1);

    QC(cplx_vec_zero)(tmp2,1, QC_LEN_P);
    QC(agx_pby_gvec_P)(1., S1bar_gvec,   tmp1,1, 0., tmp2,1);
    QC(cplx_vec_zero)(r,r_stride, QC_LEN_P);
    QC(axg_pby_gvec_P)(1., cS2_gvec,      tmp2,1, 0., r,r_stride);
  }


  __device__ void 
    QC(baryon_sigma_twopt_asymsrc_gvec)(
					QC_CPLX *r, int r_stride,
					const QC_CPLX *Fu1, int Fu1_stride,
					const QC_CPLX *Fu2, int Fu2_stride,
					const QC_CPLX *Fd,  int Fd_stride)
  {
    //  general nucleon propagator with "asymmetric" u-quarks at the source
    //  (asymmetry is fictional, needed to lift degeneracy to have a right
    //  to enumerate quarks at the source so that one of the u-propagators
    //  can be distinct from the other, e.g. if it is qbarq-insertion-sequential)
    //     source  = (dbar S1bar ubar2^T) ubar1
    //     sink    = u (u^T S2 d)
    //   Fu1 = <u ubar1>, Fu2 = <u ubar2>, Fd = <d dbar>
    //   S2, S1      diquark spin structure in the nucleon operators 
    //               at the sink and the source, respectively
    //   Tpol        nucleon polarization matrix
    //  *** Qlua equivalent code:
    //  local ut      = Fu1 * T
    //  local S1bar   = gamma{mu=3} * S1:adjoin() * gamma{mu=3}:transpose()
    //  local us2ds1b = qcd.quarkContract24(Fu2, S2 * Fd * S1bar)
    //  return (   (u1t:spintrace() * us2ds1b:spintrace()):trace()
    //           + (u1t * us2ds1b):trace() )

    QC_CPLX tmpP1[QC_LEN_P],
      tmpP2[QC_LEN_P];
    QC_CPLX S1bar_gvec[QC_LEN_G],
      tmpG[QC_LEN_G];         /* can reuse S1bar_gvec */
    
    // S1bar <- g4 . S1^dag . g4^T
    QC(cplx_vec_copy)(S1bar_gvec,1, cS1_gvec,1, QC_LEN_G);
    QC(gvec_adj)(S1bar_gvec);
    QC(gvec_sim_transf)(S1bar_gvec, 8);

    // tmp1 <- qC24(u2, s2*d*s1b)
    QC(cplx_vec_zero)(tmpP1,1, QC_LEN_P);
    QC(agx_pby_gvec_P)(1., cS2_gvec, Fd,Fd_stride, 0., tmpP1,1);
    QC(cplx_vec_zero)(tmpP2,1, QC_LEN_P);
    QC(axg_pby_gvec_P)(1., S1bar_gvec, tmpP1,1,   0., tmpP2,1);
    QC(quarkContract24)(tmpP1,1, Fu2,Fu2_stride, tmpP2,1);
    // tmp1 <- tmp1 + tmp1:spintrace(), 
    QC(P_eq_spintrace_P)(tmpP2,1, tmpP1,1);
    QC(ax_py_P)(1., tmpP2,1, tmpP1,1);

    // res[ng] <- Tr[Gamma[ng] . tmp1 . Fu1]
    QC(cplx_vec_zero)(tmpG,1, QC_LEN_G);
    QC(G_peqa_colortrace_P_dot_P)(tmpG,1, 1., tmpP1,1, Fu1,Fu1_stride);
    QC(gvec_eq_trace_gamma_dot_G)(r,r_stride, tmpG,1);
  }

  /* gres[is,js] <- Tr_c[x . y^H] = \sum_{ic} x[ic,is] * conj(y[ic,js]) */
  //- Version 1
#if 0
  inline __device__ void
  QC(G_peqa_colortrace_D_dot_aD)(QC_CPLX *gres, int gres_stride, QC_CPLX a,
                                 const Vector& x, const Vector& y){
    
    for (int is = 0 ; is < QC_Ns ; is++)
      for (int js = 0 ; js < QC_Ns ; js++){
	QC_CPLX s = 0;
#pragma unroll
	for (int ic = 0 ; ic < QC_Nc ; ic++){
	  QC_CPLX yD = y(js,ic);
	  s += x(is,ic) * conj(yD);
	}
	gres[gres_stride * QC_LIDX_G(is, js)] += a*s;
      }
  }
#endif
  //------------------------------------

  //- Version 2
#if 1
  inline __device__ void
  QC(G_peqa_colortrace_D_dot_aD)(QC_CPLX *gres, int gres_stride, QC_CPLX a,
                                 const Vector& x, const Vector& y){
#pragma unroll
    for (int ic = 0 ; ic < QC_Nc ; ic++)
#pragma unroll
      for (int is = 0 ; is < QC_Ns ; is++)
#pragma unroll
	for (int js = 0 ; js < QC_Ns ; js++){
	  gres[gres_stride * QC_LIDX_G(is, js)] += a * x(is,ic) * conj(y(js,ic));
	  //	             a * x.data[LIDX_Vector(ic,is)] * conj(y.data[LIDX_Vector(ic,js)]);
	}
  }
#endif
  //------------------------------------


  //- Version 3
#if 0
  inline __device__ void
  QC(G_peqa_colortrace_D_dot_aD)(QC_CPLX *gres, int gres_stride, QC_CPLX a,
				 const Vector& x, const Vector& y){
    QC_CPLX gres1[QC_LEN_G];
#pragma unroll
    for (int i = 0 ; i < QC_LEN_G ; i++) 
      gres1[i] = 0;
    
#pragma unroll
    for (int ic = 0 ; ic < QC_Nc ; ic++)
#pragma unroll
      for (int is = 0 ; is < QC_Ns ; is++)
#pragma unroll
	for (int js = 0 ; js < QC_Ns ; js++) {
	  gres1[gres_stride * QC_LIDX_G(is, js)] += x(is,ic) * conj(y(js,ic));
	    //	    x.data[LIDX_Vector(ic,is)] * conj(y.data[LIDX_Vector(ic,js)]);
	}
    
#pragma unroll
    for (int i = 0 ; i < QC_LEN_G ; i++)
      gres[i] += a * gres1[i];
  }
#endif
  //------------------------------------



  /* gres[is,js] += (Tr_c[u . x . y^H] = \sum_{ic,jc} u[jc,ic] * x[ic,is] * conj(y[jc,js])) */
  inline __device__ void
  QC(G_peqa_colortrace_U_dot_D_dot_aD)(
				       QC_CPLX *gres, int gres_stride, QC_CPLX a,
				       const Link& U, const Vector& x, const Vector& y)
  {
    Vector u_x = U * x;
    QC(G_peqa_colortrace_D_dot_aD)(gres, gres_stride, a, u_x, y);
  }

  //---------------------------------------------------------------------------//
  // U T I L I T Y   F U N C T I O N S   A N D   K E R N E L   W R A P P E R S //
  //---------------------------------------------------------------------------//

  void copySmatricesToSymbol(complex<QC_REAL> *S2, complex<QC_REAL> *S1){
    cudaMemcpyToSymbol(cS2_gvec, S2, sizeof(complex<QC_REAL>)*QC_LEN_G);
    cudaMemcpyToSymbol(cS1_gvec, S1, sizeof(complex<QC_REAL>)*QC_LEN_G);
  }

  void qcCopyGammaToSymbol(qcTMD_gamma gamma_h){
    cudaMemcpyToSymbol(cGamma, &gamma_h, sizeof(qcTMD_gamma));
  }


  __device__ void prepareDevicePropSite(complex<QC_REAL> *devProp, Vector *vec, bool preserveBasis){

    const int Ns = QC_Ns;
    const int Nc = QC_Nc;

    if(!preserveBasis)
      rotateVectorBasis(vec,QLUA_quda2qdp); //-- Rotate basis back to the QDP conventions

    for(int jc = 0; jc < Nc; jc++){
      for(int js = 0; js < Ns; js++){
        int vIdx = js + Ns*jc;     //-- vector index (which vector within propagator)
        for(int ic = 0; ic < Nc; ic++){
          for(int is = 0; is < Ns; is++){
            int dIdx = ic + Nc*is; //-- spin-color index within each vector

            int pIdx = QC_QUDA_LIDX_P(ic,is,jc,js);

            devProp[pIdx] = vec[vIdx].data[dIdx];
          }}}
    }//-jc

  }//-- prepareDevicePropSite


  __device__ void prepareDeviceLinkSite(complex<QC_REAL> *devLink, Link U){

    const int Nc = QC_Nc;

    for(int ic = 0; ic < Nc; ic++){
      for(int jc = 0; jc < Nc; jc++){
	int uIdx = QC_QUDA_LIDX_M(ic, jc);
	devLink[uIdx] = U(ic,jc);
      }
    }

  }//-- prepareDeviceLinkSite


  //--------------------------------------------------------------------------------------
  //--------------------------------------------------------------------------------------
  //-- Kernel wrappers

  __global__ void baryon_sigma_twopt_asymsrc_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];
    complex<QC_REAL> prop3[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    Vector vec3[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
      vec3[i] = arg->prop3[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);
    prepareDevicePropSite(prop3, vec3, arg->preserveBasis);

    qc_quda_baryon_sigma_twopt_asymsrc_gvec(Corr_dev + tid, lV,
                                            prop1, 1,
                                            prop2, 1,
                                            prop3, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void qbarq_g_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV  = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_P(Corr_dev + tid, lV,
			      prop1, 1,
			      prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void qbarq_g_P_aP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_aP(Corr_dev + tid, lV,
			       prop1, 1,
			       prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  /* local qbarq contractions */
  __global__ void qbarq_g_P_aP_gvec_kernel_vecByVec_preserveBasisTrue(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);

    for(int i=0;i<QUDA_PROP_NVEC/*TODO replace with nvec for disconnected contractions*/;i++){
      /* assuming the gamma basis is always correct */
      Vector vec1 = arg->prop1[i](x_cb, pty);
      Vector vec2 = arg->prop2[i](x_cb, pty);
      QC(G_peqa_colortrace_D_dot_aD)(Corr_dev + tid, lV, 1., vec1, vec2);
    }

    /* project onto Gamma(n=0..15)-matrices */
    QC(gvec_eq_trace_gamma_dot_G)(Corr_dev, lV, tmpG1, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void qbarq_g_P_hP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_hP(Corr_dev + tid, lV,
			       prop1, 1,
			       prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void meson_F_B_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_mgbar_P(Corr_dev + tid, lV,
				    prop1, 1,
				    prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void meson_F_aB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_mgbar_aP(Corr_dev + tid, lV,
				     prop1, 1,
				     prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  __global__ void meson_F_hB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> prop1[QC_LEN_P];
    complex<QC_REAL> prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->prop1[i](x_cb, pty);
      vec2[i] = arg->prop2[i](x_cb, pty);
    }
    prepareDevicePropSite(prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(prop2, vec2, arg->preserveBasis);

    qc_quda_contract_tr_g_P_mgbar_hP(Corr_dev + tid, lV,
				     prop1, 1,
				     prop2, 1);
  }

  //------------------------------------------------------------------------------------------
  __global__ void tmd_g_U_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> dev_prop1[QC_LEN_P];
    complex<QC_REAL> dev_prop2[QC_LEN_P];
    complex<QC_REAL> dev_link[QC_LEN_M];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->fwdProp[i](x_cb, pty);
      vec2[i] = arg->bwdProp[i](x_cb, pty);
    }
    prepareDevicePropSite(dev_prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(dev_prop2, vec2, arg->preserveBasis);

    Link U = arg->U(arg->i_mu, x_cb, pty);

    prepareDeviceLinkSite(dev_link, U);

    qc_quda_contract_tr_g_U_P_P(Corr_dev + tid, lV,
				dev_link,  1,
				dev_prop1, 1,
				dev_prop2, 1);
  }
  __global__ void tmd_g_U_P_P_gvec_kernel_gaugeExt(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> dev_prop1[QC_LEN_P];
    complex<QC_REAL> dev_prop2[QC_LEN_P];
    complex<QC_REAL> dev_link[QC_LEN_M];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = arg->fwdProp[i](x_cb, pty);
      vec2[i] = arg->bwdProp[i](x_cb, pty);
    }
    prepareDevicePropSite(dev_prop1, vec1, arg->preserveBasis);
    prepareDevicePropSite(dev_prop2, vec2, arg->preserveBasis);

    Link U;
    {      
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];
	
      U = arg->U(arg->i_mu, linkIndex(c2, arg->dimEx), pty);
    }

    prepareDeviceLinkSite(dev_link, U);

    qc_quda_contract_tr_g_U_P_P(Corr_dev + tid, lV,
				dev_link,  1,
				dev_prop1, 1,
				dev_prop2, 1);
  }
  //------------------------------------------------------------------------------------------

  /* local qbarUq contractions for TMD */
  __global__ void tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    Link U = arg->U(arg->i_mu, x_cb, pty);

    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);

    for(int i=0;i<QUDA_PROP_NVEC;i++){
      Vector vec1 = arg->fwdProp[i](x_cb, pty);
      Vector vec2 = arg->bwdProp[i](x_cb, pty);
      //      QC(G_peqa_colortrace_U_dot_D_dot_aD)(Corr_dev + tid, lV, 1., U, vec1, vec2);
      QC(G_peqa_colortrace_U_dot_D_dot_aD)(tmpG1, 1., 1., U, vec1, vec2);
    }

    /* project onto Gamma(n=0..15)-matrices */
    QC(gvec_eq_trace_gamma_dot_G)(Corr_dev + tid, lV, tmpG1, 1);

  }
  //------------------------------------------------------------------------------------------


  __global__ void tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue_gaugeExt(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * arg->volumeCB;
    int lV   = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;
    if(tid >= lV) return;

    Link U;
    { /* this condition is the same for all threads;
			       can this cause thread divergence? */
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];

      U = arg->U(arg->i_mu, linkIndex(c2, arg->dimEx), pty);
    }

    QC_CPLX tmpG1[QC_LEN_G];
    QC(cplx_vec_zero)(tmpG1, 1, QC_LEN_G);

#pragma unroll
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      Vector vec1 = arg->fwdProp[i](x_cb, pty);
      Vector vec2 = arg->bwdProp[i](x_cb, pty);
      QC(G_peqa_colortrace_U_dot_D_dot_aD)(tmpG1, 1., 1., U, vec1, vec2);
    }

    /* project onto Gamma(n=0..15)-matrices */
    QC(gvec_eq_trace_gamma_dot_G)(Corr_dev + tid, lV, tmpG1, 1);

  }
  //------------------------------------------------------------------------------------------


  extern __shared__ complex<QC_REAL> qc_U_D_aD_shared[];


  inline __device__ const qcTMD_gamma* gammaMat() {
    return reinterpret_cast<const qcTMD_gamma*>(cGamma);
  }


  /* local qbarUq contractions for TMD */
  /* compute (res)[is,js] += a * \sum_{jc} conj(v2[jc,js]) * (\sum_{ic} u[jc,ic] v1[ic,is]) */
  __global__ void tmd_g_U_D_aD_gvec_kernel_shmem16_VecByVec_preserveBasisTrue(
									      complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg)
  {
    /* blockDim = {*SITES_PER_BLOCK, parity ...} */
    /* site/parity */
    int x_cb  = blockIdx.x*blockDim.x + threadIdx.x;    /* CB site within vol4loc */
    int pty   = blockIdx.y*blockDim.y + threadIdx.y;    /* parity within vol4loc */
    int tid   = x_cb + pty * arg->volumeCB;             /* index within result buffer */
    int lV    = arg->volume;

    if (x_cb >= arg->volumeCB) return;
    if (pty  >= arg->nParity) return;
    if (tid  >= lV) return;

    /* internal indices for threads working on this site */
    const int i12   = threadIdx.z;
    int i1    = i12 / QC_Ns ;
    int i2    = i12 % QC_Ns ;

    /* shmem storage for v1,v2,u,gres */
    int isite_blk = threadIdx.y * blockDim.x + threadIdx.x;
    complex<QC_REAL> *v1    = (complex<QC_REAL> *)&(
			qc_U_D_aD_shared[QC_UDD_SITE_CPLXBUF * isite_blk]) ;
    complex<QC_REAL> *v2    = v1 + QC_LEN_D ;
    complex<QC_REAL> *u_v1  = v2 + QC_LEN_D ;
    complex<QC_REAL> *gres  = u_v1 + QC_LEN_D ;
    complex<QC_REAL> *umat  = gres + QC_LEN_G ;

    const qcTMD_gamma *gamma = gammaMat();

    /* index into GaugeField */
    int ulink_idx;
    { /* speculative calc for extendedGauge */
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];
      ulink_idx = linkIndex(c2, arg->dimEx);
    }
    ulink_idx = arg->extendedGauge ? ulink_idx : x_cb;
    
    /* load umat */
#if 0
    if (i1 < QC_Nc && i2 < QC_Nc)
      umat[QC_LIDX_M(i1,i2)] = arg->U.getData(arg->i_mu, ulink_idx, pty, i1, i2);
    //__syncthreads(); /* no need until the 1st pair v1,v2 are also read */
#else
    if (0 == i12) {
      *(reinterpret_cast<Link *>(umat)) = arg->U(arg->i_mu, ulink_idx, pty);
    }
#endif
    gres[QC_LIDX_G(i1, i2)] = 0.;
    /* loop over vectors */
#pragma unroll
    for (int i = 0 ; i < QUDA_PROP_NVEC /*TODO replace with nvec for disconnected contractions*/ ; i++){
      /* load v1, v2 */
#if 0
#define QC_LIDX_D_TR(ic, is) QC_LIDX_D(ic, is)
      if (i1 < QC_Nc) {
        v1[QC_LIDX_D(i1,i2)] = arg->fwdProp[i].getData(x_cb, pty, i2, i1); /* sic! [color, spin] <- [spin, color] */
        v2[QC_LIDX_D(i1,i2)] = arg->bwdProp[i].getData(x_cb, pty, i2, i1); /* sic! [color, spin] <- [spin, color] */
      }
#else
#define QC_LIDX_D_TR(ic, is) ((ic) + QC_Nc*(is))
      if (0 == i12) {
        *(reinterpret_cast<Vector *>(v1)) = arg->fwdProp[i](x_cb, pty);
        *(reinterpret_cast<Vector *>(v2)) = arg->bwdProp[i](x_cb, pty); 
      }
      /* FIXME change QC_LIDX_V to Vector indexing */
#endif
      __syncthreads();

      /* compute u.v1 */
      if (i1 < QC_Nc) {
        complex<QC_REAL> s = 0;

#pragma unroll
        for (int kc = 0 ; kc < QC_Nc ; kc++)
          s += umat[QC_LIDX_M(i1, kc)] * v1[QC_LIDX_D_TR(kc,i2)];

        u_v1[QC_LIDX_D_TR(i1,i2)] = s;
      }
      __syncthreads();

      /* compute v2^dag . u_v1 */
      {
        complex<QC_REAL> s = 0;

        for (int kc = 0 ; kc < QC_Nc ; kc++)
          s += conj(v2[QC_LIDX_D_TR(kc, i2)]) * u_v1[QC_LIDX_D_TR(kc,i1)];

        gres[QC_LIDX_G(i1, i2)] += s;
      }
      __syncthreads(); /* sic! avoid overwrite v2 in the next iter; sync gres before Gamma proj */
    }//- loop over prop-i


    /* proj on Gamma(ng) -> Corr_dev[tid + lV*ng] */
    int ng  = i12;
#if 1
    QC_CPLX s = 0;
#pragma unroll
    for (int is = 0 ; is < QC_Ns ; is++) {
      int js = gamma->left_ind[ng][is];
      s += gamma->left_coeff[ng][is] * gres[QC_LIDX_G(js, is)];
    }
    Corr_dev[tid + lV*ng] = s;
#else
    Corr_dev[tid + lV*ng] = gres[QC_LIDX_G(i1, i2)]; /* XXX skip Gamma-projection, for testing only! */
#endif

  }//-kernel



} //- namespace quda
