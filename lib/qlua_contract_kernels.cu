#include <qlua_contract_kernels.cuh>

namespace quda {

  //- C.K. Constant variable declarations
   __constant__ QC_CPLX cS1_gvec[QC_LEN_G];
   __constant__ QC_CPLX cS2_gvec[QC_LEN_G];

  
  /* elementary actions on gamma matrices */
  /* a <- a^\dag */
  INFUNC_ DEVFUNC_ void 
    QC(gvec_adj)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = conj(a_gvec[i]) * (QC_CPLX)(1 - 2 * qc_gamma_adj_parity(i));
  }
  /* a <- a^T */
  INFUNC_ DEVFUNC_ void 
    QC(gvec_transp)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_transp_parity(i));
  }
  /* a <- conj(a) */
  INFUNC_ DEVFUNC_ void 
    QC(gvec_conj)(QC_CPLX *a_gvec)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = conj(a_gvec[i]) * (QC_CPLX)(1 - 2 * qc_gamma_conj_parity(i));
  }
  /* a <- G(ng)^dag . a . G(ng) */
  INFUNC_ DEVFUNC_ void 
    QC(gvec_uni_transf)(QC_CPLX *a_gvec, int ng)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_uni_parity(i, ng));
  }
  /* a <- G(ng)^T . a . G(ng) */
  INFUNC_ DEVFUNC_ void 
    QC(gvec_sim_transf)(QC_CPLX *a_gvec, int ng)
  {
    for (int i = 0 ; i < QC_LEN_G ; i++)
      a_gvec[i] = a_gvec[i] * (QC_CPLX)(1 - 2 * qc_gamma_sim_parity(i, ng));
  }
  // TODO Cliffort algegra mult: (a*b)[i^j] += a[i]*b[j] * mul_sign(i,j)


  /* general BLAS-style functions for multiplying by gamma 
     having general b is ok since extra MADD cost in negl. on GPU 
  */

  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void
    QC(cplx_vec_zero)(QC_CPLX *y, int y_stride, int len)
  {
    for (int i = len ; i-- ; y += y_stride) 
      *y = 0;
  }
  /* y <- x */
  INFUNC_ DEVFUNC_ void
    QC(cplx_vec_copy)(QC_CPLX *y, int y_stride, 
		      const QC_CPLX *x, int x_stride, int len)
  {
    for (int i = len ; i-- ; x += x_stride, y += y_stride)
      *y = *x;
  }
  /* x <- a * x */
  INFUNC_ DEVFUNC_ void 
    QC(cplx_vec_scal)(QC_CPLX *x, int x_stride, QC_CPLX a, int len)
  {
    for (int i = len ; i-- ; x += x_stride)
      *x *= a;
  }
  /* y <- a * x */
  INFUNC_ DEVFUNC_ void 
    QC(cplx_vec_scal_copy)(QC_CPLX *y, int y_stride, 
			   QC_CPLX a, const QC_CPLX *x, int x_stride, int len)
  {
    for (int i = len ; i-- ; x += x_stride, y += y_stride)
      *y = *x * a;
  }
  /* y <- conj(y) */
  INFUNC_ DEVFUNC_ void
    QC(cplx_vec_conj)(QC_CPLX *y, int y_stride, int len)
  {
    for (int i = len ; i-- ; y += y_stride) 
      *y = QC_CONJ(*y);
  }


#define def_gvec_func(gvec_func, gind_func, len)			\
  INFUNC_ DEVFUNC_ void gvec_func(					\
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
#define def_quarkcontractMN(MN, A,B,C,D) INFUNC_ DEVFUNC_ void		\
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
    INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ void
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
  INFUNC_ DEVFUNC_ QC_CPLX 
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
  INFUNC_ DEVFUNC_ QC_CPLX 
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
  INFUNC_ DEVFUNC_ QC_CPLX 
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
  INFUNC_ DEVFUNC_ QC_CPLX 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ QC_CPLX
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
  INFUNC_ DEVFUNC_ void
    QC(gvec_eq_trace_gamma_dot_G)(
				  QC_CPLX *gres, int gres_stride, 
				  const QC_CPLX *g1, int g1_stride)
  {
    for (int ng = 0 ; ng < QC_LEN_G ; ng++) 
      gres[gres_stride * ng] = QC(trace_gamma_dot_G)(ng, g1,g1_stride);
  }

  /* compute res[gn] = Tr [Gamma[gn] * F * B] */
  DEVFUNC_ void 
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
  DEVFUNC_ void 
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
  DEVFUNC_ void 
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

  INFUNC_ DEVFUNC_ void 
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
  INFUNC_ DEVFUNC_ void 
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
  DEVFUNC_ void
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
  DEVFUNC_ void
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
  DEVFUNC_ void
    QC(contract_tr_g_P_mgbar_hP)(
				 QC_CPLX *gres, int gres_stride,
				 const QC_CPLX *F, int F_stride,
				 const QC_CPLX *B, int B_stride)
  {
    QC_CPLX tmpP1[QC_LEN_P];
    QC(P_eq_hP)(tmpP1,1, B,B_stride);
    QC(contract_tr_g_P_mgbar_P)(gres, gres_stride, F, F_stride, tmpP1,1);
  }


  DEVFUNC_ void 
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

  DEVFUNC_ void 
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


  DEVFUNC_ void 
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


  //---------------------------------------------------------------------------//
  // U T I L I T Y   F U N C T I O N S   A N D   K E R N E L   W R A P P E R S //
  //---------------------------------------------------------------------------//

  void copySmatricesToSymbol(complex<QC_REAL> *S2, complex<QC_REAL> *S1){
    cudaMemcpyToSymbol(cS2_gvec, S2, sizeof(complex<QC_REAL>)*QC_LEN_G);
    cudaMemcpyToSymbol(cS1_gvec, S1, sizeof(complex<QC_REAL>)*QC_LEN_G);
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

  __global__ void qtmd_g_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, TMDcontractState *TMDcs){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    int tid  = x_cb + pty * TMDcs->volumeCB;
    int lV   = TMDcs->volume;

    if (x_cb >= TMDcs->volumeCB) return;
    if (pty >= TMDcs->nParity) return;
    if(tid >= lV) return;

    complex<QC_REAL> dev_prop1[QC_LEN_P];
    complex<QC_REAL> dev_prop2[QC_LEN_P];

    Vector vec1[QUDA_PROP_NVEC];
    Vector vec2[QUDA_PROP_NVEC];
    for(int i=0;i<QUDA_PROP_NVEC;i++){
      vec1[i] = TMDcs->fwdProp[i](x_cb, pty);
      vec2[i] = TMDcs->bwdProp[i](x_cb, pty);
    }
    prepareDevicePropSite(dev_prop1, vec1, TMDcs->preserveBasis);
    prepareDevicePropSite(dev_prop2, vec2, TMDcs->preserveBasis);

    qc_quda_contract_tr_g_P_P(Corr_dev + tid, lV,
			      dev_prop1, 1,
			      dev_prop2, 1);
  }
  //------------------------------------------------------------------------------------------

} //- namespace quda
