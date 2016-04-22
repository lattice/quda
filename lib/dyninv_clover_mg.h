#ifdef	DYNAMIC_CLOVER
/*	Some variables that will become handy later	*/

#define c00_01 (conj(c01_00))
#define c00_02 (conj(c02_00))
#define c01_02 (conj(c02_01))
#define c00_10 (conj(c10_00))
#define c01_10 (conj(c10_01))
#define c02_10 (conj(c10_02))
#define c00_11 (conj(c11_00))
#define c01_11 (conj(c11_01))
#define c02_11 (conj(c11_02))
#define c10_11 (conj(c11_10))
#define c00_12 (conj(c12_00))
#define c01_12 (conj(c12_01))
#define c02_12 (conj(c12_02))
#define c10_12 (conj(c12_10))
#define c11_12 (conj(c12_11))

#define tri5  tri0
#define tri9  tri1
#define tri13 tri2
#define tri14 tri3
#define tmp0 tri0.x
#define tmp1 tri0.y
#define tmp2 tri1.x
#define tmp3 tri1.y
#define tmp4 tri2.x
#define tmp5 tri2.y
#define v1_0 tri3
#define v1_1 tri4
#define v1_2 tri6
#define v1_3 tri7
#define v1_4 tri8
#define v1_5 tri10
#define sum  tri11

#define Clv(s1,c1,s2,c2)\
(arg.C(0, parity, x_cb, s1+ch, s2+ch, c1, c2))

#define InvClv(s1,c1,s2,c2)\
c##s1##c1##_##s2##c2

template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
__device__ __host__ inline void applyInvClover(Arg &arg, int parity, int x_cb) {
    /* Applies the inverse of the clover term squared plus mu2 to the spinor */
    /* Compute (T^2 + mu2) first, then invert */
    /* We proceed by chiral blocks */

    /* FIXME: This happens because of the last final loop */
    if (fineSpin != 4 || fineColor != 3) errorQuda("Fine spin, fine color combination not supported for Dynamic Clover inversion\n");


    for (int ch = 0; ch < 4; ch += 2) {	/* Loop over chiral blocks */
	complex<Float> tri0, tri1, tri2, tri3, tri4, tri6, tri7, tri8, tri10, tri11, tri12;
	Float d, c00_00, c01_01, c02_02, c10_10, c11_11, c12_12;
	complex<Float> c01_00, c02_00, c10_00, c11_00, c12_00;
	complex<Float> c02_01, c10_01, c11_01, c12_01;
	complex<Float> c10_02, c11_02, c12_02;
	complex<Float> c11_10, c12_10;
	complex<Float> c12_11;


	tri0  = Clv(0,1,0,0)*Clv(0,0,0,0).real() + Clv(0,1,0,1)*Clv(0,1,0,0) + Clv(0,1,0,2)*Clv(0,2,0,0) + Clv(0,1,1,0)*Clv(1,0,0,0) + Clv(0,1,1,1)*Clv(1,1,0,0) + Clv(0,1,1,2)*Clv(1,2,0,0);
	tri1  = Clv(0,2,0,0)*Clv(0,0,0,0).real() + Clv(0,2,0,2)*Clv(0,2,0,0) + Clv(0,2,0,1)*Clv(0,1,0,0) + Clv(0,2,1,0)*Clv(1,0,0,0) + Clv(0,2,1,1)*Clv(1,1,0,0) + Clv(0,2,1,2)*Clv(1,2,0,0);
	tri3  = Clv(1,0,0,0)*Clv(0,0,0,0).real() + Clv(1,0,1,0)*Clv(1,0,0,0) + Clv(1,0,0,1)*Clv(0,1,0,0) + Clv(1,0,0,2)*Clv(0,2,0,0) + Clv(1,0,1,1)*Clv(1,1,0,0) + Clv(1,0,1,2)*Clv(1,2,0,0);
	tri6  = Clv(1,1,0,0)*Clv(0,0,0,0).real() + Clv(1,1,1,1)*Clv(1,1,0,0) + Clv(1,1,0,1)*Clv(0,1,0,0) + Clv(1,1,0,2)*Clv(0,2,0,0) + Clv(1,1,1,0)*Clv(1,0,0,0) + Clv(1,1,1,2)*Clv(1,2,0,0);
	tri10 = Clv(1,2,0,0)*Clv(0,0,0,0).real() + Clv(1,2,1,2)*Clv(1,2,0,0) + Clv(1,2,0,1)*Clv(0,1,0,0) + Clv(1,2,0,2)*Clv(0,2,0,0) + Clv(1,2,1,0)*Clv(1,0,0,0) + Clv(1,2,1,1)*Clv(1,1,0,0);

	d = arg.mu*arg.mu + Clv(0,0,0,0).real()*Clv(0,0,0,0).real() + norm(Clv(0,1,0,0)) + norm(Clv(0,2,0,0)) + norm(Clv(1,0,0,0)) + norm(Clv(1,1,0,0)) + norm(Clv(1,2,0,0));

	InvClv(0,0,0,0) = d;

	tri2  = Clv(0,2,0,1)*Clv(0,1,0,1).real() + Clv(0,2,0,2)*Clv(0,2,0,1) + Clv(0,2,0,0)*Clv(0,0,0,1) + Clv(0,2,1,0)*Clv(1,0,0,1) + Clv(0,2,1,1)*Clv(1,1,0,1) + Clv(0,2,1,2)*Clv(1,2,0,1);
	tri4  = Clv(1,0,0,1)*Clv(0,1,0,1).real() + Clv(1,0,1,0)*Clv(1,0,0,1) + Clv(1,0,0,0)*Clv(0,0,0,1) + Clv(1,0,0,2)*Clv(0,2,0,1) + Clv(1,0,1,1)*Clv(1,1,0,1) + Clv(1,0,1,2)*Clv(1,2,0,1);
	tri7  = Clv(1,1,0,1)*Clv(0,1,0,1).real() + Clv(1,1,1,1)*Clv(1,1,0,1) + Clv(1,1,0,0)*Clv(0,0,0,1) + Clv(1,1,0,2)*Clv(0,2,0,1) + Clv(1,1,1,0)*Clv(1,0,0,1) + Clv(1,1,1,2)*Clv(1,2,0,1);
	tri11 = Clv(1,2,0,1)*Clv(0,1,0,1).real() + Clv(1,2,1,2)*Clv(1,2,0,1) + Clv(1,2,0,0)*Clv(0,0,0,1) + Clv(1,2,0,2)*Clv(0,2,0,1) + Clv(1,2,1,0)*Clv(1,0,0,1) + Clv(1,2,1,1)*Clv(1,1,0,1);

	d = arg.mu*arg.mu + Clv(0,1,0,1).real()*Clv(0,1,0,1).real() + norm(Clv(0,0,0,1)) + norm(Clv(0,2,0,1)) + norm(Clv(1,0,0,1)) + norm(Clv(1,1,0,1)) + norm(Clv(1,2,0,1));

	InvClv(0,1,0,1) = d;
	InvClv(0,1,0,0) = tri0;	/* We freed tri0, so we can reuse it as tri5 to save memory */

	tri5  = Clv(1,0,0,2)*Clv(0,2,0,2).real() + Clv(1,0,1,0)*Clv(1,0,0,2) + Clv(1,0,0,0)*Clv(0,0,0,2) + Clv(1,0,0,1)*Clv(0,1,0,2) + Clv(1,0,1,1)*Clv(1,1,0,2) + Clv(1,0,1,2)*Clv(1,2,0,2);
	tri8  = Clv(1,1,0,2)*Clv(0,2,0,2).real() + Clv(1,1,1,1)*Clv(1,1,0,2) + Clv(1,1,0,0)*Clv(0,0,0,2) + Clv(1,1,0,1)*Clv(0,1,0,2) + Clv(1,1,1,0)*Clv(1,0,0,2) + Clv(1,1,1,2)*Clv(1,2,0,2);
	tri12 = Clv(1,2,0,2)*Clv(0,2,0,2).real() + Clv(1,2,1,2)*Clv(1,2,0,2) + Clv(1,2,0,0)*Clv(0,0,0,2) + Clv(1,2,0,1)*Clv(0,1,0,2) + Clv(1,2,1,0)*Clv(1,0,0,2) + Clv(1,2,1,1)*Clv(1,1,0,2);

	d = arg.mu*arg.mu + Clv(0,2,0,2).real()*Clv(0,2,0,2).real() + norm(Clv(0,0,0,2)) + norm(Clv(0,1,0,2)) + norm(Clv(1,0,0,2)) + norm(Clv(1,1,0,2)) + norm(Clv(1,2,0,2));

	InvClv(0,2,0,2) = d;
	InvClv(0,2,0,0) = tri1;	/* We freed tri1, so we can reuse it as tri9  to save memory */
	InvClv(0,2,0,1) = tri2;	/* We freed tri2, so we can reuse it as tri13 to save memory */

	tri9  = Clv(1,1,1,0)*Clv(1,0,1,0).real() + Clv(1,1,1,1)*Clv(1,1,1,0) + Clv(1,1,0,0)*Clv(0,0,1,0) + Clv(1,1,0,1)*Clv(0,1,1,0) + Clv(1,1,0,2)*Clv(0,2,1,0) + Clv(1,1,1,2)*Clv(1,2,1,0);
	tri13 = Clv(1,2,1,0)*Clv(1,0,1,0).real() + Clv(1,2,1,2)*Clv(1,2,1,0) + Clv(1,2,0,0)*Clv(0,0,1,0) + Clv(1,2,0,1)*Clv(0,1,1,0) + Clv(1,2,0,2)*Clv(0,2,1,0) + Clv(1,2,1,1)*Clv(1,1,1,0);

	d = arg.mu*arg.mu + Clv(1,0,1,0).real()*Clv(1,0,1,0).real() + norm(Clv(0,0,1,0)) + norm(Clv(0,1,1,0)) + norm(Clv(0,2,1,0)) + norm(Clv(1,1,1,0)) + norm(Clv(1,2,1,0));

	InvClv(1,0,1,0) = d;
	InvClv(1,0,0,0) = tri3;	/* We freed tri3, so we can reuse it as tri14 to save memory */

	tri14 = Clv(1,2,1,1)*Clv(1,1,1,1).real() + Clv(1,2,1,2)*Clv(1,2,1,1) + Clv(1,2,0,0)*Clv(0,0,1,1) + Clv(1,2,0,1)*Clv(0,1,1,1) + Clv(1,2,0,2)*Clv(0,2,1,1) + Clv(1,2,1,0)*Clv(1,0,1,1);

	d = arg.mu*arg.mu + Clv(1,1,1,1).real()*Clv(1,1,1,1).real() + norm(Clv(0,0,1,1)) + norm(Clv(0,1,1,1)) + norm(Clv(0,2,1,1)) + norm(Clv(1,0,1,1)) + norm(Clv(1,2,1,1));

	InvClv(1,1,1,1) = d;

	d = arg.mu*arg.mu + Clv(1,2,1,2).real()*Clv(1,2,1,2).real() + norm(Clv(0,0,1,2)) + norm(Clv(0,1,1,2)) + norm(Clv(0,2,1,2)) + norm(Clv(1,0,1,2)) + norm(Clv(1,1,1,2));

	InvClv(1,2,1,2) = d;

	InvClv(1,2,1,1) = tri14;
	InvClv(1,2,1,0) = tri13;
	InvClv(1,2,0,2) = tri12;
	InvClv(1,2,0,1) = tri11;
	InvClv(1,2,0,0) = tri10;
	InvClv(1,1,1,0) = tri9;
	InvClv(1,1,0,2) = tri8;
	InvClv(1,1,0,1) = tri7;
	InvClv(1,1,0,0) = tri6;
	InvClv(1,0,0,2) = tri5;
	InvClv(1,0,0,1) = tri4;

	/*	INVERSION STARTS	*/

	/* j = 0 */
	InvClv(0,0,0,0) = sqrt(InvClv(0,0,0,0));
	tmp0 = 1. / InvClv(0,0,0,0);
	InvClv(0,1,0,0) *= tmp0;
	InvClv(0,2,0,0) *= tmp0;
	InvClv(1,0,0,0) *= tmp0;
	InvClv(1,1,0,0) *= tmp0;
	InvClv(1,2,0,0) *= tmp0;

	/* k = 1 kj = 0 */
	InvClv(0,1,0,1) -= norm(InvClv(0,1,0,0));

		/* l = 2...5 kj = 0 lj = 1,3,6,10 lk = 2,4,7,11*/
		InvClv(0,2,0,1) -= InvClv(0,2,0,0)*InvClv(0,0,0,1);
		InvClv(1,0,0,1) -= InvClv(1,0,0,0)*InvClv(0,0,0,1);
		InvClv(1,1,0,1) -= InvClv(1,1,0,0)*InvClv(0,0,0,1);
		InvClv(1,2,0,1) -= InvClv(1,2,0,0)*InvClv(0,0,0,1);

	/* k = 2 kj = 1 */
	InvClv(0,2,0,2) -= norm(InvClv(0,2,0,0));

		/* l = 3...5 kj = 1 lj = 3,6,10 lk = 5,8,12*/
		InvClv(1,0,0,2) -= InvClv(1,0,0,0)*InvClv(0,0,0,2);
		InvClv(1,1,0,2) -= InvClv(1,1,0,0)*InvClv(0,0,0,2);
		InvClv(1,2,0,2) -= InvClv(1,2,0,0)*InvClv(0,0,0,2);

	/* k = 3 kj = 3 */
	InvClv(1,0,1,0) -= norm(InvClv(1,0,0,0));

		/* l = 4 kj = 3 lj = 6,10 lk = 9,13*/
		InvClv(1,1,1,0) -= InvClv(1,1,0,0)*InvClv(0,0,1,0);
		InvClv(1,2,1,0) -= InvClv(1,2,0,0)*InvClv(0,0,1,0);

	/* k = 4 kj = 6 */
	InvClv(1,1,1,1) -= norm(InvClv(1,1,0,0));

		/* l = 5 kj = 6 lj = 10 lk = 14*/
		InvClv(1,2,1,1) -= InvClv(1,2,0,0)*InvClv(0,0,1,1);

	/* k = 5 kj = 10 */
	InvClv(1,2,1,2) -= norm(InvClv(1,2,0,0));

	/* j = 1 */
	InvClv(0,1,0,1) = sqrt(InvClv(0,1,0,1));
	tmp1 = 1. / InvClv(0,1,0,1);
	InvClv(0,2,0,1) *= tmp1;
	InvClv(1,0,0,1) *= tmp1;
	InvClv(1,1,0,1) *= tmp1;
	InvClv(1,2,0,1) *= tmp1;

	/* k = 2 kj = 2 */
	InvClv(0,2,0,2) -= norm(InvClv(0,2,0,1));

		/* l = 3...5 kj = 2 lj = 4,7,11 lk = 5,8,12*/
		InvClv(1,0,0,2) -= InvClv(1,0,0,1)*InvClv(0,1,0,2);
		InvClv(1,1,0,2) -= InvClv(1,1,0,1)*InvClv(0,1,0,2);
		InvClv(1,2,0,2) -= InvClv(1,2,0,1)*InvClv(0,1,0,2);

	/* k = 3 kj = 4 */
	InvClv(1,0,1,0) -= norm(InvClv(1,0,0,1));

		/* l = 4 kj = 4 lj = 7,11 lk = 9,13*/
		InvClv(1,1,1,0) -= InvClv(1,1,0,1)*InvClv(0,1,1,0);
		InvClv(1,2,1,0) -= InvClv(1,2,0,1)*InvClv(0,1,1,0);

	/* k = 4 kj = 7 */
	InvClv(1,1,1,1) -= norm(InvClv(1,1,0,1));

		/* l = 5 kj = 7 lj = 11 lk = 14*/
		InvClv(1,2,1,1) -= InvClv(1,2,0,1)*InvClv(0,1,1,1);

	/* k = 5 kj = 11 */
	InvClv(1,2,1,2) -= norm(InvClv(1,2,0,1));

	/* j = 2 */
	InvClv(0,2,0,2) = sqrt(InvClv(0,2,0,2));
	tmp2 = 1. / InvClv(0,2,0,2);
	InvClv(1,0,0,2) *= tmp2;
	InvClv(1,1,0,2) *= tmp2;
	InvClv(1,2,0,2) *= tmp2;

	/* k = 3 kj = 5 */
	InvClv(1,0,1,0) -= norm(InvClv(1,0,0,2));

		/* l = 4 kj = 5 lj = 8,12 lk = 9,13*/
		InvClv(1,1,1,0) -= InvClv(1,1,0,2)*InvClv(0,2,1,0);
		InvClv(1,2,1,0) -= InvClv(1,2,0,2)*InvClv(0,2,1,0);

	/* k = 4 kj = 8 */
	InvClv(1,1,1,1) -= norm(InvClv(1,1,0,2));

		/* l = 5 kj = 8 lj = 12 lk = 14*/
		InvClv(1,2,1,1) -= InvClv(1,2,0,2)*InvClv(0,2,1,1);

	/* k = 5 kj = 12 */
	InvClv(1,2,1,2) -= norm(InvClv(1,2,0,2));

	/* j = 3 */
	InvClv(1,0,1,0) = sqrt(InvClv(1,0,1,0));
	tmp3 = 1. / InvClv(1,0,1,0);
	InvClv(1,1,1,0) *= tmp3;
	InvClv(1,2,1,0) *= tmp3;

	/* k = 4 kj = 9 */
	InvClv(1,1,1,1) -= norm(InvClv(1,1,1,0));

		/* l = 5 kj = 9 lj = 13 lk = 14*/
		InvClv(1,2,1,1) -= InvClv(1,2,1,0)*InvClv(1,0,1,1);

	/* k = 5 kj = 13 */
	InvClv(1,2,1,2) -= norm(InvClv(1,2,1,0));

	/* j = 4 */
	InvClv(1,1,1,1) = sqrt(InvClv(1,1,1,1));
	tmp4 = 1. / InvClv(1,1,1,1);
	InvClv(1,2,1,1) *= tmp4;

	/* k = 5 kj = 14 */
	InvClv(1,2,1,2) -= norm(InvClv(1,2,1,1));

	/* j = 5 */
	InvClv(1,2,1,2) = sqrt(InvClv(1,2,1,2));
	tmp5 = 1. / InvClv(1,2,1,2);

	/* NO INFO YET ON TR(LOG A) FOR TM-CLOVER */
	/* Accumulate trlogA */	
	/* for (int j=0;j<6;j++) trlogA += (double)2.0*log((double)(diag[j])); */

	/* Forwards substitute */

	/* k = 0 */
	v1_0 = complex<Float>(tmp0,0.);
	v1_1 = InvClv(0,1,0,0)*(-tmp1*tmp0);
	v1_2 = (InvClv(0,2,0,0)*tmp0 + InvClv(0,2,0,1)*v1_1)*(-tmp2);
	v1_3 = (InvClv(1,0,0,0)*tmp0 + InvClv(1,0,0,1)*v1_1 + InvClv(1,0,0,2)*v1_2)*(-tmp3);
	v1_4 = (InvClv(1,1,0,0)*tmp0 + InvClv(1,1,0,1)*v1_1 + InvClv(1,1,0,2)*v1_2 + InvClv(1,1,1,0)*v1_3)*(-tmp4);
	v1_5 = (InvClv(1,2,0,0)*tmp0 + InvClv(1,2,0,1)*v1_1 + InvClv(1,2,0,2)*v1_2 + InvClv(1,2,1,0)*v1_3 + InvClv(1,2,1,1)*v1_4)*(-tmp5*tmp5);

	/* Backwards substitute */

	sum = v1_4 - InvClv(1,1,1,2)*v1_5;
	v1_4 = sum*tmp4;

	sum = v1_3 - InvClv(1,0,1,1)*v1_4 - InvClv(1,0,1,2)*v1_5;
	v1_3 = sum*tmp3;

	sum = v1_2 - InvClv(0,2,1,0)*v1_3 - InvClv(0,2,1,1)*v1_4 - InvClv(0,2,1,2)*v1_5;
	v1_2 = sum*tmp2;

	sum = v1_1 - InvClv(0,1,0,2)*v1_2 - InvClv(0,1,1,0)*v1_3 - InvClv(0,1,1,1)*v1_4 - InvClv(0,1,1,2)*v1_5;
	v1_1 = sum*tmp1;

	sum = v1_0 - InvClv(0,0,0,1)*v1_1 - InvClv(0,0,0,2)*v1_2 - InvClv(0,0,1,0)*v1_3 - InvClv(0,0,1,1)*v1_4 - InvClv(0,0,1,2)*v1_5;
	v1_0 = sum*tmp0;

	InvClv(0,0,0,0) = v1_0.real();
	InvClv(0,1,0,0) = v1_1;
	InvClv(0,2,0,0) = v1_2;
	InvClv(1,0,0,0) = v1_3;
	InvClv(1,1,0,0) = v1_4;
	InvClv(1,2,0,0) = v1_5;

	/* k = 1 */
	v1_1 = complex<Float>(tmp1,0.);
	v1_2 = InvClv(0,2,0,1)*(-tmp2*tmp1);
	v1_3 = (InvClv(1,0,0,1)*tmp1 + InvClv(1,0,0,2)*v1_2)*(-tmp3);
	v1_4 = (InvClv(1,1,0,1)*tmp1 + InvClv(1,1,0,2)*v1_2 + InvClv(1,1,1,0)*v1_3)*(-tmp4);
	v1_5 = (InvClv(1,2,0,1)*tmp1 + InvClv(1,2,0,2)*v1_2 + InvClv(1,2,1,0)*v1_3 + InvClv(1,2,1,1)*v1_4)*(-tmp5*tmp5);

	/* Backwards substitute */

	/* l = 4 */
	sum = v1_4 - InvClv(1,1,1,2)*v1_5;
	v1_4 = sum*tmp4;

	sum = v1_3 - InvClv(1,0,1,1)*v1_4 - InvClv(1,0,1,2)*v1_5;
	v1_3 = sum*tmp3;

	sum = v1_2 - InvClv(0,2,1,0)*v1_3 - InvClv(0,2,1,1)*v1_4 - InvClv(0,2,1,2)*v1_5;
	v1_2 = sum*tmp2;

	sum = v1_1 - InvClv(0,1,0,2)*v1_2 - InvClv(0,1,1,0)*v1_3 - InvClv(0,1,1,1)*v1_4 - InvClv(0,1,1,2)*v1_5;
	v1_1 = sum*tmp1;

	InvClv(0,1,0,1) = v1_1.real();
	InvClv(0,2,0,1) = v1_2;
	InvClv(1,0,0,1) = v1_3;
	InvClv(1,1,0,1) = v1_4;
	InvClv(1,2,0,1) = v1_5;

	/* k = 2 */
	v1_2 = complex<Float>(tmp2,0.);
	v1_3 = InvClv(1,0,0,2)*(-tmp2*tmp3);
	v1_4 = (InvClv(1,1,0,2)*tmp2 + InvClv(1,1,1,0)*v1_3)*(-tmp4);
	v1_5 = (InvClv(1,2,0,2)*tmp2 + InvClv(1,2,1,0)*v1_3 + InvClv(1,2,1,1)*v1_4)*(-tmp5*tmp5);

	/* Backwards substitute */

	/* l = 4 */
	sum = v1_4 - InvClv(1,1,1,2)*v1_5;
	v1_4 = sum*tmp4;

	sum = v1_3 - InvClv(1,0,1,1)*v1_4 - InvClv(1,0,1,2)*v1_5;
	v1_3 = sum*tmp3;

	sum = v1_2 - InvClv(0,2,1,0)*v1_3 - InvClv(0,2,1,1)*v1_4 - InvClv(0,2,1,2)*v1_5;
	v1_2 = sum*tmp2;

	InvClv(0,2,0,2) = v1_2.real();
	InvClv(1,0,0,2) = v1_3;
	InvClv(1,1,0,2) = v1_4;
	InvClv(1,2,0,2) = v1_5;

	/* k = 3 */
	v1_3 = complex<Float>(tmp3,0.);
	v1_4 = InvClv(1,1,1,0)*(-tmp3*tmp4);
	v1_5 = (InvClv(1,2,1,0)*tmp3 + InvClv(1,2,1,1)*v1_4)*(-tmp5*tmp5);

	/* Backwards substitute */

	/* l = 4 */
	sum = v1_4 - InvClv(1,1,1,2)*v1_5;
	v1_4 = sum*tmp4;

	sum = v1_3 - InvClv(1,0,1,1)*v1_4 - InvClv(1,0,1,2)*v1_5;
	v1_3 = sum*tmp3;

	InvClv(1,0,1,0) = v1_3.real();
	InvClv(1,1,1,0) = v1_4;
	InvClv(1,2,1,0) = v1_5;

	/* k = 4 */
	v1_4 = complex<Float>(tmp4,0.);
	v1_5 = InvClv(1,2,1,1)*(-tmp4*tmp5*tmp5);

	/* Backwards substitute */

	/* l = 4 */
	sum = v1_4 - InvClv(1,1,1,2)*v1_5;
	v1_4 = sum*tmp4;

	InvClv(1,1,1,1) = v1_4.real();
	InvClv(1,2,1,1) = v1_5;

	/* k = 5 */
	InvClv(1,2,1,2) = tmp5*tmp5;

	/*	Calculate the product for the first chiral block	*/

	//Then we calculate AV = Cinv UV, so  [AV = (C^2 + mu^2)^{-1} (Clover -/+ i mu)·Vector]
	//for in twisted-clover fermions, Cinv keeps (C^2 + mu^2)^{-1}

    for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	/* Unrolled loop, supports limited values for fineSpin, fineColor and coarseColor */
	arg.AV(parity,x_cb,ch,0,ic_c)  += InvClv(0, 0, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
					+ InvClv(0, 0, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
					+ InvClv(0, 0, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
	  				+ InvClv(0, 0, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
					+ InvClv(0, 0, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
					+ InvClv(0, 0, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);

	arg.AV(parity,x_cb,ch,1,ic_c)  += InvClv(0, 1, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
	 	     			+ InvClv(0, 1, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
	      				+ InvClv(0, 1, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
					+ InvClv(0, 1, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
	      				+ InvClv(0, 1, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
	      				+ InvClv(0, 1, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);

	arg.AV(parity,x_cb,ch,2,ic_c)  += InvClv(0, 2, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
	      				+ InvClv(0, 2, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
	      				+ InvClv(0, 2, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
					+ InvClv(0, 2, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
		      			+ InvClv(0, 2, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
		      			+ InvClv(0, 2, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);

	arg.AV(parity,x_cb,ch+1,0,ic_c)+= InvClv(1, 0, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
	      				+ InvClv(1, 0, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
	      				+ InvClv(1, 0, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
					+ InvClv(1, 0, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
	      				+ InvClv(1, 0, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
	 	     			+ InvClv(1, 0, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);

	arg.AV(parity,x_cb,ch+1,1,ic_c)+= InvClv(1, 1, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
	      				+ InvClv(1, 1, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
	      				+ InvClv(1, 1, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
					+ InvClv(1, 1, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
	      				+ InvClv(1, 1, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
	 	     			+ InvClv(1, 1, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);

	arg.AV(parity,x_cb,ch+1,2,ic_c)+= InvClv(1, 2, 0, 0) * arg.UV(parity, x_cb, ch, 0, ic_c)
	      				+ InvClv(1, 2, 0, 1) * arg.UV(parity, x_cb, ch, 1, ic_c)
	      				+ InvClv(1, 2, 0, 2) * arg.UV(parity, x_cb, ch, 2, ic_c)
					+ InvClv(1, 2, 1, 0) * arg.UV(parity, x_cb, ch+1, 0, ic_c)
	      				+ InvClv(1, 2, 1, 1) * arg.UV(parity, x_cb, ch+1, 1, ic_c)
	 	     			+ InvClv(1, 2, 1, 2) * arg.UV(parity, x_cb, ch+1, 2, ic_c);
      }	// Coarse color
    }	// Chirality
}


/*	Cleanup		*/

#undef tri14
#undef tri13
#undef tri9
#undef tri5
#undef tmp0
#undef tmp1
#undef tmp2
#undef tmp3
#undef tmp4
#undef tmp5
#undef v1_0
#undef v1_1
#undef v1_2
#undef v1_3
#undef v1_4
#undef v1_5
#undef sum

#undef c00_01
#undef c00_02
#undef c01_02
#undef c00_10
#undef c01_10
#undef c02_10
#undef c00_11
#undef c01_11
#undef c02_11
#undef c10_11
#undef c00_12
#undef c01_12
#undef c02_12
#undef c10_12
#undef c11_12

#undef Clv
#undef InvClv

#endif
