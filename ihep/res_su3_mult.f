**********************************************************
**         Subroutine for SU(3)*vector multiplications  **
**                                                      **
**         They perform the following tasks:            **
**                                                      **
**         (1) su3_vect01.f:                            **
**             Syntax: call su3_vect01(a,v,w)           **
**             Task:   w = a * v,                       **
**                     a,is 3*3 complex matrix          **
**                     w,v are SU(3) vectors            **
**                                                      **
**         (2) su3_vect02.f:                            **
**             Syntax: call su3_vect02(a,v,w)           **
**             Task:   w = a^ * v,                      **
**                     a,is 3*3 complex matrix          **
**                     w,v are SU(3) vectors            **
**                                                      **
**********************************************************
**                                                      **
**    This is file     su3_vect01.f                     **
**                                                      **
**********************************************************
           subroutine su3_vect01(a,v,w)
           real*8 a(18),v(6),w(6) 

c          w(1)=a(1,1)*v(1)+a(1,2)*v(2)+a(1,3)*v(3) 
           tmp1=a( 1)*v( 1)
           tmp2=a( 2)*v( 2)
           tmp3=(a( 1)+a( 2))*(v( 1)+v( 2))
           w( 1)=tmp1-tmp2
           w( 2)=tmp3-tmp1-tmp2
           tmp1=a( 7)*v( 3)
           tmp2=a( 8)*v( 4)
           tmp3=(a( 7)+a( 8))*(v( 3)+v( 4))
           w( 1)=w( 1)+tmp1-tmp2
           w( 2)=w( 2)+tmp3-tmp1-tmp2
           tmp1=a(13)*v( 5)
           tmp2=a(14)*v( 6)
           tmp3=(a(13)+a(14))*(v( 5)+v( 6))
           w( 1)=w( 1)+tmp1-tmp2
           w( 2)=w( 2)+tmp3-tmp1-tmp2
   
c          w(2)=a(2,1)*v(1)+a(2,2)*v(2)+a(2,3)*v(3) 
           tmp1=a( 3)*v( 1)
           tmp2=a( 4)*v( 2)
           tmp3=(a( 3)+a( 4))*(v( 1)+v( 2))
           w( 3)=tmp1-tmp2
           w( 4)=tmp3-tmp1-tmp2
           tmp1=a( 9)*v( 3)
           tmp2=a(10)*v( 4)
           tmp3=(a( 9)+a(10))*(v( 3)+v( 4))
           w( 3)=w( 3)+tmp1-tmp2
           w( 4)=w( 4)+tmp3-tmp1-tmp2
           tmp1=a(15)*v( 5)
           tmp2=a(16)*v( 6)
           tmp3=(a(15)+a(16))*(v( 5)+v( 6))
           w( 3)=w( 3)+tmp1-tmp2
           w( 4)=w( 4)+tmp3-tmp1-tmp2

c          w(3)=a(3,1)*v(1)+a(3,2)*v(2)+a(3,3)*v(3) 
           tmp1=a( 5)*v( 1)
           tmp2=a( 6)*v( 2)
           tmp3=(a( 5)+a( 6))*(v( 1)+v( 2))
           w( 5)=tmp1-tmp2
           w( 6)=tmp3-tmp1-tmp2
           tmp1=a(11)*v( 3)
           tmp2=a(12)*v( 4)
           tmp3=(a(11)+a(12))*(v( 3)+v( 4))
           w( 5)=w( 5)+tmp1-tmp2
           w( 6)=w( 6)+tmp3-tmp1-tmp2
           tmp1=a(17)*v( 5)
           tmp2=a(18)*v( 6)
           tmp3=(a(17)+a(18))*(v( 5)+v( 6))
           w( 5)=w( 5)+tmp1-tmp2
           w( 6)=w( 6)+tmp3-tmp1-tmp2

           return
           end

          subroutine su3_vect02(a,v,w)
           real*8 a(18),v(6),w(6)

c          w(1)=a(1,1)*v(1)+a(1,2)*v(2)+a(1,3)*v(3) 
           tmp1=a( 1)*v( 1)
           tmp2=a( 2)*v( 2)
           tmp3=(a( 1)-a( 2))*(v( 1)+v( 2))
           w( 1)=tmp1+tmp2
           w( 2)=tmp3-tmp1+tmp2
           tmp1=a( 3)*v( 3)
           tmp2=a( 4)*v( 4)
           tmp3=(a( 3)-a( 4))*(v( 3)+v( 4))
           w( 1)=w( 1)+tmp1+tmp2
           w( 2)=w( 2)+tmp3-tmp1+tmp2
           tmp1=a( 5)*v( 5)
           tmp2=a( 6)*v( 6)
           tmp3=(a( 5)-a( 6))*(v( 5)+v( 6))
           w( 1)=w( 1)+tmp1+tmp2
           w( 2)=w( 2)+tmp3-tmp1+tmp2

c          w(2)=a(2,1)*v(1)+a(2,2)*v(2)+a(2,3)*v(3) 
           tmp1=a( 7)*v( 1)
           tmp2=a( 8)*v( 2)
           tmp3=(a( 7)-a( 8))*(v( 1)+v( 2))
           w( 3)=tmp1+tmp2
           w( 4)=tmp3-tmp1+tmp2
           tmp1=a( 9)*v( 3)
           tmp2=a(10)*v( 4)
           tmp3=(a( 9)-a(10))*(v( 3)+v( 4))
           w( 3)=w( 3)+tmp1+tmp2
           w( 4)=w( 4)+tmp3-tmp1+tmp2
           tmp1=a(11)*v( 5)
           tmp2=a(12)*v( 6)
           tmp3=(a(11)-a(12))*(v( 5)+v( 6))
           w( 3)=w( 3)+tmp1+tmp2
           w( 4)=w( 4)+tmp3-tmp1+tmp2

c          w(3)=a(3,1)*v(1)+a(3,2)*v(2)+a(3,3)*v(3) 
           tmp1=a(13)*v( 1)
           tmp2=a(14)*v( 2)
           tmp3=(a(13)-a(14))*(v( 1)+v( 2))
           w( 5)=tmp1+tmp2
           w( 6)=tmp3-tmp1+tmp2
           tmp1=a(15)*v( 3)
           tmp2=a(16)*v( 4)
           tmp3=(a(15)-a(16))*(v( 3)+v( 4))
           w( 5)=w( 5)+tmp1+tmp2
           w( 6)=w( 6)+tmp3-tmp1+tmp2
           tmp1=a(17)*v( 5)
           tmp2=a(18)*v( 6)
           tmp3=(a(17)-a(18))*(v( 5)+v( 6))
           w( 5)=w( 5)+tmp1+tmp2
           w( 6)=w( 6)+tmp3-tmp1+tmp2

           return
           end

*********************************************************
**         Subroutine for the SU(3) multiplications     **
**                                                      **
**         They perform the following tasks:            **
**                                                      **
**         (1) su3_mult01.f:                            **
**             Syntax: call su3_mult01(a,b,c)           **
**             Task:   c = a * b,                       **
**                     a,b,c are 3*3 complex matrices   **
**                                                      **
**         (2) su3_mult02.f:                            **
**             Syntax: call su3_mult02(a,b,c)           **
**             Task:   c = a * b^,                      **
**                     a,b,c are 3*3 complex matrices   **
**                                                      **
**         (3) su3_mult03.f:                            **
**             Syntax: call su3_mult03(a,b,c)           **
**             Task:   c = a^ * b,                      **
**                     a,b,c are 3*3 complex matrices   **
**                                                      **
**********************************************************
**                                                      **
**    This is file     su3_mult01.f                     **
**                                                      **
**********************************************************
           subroutine su3_mult01(a,b,c)
           complex*16 a(3,3),b(3,3),c(3,3)

           c(1,1)=a(1,1)*b(1,1)+a(1,2)*b(2,1)+a(1,3)*b(3,1)
           c(1,2)=a(1,1)*b(1,2)+a(1,2)*b(2,2)+a(1,3)*b(3,2)
           c(1,3)=a(1,1)*b(1,3)+a(1,2)*b(2,3)+a(1,3)*b(3,3)

           c(2,1)=a(2,1)*b(1,1)+a(2,2)*b(2,1)+a(2,3)*b(3,1)
           c(2,2)=a(2,1)*b(1,2)+a(2,2)*b(2,2)+a(2,3)*b(3,2)
           c(2,3)=a(2,1)*b(1,3)+a(2,2)*b(2,3)+a(2,3)*b(3,3)

           c(3,1)=a(3,1)*b(1,1)+a(3,2)*b(2,1)+a(3,3)*b(3,1)
           c(3,2)=a(3,1)*b(1,2)+a(3,2)*b(2,2)+a(3,3)*b(3,2)
           c(3,3)=a(3,1)*b(1,3)+a(3,2)*b(2,3)+a(3,3)*b(3,3)

           return
           end

           subroutine su3_mult02(a,b,c)
           complex*16 a(3,3),b(3,3),c(3,3),d(3,3)

           d(1,1)=dconjg(b(1,1))
           d(1,2)=dconjg(b(2,1))
           d(1,3)=dconjg(b(3,1))

           d(2,1)=dconjg(b(1,2))
           d(2,2)=dconjg(b(2,2))
           d(2,3)=dconjg(b(3,2))

           d(3,1)=dconjg(b(1,3))
           d(3,2)=dconjg(b(2,3))
           d(3,3)=dconjg(b(3,3))

           c(1,1)=a(1,1)*d(1,1)+a(1,2)*d(2,1)+a(1,3)*d(3,1)
           c(1,2)=a(1,1)*d(1,2)+a(1,2)*d(2,2)+a(1,3)*d(3,2)
           c(1,3)=a(1,1)*d(1,3)+a(1,2)*d(2,3)+a(1,3)*d(3,3)

           c(2,1)=a(2,1)*d(1,1)+a(2,2)*d(2,1)+a(2,3)*d(3,1)
           c(2,2)=a(2,1)*d(1,2)+a(2,2)*d(2,2)+a(2,3)*d(3,2)
           c(2,3)=a(2,1)*d(1,3)+a(2,2)*d(2,3)+a(2,3)*d(3,3)

           c(3,1)=a(3,1)*d(1,1)+a(3,2)*d(2,1)+a(3,3)*d(3,1)
           c(3,2)=a(3,1)*d(1,2)+a(3,2)*d(2,2)+a(3,3)*d(3,2)
           c(3,3)=a(3,1)*d(1,3)+a(3,2)*d(2,3)+a(3,3)*d(3,3)

           return
           end

           subroutine su3_mult03(a,b,c)
           complex*16 a(3,3),b(3,3),c(3,3),d(3,3)

           d(1,1)=dconjg(a(1,1))
           d(1,2)=dconjg(a(2,1))
           d(1,3)=dconjg(a(3,1))

           d(2,1)=dconjg(a(1,2))
           d(2,2)=dconjg(a(2,2))
           d(2,3)=dconjg(a(3,2))

           d(3,1)=dconjg(a(1,3))
           d(3,2)=dconjg(a(2,3))
           d(3,3)=dconjg(a(3,3))

           c(1,1)=d(1,1)*b(1,1)+d(1,2)*b(2,1)+d(1,3)*b(3,1)
           c(1,2)=d(1,1)*b(1,2)+d(1,2)*b(2,2)+d(1,3)*b(3,2)
           c(1,3)=d(1,1)*b(1,3)+d(1,2)*b(2,3)+d(1,3)*b(3,3)

           c(2,1)=d(2,1)*b(1,1)+d(2,2)*b(2,1)+d(2,3)*b(3,1)
           c(2,2)=d(2,1)*b(1,2)+d(2,2)*b(2,2)+d(2,3)*b(3,2)
           c(2,3)=d(2,1)*b(1,3)+d(2,2)*b(2,3)+d(2,3)*b(3,3)

           c(3,1)=d(3,1)*b(1,1)+d(3,2)*b(2,1)+d(3,3)*b(3,1)
           c(3,2)=d(3,1)*b(1,2)+d(3,2)*b(2,2)+d(3,3)*b(3,2)
           c(3,3)=d(3,1)*b(1,3)+d(3,2)*b(2,3)+d(3,3)*b(3,3)

           return
           end

           subroutine su3_mult04(a,b,c)
           complex*16 a(3,3),b(3,3),c(3,3),d(3,3),e(3,3)


           e(1,1)=dconjg(a(1,1))
           e(1,2)=dconjg(a(2,1))
           e(1,3)=dconjg(a(3,1))

           e(2,1)=dconjg(a(1,2))
           e(2,2)=dconjg(a(2,2))
           e(2,3)=dconjg(a(3,2))

           e(3,1)=dconjg(a(1,3))
           e(3,2)=dconjg(a(2,3))
           e(3,3)=dconjg(a(3,3))

           d(1,1)=dconjg(b(1,1))
           d(1,2)=dconjg(b(2,1))
           d(1,3)=dconjg(b(3,1))

           d(2,1)=dconjg(b(1,2))
           d(2,2)=dconjg(b(2,2))
           d(2,3)=dconjg(b(3,2))

           d(3,1)=dconjg(b(1,3))
           d(3,2)=dconjg(b(2,3))
           d(3,3)=dconjg(b(3,3))

           c(1,1)=e(1,1)*d(1,1)+e(1,2)*d(2,1)+e(1,3)*d(3,1)
           c(1,2)=e(1,1)*d(1,2)+e(1,2)*d(2,2)+e(1,3)*d(3,2)
           c(1,3)=e(1,1)*d(1,3)+e(1,2)*d(2,3)+e(1,3)*d(3,3)

           c(2,1)=e(2,1)*d(1,1)+e(2,2)*d(2,1)+e(2,3)*d(3,1)
           c(2,2)=e(2,1)*d(1,2)+e(2,2)*d(2,2)+e(2,3)*d(3,2)
           c(2,3)=e(2,1)*d(1,3)+e(2,2)*d(2,3)+e(2,3)*d(3,3)

           c(3,1)=e(3,1)*d(1,1)+e(3,2)*d(2,1)+e(3,3)*d(3,1)
           c(3,2)=e(3,1)*d(1,2)+e(3,2)*d(2,2)+e(3,3)*d(3,2)
           c(3,3)=e(3,1)*d(1,3)+e(3,2)*d(2,3)+e(3,3)*d(3,3)

           return
	   end 
