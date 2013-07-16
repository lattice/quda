c=====================================================================
c	set_metrix
c=====================================================================
      subroutine set_metrix(gammi,gamm5i,sig,unit,delta3,epsl)
      complex*16 unit(4,4)
      complex*16 gammi(4,4,5),gamm5i(4,4,4),sig(4,4,3)
      complex*16 delta3(3,3),epsl(3,3)
      
	
	do i = 1, 3
	  do j = 1, 3
	    delta3(i,j)   = dcmplx(0.0d0,0.0d0)
	  end do
	    delta3(i,i)   = dcmplx(1.0d0,0.0d0)
	    epsl(i,i) = dcmplx(0.0d0,0.0d0)
	    epsl(i,mod(i,3)+1)  = dcmplx(1.0d0,0.0d0)
	    epsl(i,mod(i+1,3)+1)  = dcmplx(-1.0d0,0.0d0)
	end do

 

      do i = 1, 4
	  do j = 1, 4
	    unit(i,j)   = dcmplx(0.0d0,0.0d0)
	    do ik = 1,5
	    gammi(i,j,ik)  = dcmplx(0.0d0,0.0d0)
	    end do
	    do ik =1,4
	    gamm5i(i,j,ik)  = dcmplx(0.0d0,0.0d0)
	    end do
	    do ik =1,3
	    sig(i,j,ik)  = dcmplx(0.0d0,0.0d0)
	    end do
	  end do
	    unit(i,i)   = dcmplx(1.0d0,0.0d0)
	end do

c	gamma1

	gammi(1,4,1) = -dcmplx(0.0d0,1.0d0)
	gammi(2,3,1) = -dcmplx(0.0d0,1.0d0)
	gammi(3,2,1) =  dcmplx(0.0d0,1.0d0)
	gammi(4,1,1) =  dcmplx(0.0d0,1.0d0)

c	gamma2 

	gammi(1,4,2) = -dcmplx(1.0d0,0.0d0)
	gammi(2,3,2) =  dcmplx(1.0d0,0.0d0)
	gammi(3,2,2) =  dcmplx(1.0d0,0.0d0)
	gammi(4,1,2) = -dcmplx(1.0d0,0.0d0)

c	gamma3

	gammi(1,3,3) = -dcmplx(0.0d0,1.0d0)
	gammi(2,4,3) =  dcmplx(0.0d0,1.0d0)
	gammi(3,1,3) =  dcmplx(0.0d0,1.0d0)
	gammi(4,2,3) = -dcmplx(0.0d0,1.0d0)

c	gamma4 = ga5 in ps base

	gammi(1,3,5) = -dcmplx(1.0d0,0.0d0)
	gammi(2,4,5) = -dcmplx(1.0d0,0.0d0)
	gammi(3,1,5) = -dcmplx(1.0d0,0.0d0)
	gammi(4,2,5) = -dcmplx(1.0d0,0.0d0)

c	gamma5 = ga4 in ps base

	gammi(1,1,4) =  dcmplx(1.0d0,0.0d0)
	gammi(2,2,4) =  dcmplx(1.0d0,0.0d0)
	gammi(3,3,4) = -dcmplx(1.0d0,0.0d0)
	gammi(4,4,4) = -dcmplx(1.0d0,0.0d0)

!==========================================
c	gamma51 = sig51  in ps base

	sig(1,4,1) = -dcmplx(0.0d0,1.0d0)
	sig(2,3,1) = -dcmplx(0.0d0,1.0d0)
	sig(3,2,1) = -dcmplx(0.0d0,1.0d0)
	sig(4,1,1) = -dcmplx(0.0d0,1.0d0)

c	gamma52 = sig52  in ps base

	sig(1,4,2) = -dcmplx(1.0d0,0.0d0)
	sig(2,3,2) =  dcmplx(1.0d0,0.0d0)
	sig(3,2,2) = -dcmplx(1.0d0,0.0d0)
	sig(4,1,2) =  dcmplx(1.0d0,0.0d0)

c	gamma53 = sig53  in ps base

	sig(1,3,3) = -dcmplx(0.0d0,1.0d0)
	sig(2,4,3) =  dcmplx(0.0d0,1.0d0)
	sig(3,1,3) = -dcmplx(0.0d0,1.0d0)
	sig(4,2,3) =  dcmplx(0.0d0,1.0d0)

!==========================================
c	gamma54

	gamm5i(1,3,4) = -dcmplx(1.0d0,0.0d0)
	gamm5i(2,4,4) = -dcmplx(1.0d0,0.0d0)
	gamm5i(3,1,4) =  dcmplx(1.0d0,0.0d0)
	gamm5i(4,2,4) =  dcmplx(1.0d0,0.0d0)

c	sig523 = ga51 in ps base

!	gamm5i(1,2,1) = -dcmplx(1.0d0,0.0d0)
!	gamm5i(2,1,1) = -dcmplx(1.0d0,0.0d0)
!	gamm5i(3,4,1) =  dcmplx(1.0d0,0.0d0)
!	gamm5i(4,3,1) =  dcmplx(1.0d0,0.0d0)
	gamm5i(1,4,1) = -dcmplx(0.0d0,1.0d0)
	gamm5i(2,3,1) = -dcmplx(0.0d0,1.0d0)
	gamm5i(3,2,1) = -dcmplx(0.0d0,1.0d0)
	gamm5i(4,1,1) = -dcmplx(0.0d0,1.0d0)

c	sig531 = ga52 in ps base 

!	gamm5i(1,2,2) =  dcmplx(0.0d0,1.0d0)
!	gamm5i(2,1,2) = -dcmplx(0.0d0,1.0d0)
!	gamm5i(3,4,2) = -dcmplx(0.0d0,1.0d0)
!	gamm5i(4,3,2) =  dcmplx(0.0d0,1.0d0)
	gamm5i(1,4,2) = -dcmplx(1.0d0,0.0d0)
	gamm5i(2,3,2) =  dcmplx(1.0d0,0.0d0)
	gamm5i(3,2,2) = -dcmplx(1.0d0,0.0d0)
	gamm5i(4,1,2) =  dcmplx(1.0d0,0.0d0)

c	sig512 = ga53 in ps base
	
!	gamm5i(1,1,3) = -dcmplx(1.0d0,0.0d0)
!	gamm5i(2,2,3) =  dcmplx(1.0d0,0.0d0)
!	gamm5i(3,3,3) =  dcmplx(1.0d0,0.0d0)
!	gamm5i(4,4,3) = -dcmplx(1.0d0,0.0d0)
	gamm5i(1,3,3) = -dcmplx(0.0d0,1.0d0)
	gamm5i(2,4,3) =  dcmplx(0.0d0,1.0d0)
	gamm5i(3,1,3) = -dcmplx(0.0d0,1.0d0)
	gamm5i(4,2,3) =  dcmplx(0.0d0,1.0d0)
	
      
      END
