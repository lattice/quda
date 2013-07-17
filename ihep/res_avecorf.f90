!------------------------------------------------------------------------------
      	subroutine avecorf(g1, ncon, nt, nboot, i_nmo,i_num,result)
!------------------------------------------------------------------------------

        implicit real*8 (a-h,o-z)


      	real*8, dimension(nt, ncon)   :: g1
	real*8, dimension(4,nt)   :: result
        real*8, dimension(nt)   :: simple
      	real*8, dimension(ncon, nt)   :: d1
      	real*8, dimension(0:nboot,nt) :: av1,mav1
      	real*8, dimension(nboot)      :: xboot,xord
      	real*8                        :: ave, sig,ap,smax
	integer :: isign

      	real*8, dimension(nt) :: gave,gsig,gberr,mave,mberr,gavep,gavep_err

!      	common /seed/ iseedin
      	iseed = 1

      	do iboot=0,nboot

! if iboot = 0 use original sample, else construct bootstrap sample

            do it =1,nt
                simple(it) =0.0d0
              do iaa=1,ncon
                simple(it)=simple(it)+g1(it,iaa)
              enddo
            enddo

        isign=1
        if(dabs(simple(4)+simple(nt-4+2)).lt.dabs(simple(4))) isign=-1
!       use sinh to fit instand of cosh,if simple(it) = -simple(nt-it+2)
	if(dabs(simple(4)).gt.1000*dabs(simple(nt-4+2))) isign=0

          if(iboot .eq. 0)then

            do iaa =1, ncon
              do it=1,nt
                d1(iaa,it) = g1(it,iaa)
              enddo
            enddo

          else

            do iaa =1, ncon
              ipk = ipick(iseed,ncon)
              do it=1,nt
                d1(iaa,it) = g1(it,ipk)
              enddo
            enddo

          endif

! calculate average cor. fcn. and mass function for current sample

          do it = 1,nt

            call avedat(d1(1,it), ncon, ave, sig)

            av1(iboot,it) = ave

            if(iboot .eq. 0)then
              gave(it) = ave
              gsig(it) = sig
            endif

          enddo

	  smax = 10.0d0
          do it = 2,nt 
 	    gs1 = av1(iboot,it-1)
            gs2 = av1(iboot,it)
!----------------------------------------------------------------------
!      The last two variables in the following subroutine is the 
!      possible minimum and maximum of the effetive mass, which 
!      guarantee the Newton solver works smoothly.
!----------------------------------------------------------------------
	    if(isign.eq.1) then
	    call eff_cosh(nt, it-2, gs1, gs2, eff_mass, 0.0d0, smax)
	    else	  
	    call eff_sinh(nt,it-2,gs1,gs2,eff_mass, 0.0d0, smax)
	    end if
           if(isign.eq.0) eff_mass =  dlog(gs1/gs2)

	    mav1(iboot, it-1) = eff_mass
!	    write(*,100), it, gs1, gs2, eff_mass, iboot
            gs1 = gs2

            if(iboot .eq. 0)then
              mave(it-1) = mav1(iboot,it-1)
            endif

          enddo

          mav1(iboot,nt) = 1.0
          mave(nt) = 0.0

      	enddo
!100 	format(1x, i6, 3e14.6 i4)

! calculate bootstrap errors
       	do it = 1,nt
          do iboot = 0,nboot
!             write(89,*)iboot,av1(iboot,it)
          enddo
        enddo


      	do it=1,nt

          do iboot=1,nboot
            xboot(iboot) = av1(iboot,it)
          enddo

          call order(xboot,xord,nboot)

          iup = 0.85*nboot
          idwn = 0.16*nboot

          gberr(it)= 0.5*(xord(iup)-xord(idwn))

          do iboot=1,nboot
            xboot(iboot) = mav1(iboot,it)
          enddo

          call order(xboot,xord,nboot)
          iup = 0.85*nboot
          idwn = 0.16*nboot
!	  print*, xord(iup),xord(idwn)
          mberr(it)= 0.5*(xord(iup)-xord(idwn))

        enddo

! write out correlation function for original sample

!        write(88,*) 'mass index = ', i_num
!        write(88,'(i4,2e15.7)') (it,gave(it),gberr(it),it=1,nt)

! write out mass function for original sample

!        write(12,'(/"   Effective masses "/)')
!	write(12,*) '-----------------------------'
!        write(12,*) 'mass index = ', i_num
!        write(12,'(i4,2e15.7)') (it,mave(it),mberr(it),it=1,nt)

	do it = 1,nt
	result(1,it)= mave(it)
	result(2,it)= mberr(it)
	result(3,it)= gave(it)
	result(4,it)= gberr(it)
	end do

!	write(*,'(i4,2e15.7)') (it,mave(it),mberr(it),it=1,nt)

      	end subroutine avecorf
!!=======================================================================
!
!      	subroutine avedat(dat, ncon, ave, sig)
!
!      	implicit real*8(a-h,o-z)
!      
!      	real*8, dimension(ncon) :: dat
!      	real*8 :: temp, ave, sig
!
!!  first calculate mean values
!
!       	ave=0.0d0
!
!       	do icon =1,ncon
!          ave= ave + dat(icon)
!       	enddo
!
!       	ave = ave/dble(ncon)
!
!        temp = 0.0d0
!
!        do icon =1,ncon
!          temp = temp + (dat(icon)-ave)*(dat(icon)-ave)
!        enddo
!       
!	if(ncon .gt. 1)then
!          sig = dsqrt(temp/dfloat(ncon-1)/dfloat(ncon))
!        else
!          sig = 0.0
!        endif
!
!        end subroutine avedat
!
!=======================================================================

      	subroutine eff_cosh(nt, it, a, b, c, x1, x2)

      	implicit real*8(a-h,o-z)
      	integer, parameter :: nmax = 1000
      	real*8,  parameter :: erro = 0.0000001d0
      	integer    :: it, nt, iter
      	real*8 :: a, b, c, val, pt1, pt2
      	real*8 :: x1, x2, rtn, dx
      	real*8 :: fun, dfdx, ch1, ch2, sh1, sh2

      	pt1 = dble(nt/2-it)
      	pt2 = dble(nt/2-it-1)

      	rtn = 0.5d0 * ( x1 + x2 )

      	do iter = 1, nmax

          ch1 = dcosh( pt1 * rtn )
          ch2 = dcosh( pt2 * rtn )
          sh1 = dsinh( pt1 * rtn )
          sh2 = dsinh( pt2 * rtn )

          fun  = a * ch2 - b * ch1
          dfdx = a * sh2 * pt2 - b * sh1 * pt1
          dx   = fun/dfdx
          rtn  = rtn - dx

          if (dabs(dx) .le. erro) goto 100

      	end do

 100  	c = dabs(rtn)

      	end subroutine eff_cosh
!=======================================================================

        subroutine eff_sinh(nt, it, a, b, c, x1, x2)

        implicit real*8(a-h,o-z)
        integer, parameter :: nmax = 1000
        real*8,  parameter :: erro = 0.0000001d0
        integer    :: it, nt, iter
        real*8 :: a, b, c, val, pt1, pt2
        real*8 :: x1, x2, rtn, dx
        real*8 :: fun, dfdx, ch1, ch2, sh1, sh2

        pt1 = dble(nt/2-it)
        pt2 = dble(nt/2-it-1)

        rtn = 0.5d0 * ( x1 + x2 )

        do iter = 1, nmax

          ch1 = dcosh( pt1 * rtn )
          ch2 = dcosh( pt2 * rtn )
          sh1 = dsinh( pt1 * rtn )
          sh2 = dsinh( pt2 * rtn )

          fun  = a * sh2 - b * sh1
          dfdx = a * ch2 * pt2 - b * ch1 * pt1
          dx   = fun/dfdx
          rtn  = rtn - dx

          if (dabs(dx) .le. erro) goto 100

        end do

 100    c = dabs(rtn)

        end subroutine eff_sinh
!!=======================================================================
!
!      	subroutine order(a,b,n)
!
!      	implicit real*8(a-h,o-z)
!
!      	real*8, dimension(n) :: a,b
!
!      	b(1) = a(1)
!
!      	do i=2,n
!        do j=1,i-1
!          if(a(i) .lt. b(j))then
!            do l=i,j+1,-1
!              b(l) = b(l-1)
!            enddo
!            b(j) = a(i)
!            exit
!          end if
!          b(i) = a(i)
!        enddo
!      	enddo
!
!      	end subroutine order
!
!!======================================================================
!        integer function ipick(iseed,mmax)
!
!!	implicit real*8(a-h,o-z)
!
!        xmax=dfloat(mmax)
!        x = uran(iseed)*xmax
!        ipick = x
!        ipick = ipick+1
!
!        return
!        end
!!=======================================================================
!
!        REAL FUNCTION URAN(SEED)
!
!	IMPLICIT REAL*8 (A-H, O-Z)
!
!!       UNIRAN RANDOM NUMBER GENERATOR
!        INTEGER B2E15,B2E16,HI15,HI31,LOW15,LOWPRD,MODLUS
!        INTEGER MULT1,MULT2,OVFLOW,SEED
!        DATA MULT1,MULT2/24112,26143/
!        DATA B2E15,B2E16,MODLUS/32768,65536,2147483647/
!
!        HI15=SEED/B2E16
!        LOWPRD=(SEED-HI15*B2E16)*MULT1
!        LOW15=LOWPRD/B2E16
!        HI31=HI15*MULT1+LOW15
!        OVFLOW=HI31/B2E15
!
!      
! SEED=(((LOWPRD-LOW15*B2E16)-MODLUS)+(HI31-OVFLOW*B2E15)*B2E16)+OVFLOW
!
!        IF(SEED.LT.0)SEED=SEED+MODLUS
!
!        HI15=SEED/B2E16
!        LOWPRD=(SEED-HI15*B2E16)*MULT2
!        LOW15=LOWPRD/B2E16
!        HI31=HI15*MULT2+LOW15
!        OVFLOW=HI31/B2E15
!
! SEED=(((LOWPRD-LOW15*B2E16)-MODLUS)+(HI31-OVFLOW*B2E15)*B2E16)+OVFLOW
!
!        IF(SEED.LT.0)SEED=SEED+MODLUS
!
!        URAN=(2*(SEED/256)+1)/16777216.0
!        RETURN
!        END
!!=======================================================================
!
!!=======================================================================




