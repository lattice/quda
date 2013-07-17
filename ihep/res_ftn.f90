!=======================================================================

        subroutine avedat(dat, ncon, ave, sig)

        implicit real*8(a-h,o-z)

        real*8, dimension(ncon) :: dat
        real*8 :: temp, ave, sig

!  first calculate mean values

        ave=0.0d0

        do icon =1,ncon
          ave= ave + dat(icon)
        enddo

        ave = ave/dble(ncon)

        temp = 0.0d0

        do icon =1,ncon
          temp = temp + (dat(icon)-ave)*(dat(icon)-ave)
        enddo

        if(ncon .gt. 1)then
          sig = dsqrt(temp/dfloat(ncon-1)/dfloat(ncon))
        else
          sig = 0.0
        endif

        end subroutine avedat


!=======================================================================
        REAL FUNCTION URAN(SEED)
!       UNIRAN RANDOM NUMBER GENERATOR
        INTEGER B2E15,B2E16,HI15,HI31,LOW15,LOWPRD,MODLUS
        INTEGER MULT1,MULT2,OVFLOW,SEED
        DATA MULT1,MULT2/24112,26143/
        DATA B2E15,B2E16,MODLUS/32768,65536,2147483647/

        HI15=SEED/B2E16
        LOWPRD=(SEED-HI15*B2E16)*MULT1
        LOW15=LOWPRD/B2E16
        HI31=HI15*MULT1+LOW15
        OVFLOW=HI31/B2E15
        SEED=(((LOWPRD-LOW15*B2E16)-MODLUS)+(HI31-OVFLOW*B2E15)*B2E16)+OVFLOW

        IF(SEED.LT.0)SEED=SEED+MODLUS

        HI15=SEED/B2E16
        LOWPRD=(SEED-HI15*B2E16)*MULT2
        LOW15=LOWPRD/B2E16
        HI31=HI15*MULT2+LOW15
        OVFLOW=HI31/B2E15
        SEED=(((LOWPRD-LOW15*B2E16)-MODLUS)+(HI31-OVFLOW*B2E15)*B2E16)+OVFLOW

        IF(SEED.LT.0)SEED=SEED+MODLUS

        URAN=(2*(SEED/256)+1)/16777216.0
        RETURN
        END
!======================================================================
        integer function ipick(iseed,mmax)
!        implicit real*8(a-h,o-z)
        xmax=dfloat(mmax)
        x = uran(iseed)*xmax
!        write(*,*)x
        ipick = x
        ipick = ipick+1

        return
        end
!=======================================================================
      subroutine order(a,b,n)

! order elements in a array
      implicit real*8(a-h,o-z)
      real*8, dimension(n) :: a,b

      b(1) = a(1)

      do i=2,n
! check element i against elements already in output array
        do j=1,i-1
          if(a(i) .lt. b(j))then
! if a(i) less than b(j), lower position of elements >= j in b
            do l=i,j+1,-1
              b(l) = b(l-1)
            enddo
! insert a(i) in j'th position and exit this loop
            b(j) = a(i)
            exit
          end if
          b(i) = a(i)
        enddo
      enddo

      end subroutine order

