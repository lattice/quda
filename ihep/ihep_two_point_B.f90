      subroutine two_point_B(prop,u,ns,r,&
                twopf,ntime)

      implicit none
      integer ns,r
      integer nsp,ntime
      complex*16 prop(3,3,4,4,ns*ns*ns*ntime)
      complex*16 propt1(3,3,4,4)
      complex*16 propt2(3,3,4,4)
      complex*16 u(9,4,ns*ns*ns*ntime)
      complex*16 u_eq(9,4,ns*ns*ns)
      complex*16 u3(9,4,ns*ns*ns*3)
      complex*16 twopf(ntime)

      complex*16 unit(4,4),delta3(3,3),epsl(3,3)
      complex*16 gammi(4,4,5),gamm5i(4,4,4),sig5(4,4,3)
      complex*16 eipx,trace,u_s,tmp,plq
      integer iup(3,ns*ns*ns),idn(3,ns*ns*ns),ixyz(3,ns*ns*ns)
      integer id,id1,id2,isp,isp_n,icount,irx,iry,irz,ix,iy,iz
      integer it,ic1,ic2,ic3,is1,is2

      integer isize
      real*8 us2
      complex*16 bfield(3,3,3,ns*ns*ns,ntime),efield(3,3,3,ns*ns*ns)
      integer ine(8,ns*ns*ns*3)

      nsp=ns*ns*ns
      call set_metrix(gammi,gamm5i,sig5,unit,delta3,epsl)
      u_s=1.0d0
      call set_geo(ns,ixyz,iup,idn)

      !call do_plq(u,iup,idn,nsp,trace)
      !write(*,"(a,f13.5)") "plq=",dble(trace)

      isize = 1
      us2=1
      do ix=1,nsp
        iy = ix+nsp
        ine(1,iy)=iy
        ine(5,iy)=iy
        do id=1,3
          ine(id+1,iy)=iup(id,ix)+nsp
          ine(id+5,iy)=idn(id,ix)+nsp
        end do
      end do

      do it=0,ntime-1
        u3=0.0;		
        do ix=1,nsp
          u3(1:9,1:4,nsp+ix)=u(1:9,1:4,it*nsp+ix)
        enddo
        call get_fmunu(u3,ine,isize,nsp,us2,bfield(1,1,1,1,it+1),efield)
      end do

	twopf=0.0d0

	do it=0,ntime-1
	do isp=1,nsp
          tmp = 0.0d0
          trace = 0.0d0
          propt1(1:3,1:3,1:4,1:4)=prop(1:3,1:3,1:4,1:4,it*nsp+isp)
          icount=0
          do irx=-r,r
          do iry=-r,r
          do irz=-r,r
            if(irx*irx + iry*iry +irz*irz .eq. r) then
              icount = icount +1
              ix = mod((ixyz(1,isp) +irx -1 +ns),ns)
              iy = mod((ixyz(2,isp) +iry -1 +ns),ns)
              iz = mod((ixyz(3,isp) +irz -1 +ns),ns)
              isp_n = ns*(ns*iz +iy) +ix +1


            do id1=1,3
                propt2 = 0.0d0
                do ic1=1,3
                do ic2=1,3
                do ic3=1,3
                  propt2(ic1,ic3,1:4,1:4)=propt2(ic1,ic3,1:4,1:4) +&
                  bfield(ic1,ic2,id1,isp_n,it+1)*propt1(ic2,ic3,1:4,1:4)
                enddo
                enddo
                enddo
              call do_trace(propt1,propt2,gamm5i(1,1,id1),unit,trace)
              tmp = tmp+ trace
            enddo
            endif
          enddo
          enddo
          enddo
          tmp = tmp/(icount*3)
          twopf(it+1) = twopf(it+1) + tmp
	end do
	end do
      return
      end
!-----------------------------------

