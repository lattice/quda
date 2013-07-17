      subroutine three_point(prop,propseq,u,ns,r,&
                threepf,ntime,myid)

      implicit none

      integer ns,r,nsp,ntime,myid
      complex*16 prop(3,3,4,4,ns*ns*ns*ntime)
      complex*16 propt1(3,3,4,4)
      complex*16 propt2(3,3,4,4)
      complex*16 propseq(3,3,4,4,ns*ns*ns*ntime)
      complex*16 u(9,4,ns*ns*ns*ntime)
      complex*16 u_eq(9,4,ns*ns*ns)
      complex*16 u3(9,4,ns*ns*ns*3)
      complex*16 threepf(ntime*2)
      
      complex*16 unit(4,4),delta3(3,3),epsl(3,3)
      complex*16 gammi(4,4,5),gamm5i(4,4,4),sig5(4,4,3)
      complex*16 eipx,trace(2),u_s,tmp(2),plq
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

!      u_eq(1:9,1:4,1:nsp) = u(1:9,1:4,1:nsp)
!      call do_plq(u_eq,iup,idn,nsp,plq)
!      if(myid.eq.0)write(*,*) "plq=",plq

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

!      if(myid.eq.0) write(*,*) "bfield=",bfield(1,1,1,1)
!      do ix=1,nsp
!      if(myid.eq.0) write(*,*) ix, "bfield=",bfield(1,1,1,ix)
!      enddo
!      write(*,*) "bfield=",bfield(1,1,1,1)

	threepf=0.0d0

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
              call do_trace(propt1,propseq(1,1,1,1,it*nsp+isp_n),&
                         gamm5i(1,1,id1),gamm5i(1,1,id1),trace(1))
              tmp(1) = tmp(1) + trace(1)
              enddo

            do id1=1,3
                propt2 = 0.0d0
                do ic1=1,3
                do ic2=1,3
                do ic3=1,3
                  propt2(ic1,ic3,1:4,1:4)=propt2(ic1,ic3,1:4,1:4) +&
                  bfield(ic1,ic2,id1,isp_n,it+1)*propseq(ic2,ic3,1:4,1:4,it*nsp+isp)
                enddo
                enddo
                enddo
              call do_trace(propt1,propt2,gamm5i(1,1,id1),unit,trace(2))
              tmp(2) = tmp(2) + trace(2)
            enddo
            endif
          enddo
          enddo
          enddo
          tmp(1:2) = tmp(1:2)/(icount*3)
          threepf(it*2+1) = threepf(it*2+1) + tmp(1)
          threepf(it*2+2) = threepf(it*2+2) + tmp(2)
	end do
	end do
      return
      end

!------------------------------------------------
!prop[ic_sink,ic_source,is_sink,is_source,insp]
!do_trace(prop(dag),prop,gamma_source,gamma_sink)
!------------------------------------------------
      subroutine do_trace(prop1,prop2,gamma1,gamma2,trace)

	complex*16 prop1(3,3,4,4),prop2(3,3,4,4)
	complex*16 gamma1(4,4),gamma2(4,4)
	complex*16 trace
	complex*16 ga1(4),ga2(4)
	integer iga1(4),iga2(4)

	trace=0.0d0

	do id4=1,4
	do id3=1,4
	if(cdabs(gamma1(id3,id4)).gt.0.001d0) then
		do id2=1,4
		do id1=1,4
		if(cdabs(gamma2(id1,id2)).gt.0.001d0) then
		do ic2=1,3
		do ic1=1,3
		trace=trace+dconjg(prop1(ic1,ic2,id1,id4))*gamma2(id1,id2)*&
			prop2(ic1,ic2,id2,id3)*gamma1(id3,id4)
		end do
		end do
		end if
		end do
		end do
	end if
	end do
	end do

	return

	end
