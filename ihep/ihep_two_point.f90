      subroutine two_point(prop,u,ns,nmom,&
                twopf,ntime)

      implicit none
      integer ns,nmom
      integer nsp,ntime
      complex*16 prop(3,3,4,4,ns*ns*ns*ntime)
      complex*16 u(3,3,4,ns*ns*ns)
      complex*16 twopf(nmom*ntime)

      complex*16 unit(4,4),delta3(3,3),epsl(3,3)
      complex*16 gammi(4,4,5),gamm5i(4,4,4),sig5(4,4,3)
      complex*16 eipx,trace,u_s,tmp
      integer iup(3,ns*ns*ns),idn(3,ns*ns*ns),ixyz(3,ns*ns*ns)
      real*8  fmom,px,py,pz
      integer it
      integer id1,id2,isp,isp_n,imom,icount,ipx,ipy,ipz

	nsp=ns*ns*ns
        call set_metrix(gammi,gamm5i,sig5,unit,delta3,epsl)
	u_s=1.0d0
	call set_geo(ns,ixyz,iup,idn)

!	call do_plq(u,iup,idn,nsp,trace)
!	if(myid.eq.0)write(*,"(a,f13.5)") "plq=",dble(trace)

	twopf=0.0d0
	fmom=2*3.14159265358979323d0/ns

	do it=0,ntime-1
	do isp=1,nsp
        tmp = 0
        do id1=1,3
          call do_trace(prop(1,1,1,1,it*nsp+isp),prop(1,1,1,1,it*nsp+isp),&
                        gamm5i(1,1,id1),gamm5i(1,1,id1),trace)
          tmp = tmp + trace
        enddo
        trace = tmp/3

        do imom=0,nmom-1
          icount=0
          eipx=0
          do ipx=0,imom
          do ipy=0,imom
          do ipz=0,imom
            if(ipx*ipx + ipy*ipy +ipz*ipz .eq. imom) then
              icount = icount +1
              px = ipx*fmom 
              py = ipy*fmom 
              pz = ipz*fmom
              eipx = eipx +dcos((ixyz(1,isp)-1)*px +&
                   (ixyz(2,isp)-1)*py +(ixyz(3,isp)-1)*pz)
            endif
          enddo
          enddo
          enddo
        twopf(it*nmom+imom+1)=twopf(it*nmom+imom+1)+ trace*eipx/icount
        enddo

        enddo
        enddo
 
      return
 
      end
!-----------------------------------

