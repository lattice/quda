      subroutine set_geo(ns,ixyz,iup,idn)

        implicit none

	integer nsp
	integer ixyz(3,ns*ns*ns),iup(3,ns*ns*ns),idn(3,ns*ns*ns)
	integer isp,ix,iy,iz
	integer ns

	nsp=ns*ns*ns

	do ix=0,ns-1
	do iy=0,ns-1
	do iz=0,ns-1
	 isp=ns*(ns*iz+iy)+ix+1
	 ixyz(1,isp)=ix+1
	 ixyz(2,isp)=iy+1
	 ixyz(3,isp)=iz+1

	 if(ix.eq.ns-1) then
	 iup(1,isp)=isp-ns+1
 	 else
	 iup(1,isp)=isp+1
	 end if
	 if(ix.eq.0) then
	 idn(1,isp)=isp+ns-1
 	 else
	 idn(1,isp)=isp-1
	 end if

	 if(iy.eq.ns-1) then
	 iup(2,isp)=isp-(ns-1)*ns
 	 else
	 iup(2,isp)=isp+ns
	 end if
	 if(iy.eq.0) then
	 idn(2,isp)=isp+(ns-1)*ns
 	 else
	 idn(2,isp)=isp-ns
	 end if

	 if(iz.eq.ns-1) then
	 iup(3,isp)=isp-(ns-1)*ns*ns
 	 else
	 iup(3,isp)=isp+ns*ns
	 end if
	 if(iz.eq.0) then
	 idn(3,isp)=isp+(ns-1)*ns*ns
 	 else
	 idn(3,isp)=isp-ns*ns
	 end if
	
	end do
	end do
	end do

	return

        END
