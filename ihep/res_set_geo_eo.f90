      subroutine set_geo_eo(ns,ixyz,iup,idn)

        implicit none

        integer nsp
        integer ixyz(3,ns*ns*ns),iup(3,ns*ns*ns),idn(3,ns*ns*ns)
        integer isp,ix,iy,iz
        integer ns

!*******************************************************************
!*****          This is the geometry subroutine               ******
!*****  Three volume lattice points are labled                ******
!*****  as the following:
!*****                      
!*****  For 1<= ix1 <=nspace_x
!*****      1<= ix2 <=nspace_y
!*****      1<= ix3 <=nspace_z
!*****  The point is labeled as:
!*****                          
!*****    (ix1+1)/2 +(ix2-1)*(nspace/2)+(ix3-1)*(nspace/2)*nspace+vol3_h 
!*****                  for  ix1+ix2+ix3= even
!*****                          
!*****    (ix1+1)/2 +(ix2-1)*(nspace/2)+(ix3-1)*(nspace/2)*nspace
!*****                  for  ix1+ix2+ix3= odd 
!*****                          
!*******************************************************************
       integer ix1,ix2,ix3
       integer ixp1,ixp2,ixp3
       integer ixm1,ixm2,ixm3
	integer nspace_x,nspace_y,nspace_z,ieven
	integer nspace_x2,nspace_y2,nspace_z2,nsp_half

	nspace_x=ns
	nspace_y=ns
	nspace_z=ns
	nspace_x2=nspace_x/2
	nspace_y2=nspace_y/2
	nspace_z2=nspace_z/2
	nsp_half=nspace_x*nspace_y*nspace_z2
	

!   Now the 3-volume stuff

       do ix3=1,nspace_z
        do ix2=1,nspace_y
         do ix1=1,nspace_x
         
         iy=ix1+((ix2-1)+(ix3-1)*nspace_y)*nspace_z
! The neighbors : +-direction 1
         ixp1=ix1+1
         if(ixp1.gt.nspace_x) ixp1=1
         ixm1=ix1-1
         if(ixm1.lt.1) ixm1=nspace_x
! The neighbors : +-direction 2
         ixp2=ix2+1
         if(ixp2.gt.nspace_y) ixp2=1
         ixm2=ix2-1
         if(ixm2.lt.1) ixm2=nspace_y
! The neighbors : +-direction 3
         ixp3=ix3+1
         if(ixp3.gt.nspace_z) ixp3=1
         ixm3=ix3-1
         if(ixm3.lt.1) ixm3=nspace_z

! Now fill in the geometry arrays     

         ieven=ix1+ix2+ix3

         if(mod(ieven,2).eq.0) then
          ix=(ix1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y+nsp_half
         ixyz(1,ix)=ix1
         ixyz(2,ix)=ix2
         ixyz(3,ix)=ix3
         iup(1,ix)=(ixp1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y
         idn(1,ix)=(ixm1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y
         iup(2,ix)=(ix1+1)/2 &
           +(ixp2-1)*(nspace_x2)&
           +(ix3 -1)*(nspace_x2)*nspace_y
         idn(2,ix)=(ix1+1)/2 &
           +(ixm2-1)*(nspace_x2)&
           +(ix3 -1)*(nspace_x2)*nspace_y
         iup(3,ix)=(ix1+1)/2 &
           +(ix2 -1)*(nspace_x2)&
           +(ixp3-1)*(nspace_x2)*nspace_y
         idn(3,ix)=(ix1+1)/2 &
           +(ix2 -1)*(nspace_x2)&
           +(ixm3-1)*(nspace_x2)*nspace_y
        else
         ix=(ix1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y
         ixyz(1,ix)=ix1
         ixyz(2,ix)=ix2
         ixyz(3,ix)=ix3
         iup(1,ix)=(ixp1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y+nsp_half
         idn(1,ix)=(ixm1+1)/2 &
           +(ix2-1)*(nspace_x2)&
           +(ix3-1)*(nspace_x2)*nspace_y+nsp_half
         iup(2,ix)=(ix1+1)/2 &
           +(ixp2-1)*(nspace_x2)&
           +(ix3 -1)*(nspace_x2)*nspace_y+nsp_half
         idn(2,ix)=(ix1+1)/2 &
           +(ixm2-1)*(nspace_x2)&
           +(ix3 -1)*(nspace_x2)*nspace_y+nsp_half
         iup(3,ix)=(ix1+1)/2 &
           +(ix2 -1)*(nspace_x2)&
           +(ixp3-1)*(nspace_x2)*nspace_y+nsp_half
         idn(3,ix)=(ix1+1)/2 &
           +(ix2 -1)*(nspace_x2)&
           +(ixm3-1)*(nspace_x2)*nspace_y+nsp_half
         endif

         enddo
        enddo
       enddo

       return
       end

