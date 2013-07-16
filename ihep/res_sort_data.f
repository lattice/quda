
        subroutine sort_gauge_quda(u,nsp,myid)

	 integer nsp,myid
         real*8 u(18*4*nsp)
         real*8 v(18*4*nsp)
         integer ierr,nnn,lrec,iread,it,k,isp,ix,id,ic,ipart
         integer ic1,ic2,id1,id2
         integer ind(nsp),ns

	ns=nsp**(1.0d0/3)
	if(ns*ns*ns.lt.nsp) ns=ns+1
	if(ns*ns*ns.gt.nsp) ns=ns-1
	if(ns*ns*ns.ne.nsp) print*, ns,nsp

	do iz=0,ns-1
	do iy=0,ns-1
	do ix=0,ns-1
		isp=ix+ns*iy+ns*ns*iz
		isp1=mod(ix+iy+iz,2)*nsp/2+isp/2+1
		ind(isp+1)=isp1
	end do
	end do
	end do
	
         
	   do isp=0,nsp-1
	   do id=0,3
           do ic=1,18
           ic1=mod(ic-1,6)/2
           ic2=(ic-1)/6
           ic0=mod(ic-1,2)+ic2*2+ic1*6+1
               v(ic0+mod(id+1,4)*18+isp*72)=
     &			u(ic+(ind(isp+1)-1)*18+id*nsp*18)
           end do
           end do
           end do

	   do isp=1,nsp*4*18           
              u(isp)=v(isp)
           end do

!	if(myid.eq.0) print*,"sort gauge done"

	return
       end

        subroutine sort_prop_quda(u,nsp,myid)

	 integer nsp
         real*8 u(6*3*4*4*nsp)
         real*8 v(6*3*4*4*nsp)
         integer ind(nsp),ns

	ns=nsp**(1.0d0/3)
	if(ns*ns*ns.lt.nsp) ns=ns+1
	if(ns*ns*ns.gt.nsp) ns=ns-1
	if(ns*ns*ns.ne.nsp) print*, ns,nsp

	do iz=0,ns-1
	do iy=0,ns-1
	do ix=0,ns-1
		isp=ix+ns*iy+ns*ns*iz
		isp1=mod(ix+iy+iz,2)*nsp/2+isp/2+1
		ind(isp+1)=isp1
	end do
	end do
	end do
	
         
	   do isp=0,nsp-1
	   do id2=0,3
	   do id1=0,3
           do ic2=0,2
           do ic1=1,6
               v(ic1+ic2*6+id1*18+id2*72+isp*288)=
     &		u(ic1+id1*6+(ind(isp+1)-1)*24+ic2*24*nsp+id2*72*nsp)
           end do
           end do
           end do
           end do
           end do

	   do isp=1,nsp*288
              u(isp)=v(isp)
           end do

!	if(myid.eq.0) print*,"sort prop done"

	return
       end

        subroutine sort_wilvec_quda(u,nsp,myid)

	 integer nsp
         real*8 u(6*4*nsp)
         real*8 v(6*4*nsp)
         integer ind(nsp),ns

	ns=nsp**(1.0d0/3)
	if(ns*ns*ns.lt.nsp) ns=ns+1
	if(ns*ns*ns.gt.nsp) ns=ns-1
	if(ns*ns*ns.ne.nsp) print*, ns,nsp

	do iz=0,ns-1
	do iy=0,ns-1
	do ix=0,ns-1
		isp=ix+ns*iy+ns*ns*iz
		isp1=mod(ix+iy+iz,2)*nsp/2+isp/2+1
		ind(isp+1)=isp1
	end do
	end do
	end do
	
         
	   do isp=0,nsp-1
	   do id1=0,3
           do ic1=1,6
               v(ic1+id1*6+isp*24)=
     &		u(ic1+id1*6+(ind(isp+1)-1)*24)
           end do
           end do
           end do

	   do isp=1,nsp*24
              u(isp)=v(isp)
           end do

!	if(myid.eq.0) print*,"sort prop done"

	return
       end

        subroutine sort_gauge_kentucky(u,nsp)

	 integer nsp
         real*8 u(18*4*nsp)
         real*8 v(18*4*nsp)
         integer ierr,nnn,lrec,iread,it,k,isp,ix,id,ic,ipart
         integer ic1,ic2,id1,id2
         
	   do isp=0,nsp-1
	   do id=0,3
           do ic=1,18
               v(ic+mod(id+1,4)*18+isp*72)=u(isp+1+(ic-1)*nsp+id*nsp*18)
           end do
           end do
           end do

	   do isp=1,nsp*4*18           
              u(isp)=v(isp)
           end do

	return
       end

        subroutine sort_prop_kentucky(u,nsp)

	 integer nsp
         real*8 u(6*3*4*4*nsp)
         real*8 v(6*3*4*4*nsp)
         
	   do isp=0,nsp-1
	   do id2=0,3
	   do id1=0,3
           do ic2=0,2
           do ic1=0,2
           do ipart=0,1
               v(ipart+1+ic1*2+ic2*6+id1*18+id2*72+isp*288)=
     &		u(isp+1+nsp*(3*(4*(2*(3*id2+ic2)+ipart)+id1)+ic1))
           end do
           end do
           end do
           end do
           end do
           end do

	   do isp=1,nsp*16*18
              u(isp)=v(isp)
           end do

	return
       end

        subroutine sort_wilvec_kentucky(u,nsp)

	 integer nsp
         real*8 u(6*4*nsp)
         real*8 v(6*4*nsp)
         
	   do isp=0,nsp-1
	   do id1=0,3
           do ic1=0,2
           do ipart=0,1
               v(ipart+1+ic1*2+id1*6+isp*24)=
     &		u(isp+1+nsp*(3*(4*ipart+id1)+ic1))
           end do
           end do
           end do
           end do

	   do isp=1,nsp*24
              u(isp)=v(isp)
           end do

	return
       end

	subroutine print_for(str)
	
	character*100 str
	
	print*,str  
	
	end
