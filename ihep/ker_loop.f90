	subroutine do_plq(v_l,iup,idn,nsp,trace)

	integer nsp,ieo
	integer iup(3,nsp),idn(3,nsp)
	complex*16 v_l(9,4,nsp)
	complex*16 trace,clover(9)
	integer ige(8,nsp)
	integer ine(8,nsp),para(10)
	integer :: isqr(10)=[1,1,1,9,9, 9,9,9,9,9]


	do ix=1,nsp
	ige(1,ix)=ix
	ige(5,ix)=ix
	ine(1,ix)=ix
	ine(5,ix)=ix
	do id=1,3
	ige(id+1,ix)=ix
	ige(id+5,ix)=idn(id,ix)
	ine(id+1,ix)=iup(id,ix)
	ine(id+5,ix)=idn(id,ix)
	end do
	end do

	trace=0.0d0

	do ix=1,nsp
	do nn=1,3
          mu=mod(nn,3)+2
          nu=mod(nn+1,3)+2

           do id=1,4
                para=[mu,nu,ix,1,nsp,id-1,0,0,0,0]
                call do_loop(para,v_l,ine,ige,clover,isqr)
                trace=trace+clover(1)+clover(5)+clover(9)
           end do
	end do
	end do
	
	trace=trace/(nsp*3*3*4)

	end


!       This is the subroutine to calculate the value of F_{\mu\nu}.
!
!       The definition of F_{\mu\nu} is the following:
!
!                 F = (i/8) ( C - C^ )
!
!       where C is the sum of four leaves of clover.
!       so, F is defined to be hermitian.
!
!       The values are stored in array
!       f_time (3,3,3,nsp)
!       f_space(3,3,3,nsp)
!
        subroutine get_fmunu(v_l,ine,isize,nsp,us2,bfield,efield)

        implicit none

        integer nsp,isize,nsize
        integer ine(8,nsp*(isize*2+1))
        integer ige(8,nsp*(isize*2+1))
        complex*16 v_l(9,4,nsp*(isize*2+1))
        complex*16 bfield(3,3,3,nsp),efield(3,3,3,nsp)
        integer :: isqr(10)=[1,1,1,9,9, 9,9,9,9,9]
        integer :: irect(20)=[0,1,1,0,1, 9,9,9,9,9, 1,0,1,1,0, 9,9,9,9,9]
        integer para(10)
        integer ix,iy,i,id,mu,nu,idr,nn
        real*8 us2,fac(2)

        complex*16 aux_add(3,3)
        complex*16 clover(3,3),clover2(3,3)

!        write(*,*) 'Calling getclover...'
!        fac(1)=1.0d0
!	fac(2)=us2
	fac=[1.0d0,us2]

        nsize=isize*2+1

        do ix=1,nsp*nsize
        do id=1,4
        ige(id  ,ix)=ix
        ige(id+4,ix)=ine(id+4,ix)
        end do
        end do

        do ix=1,nsp
        iy=ix+isize*nsp
!****** Color-electric componets first: mu=1,nu=2,3,4
        mu=1
        do nu=2,4

           aux_add=0.0d0
           do id=1,4
                para=[nu,mu,iy,nsize,nsp,id-1,0,0,0,0]
                call do_loop(para,v_l,ine,ige,clover,isqr)
                aux_add=aux_add+clover
           end do

           call do_im_su3(aux_add,clover)

           if(isize.eq.1) then

              efield(1:3,1:3,nu-1,ix)=0.125d0*clover(1:3,1:3)

           else

             aux_add=0.0d0
             do i=1,2
             do id=1,4
             if(mod(id+i,2).eq.0) then
                para=[nu,mu,iy,nsize,nsp,id-1,0,0,0,0]
                call do_loop(para,v_l,ine,ige,clover2,irect(i*10-9))
                aux_add=aux_add+clover2*fac(mod(id+i,2)+1)
             end if
             end do
             end do

             call do_im_su3(aux_add,clover2)

             efield(1:3,1:3,nu-1,ix)=0.125d0*(8*clover(1:3,1:3)-clover2(1:3,1:3)/us2)
!            efield(1:3,1:3,nu-1,ix)=0.125d0*(5*clover(1:3,1:3)-clover2(1:3,1:3)/us2)/3

           end if
           

        enddo

!****** Color-magnetic componets 
        do nn=1,3
          mu=mod(nn,3)+2
          nu=mod(nn+1,3)+2

           aux_add=0.0d0
           do id=1,4
                para=[mu,nu,iy,nsize,nsp,id-1,0,0,0,0]
                call do_loop(para,v_l,ine,ige,clover,isqr)
                aux_add=aux_add+clover
           end do

           call do_im_su3(aux_add,clover)

           if(isize.eq.1) then

              bfield(1:3,1:3,nn,ix)=0.125d0*clover(1:3,1:3)

           else

             aux_add=0.0d0
             do i=1,2
             do id=1,4
                para=[mu,nu,iy,nsize,nsp,id-1,0,0,0,0]
                call do_loop(para,v_l,ine,ige,clover2,irect(i*10-9))
                aux_add=aux_add+clover2
             end do
             end do

             call do_im_su3(aux_add,clover2)

             bfield(1:3,1:3,nn,ix)=0.125d0*(5*clover(1:3,1:3)-clover2(1:3,1:3)/us2)/3

           end if

        enddo

         enddo

         return
         end 

        subroutine do_im_su3(b,a)

        implicit none

        complex*16 a(3,3),b(3,3)
        real*8 rrr,ttt
        complex*16 tr

         a(1,1)=dcmplx(-2.0d0*dimag(b(1,1)),0.0d0)
         rrr=dimag(b(1,2))+dimag(b(2,1))
         ttt=dreal(b(1,2))-dreal(b(2,1))
         a(1,2)=dcmplx(-rrr,ttt)
         rrr=dimag(b(1,3))+dimag(b(3,1))
         ttt=dreal(b(1,3))-dreal(b(3,1))
         a(1,3)=dcmplx(-rrr,ttt)
         a(2,2)=dcmplx(-2.*dimag(b(2,2)),0.0d0)
         rrr=dimag(b(2,3))+dimag(b(3,2))
         ttt=dreal(b(2,3))-dreal(b(3,2))
         a(2,3)=dcmplx(-rrr,ttt)
         a(3,3)=dcmplx(-2.*dimag(b(3,3)),0.0d0)
         a(2,1)=dconjg(a(1,2))
         a(3,1)=dconjg(a(1,3))
         a(3,2)=dconjg(a(2,3))

         tr = (a(1,1)+a(2,2)+a(3,3))/3.00d0
         a(1,1)=a(1,1)-tr
         a(2,2)=a(2,2)-tr
         a(3,3)=a(3,3)-tr
        return
        end

        subroutine do_loop(para,v_l,ine,ige,clover,irule)

        implicit none

        integer para(10)
        complex*16 v_l(9,4,para(4)*para(5))
        integer ine(8,para(4)*para(5)),ige(8,para(4)*para(5))
        complex*16 clover(9),temp(9)
        integer irule(10)
        integer mu,nu,iy,nsize,nsp,idr
        integer index1,index2,idag,iz,i,idag2,iw,iv

        mu=para(1)
        nu=para(2)
        iy=para(3)
        nsize=para(4)
        nsp=para(5)
        idr=para(6)

        idag=idr/2
        index1=mod(idr+1,2)*mu+mod(idr,2)*nu
        iz=ige(index1+idag*4,iy)
        clover(1:9)=v_l(1:9,index1,iz)

        do i=1,10
        if(irule(i).ne.9) then
        idr=mod(idr+irule(i)+4,4)
        idag2=idr/2
        index2=mod(idr+1,2)*mu+mod(idr,2)*nu
        iv=ine(index1+idag*4,iy)
        iw=ige(index2+idag2*4,iv)
        if(i.gt.1)idag=0
         SELECT CASE (idag*2+idag2)
           CASE (0)
                call su3_mult01(clover,v_l(1,index2,iw),temp)
           CASE (1)
                call su3_mult02(clover,v_l(1,index2,iw),temp)
           CASE (2)
                call su3_mult03(clover,v_l(1,index2,iw),temp)
           CASE (3)
                call su3_mult04(clover,v_l(1,index2,iw),temp)
           CASE DEFAULT
         end SELECT
!       if(para(6).eq.0)call print_su3(temp)
        clover=temp
        idag=idag2
        index1=index2
        iy=iv
        end if
        end do

        return
        end
        
        subroutine print_su3(a)
        
        complex*16 a(3,3)

        do i=1,3
        write(*,'(2f13.5,a,2f13.5,a,2f13.5)')a(1,i),' ',a(2,i),' ',a(3,i)
        end do
        print*, ' '
        return
        end
