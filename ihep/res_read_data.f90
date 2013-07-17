
        subroutine read_data(filename,filetype,blockid,&
     		ntime,nsp0,ueq,size,size2,comm_block)

        implicit none

         include 'mpif.h'

!         include 'quark_parameter.f'
!         include 'quark_variable.f'
!         include 'quark_common.f'
	 integer ntime(3),nsp0(2),size,size2,comm_block
         character*100 filename
         real*8 ueq(size*nsp0(1)*size2)
         integer numprocs,idfwd, idbwd, nid,blockid,&
                myblock,itimes
         INTEGER ISTATUS(MPI_STATUS_SIZE),filetype
         integer ierr,nnn,lrec,iread,it,k,isp,ix,id,ic,ipart,myTime
         real*8 u0(nsp0(1)*size2)
         integer ic1,ic2,nsp,neo,id1
         
         integer nt,it1,it2,nt_r
         
         nt=ntime(1)
         it1=ntime(2)
         it2=ntime(3)
         nt_r=it2-it1+1

	 neo=nsp0(2)
	 nsp=nsp0(1)
         
         lrec = 8*nsp*size2
	nnn=lrec/8
!	if(blockid.eq.0) print*,nt,it1,it2,size,nsp,neo

         if(blockid.eq.0) then
	   if(filetype.eq.0)then!big endian
             open(unit=92, file=filename, form='unformatted', &
                   access='direct',recl=lrec,convert='big_endian')
     	   end if
	   if(filetype.eq.1)then!little endian
             open(unit=92, file=filename, form='unformatted', &
                   access='direct',recl=lrec,convert='little_endian')
     	   end if
         endif

!	call MPI_BARRIER(comm_block,ierr)

	if(blockid.lt.nt_r) then
           do id = 0,size-1
           do it = it1,it2

             if(blockid .eq. 0) then
               read(92, rec=it+1+nt*id) (u0(k), k=1,nnn)
             endif

             if(it .gt. it1)then
               nid = it-it1
               if(blockid .eq. 0) then
                 CALL MPI_SEND(u0, nnn, MPI_REAL8,&
                          nid, nid, comm_block, ierr)
               end if
               if(blockid .eq. nid) then
                 CALL MPI_RECV(u0,nnn, MPI_REAL8,&
                          0, nid, comm_block,ISTATUS, ierr)
               end if
             end if

             if(blockid .eq.it-it1)then
		if(mod(id,2).eq.0) then
			id1=id+1
		else
			id1=id-1
		end if !for eo_precondition
		if(neo.eq.0.or.mod(it,2).eq.0) id1=id !for standard one
              do isp = 1, nnn
                 ueq(nnn*id1+isp) = u0(isp)
              end do
             end if

           end do
           end do
	end if

	itimes=mod(blockid,nt_r)

	call MPI_COMM_SIZE(comm_block, numprocs, ierr )	
	if(numprocs.gt.nt_r) then
	Call MPI_COMM_SPLIT(comm_block,itimes,blockid,myTime,ierr)	
	call MPI_BARRIER(myTime,ierr)
	call MPI_Bcast(ueq,size*nsp*size2,MPI_REAL8,0,myTime,ierr)	
	call MPI_BARRIER(myTime,ierr)
	end if
  
	   if(blockid .eq. 0) close(92)

	return
       end
