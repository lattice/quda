//#include	<mpi.h>

namespace quda {

#include	"covDev.h"	//Covariant derivative definitions

#ifdef MULTI_GPU
	#include	<mpi.h>
#endif


void	covDevQuda	(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, const int *commOverride)
{
//	bindSpinorTex(in->Bytes(), in->NormBytes(), (double2*)in->V(), (float*)in->Norm()); //for multi-gpu usage
	bindSpinorTex<double2>(in); //for multi-gpu usage

	dslashParam.parity	 = parity;
	dslashParam.kernel_type	 = INTERIOR_KERNEL;
	dslashParam.threads	 = in->Volume();

#ifdef MULTI_GPU
        //Aux parameters for memory operations:
	const int	tghostVolume	 = in->X(0)*in->X(1)*in->X(2);		// / 2tslice size without padding and for parity spinor!

	const int	Nvec		 = 2;					// for double2
	const int	Nint		 = in->Ncolor()*in->Nspin()*2;		// degrees of freedom	
	const int	Npad		 = Nint / Nvec;				// number Nvec buffers we have
	const int	Nt_minus1_offset = (in->Volume() - tghostVolume);	// Vh-Vsh	

	//MPI parameters:
	const int	my_rank		 = comm_rank();

	//Neighbour  rank in t direction:
	Topology	*Topo		 = comm_default_topology();

	const int Cfwd[QUDA_MAX_DIM]	 = {0, 0, 0, +1};
	const int Cbck[QUDA_MAX_DIM]	 = {0, 0, 0, -1};

	const int	fwd_neigh_t_rank = comm_rank_displaced	(Topo, Cfwd);
	const int	bwd_neigh_t_rank = comm_rank_displaced	(Topo, Cbck);
#endif

	void *gauge0, *gauge1;
	bindGaugeTex	(gauge, parity, &gauge0, &gauge1);

	if (in->Precision() != gauge.Precision())
		errorQuda	("Mixing gauge and spinor precision not supported");

#if (__COMPUTE_CAPABILITY__ >= 130)
	if	(in->Precision() == QUDA_DOUBLE_PRECISION)
	{
		dim3 gridBlock(64, 1, 1);
		dim3 gridDim( (dslashParam.threads+gridBlock.x-1) / gridBlock.x, 1, 1);

//		if	(reconstruct == QUDA_RECONSTRUCT_NO)

		switch	(mu)
		{
			case	0:
			covDevM012Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(), dslashParam);
			break;

			case	1:
			covDevM112Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(),  dslashParam);
			break;

			case	2:
			covDevM212Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(),  dslashParam);
			break;

			case	3:
			covDevM312Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(), dslashParam);
			break;

			case	4:
			covDevM012DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(), dslashParam);
			break;

			case	5:
			covDevM112DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(),  dslashParam);
			break;

			case	6:
			covDevM212DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(),  dslashParam);
			break;

			case	7:
			covDevM312DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) in->V(), dslashParam);
			break;
		}

//		regSize = sizeof(double);
	}
	else
		errorQuda("Single or half precision not supported");    
#else
	errorQuda("Double precision not supported on this GPU");
#endif

#ifdef MULTI_GPU	
	int		send_t_rank, recv_t_rank;
	int		rel, nDimComms = 4;


	if	(comm_size() > 1)
	{
//		comm_barrier		();

		if	(mu == 3)
		{
			send_t_rank	 = bwd_neigh_t_rank;
			recv_t_rank	 = fwd_neigh_t_rank;
			rel		 = +1;  
		}	  
		else if	(mu == 7)
		{
			send_t_rank	 = fwd_neigh_t_rank;
			recv_t_rank	 = bwd_neigh_t_rank;	  
			rel		 = -1;  
		}
		else
		{
			unbindGaugeTex		(gauge);	  
			return;
		}
	}
	else
	{
		unbindGaugeTex		(gauge);	  
		return;
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Something went wrong in rank %d", my_rank);
		exit	(-1);
	}

	MPI_Request	*sreq	 = (MPI_Request *) malloc(sizeof(MPI_Request));
	MPI_Request	*rreq	 = (MPI_Request *) malloc(sizeof(MPI_Request));	

	if	((sreq == NULL)||(rreq == NULL))
	{
		printf	("Error in rank %d: Unable to allocate memory for MPI requests\n", my_rank);
		exit	(-1);
	}

	//send buffers in t-dir:
	void	*send_t	 = 0;
	if	(cudaHostAlloc(&send_t, Nint * tghostVolume * sizeof(double), 0) != cudaSuccess)
	{
		printf	("Error in rank %d: Unable to allocate %d bytes for MPI requestsi (send_t)\n", my_rank, Nint*tghostVolume*sizeof(double));
		exit	(-1);
	}

	//recv buffers in t-dir:
	void	*recv_t	 = 0;
	if	(cudaHostAlloc(&recv_t, Nint * tghostVolume * sizeof(double), 0) != cudaSuccess)
	{
		printf	("Error in rank %d: Unable to allocate %d bytes for MPI requests (recv_t)\n", my_rank, Nint*tghostVolume*sizeof(double));
		exit	(-1);
	}
	
	//ghost buffer on gpu:
	void	*ghostTBuffer;
	cudaMalloc(&ghostTBuffer, Nint * tghostVolume * sizeof(double));	


	//Collect t-slice faces on the host (befor MPI send):
	int	tface_offset	 = Nt_minus1_offset;

	//Recv buffers from neighbor:
//        unsigned long	recv_req	 = comm_recv_from_rank(recv_t, Nint * tghostVolume * sizeof(double), recv_t_rank, rreq);	
//	comm_recv_from_rank(recv_t, Nint * tghostVolume * sizeof(double), recv_t_rank, &rreq);
	
	void	*sendTFacePtr	 = mu == 3 ? (char*)in->V() : mu == 7 ? (char*)in->V() + tface_offset*Nvec*sizeof(double) : NULL;	//Front face -> 3, back face -> 7

	size_t	len		 = tghostVolume*Nvec*sizeof(double);     
	size_t	spitch		 = in->Stride()*Nvec*sizeof(double);

	cudaMemcpy2DAsync(send_t, len, sendTFacePtr, spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
	cudaStreamSynchronize(streams[0]);

//	printf	("Envío  (%d %d) %d -> %d %le\n", parity, mu, my_rank, send_t_rank, ((double*) send_t)[0]);

	for	(int i=0;i<4;i++)
	{
		dslashParam.ghostDim[i]		 = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
		dslashParam.ghostOffset[i]	 = 0;
		dslashParam.ghostNormOffset[i]	 = 0;
		dslashParam.commDim[i]		 = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
	}	
	
        //Send buffers to neighbors:

	MsgHandle	*mh_send[4];
	MsgHandle	*mh_from[4];

	int		nbytes[4];

	nbytes[0]	 = 0;
	nbytes[1]	 = 0;
	nbytes[2]	 = 0;
	nbytes[3]	 = Nint * tghostVolume * sizeof(double);

	for	(int i=3; i<nDimComms; i++)
	{
		mh_send[i]	= comm_declare_send_relative	(send_t, i, rel,      nbytes[i]);
		mh_from[i]	= comm_declare_receive_relative	(recv_t, i, rel*(-1), nbytes[i]);
	}

	for	(int i=3; i<nDimComms; i++)
	{
		comm_start	(mh_from[i]);
		comm_start	(mh_send[i]);
	}
	
	for	(int i=3; i<nDimComms; i++)
	{
		comm_wait	(mh_send[i]);
		comm_wait	(mh_from[i]);
	}
	
	for	(int i=3; i<nDimComms; i++)
	{
		comm_free	(mh_send[i]);
		comm_free	(mh_from[i]);
	}
/*
	const int	chunkSz	 = 1024;		//512,1024 OK 2048 fails
	const int	tMAX	 = (Nint*tghostVolume)/chunkSz;

	for	(int i=0; i<tMAX; i++)
	{
		comm_barrier	();
		MPI_Irecv	((void*)((double*)recv_t+chunkSz*i), chunkSz, MPI_DOUBLE, recv_t_rank, MPI_ANY_TAG, MPI_COMM_WORLD, rreq);
		MPI_Isend	((void*)((double*)send_t+chunkSz*i), chunkSz, MPI_DOUBLE, send_t_rank,       10000, MPI_COMM_WORLD, sreq);	

		MsgHandle	*iSend	 = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
		MsgHandle	*iRecv	 = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

		iSend.request	= sreq;
		iRecv.request	= rreq;

		comm_wait	(&iRecv);
		comm_wait	(&iSend);

		free(iSend);
		free(iRecv);
	}

	free	(rreq);
	free	(sreq);
*/
	//Send buffers to GPU:
	cudaMemcpy(ghostTBuffer, recv_t, Nint * tghostVolume * sizeof(double), cudaMemcpyHostToDevice);
	
	//start execution
	//define exec domain
	dslashParam.kernel_type	 = EXTERIOR_KERNEL_T;
	dslashParam.threads	 = tghostVolume;
	
	cudaBindTexture		(0, spinorTexDouble, (double2*)ghostTBuffer, Nint*tghostVolume*sizeof(double));

	dim3	gridBlock(64, 1, 1);
	dim3	gridDim((dslashParam.threads+gridBlock.x-1) / gridBlock.x, 1, 1);	

	switch	(mu)
	{
		case	3:
		covDevM312Kernel<EXTERIOR_KERNEL_T><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) ghostTBuffer, dslashParam);
		break;
		case	7:
		covDevM312DaggerKernel<EXTERIOR_KERNEL_T><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((double2*) out->V(), (const double2*) gauge0, (const double2*) gauge1, (const double2*) ghostTBuffer, dslashParam);
		break;
	}	

	cudaFree(ghostTBuffer);
	cudaFreeHost(send_t);
	cudaFreeHost(recv_t);
#endif

	cudaUnbindTexture	(spinorTexDouble);
	unbindGaugeTex		(gauge);

	cudaDeviceSynchronize	();
	checkCudaError		();
}

}
