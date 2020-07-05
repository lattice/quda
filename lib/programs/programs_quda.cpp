#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

#include <interface_quda.h>
#include <programs_quda.h>

using namespace quda;

//!< Profiler for sink projection
static TimeProfile profileSinkProject("sinkProjectQuda");

//!< Profiler for baryon kernel
static TimeProfile profileBaryonKernel("baryonKernelQuda");

//!< Profiler for baryon kernel mode triplets A
static TimeProfile profileBaryonKernelModeTripletsA("baryonKernelModeTripletsAQuda");

//!< Profiler for baryon kernel mode triplets B
static TimeProfile profileBaryonKernelModeTripletsB("baryonKernelModeTripletsBQuda");

//!< Profiler for accumulate evecs
static TimeProfile profileAccumulateEvecs("accumulateEvecsQuda");

//!< Profiler for color contract
static TimeProfile profileColorContract("colorContractQuda");

//!< Profiler for color cross
static TimeProfile profileColorCross("colorCrossQuda");

void laphSinkProject(void *host_quark, void **host_evec, double _Complex *host_sinks,
		     QudaInvertParam inv_param, unsigned int nEv, const int X[4])
{
  profileSinkProject.TPSTART(QUDA_PROFILE_TOTAL);
  profileSinkProject.TPSTART(QUDA_PROFILE_INIT);
  
  // Parameter object describing the sources and smeared quarks
  ColorSpinorParam cpu_quark_param(host_quark, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  
  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField*> quark;
  cpu_quark_param.v = host_quark;
  quark.push_back(ColorSpinorField::Create(cpu_quark_param));
  
  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  // Switch to spin 1
  cpu_evec_param.nSpin = 1;
  // QUDA style wrapper around the host data
  std::vector<ColorSpinorField*> evec;
  evec.reserve(nEv);
  for (unsigned int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec.push_back(ColorSpinorField::Create(cpu_evec_param));
  }
  
  // Create device vectors
  ColorSpinorParam cuda_quark_param(cpu_quark_param);
  cuda_quark_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_quark_param.create = QUDA_ZERO_FIELD_CREATE;
  cuda_quark_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_quark;
  quda_quark.push_back(ColorSpinorField::Create(cuda_quark_param));
  
  // Create device vectors for evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param);
  cuda_evec_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_evec_param.create = QUDA_ZERO_FIELD_CREATE;
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  cuda_evec_param.nSpin = 1;
  std::vector<ColorSpinorField *> quda_evec;
  quda_evec.push_back(ColorSpinorField::Create(cuda_evec_param));
  profileSinkProject.TPSTOP(QUDA_PROFILE_INIT);  
  
  // Copy quark field from host to device
  profileSinkProject.TPSTART(QUDA_PROFILE_H2D);
  *quda_quark[0] = *quark[0];
  profileSinkProject.TPSTOP(QUDA_PROFILE_H2D);
  
   // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  
  std::complex<double>* hostSinkPtr = reinterpret_cast<std::complex<double>*>(host_sinks);
  
  // Iterate over all EV and call 1x1 kernel for now
  for (unsigned int iEv=0; iEv<nEv; ++iEv) {
    profileSinkProject.TPSTART(QUDA_PROFILE_H2D);
    *quda_evec[0] = *evec[iEv];
    profileSinkProject.TPSTOP(QUDA_PROFILE_H2D);

    // We now perfrom the projection onto the eigenspace. The data
    // is placed in host_sinks in  T, spin order 
    profileSinkProject.TPSTART(QUDA_PROFILE_COMPUTE);
    evecProjectSumQuda(*quda_quark[0], *quda_evec[0], hostSinkPtr);
    profileSinkProject.TPSTOP(QUDA_PROFILE_COMPUTE);
    
    // Advance result pointer to next EV position
    hostSinkPtr += 4*X[3];
  }
  
  // Clean up memory allocations
  profileSinkProject.TPSTART(QUDA_PROFILE_FREE);
  delete quark[0];
  delete quda_quark[0];
  for (unsigned int iEv=0; iEv<nEv; ++iEv) delete evec[iEv];
  delete quda_evec[0];
  profileSinkProject.TPSTOP(QUDA_PROFILE_FREE);
  profileSinkProject.TPSTOP(QUDA_PROFILE_TOTAL);
}

 void laphBaryonKernel(int n1, int n2, int n3, int nMom,
		       double _Complex *host_coeffs1, 
		       double _Complex *host_coeffs2, 
		       double _Complex *host_coeffs3,
		       double _Complex *host_mom, 
		       int nEv, void **host_evec, 
		       void *retArr,
		       int blockSizeMomProj,
		       const int X[4]) {
  
  profileBaryonKernel.TPSTART(QUDA_PROFILE_TOTAL);
  profileBaryonKernel.TPSTART(QUDA_PROFILE_INIT);

  QudaInvertParam inv_param = newQudaInvertParam();
  
  inv_param.dslash_type = QUDA_WILSON_DSLASH;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_SOLVE;
  
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  
  // PADDING
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;
  
  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;


  // Create host pointers for the data device side objects.
  //--------------------------------------------------------------------------------
  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  
  // QUDA style wrapper around the host evecs
  std::vector<ColorSpinorField*> evec;
  cpu_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  evec.reserve(nEv);
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec.push_back(ColorSpinorField::Create(cpu_evec_param));
  }

  // Allocate device memory for evecs. This is done to ensure a contiguous
  // chunk of memory is used.
  int nSites = X[0] * X[1] * X[2];
  size_t data_evec_bytes = nEv * 3 * nSites * 2 * evec[0]->Precision();
  void *d_evec = pool_device_malloc(data_evec_bytes);

  // Create device vectors for evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param);
  cuda_evec_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_evec;
  for (int i=0; i<nEv; i++) {
    cuda_evec_param.v = (std::complex<double>*)d_evec + 3*nSites*i;
    quda_evec.push_back(ColorSpinorField::Create(cuda_evec_param));
  }
  
  // Create device q1 vectors
  ColorSpinorParam cuda_q1_param(cpu_evec_param);
  cuda_q1_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_q1_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_q1;
  for(int i=0; i<n1; i++) {
    quda_q1.push_back(ColorSpinorField::Create(cuda_q1_param));
  }

  // Allocate device memory for q2. This is done to ensure a contiguous
  // chunk of memory is used.
  size_t data_q2_bytes = n2 * 3 * nSites * 2 * evec[0]->Precision();
  void *d_q2 = pool_device_malloc(data_q2_bytes);

  // Create device q2 vectors, aliasing d_q2;
  ColorSpinorParam cuda_q2_param(cpu_evec_param);
  cuda_q2_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_q2_param.create = QUDA_REFERENCE_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_q2;
  for(int i=0; i<n2; i++) {
    cuda_q2_param.v = (std::complex<double>*)d_q2 + 3*nSites*i;
    quda_q2.push_back(ColorSpinorField::Create(cuda_q2_param));
  }

  // Allocate device memory for q3. This is done to ensure a contiguous
  // chunk of memory is used.
  size_t data_q3_bytes = n3 * 3 * nSites * 2 * evec[0]->Precision();
  void *d_q3 = pool_device_malloc(data_q3_bytes);

  // Create device q3 vectors, aliasing d_q3.
  ColorSpinorParam cuda_q3_param(cpu_evec_param);
  cuda_q3_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_q3_param.create = QUDA_REFERENCE_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_q3;
  for(int i=0; i<n3; i++) {
    cuda_q3_param.v = (std::complex<double>*)d_q3 + 3*nSites*i;
    quda_q3.push_back(ColorSpinorField::Create(cuda_q3_param));
  }

  // Create device diquark vector
  ColorSpinorParam cuda_diq_param(cpu_evec_param);
  cuda_diq_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_diq_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_diq;
  quda_diq.push_back(ColorSpinorField::Create(cuda_diq_param));

  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }  

  std::complex<double>* hostCoeffs1Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs1);
  std::complex<double>* hostCoeffs2Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs2);
  std::complex<double>* hostCoeffs3Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs3);
  std::complex<double>* hostMomPtr = reinterpret_cast<std::complex<double>*>(host_mom);

  // Make a multiBLAS friendly array for coeffs1 
  std::vector<Complex> coeffs1(n1*nEv);
  for(int j=0; j<n1; j++) {
    for(int i=0; i<nEv; i++) {
      coeffs1[i*n1 + j] = hostCoeffs1Ptr[j*nEv + i];
    }
  }
  
  // Device side arrays for coeff2 and coeffs3, the momentum array, the return array,
  // and a temp.
  size_t data_coeffs2_bytes = n2 * nEv * 2 * quda_evec[0]->Precision();
  void *d_coeffs2 = pool_device_malloc(data_coeffs2_bytes);

  size_t data_coeffs3_bytes = n3 * nEv * 2 * quda_evec[0]->Precision();
  void *d_coeffs3 = pool_device_malloc(data_coeffs3_bytes);

  size_t data_tmp_bytes = blockSizeMomProj * X[0] * X[1] * X[2] * 2 * quda_q3[0]->Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);

  size_t data_ret_bytes = nMom * n1 * n2 * n3 * 2 * quda_q3[0]->Precision();
  void *d_ret = pool_device_malloc(data_ret_bytes);

  size_t data_mom_bytes = nMom * nSites * 2 * quda_q3[0]->Precision();
  void *d_mom = pool_device_malloc(data_mom_bytes);

  profileBaryonKernel.TPSTOP(QUDA_PROFILE_INIT);  
  //--------------------------------------------------------------------------------

  // Copy host data to device
  profileBaryonKernel.TPSTART(QUDA_PROFILE_H2D);
  for (int i=0; i<nEv; i++) *quda_evec[i] = *evec[i];
  qudaMemcpy(d_coeffs2, hostCoeffs2Ptr, data_coeffs2_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs3, hostCoeffs3Ptr, data_coeffs3_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_mom, hostMomPtr, data_mom_bytes, cudaMemcpyHostToDevice);  
  profileBaryonKernel.TPSTOP(QUDA_PROFILE_H2D);

  // Construct momenta
  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;  
  QudaCublasParam cublas_param_init = newQudaCublasParam();
  cublas_param_init.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_init.trans_b = QUDA_CUBLAS_OP_N;
  cublas_param_init.m = n2;
  cublas_param_init.n = 3 * nSites;
  cublas_param_init.k = nEv;
  cublas_param_init.lda = nEv;
  cublas_param_init.ldb = 3 * nSites;
  cublas_param_init.ldc = 3 * nSites;
  cublas_param_init.c_offset = 0;
  cublas_param_init.batch_count = 1;
  cublas_param_init.alpha = (__complex__ double)alpha;  
  cublas_param_init.beta  = (__complex__ double)beta;
  cublas_param_init.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_init.data_type = QUDA_CUBLAS_DATATYPE_Z;
  checkCublasParam(&cublas_param_init);
 
  QudaCublasParam cublas_param_mom_sum = newQudaCublasParam();
  cublas_param_mom_sum.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_CUBLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = n1*n2*n3;
  cublas_param_mom_sum.c_offset = 0;
  cublas_param_mom_sum.batch_count = 1;
  cublas_param_mom_sum.alpha = (__complex__ double)alpha;  
  cublas_param_mom_sum.beta  = (__complex__ double)beta;
  cublas_param_mom_sum.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_CUBLAS_DATATYPE_Z;

  profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);
  cublas::BatchGEMM(d_evec, d_coeffs2, d_q2, cublas_param_init, QUDA_CUDA_FIELD_LOCATION);
  cublas_param_init.m = n3;
  cublas::BatchGEMM(d_evec, d_coeffs3, d_q3, cublas_param_init, QUDA_CUDA_FIELD_LOCATION);
  profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);

  
  // Perfrom the caxpy to compute all q1 vectors
  profileAccumulateEvecs.TPSTART(QUDA_PROFILE_COMPUTE);
  blas::caxpy(coeffs1.data(), quda_evec, quda_q1);
  profileAccumulateEvecs.TPSTOP(QUDA_PROFILE_COMPUTE);
  int nInBlock = 0;
  for (int dil1=0; dil1<n1; dil1++) {
    for (int dil2=0; dil2<n2; dil2++) {

      profileColorCross.TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(*quda_q1[dil1], *quda_q2[dil2], *quda_diq[0]);
      profileColorCross.TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int dil3=0; dil3<n3; dil3++) {
	profileColorContract.TPSTART(QUDA_PROFILE_COMPUTE);	
	colorContractQuda(*quda_diq[0], *quda_q3[dil3], 
			  (std::complex<double>*)d_tmp + nSites*nInBlock);
	profileColorContract.TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;

	if (nInBlock == blockSizeMomProj || ((dil1+1 == n1) && (dil2+1 == n2) && (dil3+1 == n3))) {

	  cublas_param_mom_sum.n = nInBlock;
	  cublas_param_mom_sum.c_offset = (dil1*n2 + dil2)*n3 + dil3 - nInBlock + 1;;
	  profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);	  
	  cublas::BatchGEMM(d_tmp, d_mom, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);	  
	  nInBlock = 0;
	}
      }
    }
  }

  // Copy return array back to host
  profileBaryonKernel.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(retArr, d_ret, data_ret_bytes, cudaMemcpyDeviceToHost);  
  profileBaryonKernel.TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  profileBaryonKernel.TPSTART(QUDA_PROFILE_FREE);
  for (int i=0; i<n1; i++) delete quda_q1[i];
  for (int i=0; i<n2; i++) delete quda_q2[i];
  for (int i=0; i<n3; i++) delete quda_q3[i];

  for (int i=0; i<nEv; i++) {
    delete evec[i];
    delete quda_evec[i];
  }
  delete quda_diq[0];
  pool_device_free(d_coeffs2);
  pool_device_free(d_q2);
  pool_device_free(d_coeffs3);
  pool_device_free(d_q3);
  pool_device_free(d_evec);

  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);

  profileBaryonKernel.TPSTOP(QUDA_PROFILE_FREE);
  profileBaryonKernel.TPSTOP(QUDA_PROFILE_TOTAL);
}

 void laphBaryonKernelComputeModeTripletA(int nMom, int nEv, void **host_evec, 
					  double _Complex *host_mom,
					  void *retArr,
					  int blockSizeMomProj,
					  const int X[4]) {
  
   
  profileBaryonKernelModeTripletsA.TPSTART(QUDA_PROFILE_TOTAL);
  profileBaryonKernelModeTripletsA.TPSTART(QUDA_PROFILE_INIT);
  
  QudaInvertParam inv_param = newQudaInvertParam();
  
  inv_param.dslash_type = QUDA_WILSON_DSLASH;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_SOLVE;
  
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  
  // PADDING
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;
  
  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;


  // Create host pointers for the data device side objects.
  //--------------------------------------------------------------------------------
  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  
  // QUDA style wrapper around the host evecs
  std::vector<ColorSpinorField*> evec;
  cpu_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  evec.reserve(nEv);
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec.push_back(ColorSpinorField::Create(cpu_evec_param));
  }

  // Allocate device memory for evecs. This is done to ensure a contiguous
  // chunk of memory is used.
  int nSites = X[0] * X[1] * X[2];
  size_t data_evec_bytes = nEv * 3 * nSites * 2 * evec[0]->Precision();
  void *d_evec = pool_device_malloc(data_evec_bytes);

  // Create device vectors for evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param);
  cuda_evec_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_evec;
  for (int i=0; i<nEv; i++) {
    cuda_evec_param.v = (std::complex<double>*)d_evec + 3*nSites*i;
    quda_evec.push_back(ColorSpinorField::Create(cuda_evec_param));
  }
  
  // Create device diquark vector
  ColorSpinorParam cuda_diq_param(cpu_evec_param);
  cuda_diq_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_diq_param.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_diq;
  quda_diq.push_back(ColorSpinorField::Create(cuda_diq_param));

  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }  

  // Device side temp array (complBuf in chroma_laph)
  size_t data_tmp_bytes = blockSizeMomProj * X[0] * X[1] * X[2] * 2 * quda_evec[0]->Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  
  // A second temp array (tmpBuf in chroma_laph) This will be returned for a
  // globalChunkedSumArray (QDP)
  size_t nEvChoose3 = nEv*(nEv-1)/2*(nEv-2)/3;
  size_t data_ret_bytes = nEvChoose3 * nMom * 2 * quda_evec[0]->Precision();
  void *d_ret = pool_device_malloc(data_ret_bytes);
  
  // Array of momenta
  std::complex<double>* hostMomPtr = reinterpret_cast<std::complex<double>*>(host_mom); 
  size_t data_mom_bytes = nMom * nSites * 2 * quda_evec[0]->Precision();
  void *d_mom = pool_device_malloc(data_mom_bytes);

  profileBaryonKernelModeTripletsA.TPSTOP(QUDA_PROFILE_INIT);  
  //--------------------------------------------------------------------------------
  
  // Copy host data to device
  profileBaryonKernelModeTripletsA.TPSTART(QUDA_PROFILE_H2D);
  for (int i=0; i<nEv; i++) *quda_evec[i] = *evec[i];
  qudaMemcpy(d_mom, hostMomPtr, data_mom_bytes, cudaMemcpyHostToDevice);  
  profileBaryonKernelModeTripletsA.TPSTOP(QUDA_PROFILE_H2D);

  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;    
  QudaCublasParam cublas_param_mom_sum = newQudaCublasParam();
  cublas_param_mom_sum.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_CUBLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = nEvChoose3;
  cublas_param_mom_sum.c_offset = 0;
  cublas_param_mom_sum.batch_count = 1;
  cublas_param_mom_sum.alpha = (__complex__ double)alpha;  
  cublas_param_mom_sum.beta  = (__complex__ double)beta;
  cublas_param_mom_sum.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_CUBLAS_DATATYPE_Z;
  
  int nInBlock = 0;  
  int blockStart = 0;
  for (int aEv=0; aEv<nEv; aEv++) {
    for (int bEv=aEv+1; bEv<nEv; bEv++) {
      
      profileColorCross.TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(*quda_evec[aEv], *quda_evec[bEv], *quda_diq[0]);
      profileColorCross.TPSTOP(QUDA_PROFILE_COMPUTE);

      for (int cEv=bEv+1; cEv<nEv; cEv++) {

	profileColorContract.TPSTART(QUDA_PROFILE_COMPUTE);	
	colorContractQuda(*quda_diq[0], *quda_evec[cEv], 
			  (std::complex<double>*)d_tmp + nSites*nInBlock);
	profileColorContract.TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;
	//printfQuda("aEv = %d bEv = %d cEv = %d\n", aEv, bEv, cEv); 
	if (nInBlock == blockSizeMomProj) {

	  cublas_param_mom_sum.n = nInBlock;
	  cublas_param_mom_sum.c_offset = blockStart;
	  profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);	  
	  //printfQuda("cuBLAS start\n"); 
	  cublas::BatchGEMM(d_tmp, d_mom, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  //printfQuda("cuBLAS end\n"); 
	  profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);	  
	  blockStart += nInBlock;
	  nInBlock = 0;
	}
      }
    }
  }

  // leftover momentum projection
  if (nInBlock > 0) {
    //printfQuda("cuBLAS remainder start\n"); 
    cublas_param_mom_sum.n = nInBlock;
    cublas_param_mom_sum.c_offset = blockStart;
    profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);	  
    cublas::BatchGEMM(d_tmp, d_mom, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
    profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);	  
    blockStart = 0;
    nInBlock = 0;
    //printfQuda("cuBLAS remainder end\n"); 
  }

  // Copy return array back to host
  profileBaryonKernelModeTripletsA.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(retArr, d_ret, data_ret_bytes, cudaMemcpyDeviceToHost);  
  profileBaryonKernelModeTripletsA.TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  profileBaryonKernelModeTripletsA.TPSTART(QUDA_PROFILE_FREE);
  for (int i=0; i<nEv; i++) {
    delete evec[i];
    delete quda_evec[i];
  }
  delete quda_diq[0];

  pool_device_free(d_evec);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);

  profileBaryonKernelModeTripletsA.TPSTOP(QUDA_PROFILE_FREE);
  profileBaryonKernelModeTripletsA.TPSTOP(QUDA_PROFILE_TOTAL);
  

}

























































 void laphBaryonKernelComputeModeTripletB(int n1, int n2, int n3, int nMom,
					  double _Complex *host_coeffs1, 
					  double _Complex *host_coeffs2, 
					  double _Complex *host_coeffs3,
					  double _Complex *host_mom, 
					  double _Complex *host_mode_trip_buf,
					  int nEv, void **host_evec, 
					  void *retArr,
					  const int X[4]) {
   
   
  profileBaryonKernelModeTripletsB.TPSTART(QUDA_PROFILE_TOTAL);
  profileBaryonKernelModeTripletsB.TPSTART(QUDA_PROFILE_INIT);
  
  QudaInvertParam inv_param = newQudaInvertParam();
  
  inv_param.dslash_type = QUDA_WILSON_DSLASH;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_SOLVE;
  
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  
  // PADDING
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;
  
  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  // number of EV indices (in first position) that this rank deals with
  int nRanks = comm_size();  
  printfQuda("comm_size() = %d\n", nRanks);
  fflush(stdout);
  int nSubEv = nEv / nRanks;
  printfQuda("nSubEv = %d\n", nSubEv);
  fflush(stdout);  

  // Create host pointers for the data device side objects.
  //--------------------------------------------------------------------------------
  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evec, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_evec_param.nSpin = 1;
  
  // QUDA style wrapper around the host evecs
  std::vector<ColorSpinorField*> evec;
  cpu_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  evec.reserve(nEv);
  for (int iEv=0; iEv<nEv; ++iEv) {
    cpu_evec_param.v = host_evec[iEv];
    evec.push_back(ColorSpinorField::Create(cpu_evec_param));
  }

  // Allocate device memory for evecs. This is done to ensure a contiguous
  // chunk of memory is used.
  int nSites = X[0] * X[1] * X[2];
  size_t data_evec_bytes = nEv * 3 * nSites * 2 * evec[0]->Precision();
  void *d_evec = pool_device_malloc(data_evec_bytes);

  // Create device vectors for evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param);
  cuda_evec_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_evec_param.create = QUDA_REFERENCE_FIELD_CREATE;
  cuda_evec_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_evec;
  for (int i=0; i<nEv; i++) {
    cuda_evec_param.v = (std::complex<double>*)d_evec + 3*nSites*i;
    quda_evec.push_back(ColorSpinorField::Create(cuda_evec_param));
  }

  std::complex<double>* hostCoeffs1Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs2);
  std::complex<double>* hostCoeffs2Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs2);
  std::complex<double>* hostCoeffs3Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs3);
  std::complex<double>* hostMomPtr = reinterpret_cast<std::complex<double>*>(host_mom);
  std::complex<double>* hostModeTripBufPtr = reinterpret_cast<std::complex<double>*>(host_mode_trip_buf);
  
  // Device side arrays
  size_t data_coeffs1_bytes = n1 * nEv * 2 * quda_evec[0]->Precision();
  printfQuda("n1 = %d, nEv = %d, 2*prec = %d, bytes = %zu\n", n1, nEv, 2 * quda_evec[0]->Precision(), data_coeffs1_bytes); 
  void *d_coeffs1 = pool_device_malloc(data_coeffs1_bytes);

  size_t data_coeffs2_bytes = n2 * nEv * 2 * quda_evec[0]->Precision();
  printfQuda("n2 = %d, nEv = %d, 2*prec = %d, bytes = %zu\n", n2, nEv, 2 * quda_evec[0]->Precision(), data_coeffs2_bytes); 
  void *d_coeffs2 = pool_device_malloc(data_coeffs2_bytes);

  size_t data_coeffs3_bytes = n3 * nEv * 2 * quda_evec[0]->Precision();
  printfQuda("n3 = %d, nEv = %d, 2*prec = %d, bytes = %zu\n", n3, nEv, 2 * quda_evec[0]->Precision(), data_coeffs3_bytes); 
  void *d_coeffs3 = pool_device_malloc(data_coeffs3_bytes);
  
  size_t data_mtb_bytes = (size_t)(nMom * nSubEv * nEv * nEv * 2 * quda_evec[0]->Precision());
  data_mtb_bytes /= 1024;
  data_mtb_bytes /= 1024;
  data_mtb_bytes /= 28512;
  
  printfQuda("nMom = %d, nSubEv = %d, nEv = %d, nEv = %d, 2*prec = %d, bytes = %zu\n", nMom, nSubEv, nEv, nEv, 2 * quda_evec[0]->Precision(), data_mtb_bytes); 
  //fflush(stdout);
  //exit(0);
  void *d_mtb = pool_device_malloc(data_mtb_bytes);
  //void *d_mtb = pool_device_malloc(8);
  
  size_t data_tmp_bytes = nSubEv * n2 * n3 * 2 * quda_evec[0]->Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);

  size_t data_q3_bytes = nMom * nSubEv * nEv * n3 * quda_evec[0]->Precision();
  void *d_q3 = pool_device_malloc(data_q3_bytes);
    
  size_t data_ret_bytes = nMom * n1 * n2 * n3 * 2 * quda_evec[0]->Precision();
  void *d_ret = pool_device_malloc(data_ret_bytes);

  size_t data_mom_bytes = nMom * nSites * 2 * quda_evec[0]->Precision();
  void *d_mom = pool_device_malloc(data_mom_bytes);

  profileBaryonKernelModeTripletsB.TPSTOP(QUDA_PROFILE_INIT);  
  //--------------------------------------------------------------------------------

  // Copy host data to device
  profileBaryonKernelModeTripletsB.TPSTART(QUDA_PROFILE_H2D);  
  for (int i=0; i<nEv; i++) *quda_evec[i] = *evec[i];
  qudaMemcpy(d_coeffs1, hostCoeffs1Ptr, data_coeffs1_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs2, hostCoeffs2Ptr, data_coeffs2_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs3, hostCoeffs3Ptr, data_coeffs3_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_mtb, hostModeTripBufPtr, data_mtb_bytes, cudaMemcpyHostToDevice);  
  qudaMemcpy(d_mom, hostMomPtr, data_mom_bytes, cudaMemcpyHostToDevice);  
  profileBaryonKernelModeTripletsB.TPSTOP(QUDA_PROFILE_H2D);

  // All cuBLAS use these alpha and beta values
  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;


  QudaCublasParam cublas_param_1 = newQudaCublasParam();
  cublas_param_1.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_1.trans_b = QUDA_CUBLAS_OP_T;
  cublas_param_1.m = nMom*nSubEv*nEv;
  cublas_param_1.n = n3;
  cublas_param_1.k = nEv;
  cublas_param_1.lda = nEv;
  cublas_param_1.ldb = nEv;
  cublas_param_1.ldc = n3;
  cublas_param_1.c_offset = 0;
  cublas_param_1.batch_count = 1;
  cublas_param_1.alpha = (__complex__ double)alpha;  
  cublas_param_1.beta  = (__complex__ double)beta;
  cublas_param_1.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_1.data_type = QUDA_CUBLAS_DATATYPE_Z;

  profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);
  cublas::BatchGEMM(d_mtb, d_coeffs3, d_q3, cublas_param_1, QUDA_CUDA_FIELD_LOCATION);
  profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);

  
  QudaCublasParam cublas_param_2 = newQudaCublasParam();
  cublas_param_2.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_2.trans_b = QUDA_CUBLAS_OP_N;
  cublas_param_2.m = n2;
  cublas_param_2.n = n3;
  cublas_param_2.k = nEv;
  cublas_param_2.lda = nEv;
  cublas_param_2.ldb = n3;
  cublas_param_2.ldc = n3;
  cublas_param_2.c_offset = 0;
  cublas_param_2.batch_count = nSubEv;
  cublas_param_2.alpha = (__complex__ double)alpha;  
  cublas_param_2.beta  = (__complex__ double)beta;
  cublas_param_2.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_2.data_type = QUDA_CUBLAS_DATATYPE_Z;

  QudaCublasParam cublas_param_3 = newQudaCublasParam();
  cublas_param_3.trans_a = QUDA_CUBLAS_OP_N;
  cublas_param_3.trans_b = QUDA_CUBLAS_OP_N;
  cublas_param_3.m = n1;
  cublas_param_3.n = n2*n3;
  cublas_param_3.k = nSubEv;
  cublas_param_3.lda = nEv;
  cublas_param_3.ldb = n2*n3;
  cublas_param_3.ldc = n2*n3;
  cublas_param_3.b_offset = 0;
  cublas_param_3.c_offset = 0;
  cublas_param_3.batch_count = 1;
  cublas_param_3.alpha = (__complex__ double)alpha;  
  cublas_param_3.beta  = (__complex__ double)beta;
  cublas_param_3.data_order = QUDA_CUBLAS_DATAORDER_ROW;
  cublas_param_3.data_type = QUDA_CUBLAS_DATATYPE_Z;

  profileCuBLAS.TPSTART(QUDA_PROFILE_COMPUTE);
  for(int i=0; i<nMom; i++) {
    for(int k=0; k<nSubEv; k++) {
      cublas_param_2.b_offset = (i * nSubEv + k) * nEv * n3;
      cublas_param_2.c_offset = k * n2 * n3;
      cublas::BatchGEMM(d_coeffs2, d_q3, d_tmp, cublas_param_2, QUDA_CUDA_FIELD_LOCATION);
    }
    cublas_param_3.c_offset = i * n1 * n2 * n3;
    cublas::BatchGEMM(d_coeffs1, d_tmp, d_ret, cublas_param_3, QUDA_CUDA_FIELD_LOCATION);
  }
  profileCuBLAS.TPSTOP(QUDA_PROFILE_COMPUTE);
  

  // Copy return array back to host
  profileBaryonKernelModeTripletsB.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(retArr, d_ret, data_ret_bytes, cudaMemcpyDeviceToHost);  
  profileBaryonKernelModeTripletsB.TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  profileBaryonKernelModeTripletsB.TPSTART(QUDA_PROFILE_FREE);
  for (int i=0; i<nEv; i++) {
    delete evec[i];
    delete quda_evec[i];
  }
  
  pool_device_free(d_evec);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);
  pool_device_free(d_q3);
  
  profileBaryonKernelModeTripletsB.TPSTOP(QUDA_PROFILE_FREE);
  profileBaryonKernelModeTripletsB.TPSTOP(QUDA_PROFILE_TOTAL);
  
}
