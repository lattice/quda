#include <quda.h>
#include <timer.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <contract_quda.h>

using namespace quda;

// Forward declarations for profiling and parameter checking
// The helper functions are defined in interface_quda.cpp
void checkBLASParam(QudaBLASParam &param);
TimeProfile &getProfileSinkProject();
TimeProfile &getProfileBaryonKernel();
TimeProfile &getProfileBaryonKernelModeTripletsA();
TimeProfile &getProfileBaryonKernelModeTripletsB();
TimeProfile &getProfileAccumulateEvecs();
TimeProfile &getProfileColorContract();
TimeProfile &getProfileColorCross();
TimeProfile &getProfileBLAS();
TimeProfile &getProfileCurrentKernel();

void laphSinkProject(void *host_quark, void **host_evec, double _Complex *host_sinks,
		     QudaInvertParam inv_param, unsigned int nEv, const int X[4])
{
  getProfileSinkProject().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileSinkProject().TPSTART(QUDA_PROFILE_INIT);
  
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
  getProfileSinkProject().TPSTOP(QUDA_PROFILE_INIT);  
  
  // Copy quark field from host to device
  getProfileSinkProject().TPSTART(QUDA_PROFILE_H2D);
  *quda_quark[0] = *quark[0];
  getProfileSinkProject().TPSTOP(QUDA_PROFILE_H2D);
  
  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }
  
  std::complex<double>* hostSinkPtr = reinterpret_cast<std::complex<double>*>(host_sinks);

  // Iterate over all EV and call 1x1 kernel for now
  for (unsigned int iEv=0; iEv<nEv; ++iEv) {
    getProfileSinkProject().TPSTART(QUDA_PROFILE_H2D);
    *quda_evec[0] = *evec[iEv];
    getProfileSinkProject().TPSTOP(QUDA_PROFILE_H2D);

    // We now perfrom the projection onto the eigenspace. The data
    // is placed in host_sinks in  T, spin order 
    getProfileSinkProject().TPSTART(QUDA_PROFILE_COMPUTE);
    evecProjectQuda(*quda_quark[0], *quda_evec[0], (void*)hostSinkPtr);
    getProfileSinkProject().TPSTOP(QUDA_PROFILE_COMPUTE);
    
    // Advance result pointer to next EV position
    hostSinkPtr += 4*X[3];
  }
  
  // Clean up memory allocations
  getProfileSinkProject().TPSTART(QUDA_PROFILE_FREE);
  delete quark[0];
  delete quda_quark[0];
  for (unsigned int iEv=0; iEv<nEv; ++iEv) delete evec[iEv];
  delete quda_evec[0];
  getProfileSinkProject().TPSTOP(QUDA_PROFILE_FREE);
  getProfileSinkProject().TPSTOP(QUDA_PROFILE_TOTAL);
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
  
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_INIT);

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
  ColorSpinorParam cuda_q1_param(cuda_evec_param);
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
  ColorSpinorParam cuda_q2_param(cuda_evec_param);
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
  ColorSpinorParam cuda_q3_param(cuda_evec_param);
  cuda_q3_param.create = QUDA_REFERENCE_FIELD_CREATE;
  std::vector<ColorSpinorField *> quda_q3;
  for(int i=0; i<n3; i++) {
    cuda_q3_param.v = (std::complex<double>*)d_q3 + 3*nSites*i;
    quda_q3.push_back(ColorSpinorField::Create(cuda_q3_param));
  }

  // Create device diquark vector
  ColorSpinorParam cuda_diq_param(cuda_evec_param);
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

  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_INIT);  
  //--------------------------------------------------------------------------------

  // Copy host data to device
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_H2D);
  for (int i=0; i<nEv; i++) *quda_evec[i] = *evec[i];
  qudaMemcpy(d_coeffs2, hostCoeffs2Ptr, data_coeffs2_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs3, hostCoeffs3Ptr, data_coeffs3_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_mom, hostMomPtr, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_H2D);

  // Construct momenta
  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;  
  QudaBLASParam cublas_param_init = newQudaBLASParam();
  cublas_param_init.trans_a = QUDA_BLAS_OP_N;
  cublas_param_init.trans_b = QUDA_BLAS_OP_N;
  cublas_param_init.m = n2;
  cublas_param_init.n = 3 * nSites;
  cublas_param_init.k = nEv;
  cublas_param_init.lda = nEv;
  cublas_param_init.ldb = 3 * nSites;
  cublas_param_init.ldc = 3 * nSites;
  cublas_param_init.a_stride = nEv * nEv;
  cublas_param_init.b_stride = 3 * nSites * 3 * nSites;
  cublas_param_init.b_stride = 3 * nSites * 3 * nSites;
  cublas_param_init.c_offset = 0;
  cublas_param_init.batch_count = 1;
  cublas_param_init.alpha = (__complex__ double)alpha;  
  cublas_param_init.beta  = (__complex__ double)beta;
  cublas_param_init.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_init.data_type = QUDA_BLAS_DATATYPE_Z;
  checkBLASParam(cublas_param_init);
 
  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = n1*n2*n3;
  cublas_param_mom_sum.a_stride = nSites * nSites;
  cublas_param_mom_sum.c_offset = 0;
  cublas_param_mom_sum.batch_count = 1;
  cublas_param_mom_sum.alpha = (__complex__ double)alpha;  
  cublas_param_mom_sum.beta  = (__complex__ double)beta;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;

  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);
  blas_lapack::native::stridedBatchGEMM(d_coeffs2, d_evec, d_q2, cublas_param_init, QUDA_CUDA_FIELD_LOCATION);
  cublas_param_init.m = n3;
  blas_lapack::native::stridedBatchGEMM(d_coeffs3, d_evec, d_q3, cublas_param_init, QUDA_CUDA_FIELD_LOCATION);
  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);
  
  // Perfrom the caxpy to compute all q1 vectors
  getProfileAccumulateEvecs().TPSTART(QUDA_PROFILE_COMPUTE);
  blas::caxpy(coeffs1.data(), quda_evec, quda_q1);
  getProfileAccumulateEvecs().TPSTOP(QUDA_PROFILE_COMPUTE);
  int nInBlock = 0;
  for (int dil1=0; dil1<n1; dil1++) {
    for (int dil2=0; dil2<n2; dil2++) {

      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(*quda_q1[dil1], *quda_q2[dil2], *quda_diq[0]);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);
      for (int dil3=0; dil3<n3; dil3++) {
	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);	
	colorContractQuda(*quda_diq[0], *quda_q3[dil3], 
			  (std::complex<double>*)d_tmp + nSites*nInBlock);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;

	if (nInBlock == blockSizeMomProj || ((dil1+1 == n1) && (dil2+1 == n2) && (dil3+1 == n3))) {
	  // To gauge how to block the calls to remove launch latency.
	  //printfQuda("dil1 = %d, dil2 = %d, dil3 = %d, nInBlock = %d\n", dil1, dil2, dil3, nInBlock);
	  cublas_param_mom_sum.n = nInBlock;
	  cublas_param_mom_sum.c_offset = (dil1*n2 + dil2)*n3 + dil3 - nInBlock + 1;
          cublas_param_mom_sum.b_stride = nInBlock * nSites;
          cublas_param_mom_sum.c_stride = nInBlock * nSites;
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);	  
	  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);	  
	  nInBlock = 0;
	}
      }
    }
  }

  // Copy return array back to host
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(retArr, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  getProfileBaryonKernel().TPSTART(QUDA_PROFILE_FREE);
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

  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_FREE);
  getProfileBaryonKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}

// GOOD
void laphBaryonKernelComputeModeTripletA(int nMom, int nEv, int blockSizeMomProj,
                                         void **host_evec, 
                                         double _Complex *host_mom,
                                         double _Complex *return_arr,
                                         const int X[4])
{  
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_INIT);
  
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

  size_t nEvChoose3 = nEv*(nEv-1)/2*(nEv-2)/3;

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
  
  std::complex<double>* hostMomPtr = reinterpret_cast<std::complex<double>*>(host_mom); 
  std::complex<double>* retArrPtr = reinterpret_cast<std::complex<double>*>(return_arr); 

  // Device side arrays
  //-------------------------------------------------------
  size_t total_bytes = 0;
  size_t OneGB = 1024;
  OneGB *= 1024;
  OneGB *= 1024;
  
  // Device side temp array (complBuf in chroma_laph)
  size_t data_tmp_bytes = blockSizeMomProj * X[0] * X[1] * X[2] * 2 * quda_evec[0]->Precision();
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
  total_bytes += data_tmp_bytes;
  printfQuda("d_tmp bytes = %fGB total_bytes = %fGB\n", (double)data_tmp_bytes/(OneGB), (double)total_bytes/(OneGB)); 

  // A second temp array (tmpBuf in chroma_laph) This will be returned for a
  // globalChunkedSumArray (QDP)
  size_t data_ret_bytes = nEvChoose3 * nMom * 2 * quda_evec[0]->Precision();
  void *d_ret = pool_device_malloc(data_ret_bytes);
  total_bytes += data_ret_bytes;
  printfQuda("d_ret bytes = %fGB total_bytes = %fGB\n", (double)data_ret_bytes/(OneGB), (double)total_bytes/(OneGB)); 
  
  size_t data_mom_bytes = nMom * nSites * 2 * quda_evec[0]->Precision();
  void *d_mom = pool_device_malloc(data_mom_bytes);
  total_bytes += data_mom_bytes;
  printfQuda("d_mom bytes = %fGB total_bytes = %fGB\n", (double)data_mom_bytes/(OneGB), (double)total_bytes/(OneGB)); 

  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_INIT);
  //--------------------------------------------------------------------------------
  
  // Copy host data to device
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_H2D);
  for (int i=0; i<nEv; i++) *quda_evec[i] = *evec[i];
  qudaMemcpy(d_mom, hostMomPtr, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_H2D);

  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;    

  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = nMom;
  cublas_param_mom_sum.k = nSites;
  cublas_param_mom_sum.n = blockSizeMomProj;
  cublas_param_mom_sum.lda = nSites;
  cublas_param_mom_sum.ldb = nSites;
  cublas_param_mom_sum.ldc = nEvChoose3;
  cublas_param_mom_sum.a_stride = nSites * nSites;
  cublas_param_mom_sum.batch_count = 1;
  cublas_param_mom_sum.alpha = (__complex__ double)alpha;  
  cublas_param_mom_sum.beta  = (__complex__ double)beta;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;
    
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_TOTAL); 

  int nInBlock = 0;  
  int blockStart = 0;
  for (int aEv=0; aEv<nEv; aEv++) {
    for (int bEv=aEv+1; bEv<nEv; bEv++) {
      
      getProfileColorCross().TPSTART(QUDA_PROFILE_COMPUTE);
      colorCrossQuda(*quda_evec[aEv], *quda_evec[bEv], *quda_diq[0]);
      getProfileColorCross().TPSTOP(QUDA_PROFILE_COMPUTE);

      for (int cEv=bEv+1; cEv<nEv; cEv++) {

	getProfileColorContract().TPSTART(QUDA_PROFILE_COMPUTE);
	colorContractQuda(*quda_diq[0], *quda_evec[cEv], 
			  (std::complex<double>*)d_tmp + nSites*nInBlock);
	getProfileColorContract().TPSTOP(QUDA_PROFILE_COMPUTE);
	nInBlock++;

	if (nInBlock == blockSizeMomProj) {
	  // To gauge how to block the calls to remove launch latency.
	  //printfQuda("aEv = %d, bEv = %d, cEv = %d, nInBlock = %d\n", aEv, bEv, cEv, nInBlock);
	  cublas_param_mom_sum.n = nInBlock;
	  cublas_param_mom_sum.c_offset = blockStart;
          cublas_param_mom_sum.b_stride = nSites * nInBlock;
          cublas_param_mom_sum.c_stride = nEvChoose3 * nInBlock;
	  getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);  
	  blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
	  getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);  
	  blockStart += nInBlock;
	  nInBlock = 0;
	}
      }
    }
  }

  // leftover momentum projection
  if (nInBlock > 0) {
    cublas_param_mom_sum.n = nInBlock;
    cublas_param_mom_sum.c_offset = blockStart;
    getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);	  
    blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
    getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);	  
    blockStart = 0;
    nInBlock = 0;
  }

  // Copy return array back to host
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_TOTAL); 
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(retArrPtr, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_D2H);
  
  // Clean up memory allocations
  getProfileBaryonKernelModeTripletsA().TPSTART(QUDA_PROFILE_FREE);
  for (int i=0; i<nEv; i++) {
    delete evec[i];
    delete quda_evec[i];
  }
  delete quda_diq[0];

  pool_device_free(d_evec);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  pool_device_free(d_ret);

  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_FREE);
  getProfileBaryonKernelModeTripletsA().TPSTOP(QUDA_PROFILE_TOTAL);
}
 
// Make this a class, save on malloc and memcopy
void *d_mtb = nullptr;
bool mtb_loaded = false;

void laphBaryonKernelComputeModeTripletB(int n1, int n2, int n3, int nMom, int nEv,
                                         double _Complex *host_coeffs1, 
                                         double _Complex *host_coeffs2, 
                                         double _Complex *host_coeffs3,
                                         double _Complex *host_mode_trip_buf,
                                         double _Complex *host_ret_arr)
{
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_INIT);
   
  // number of EV indices (in first position) that this rank deals with
  int nRanks = comm_size();  
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("comm_size() = %d\n", nRanks);
  fflush(stdout);
  int nSubEv = nEv / nRanks;
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("nSubEv = %d\n", nSubEv);
  fflush(stdout);
  int iRank = comm_rank();

  // check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }  
   
  std::complex<double>* hostCoeffs1Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs1);
  std::complex<double>* hostCoeffs2Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs2);
  std::complex<double>* hostCoeffs3Ptr = reinterpret_cast<std::complex<double>*>(host_coeffs3);
  std::complex<double>* hostModeTripBufPtr = reinterpret_cast<std::complex<double>*>(host_mode_trip_buf);
  std::complex<double>* hostRetArrPtr = reinterpret_cast<std::complex<double>*>(host_ret_arr);
   
  // Device side arrays
  //-------------------------------------------------------
  // We will define all the array sizes here, then malloc and free
  // at optimal points in the workflow.
  size_t total_bytes = 0;
  size_t OneGB = 1024;
  OneGB *= 1024;
  OneGB *= 1024;

  size_t data_coeffs1_bytes = n1 * nEv * 2 * QUDA_DOUBLE_PRECISION;
  size_t data_coeffs2_bytes = n2 * nEv * 2 * QUDA_DOUBLE_PRECISION;
  size_t data_coeffs3_bytes = n3 * nEv * 2 * QUDA_DOUBLE_PRECISION;
  
  size_t data_mtb_bytes = nMom;
  data_mtb_bytes *= nSubEv;
  data_mtb_bytes *= nEv;
  data_mtb_bytes *= nEv;
  data_mtb_bytes *= 2 * QUDA_DOUBLE_PRECISION;

  size_t data_q3_bytes = nMom;
  data_q3_bytes *= nSubEv;
  data_q3_bytes *= nEv;
  data_q3_bytes *= n3;
  data_q3_bytes *= 2 * QUDA_DOUBLE_PRECISION;
  
  size_t data_tmp_bytes = nSubEv * n2 * n3 * 2 * QUDA_DOUBLE_PRECISION;
  size_t data_ret_bytes = nMom * n1 * n2 * n3 * 2 * QUDA_DOUBLE_PRECISION;
  //--------------------------------------------------------------------------------

  // Allocate required memory
  if(!mtb_loaded) {
    total_bytes += data_mtb_bytes;
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("mtb bytes = %fGB, total = %fGB\n", (double)data_mtb_bytes/(OneGB), (double)total_bytes/(OneGB));
    d_mtb = pool_device_malloc(data_mtb_bytes);
  }
  total_bytes += data_q3_bytes;  
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("q3 bytes = %fGB, total = %fGB\n", (double)data_q3_bytes/(OneGB), (double)total_bytes/(OneGB));  
  void *d_q3 = pool_device_malloc(data_q3_bytes);

  total_bytes += data_coeffs3_bytes;
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("coeffs3 bytes = %fGB, total = %fGB\n", (double)data_coeffs3_bytes/(OneGB), (double)total_bytes/(OneGB)); 
  void *d_coeffs3 = pool_device_malloc(data_coeffs3_bytes);

  // All cuBLAS use these alpha and beta values
  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;

  QudaBLASParam cublas_param_1 = newQudaBLASParam();
  cublas_param_1.trans_a = QUDA_BLAS_OP_N;
  cublas_param_1.trans_b = QUDA_BLAS_OP_T;
  cublas_param_1.m = nMom*nSubEv*nEv;
  cublas_param_1.n = n3;
  cublas_param_1.k = nEv;
  cublas_param_1.lda = nEv;
  cublas_param_1.ldb = nEv;
  cublas_param_1.ldc = n3;
  cublas_param_1.a_stride = nEv * nEv;
  cublas_param_1.b_stride = nEv * n3;
  cublas_param_1.c_stride = n3 * n3;
  cublas_param_1.c_offset = 0;
  cublas_param_1.batch_count = 1;
  cublas_param_1.alpha = (__complex__ double)alpha;  
  cublas_param_1.beta  = (__complex__ double)beta;
  cublas_param_1.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_1.data_type = QUDA_BLAS_DATATYPE_Z;
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_INIT);

  // Copy required host data to device
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_H2D);  
  qudaMemcpy(d_coeffs3, hostCoeffs3Ptr, data_coeffs3_bytes, qudaMemcpyHostToDevice);  
  if(!mtb_loaded) {
    qudaMemcpy(d_mtb, hostModeTripBufPtr, data_mtb_bytes, qudaMemcpyHostToDevice);  
    mtb_loaded = true;
  }
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_H2D);

  // Compute ZGEMM 1:
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_COMPUTE);
  blas_lapack::native::stridedBatchGEMM(d_mtb, d_coeffs3, d_q3, cublas_param_1, QUDA_CUDA_FIELD_LOCATION);
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GEMM 1 Success!\n");
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_COMPUTE);
   
  // d_coeffs3, d_mtb no longer needed.
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_coeffs3);
  total_bytes -= data_coeffs3_bytes;
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_FREE);
   
  // Allocate the rest of the arrays
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_INIT);
  total_bytes += (data_coeffs1_bytes);
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("coeffs1_arr bytes = %fGB, total = %fGB\n", ((double)data_coeffs1_bytes)/(OneGB), (double)total_bytes/(OneGB));  
  void *d_coeffs1 = pool_device_malloc(data_coeffs1_bytes);
   
  total_bytes += (data_coeffs2_bytes);
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("coeffs2 bytes = %fGB, total = %fGB\n", (double)data_coeffs2_bytes/(OneGB), (double)total_bytes/(OneGB));   
  void *d_coeffs2 = pool_device_malloc(data_coeffs2_bytes);

  total_bytes += (data_tmp_bytes);
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("tmp bytes = %fGB, total = %fGB\n", ((double)data_tmp_bytes)/(OneGB), (double)total_bytes/(OneGB));   
  void *d_tmp = pool_device_malloc(data_tmp_bytes);
   
  total_bytes += data_ret_bytes;
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("ret bytes = %fGB, total = %fGB\n", (double)data_ret_bytes/(OneGB), (double)total_bytes/(OneGB));  
  void *d_ret = pool_device_malloc(data_ret_bytes);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_INIT);

  // Copy host data to device
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_H2D);  
  qudaMemcpy(d_coeffs1, hostCoeffs1Ptr, data_coeffs1_bytes, qudaMemcpyHostToDevice);  
  qudaMemcpy(d_coeffs2, hostCoeffs2Ptr, data_coeffs2_bytes, qudaMemcpyHostToDevice);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_H2D);

  // Initialise teh final ZGEMMs
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_INIT);
  QudaBLASParam cublas_param_2 = newQudaBLASParam();
  cublas_param_2.trans_a = QUDA_BLAS_OP_N;
  cublas_param_2.trans_b = QUDA_BLAS_OP_N;
  cublas_param_2.m = n2;
  cublas_param_2.n = n3;
  cublas_param_2.k = nEv;
  cublas_param_2.lda = nEv;
  cublas_param_2.ldb = n3;
  cublas_param_2.ldc = n3;
  cublas_param_2.a_stride = 0; // Instruct cuBLAS to use the only the data in d_coeffs2 (single batch sized array)
  cublas_param_2.b_stride = n3 * n3;
  cublas_param_2.c_stride = n3 * n3;
  cublas_param_2.batch_count = nSubEv;
  cublas_param_2.alpha = (__complex__ double)alpha;  
  cublas_param_2.beta  = (__complex__ double)beta;
  cublas_param_2.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_2.data_type = QUDA_BLAS_DATATYPE_Z;

  QudaBLASParam cublas_param_3 = newQudaBLASParam();
  cublas_param_3.trans_a = QUDA_BLAS_OP_N;
  cublas_param_3.trans_b = QUDA_BLAS_OP_N;
  cublas_param_3.m = n1;
  cublas_param_3.n = n2*n3;
  cublas_param_3.k = nSubEv;
  cublas_param_3.lda = nEv;
  cublas_param_3.ldb = n2*n3;
  cublas_param_3.ldc = n2*n3;   
  cublas_param_3.a_stride = 0; // Instruct cuBLAS to use the only the data in d_coeffs1 (single batch sized array) 
  cublas_param_3.b_stride = n2*n3 * n2*n3;
  cublas_param_3.c_stride = n2*n3 * n2*n3;
  cublas_param_3.batch_count = 1;
  cublas_param_3.alpha = (__complex__ double)alpha;  
  cublas_param_3.beta  = (__complex__ double)beta;
  cublas_param_3.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_3.data_type = QUDA_BLAS_DATATYPE_Z;
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_INIT);

  // Compute ZGEMMs
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_COMPUTE);   
  for(int i=0; i<nMom; i++) {
    cublas_param_2.b_offset = i * nSubEv * nEv * n3;
    blas_lapack::native::stridedBatchGEMM(d_coeffs2, d_q3, d_tmp, cublas_param_2, QUDA_CUDA_FIELD_LOCATION);
    cublas_param_3.a_offset = iRank * nSubEv;
    cublas_param_3.c_offset = i * n1 * n2 * n3;
    blas_lapack::native::stridedBatchGEMM(d_coeffs1, d_tmp, d_ret, cublas_param_3, QUDA_CUDA_FIELD_LOCATION);
  }
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GEMM 2+3 Success!\n");  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_COMPUTE);
   
  // Copy return array to host
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(hostRetArrPtr, d_ret, data_ret_bytes, qudaMemcpyDeviceToHost);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_D2H);
      
  // Clean up all remaining memory allocations
  getProfileBaryonKernelModeTripletsB().TPSTART(QUDA_PROFILE_FREE);
  pool_device_free(d_coeffs1);
  pool_device_free(d_coeffs2);
  pool_device_free(d_tmp);
  pool_device_free(d_q3);
  pool_device_free(d_ret);  
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_FREE);

  saveTuneCache();
  getProfileBaryonKernelModeTripletsB().TPSTOP(QUDA_PROFILE_TOTAL);  
}

void laphBaryonKernelComputeModeTripletEnd() {   
  if(mtb_loaded) pool_device_free(d_mtb);
  saveTuneCache();
}

void laphCurrentKernel(int n1, int n2, int n_mom,
                       int block_size_mom_proj,
		       void **host_quark,
		       void **host_quark_bar,
		       int *host_mom,
		       void *ret_arr,
		       const int X[4]) {
  
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_TOTAL);
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_INIT);
  
  // Check we are safe to cast into a Complex (= std::complex<double>)
  if (sizeof(Complex) != sizeof(double _Complex)) {
    errorQuda("Irreconcilable difference between interface and internal complex number conventions");
  }

  std::complex<double>* host_mom_ptr = reinterpret_cast<std::complex<double>*>(host_mom);
  
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

  // Some common variables
  size_t n_color = 3;
  size_t n_spatial_sites = X[0] * X[1] * X[2];
  size_t n_sites = n_spatial_sites * X[3];
  QudaPrecision precision = QUDA_DOUBLE_PRECISION;
  
  // Create host pointers for the data device side objects.
  //--------------------------------------------------------------------------------
  // Parameter object describing quark
  ColorSpinorParam cpu_quark_param(host_quark, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.nSpin = 1;
  
  // QUDA style wrapper around the host quark
  std::vector<ColorSpinorField*> quark;
  cpu_quark_param.create = QUDA_REFERENCE_FIELD_CREATE;
  quark.reserve(n2);
  for (int i=0; i<n2; i++) {
    cpu_quark_param.v = host_quark[i];
    quark.push_back(ColorSpinorField::Create(cpu_quark_param));
  }

  // Allocate device memory for quark. This is done to ensure a contiguous
  // chunk of memory is used.
  // vectors * colours * spatial sites * complex * precision
  size_t data_quark_bytes = n2 * n_color * n_sites * 2 * precision;
  void *d_quark = pool_device_malloc(data_quark_bytes);

  // Create device vectors for quarks
  ColorSpinorParam cuda_quark_param(cpu_quark_param);
  cuda_quark_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_quark_param.create = QUDA_REFERENCE_FIELD_CREATE;
  cuda_quark_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_quark;
  for (int i=0; i<n2; i++) {
    cuda_quark_param.v = (std::complex<double>*)d_quark + n_color*n_sites*i;
    quda_quark.push_back(ColorSpinorField::Create(cuda_quark_param));
  }

  // Repeat for quark_bar
  ColorSpinorParam cpu_quark_bar_param(host_quark_bar, inv_param, X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_bar_param.nSpin = 1;
  
  // QUDA style wrapper around the host quark_bar
  std::vector<ColorSpinorField*> quark_bar;
  cpu_quark_bar_param.create = QUDA_REFERENCE_FIELD_CREATE;
  quark_bar.reserve(n1);
  for (int i=0; i<n1; i++) {
    cpu_quark_bar_param.v = host_quark_bar[i];
    quark_bar.push_back(ColorSpinorField::Create(cpu_quark_bar_param));
  }

  // Allocate device memory for quark_bar. This is done to ensure a contiguous
  // chunk of memory is used.
  // vectors * colours * spatial sites * complex * precision
  size_t data_quark_bar_bytes = n1 * n_color * n_sites * 2 * precision;
  void *d_quark_bar = pool_device_malloc(data_quark_bar_bytes);

  // Create device vectors for quark_bar
  ColorSpinorParam cuda_quark_bar_param(cpu_quark_bar_param);
  cuda_quark_bar_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_quark_bar_param.create = QUDA_REFERENCE_FIELD_CREATE;
  cuda_quark_bar_param.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);
  std::vector<ColorSpinorField *> quda_quark_bar;
  for (int i=0; i<n1; i++) {
    cuda_quark_bar_param.v = (std::complex<double>*)d_quark_bar + n_color*n_sites*i;
    quda_quark_bar.push_back(ColorSpinorField::Create(cuda_quark_bar_param));
  }
  
  // Device array to hold the entire return array
  size_t data_ret_bytes = n_mom * X[3] * n1 * n2 * 2 * precision;
  void *d_ret = pool_device_malloc(data_ret_bytes);

  // Device array to hold the inner production
  size_t data_tmp_bytes = block_size_mom_proj * n_sites * 2 * precision;
  void *d_tmp = pool_device_malloc(data_tmp_bytes);

  // Device array to hold the momentum
  size_t data_mom_bytes = n_mom * n_spatial_sites * 2 * precision;
  void *d_mom = pool_device_malloc(data_mom_bytes);

  std::vector<int> momenta(n_mom * 3, 0);
  for(int i=0; i<3*n_mom; i++) momenta[i] = host_mom[i];

  //void cblas_zgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, 
  //const void *alpha, const void *a, const MKL_INT lda, const void *b, const MKL_INT ldb, const void *beta, void *c, const MKL_INT ldc);
  // cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nMom, nT*nInBlock, nSpatSites, &alpha, momBuf, nSpatSites, complBuf, nSpatSites, &beta, tmpResP, nT*nInBlock);

  __complex__ double alpha = 1.0;
  __complex__ double beta = 0.0;    

  QudaBLASParam cublas_param_mom_sum = newQudaBLASParam();
  cublas_param_mom_sum.trans_a = QUDA_BLAS_OP_N;
  cublas_param_mom_sum.trans_b = QUDA_BLAS_OP_T;
  cublas_param_mom_sum.m = n_mom;
  cublas_param_mom_sum.k = n_spatial_sites;
  cublas_param_mom_sum.lda = n_spatial_sites;
  cublas_param_mom_sum.ldb = n_spatial_sites;
  cublas_param_mom_sum.batch_count = 1;
  cublas_param_mom_sum.alpha = (__complex__ double)alpha;  
  cublas_param_mom_sum.beta  = (__complex__ double)beta;
  cublas_param_mom_sum.data_order = QUDA_BLAS_DATAORDER_ROW;
  cublas_param_mom_sum.data_type = QUDA_BLAS_DATATYPE_Z;

  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_INIT);
  //--------------------------------------------------------------------------------

  std::vector<Complex> ret_arr_tmp(n_mom * X[3] * n1 * n2, 0.0);

  // Copy host data to device
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_H2D);
  for (int i=0; i<n2; i++) *quda_quark[i] = *quark[i];
  for (int i=0; i<n1; i++) *quda_quark_bar[i] = *quark_bar[i];
  // For the moment, use the chroma_laph defined momenta, then compute on host
  qudaMemcpy(d_mom, host_mom_ptr, data_mom_bytes, qudaMemcpyHostToDevice);  
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_H2D);

  int n_in_block = 0;
  for (int dil1=0; dil1<n1; dil1++) {
    for (int dil2=0; dil2<n2; dil2++) {

      std::vector<Complex> mom_mode_data(n_mom * X[3], 0.0);
      getProfileCurrentKernel().TPSTART(QUDA_PROFILE_COMPUTE);
      innerProductQuda(*quda_quark_bar[dil1], *quda_quark[dil2], 
                       (std::complex<double>*)d_tmp + n_sites*n_in_block);
      n_in_block++;
      getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_COMPUTE);
      
      if (n_in_block == block_size_mom_proj || ((dil1+1 == n1) && (dil2+1 == n2))) {
        cublas_param_mom_sum.n = n_in_block * X[3];
        // c offset dictates where in the d_ret array we place the result. Each C matrix is n_in_block * n_mom * X[3] in size, consistent with m * ldc.               
        cublas_param_mom_sum.c_offset = n_mom * X[3] * n_in_block;
        cublas_param_mom_sum.ldb = X[3] * n_in_block;
        cublas_param_mom_sum.ldc = X[3] * n_in_block;
        getProfileBLAS().TPSTART(QUDA_PROFILE_COMPUTE);	  
        blas_lapack::native::stridedBatchGEMM(d_mom, d_tmp, d_ret, cublas_param_mom_sum, QUDA_CUDA_FIELD_LOCATION);
        getProfileBLAS().TPSTOP(QUDA_PROFILE_COMPUTE);	  
        n_in_block = 0;
      }
    }
  }
  
  // Copy device data back to host
  qudaMemcpy((void*)&ret_arr_tmp[0], d_ret, data_ret_bytes, qudaMemcpyHostToDevice);
  // Copy into return array 
  memcpy(ret_arr, ret_arr_tmp.data(), sizeof(Complex) * n_mom * X[3] * n1 * n2);
  
  // Clean up memory allocations
  getProfileCurrentKernel().TPSTART(QUDA_PROFILE_FREE);
  for (int i=0; i<n1; i++) {
    delete quda_quark[i];
    delete quark[i];
  }

  for (int i=0; i<n2; i++) {
    delete quda_quark_bar[i];
    delete quark_bar[i];
  }

  pool_device_free(d_quark);
  pool_device_free(d_quark_bar);
  pool_device_free(d_ret);
  pool_device_free(d_tmp);
  pool_device_free(d_mom);
  
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_FREE);
  getProfileCurrentKernel().TPSTOP(QUDA_PROFILE_TOTAL);
}
