#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <misc.h>

#include <zfp.h>

#include <comm_quda.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);
  
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

/*
void setGaugeParam(QudaGaugeParam &gauge_param) {

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}
*/

// compress and decompress contiguous blocks
static double process4D(double* buffer, uint blocks, double tolerance, uint block_size)
{
  zfp_stream* zfp;   /* compressed stream */
  bitstream* stream; /* bit stream to write to or read from */
  size_t* offset;    /* per-block bit offset in compressed stream */
  double* ptr;       /* pointer to block being processed */
  size_t bufsize;    /* byte size of uncompressed storage */
  size_t zfpsize;    /* byte size of compressed stream */
  uint minbits;      /* min bits per block */
  uint maxbits;      /* max bits per block */
  uint maxprec;      /* max precision */
  int minexp;        /* min bit plane encoded */
  uint bits;         /* size of compressed block */
  uint i;
  uint block_dim = block_size * block_size * block_size * block_size;
  
  /* maintain offset to beginning of each variable-length block */
  offset = (size_t*)malloc(blocks * sizeof(size_t));

  /* associate bit stream with same storage as input */
  bufsize = blocks * block_dim * sizeof(*buffer);
  stream = stream_open(buffer, bufsize);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(stream);

  /* set tolerance for fixed-accuracy mode */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* set maxbits to guard against prematurely overwriting the input */
  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  maxbits = block_dim * sizeof(*buffer) * CHAR_BIT;
  zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);

  /* compress one block at a time in sequential order */
  ptr = buffer;
  for (i = 0; i < blocks; i++) {
    offset[i] = stream_wtell(stream);
    bits = zfp_encode_block_double_4(zfp, ptr);
    if (!bits) {
      fprintf(stderr, "compression failed\n");
      return 0;
    }
    if(verbosity >= QUDA_DEBUG_VERBOSE) printf("block #%u offset=%4u size=%4u\n", i, (uint)offset[i], bits);
    ptr += block_dim;
  }
  /* important: flush any buffered compressed bits */
  stream_flush(stream);

  /* print out size */
  zfpsize = stream_size(stream);
  if(verbosity >= QUDA_DEBUG_VERBOSE) printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);

  /* decompress one block at a time in reverse order */
  for (i = blocks; i--;) {
    ptr -= block_dim;
    stream_rseek(stream, offset[i]);
    if (!zfp_decode_block_double_4(zfp, ptr)) {
      fprintf(stderr, "decompression failed\n");
      return 0;
    }
  }
  
  /* clean up */
  zfp_stream_close(zfp);
  stream_close(stream);
  free(offset);
  
  return (1.0*((uint)zfpsize))/(uint)bufsize;
}

// compress and decompress contiguous blocks
static double process3D(double* buffer, uint blocks, double tolerance, uint block_size)
{
  zfp_stream* zfp;   /* compressed stream */
  bitstream* stream; /* bit stream to write to or read from */
  size_t* offset;    /* per-block bit offset in compressed stream */
  double* ptr;       /* pointer to block being processed */
  size_t bufsize;    /* byte size of uncompressed storage */
  size_t zfpsize;    /* byte size of compressed stream */
  uint minbits;      /* min bits per block */
  uint maxbits;      /* max bits per block */
  uint maxprec;      /* max precision */
  int minexp;        /* min bit plane encoded */
  uint bits;         /* size of compressed block */
  uint i;
  uint block_dim = block_size * block_size * block_size;
  
  /* maintain offset to beginning of each variable-length block */
  offset = (size_t*)malloc(blocks * sizeof(size_t));

  /* associate bit stream with same storage as input */
  bufsize = blocks * block_dim * sizeof(*buffer);
  stream = stream_open(buffer, bufsize);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(stream);

  /* set tolerance for fixed-accuracy mode */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* set maxbits to guard against prematurely overwriting the input */
  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  maxbits = block_dim * sizeof(*buffer) * CHAR_BIT;
  zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);

  /* compress one block at a time in sequential order */
  ptr = buffer;
  for (i = 0; i < blocks; i++) {
    offset[i] = stream_wtell(stream);
    bits = zfp_encode_block_double_3(zfp, ptr);
    if (!bits) {
      fprintf(stderr, "compression failed\n");
      return 0;
    }
    if(verbosity >= QUDA_VERBOSE && bits > 1) printf("block #%u offset=%4u size=%4u\n", i, (uint)offset[i], bits);
    ptr += block_dim;
  }
  /* important: flush any buffered compressed bits */
  stream_flush(stream);

  /* print out size */
  zfpsize = stream_size(stream);
  if(verbosity >= QUDA_VERBOSE) printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);

  /* decompress one block at a time in reverse order */
  for (i = blocks; i--;) {
    ptr -= block_dim;
    stream_rseek(stream, offset[i]);
    if (!zfp_decode_block_double_3(zfp, ptr)) {
      fprintf(stderr, "decompression failed\n");
      return 0;
    }
  }

  /* clean up */
  zfp_stream_close(zfp);
  stream_close(stream);
  free(offset);
  
  return (1.0*((uint)zfpsize))/(uint)bufsize;
}


int main(int argc, char **argv)
{

  auto app = make_app();
  add_su3_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"APE", 0}, {"Stout", 1}, {"Over-Improved Stout", 2}, {"Wilson Flow", 3}, {"Fundamental Rep", 4}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) 
    prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) 
    link_recon_sloppy = link_recon;

  setGaugeParam(gauge_param);
  setDims(gauge_param.X);

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // All user inputs now defined
  display_test_info();
  
  // call srand() with a rank-dependent seed
  initRand();  
  
  void *gauge[4], *gauge_orig[4], *gauge_new[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    gauge_orig[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    gauge_new[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    for(int i = 0; i < V * gauge_site_size; i++) ((double*)gauge_new[dir])[i] = 0.0;
  }
   
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  constructHostGaugeField(gauge_orig, gauge_param, argc, argv);

  // Exponentiate the gauge field for measurements.
  if(fund_gauge) exponentiateHostGaugeField(gauge, su3_taylor_N, prec);
  
  // Load the exponentiated gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette gauge precise is %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0], plaq[1],
             plaq[2]);

#ifdef GPU_GAUGE_TOOLS

  // Topological charge and gauge energy
  //------------------------------------
  // Size of floating point data
  size_t data_size = prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
  size_t array_size = V * data_size;
  void *qDensity = malloc(array_size);
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param.qcharge_density = qDensity;

  gaugeObservablesQuda(&param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Computed Etot, Es, Et, Q is\n%.16e %.16e, %.16e %.16e\nDone in %g secs\n", param.energy[0],
             param.energy[1], param.energy[2], param.qcharge, time0);

  // Use ZFP to compress then decpompress a gauge field in the fundamental rep
  //--------------------------------------------------------------------------

  time0 = -((double)clock()); 
  double tolerance = su3_comp_tol;
  
  double* array4D;
  double* orig4D;
  double* buffer4D;

  double* array3D;
  double* orig3D;
  double* buffer3D;

  uint block_size = su3_comp_block_size;
  uint block_dim4D = block_size * block_size * block_size * block_size;
  uint block_dim3D = block_size * block_size * block_size;
  uint bx = xdim / block_size;
  uint by = ydim / block_size;
  uint bz = zdim / block_size;
  uint bt = tdim / block_size;
  uint nx = block_size * bx;
  uint ny = block_size * by;
  uint nz = block_size * bz;
  uint nt = block_size * bt;
  uint n4D = nx * ny * nz * nt;
  uint n3D = nx * ny * nz;
  uint blocks4D = bx * by * bz * bt;
  uint blocks3D = bx * by * bz;
  size_t x, y, z, t, idx;
  size_t i, j, k, l, b;
  double comp_ratio = 0.0;
  
  array4D = (double*)malloc(n4D * sizeof(double));
  orig4D = (double*)malloc(n4D * sizeof(double));
  buffer4D = (double*)malloc(blocks4D * block_dim4D * sizeof(double));

  if(verbosity >= QUDA_DEBUG_VERBOSE) {
    printfQuda("blocks4D * block_dim4D = %u * %u = %u\n", blocks4D , block_dim4D, blocks4D * block_dim4D);
    printfQuda("size of ntot = %u\n", n4D);
    printfQuda("size of array = %lu\n", n4D * sizeof(double));
    printfQuda("size of orig = %lu\n", n4D * sizeof(double));
    printfQuda("size of buffer = %lu\n", blocks4D * block_dim4D * sizeof(double));
  }
  
  // Loop over dimensions and fundamental coeffs.
  // For the purposes of testing, we loop over all 18 real
  // coeffs of the hermitian matrix. In practise, we need
  // only perform the compression on the upper triangular
  // and real diagonal.
  
  int Nc = 3;
  for(int dim=0; dim<4; dim++) {
    for(int elem = 0; elem < 2*Nc*Nc; elem++) {
      
      // Initialize array to be compressed
      for (t = 0; t < nt; t++) {
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      size_t idx = x + nx*y + nx*ny*z + nx*ny*nz*t;
	      size_t parity = idx % 2;
	      double u = ((double *)gauge_orig[dim])[Vh * gauge_site_size * parity + (idx/2) * gauge_site_size + elem];
	      array4D[idx] = u;
	      orig4D[idx] = u;
	    }
	  }
	}
      }
      
      // Reorganise array into NxNxNxN blocks   
      idx = 0;
      for (b = 0; b < blocks4D; b++) {
	for (l = 0; l < block_size; l++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	  
		buffer4D[i + block_size * (j + block_size * (k + block_size * (l + block_size * b)))] = array4D[idx];
		idx++;	  
	      }
	    }
	  }
	}
      }

      // Apply compression
      comp_ratio += process4D(buffer4D, blocks4D, tolerance, block_size);
      if(verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("comp_ratio %d = %e\n", 2*Nc*Nc * dim + elem, comp_ratio);
      
      // Reorganise blocks into array 
      idx = 0;
      for (b = 0; b < blocks4D; b++) {
	for (l = 0; l < block_size; l++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	    
		array4D[idx] = buffer4D[i + block_size * (j + block_size * (k + block_size * (l + block_size * b)))];
		idx++;
	      }
	    }
	  }
	}
      }
      
      // Replace gauge data with decompressed data
      for (t = 0; t < nt; t++) {
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      idx = x + nx*y + nx*ny*z +nx*ny*nz*t;
	      int parity = idx % 2;
	      ((double *)gauge_new[dim])[Vh * gauge_site_size * parity + (idx/2) * gauge_site_size + elem] = array4D[idx];
	    }
	  }
	}
      }
    }
  }
  
  // Exponentiate the reconstructed links
  if(fund_gauge) exponentiateHostGaugeField(gauge_new, su3_taylor_N, prec);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for compression and decompression = %g (%g per link)\n", time0, time0/(4 * xdim*ydim*zdim*tdim));
  printfQuda("Average compression ratio = %f (%.2fx)\n", comp_ratio/(4 * 2*Nc*Nc), 1.0/(comp_ratio/(4 * 2*Nc*Nc)));
  
  // Load the reconstrcuted gauge to the device
  loadGaugeQuda((void *)gauge_new, &gauge_param);
  
  // Compute differences of gauge obserables using the reconstructed field
  double plaq_recon[3];
  plaqQuda(plaq_recon);  
  QudaGaugeObservableParam param_recon = newQudaGaugeObservableParam();
  param_recon.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param_recon.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param_recon.qcharge_density = qDensity;
  gaugeObservablesQuda(&param_recon);

  printfQuda("\nComputed gauge obvervables: compressed->decompressed:\n");
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq_recon[0], plaq_recon[1],
	     plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param_recon.energy[0],
	     param_recon.energy[1], param_recon.energy[2], param_recon.qcharge);

  
  printfQuda("\nComputed gauge obvervables: original - compressed->decompressed at tol %e:\n", su3_comp_tol);
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0] - plaq_recon[0], plaq[1] - plaq_recon[1],
             plaq[2] - plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param.energy[0] - param_recon.energy[0],
             param.energy[1] - param_recon.energy[1], param.energy[2] - param_recon.energy[2], param.qcharge - param_recon.qcharge);
  
  free(buffer4D);
  free(array4D);
  free(orig4D);

  time0 = -((double)clock());
  comp_ratio = 0.0;
  
  array3D = (double*)malloc(n3D * sizeof(double));
  orig3D = (double*)malloc(n3D * sizeof(double));
  buffer3D = (double*)malloc(blocks3D * block_dim3D * sizeof(double));
  for(int d = 0; d<4; d++)
    for(int i = 0; i < V * gauge_site_size; i++) ((double*)gauge_new[d])[i] = 0.0;

  if(verbosity >= QUDA_DEBUG_VERBOSE) {
    printfQuda("size of ntot = %u\n", n3D);
    printfQuda("size of array = %lu\n", n3D * sizeof(double));
    printfQuda("size of orig = %lu\n", n3D * sizeof(double));
    printfQuda("size of buffer = %lu\n", blocks3D * block_dim3D * sizeof(double));
  }
  
  // Loop over dimensions, time slices, and fundamental coeffs.
  // For the purposes of testing, we loop over all 18 real
  // coeffs of the hermitian matrix. In practise, we need
  // only perform the compression on the upper traingular
  // and real diagonal.
  
  for(int dim=0; dim<3; dim++) {
    for(int t=0; t<tdim; t++) {
      for(int elem = 0; elem < 2*Nc*Nc; elem++) {
	
	// Initialize array to be compressed
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      int idx = x + nx*y + nx*ny*z;
	      int parity = idx % 2;
	      double u = ((double *)gauge_orig[dim])[Vh * gauge_site_size * parity + ((t*n3D + idx)/2) * gauge_site_size + elem];
	      array3D[idx] = u;
	      orig3D[idx] = u;
	    }
	  }
	}

	// Reorganise array into NxNxN blocks   
	idx = 0;
	for (b = 0; b < blocks3D; b++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	  
		buffer3D[i + block_size * (j + block_size * (k + block_size * b))] = array3D[idx];
		idx++;	  
	      }
	    }
	  }
	}
		
	// Apply compression
	comp_ratio += process3D(buffer3D, blocks3D, tolerance, block_size);
	if(verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("comp_ratio %d = %e\n", 2*Nc*Nc * dim + elem, comp_ratio);
	
	// Reorganise blocks into array 
	idx = 0;
	for (b = 0; b < blocks3D; b++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	    
		array3D[idx] = buffer3D[i + block_size * (j + block_size * (k + block_size * b))];
		idx++;
	      }
	    }
	  }
	}
	
	// Replace gauge data with decompressed data
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      idx = x + nx*y + nx*ny*z;
	      int parity = idx % 2;
	      ((double *)gauge_new[dim])[Vh * gauge_site_size * parity + ((t*n3D + idx)/2) * gauge_site_size + elem] = array3D[idx];
	    }
	  }
	}
      }
    }
  }

  // Replace all temporal links
  for(int elem = 0; elem < 2*Nc*Nc; elem++) {
    for(int t=0; t<tdim; t++) {
      for (z = 0; z < nz; z++) {
	for (y = 0; y < ny; y++) {
	  for (x = 0; x < nx; x++) {      
	    idx = x + nx*y + nx*ny*z + nx*ny*nz*t;
	    int parity = idx % 2;
	    ((double *)gauge_new[3])[Vh * gauge_site_size * parity + (idx/2) * gauge_site_size + elem] = ((double *)gauge_orig[3])[Vh * gauge_site_size * parity + (idx/2) * gauge_site_size + elem];
	  }
	}
      }
    }
  }
  
  // Exponentiate the reconstructed links
  if(fund_gauge) exponentiateHostGaugeField(gauge_new, su3_taylor_N, prec);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for compression and decompression = %g (%g per link)\n", time0, time0/(3 * xdim*ydim*zdim*tdim));
  printfQuda("Total compression ratio = %f (%.2fx)\n", comp_ratio/(3 * 2*Nc*Nc * tdim), 1.0/(comp_ratio/(3 * 2*Nc*Nc * tdim)));
  
  // Load the reconstructed gauge to the device
  loadGaugeQuda((void *)gauge_new, &gauge_param);
  
  // Compute differences of gauge observables using the reconstructed field
  plaqQuda(plaq_recon);
  param_recon.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param_recon.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param_recon.qcharge_density = qDensity;
  gaugeObservablesQuda(&param_recon);

  printfQuda("\nComputed gauge obvervables: compressed->decompressed:\n");
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq_recon[0], plaq_recon[1],
	     plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param_recon.energy[0],
	     param_recon.energy[1], param_recon.energy[2], param_recon.qcharge);

  
  printfQuda("\nComputed gauge obvervables: original - compressed->decompressed at tol %e:\n", su3_comp_tol);
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0] - plaq_recon[0], plaq[1] - plaq_recon[1],
             plaq[2] - plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param.energy[0] - param_recon.energy[0],
             param.energy[1] - param_recon.energy[1], param.energy[2] - param_recon.energy[2], param.qcharge - param_recon.qcharge);
  
  free(buffer3D);
  free(array3D);
  free(orig3D);
  
#else
  printfQuda("Skipping other gauge tests since gauge tools have not been compiled\n");
#endif

  if (verify_results) check_gauge(gauge, gauge_orig, 1e-3, gauge_param.cpu_prec);

  freeGaugeQuda();
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(gauge_orig[dir]);
    free(gauge_new[dir]);
  }

  finalizeComms();
  return 0;
}
