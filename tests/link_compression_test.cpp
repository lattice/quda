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

// compress and decompress contiguous blocks
static double process(double* buffer, uint blocks, double tolerance, uint block_size)
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

  /* maintain offset to beginning of each variable-length block */
  offset = (size_t*)malloc(blocks * sizeof(size_t));

  /* associate bit stream with same storage as input */
  bufsize = blocks * block_size * block_size * block_size * sizeof(*buffer);
  stream = stream_open(buffer, bufsize);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(stream);

  /* set tolerance for fixed-accuracy mode */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* set maxbits to guard against prematurely overwriting the input */
  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  maxbits = block_size * block_size * block_size * sizeof(*buffer) * CHAR_BIT;
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
    if(verbosity >= QUDA_VERBOSE) printf("block #%u offset=%4u size=%4u\n", i, (uint)offset[i], bits);
    ptr += block_size * block_size * block_size;
  }
  /* important: flush any buffered compressed bits */
  stream_flush(stream);

  /* print out size */
  zfpsize = stream_size(stream);
  if(verbosity >= QUDA_VERBOSE) printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);

  /* decompress one block at a time in reverse order */
  for (i = blocks; i--;) {
    ptr -= block_size * block_size * block_size;
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

  return 1.0*((uint)zfpsize)/(uint)bufsize;
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

  void *gauge[4], *new_gauge[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    new_gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // All user inputs now defined
  display_test_info();
  
  // call srand() with a rank-dependent seed
  initRand();

  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Save the gauge into a CPU array
  saveGaugeQuda(new_gauge, &gauge_param);

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
  double* array;
  double* orig;
  double* buffer;
  uint block_size = su3_comp_block_size;
  uint bx = xdim / block_size;
  uint by = ydim / block_size;
  uint bz = zdim / block_size;
  uint spatial_block_size = xdim*ydim*zdim;
  uint nx = block_size * bx;
  uint ny = block_size * by;
  uint nz = block_size * bz;
  uint blocks = bx * by * bz;
  uint x, y, z, idx;
  uint i, j, k, b;
  double L2 = 0.0, diff = 0.0, comp_ratio = 0.0;
  
  array = (double*)malloc(nx * ny * nz * sizeof(double));
  orig = (double*)malloc(nx * ny * nz * sizeof(double));
  buffer = (double*)malloc(blocks * block_size * block_size * block_size * sizeof(double));
  
  // Loop over dimensions, time slices, and fundamental coeffs.
  // For the purposes of testing, we loop over all 18 real
  // coeffs of the hermitian matrix. In practise, we need
  // only perfrom the compression on the upper traingular
  // and real diagonal.
  
  int Nc = 3;
  for(int dim=0; dim<3; dim++) {
    for(int t=0; t<tdim; t++) {
      for(int elem = 0; elem < 2*Nc*Nc; elem++) {
	
	// Initialize array to be compressed       
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      int idx = x + nx*y + nx*ny*z;
	      int parity = idx % 2;
	      double u = ((double *)gauge[dim])[Vh * gauge_site_size * parity + (idx + t*spatial_block_size)/2 * gauge_site_size + elem];
	      array[idx] = u;
	      orig[idx] = u;
	    }
	  }
	}
	
	// Reorganise array into NxNxN blocks   
	idx = 0;
	for (b = 0; b < blocks; b++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	  
		buffer[i + block_size * (j + block_size * (k + block_size * b))] = array[idx];
		idx++;	  
	      }
	    }
	  }
	}

	// Apply compression
	comp_ratio += process(buffer, blocks, tolerance, 4);
	
	// Reorganise blocks into array 
	idx = 0;
	for (b = 0; b < blocks; b++) {
	  for (k = 0; k < block_size; k++) {
	    for (j = 0; j < block_size; j++) {
	      for (i = 0; i < block_size; i++) {	    
		array[idx] = buffer[i + block_size * (j + block_size * (k + block_size * b))];
		idx++;
	      }
	    }
	  }
	}
	
	// Diff of modified array with original 
	diff = 0.0;
	L2 = 0.0;
	idx = 0;
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {
	      diff = array[x + nx*y + nx*ny*z] - orig[x + nx*y + nx*ny*z];
	      L2 += diff*diff;
	      idx++;
	      if(abs(diff) > tolerance) printf("decompressed array diff[%d] = %e\n", idx, diff);
	    }
	  }
	}
	
	//printf("decompressed array diff = %e\n", L2);
	
	// Replace gauge data with decompressed data 
	for (z = 0; z < nz; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {      
	      idx = x + nx*y + nx*ny*z;
	      int parity = idx % 2;
	      ((double *)gauge[dim])[Vh * gauge_site_size * parity + (t*spatial_block_size + idx)/2 * gauge_site_size + elem] = array[idx];
	    }
	  }
	}
      }
    }
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for compression and decompression = %g (%g per link)\n", time0, time0/(3 * spatial_block_size));
  printfQuda("Total compression ratio = %f (%.2fx)\n", comp_ratio/(3 * 18 * tdim), 1.0/(comp_ratio/(3 * 18 * tdim)));
  
  // Reload the gauge to the device
  loadGaugeQuda((void *)gauge, &gauge_param);
  
  // Compute differences of gauge obserables using the reconstructed field
  double plaq_recon[3];
  plaqQuda(plaq_recon);  
  QudaGaugeObservableParam param_recon = newQudaGaugeObservableParam();
  param_recon.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param_recon.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param_recon.qcharge_density = qDensity;
  gaugeObservablesQuda(&param_recon);

  printfQuda("\nComputed gauge obvervables: original - compressed->decompressed:\n");
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0] - plaq_recon[0], plaq[1] - plaq_recon[1],
             plaq[2] - plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param.energy[0] - param_recon.energy[0],
             param.energy[1] - param_recon.energy[1], param.energy[2] - param_recon.energy[2], param.qcharge - param_recon.qcharge);
  
  free(buffer);
  free(array);
  free(orig);
  
#else
  printfQuda("Skipping other gauge tests since gauge tools have not been compiled\n");
#endif

  if (verify_results) check_gauge(gauge, new_gauge, 1e-3, gauge_param.cpu_prec);

  freeGaugeQuda();
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(new_gauge[dir]);
  }

  finalizeComms();
  return 0;
}
