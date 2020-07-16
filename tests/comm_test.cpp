#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h> // convenient quark field container

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#include <communicator_quda.h>

#include <gauge_field.h>

#include <timer.h>

#include <gauge_tools.h>

static quda::TimeProfile profileExtendedGauge("createExtendedGaugeFieldCommTest");

extern quda::cudaGaugeField *gaugePrecise;

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension "
             "Ls_dimension   dslash_type  normalization\n");
  printfQuda(
    "%6s   %6s          %d     %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
    get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type), get_recon_str(link_recon),
    get_recon_str(link_recon_sloppy), get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
    get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  if (inv_multigrid) {
    printfQuda("MG parameters\n");
    printfQuda(" - number of levels %d\n", mg_levels);
    for (int i = 0; i < mg_levels - 1; i++) {
      printfQuda(" - level %d number of null-space vectors %d\n", i + 1, nvec[i]);
      printfQuda(" - level %d number of pre-smoother applications %d\n", i + 1, nu_pre[i]);
      printfQuda(" - level %d number of post-smoother applications %d\n", i + 1, nu_post[i]);
    }

    printfQuda("MG Eigensolver parameters\n");
    for (int i = 0; i < mg_levels; i++) {
      if (low_mode_check || mg_eig[i]) {
        printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
        printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
        if (mg_eig_type[i] == QUDA_EIG_BLK_TR_LANCZOS)
          printfQuda(" - eigenvector block size %d\n", mg_eig_block_size[i]);
        printfQuda(" - level %d number of eigenvectors requested nConv %d\n", i + 1, nvec[i]);
        printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_nEv[i]);
        printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_nKr[i]);
        printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
        printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
        printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1,
                   mg_eig_use_dagger[i] ? "true" : "false", mg_eig_use_normop[i] ? "true" : "false");
        if (mg_eig_use_poly_acc[i]) {
          printfQuda(" - level %d Chebyshev polynomial degree %d\n", i + 1, mg_eig_poly_deg[i]);
          printfQuda(" - level %d Chebyshev polynomial minumum %e\n", i + 1, mg_eig_amin[i]);
          if (mg_eig_amax[i] <= 0)
            printfQuda(" - level %d Chebyshev polynomial maximum will be computed\n", i + 1);
          else
            printfQuda(" - level %d Chebyshev polynomial maximum %e\n", i + 1, mg_eig_amax[i]);
        }
        printfQuda("\n");
      }
    }
  }

  if (inv_deflate) {
    printfQuda("\n   Eigensolver parameters\n");
    printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
    printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
    if (eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", eig_block_size);
    printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
    printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
    printfQuda(" - size of Krylov space %d\n", eig_nKr);
    printfQuda(" - solver tolerance %e\n", eig_tol);
    printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
    if (eig_compute_svd) {
      printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
      printfQuda(" - ***********************************************************\n");
      printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
      printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
      printfQuda(" - ***********************************************************\n");
    } else {
      printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
                 eig_use_normop ? "true" : "false");
    }
    if (eig_use_poly_acc) {
      printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
      printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
      if (eig_amax <= 0)
        printfQuda(" - Chebyshev polynomial maximum will be computed\n");
      else
        printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
    }
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void plaquette_gauge(quda::GaugeField &g)
{
  int R[4] = {2 * comm_dim_partitioned(0), 2 * comm_dim_partitioned(1), 2 * comm_dim_partitioned(2),
              2 * comm_dim_partitioned(3)};

  quda::cudaGaugeField *data
    = quda::createExtendedGauge(*reinterpret_cast<quda::cudaGaugeField *>(&g), R, profileExtendedGauge);
  double3 plaq3 = quda::plaquette(*data);
  printf("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq3.x, plaq3.y, plaq3.z);

  delete data;
}

namespace quda
{
  void copyOffsetGauge(GaugeField &out, const GaugeField &in, const int offset[4]);

  void send_field(GaugeField &send_field, int dst_rank, int tag)
  {
    // TODO: For now only the cpu-gpu communication is implemented.
    printf("rank %d sending to %d: tag = %d\n", comm_rank(), dst_rank, tag);

    size_t bytes = send_field.Bytes();

    void *send_buffer_h = pinned_malloc(bytes);

    cudaMemcpy(send_buffer_h, send_field.Gauge_p(), bytes, cudaMemcpyDeviceToHost);

    auto mh_send = comm_declare_send_rank(send_buffer_h, dst_rank, tag, bytes);

    comm_start(mh_send);

    // comm_free(mh_send);
  }

  void recv_field(GaugeField &recv_field, int src_rank, int tag)
  {
    // TODO: For now only the cpu-gpu communication is implemented.
    printf("rank %d receiving from %d: tag = %d\n", comm_rank(), src_rank, tag);

    size_t bytes = recv_field.Bytes();

    void *recv_buffer_h = pinned_malloc(bytes);

    auto mh_recv = comm_declare_recv_rank(recv_buffer_h, src_rank, tag, bytes);

    comm_start(mh_recv);
    comm_wait(mh_recv);

    cudaMemcpy(recv_field.Gauge_p(), recv_buffer_h, bytes, cudaMemcpyHostToDevice);

    comm_free(mh_recv);
    host_free(recv_buffer_h);
  }

} // namespace quda

int main(int argc, char **argv)
{
  // Parse command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // All parameters have been set. Display the parameters via stdout
  display_test_info();

  // Initialize the QUDA library
  initQuda(device);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();

  setInvertParam(inv_param);

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }
  setSpinorSiteSize(24);

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Compute plaquette as a sanity check
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  quda::GaugeFieldParam param(*gaugePrecise);
  quda::GaugeField *buffer_gauge = quda::GaugeField::Create(param);

  param.x[3] *= 2;
  param.pad *= 2;
  quda::GaugeField *collected_gauge = quda::GaugeField::Create(param);

  int rank = comm_rank();

  for (int i = 0; i < 2; i++) {
    send_field(*gaugePrecise, 0, rank * 2 + 0); // 0 -> 0; 1 -> 0
    send_field(*gaugePrecise, 1, rank * 2 + 1); // 0 -> 1; 1 -> 1
    {
      recv_field(*buffer_gauge, 0, 0 * 2 + rank);
      int offset[4] = {0, 0, 0, 0 * param.x[3] / 2};
      quda::copyOffsetGauge(*collected_gauge, *buffer_gauge, offset);
    }

    {
      recv_field(*buffer_gauge, 1, 1 * 2 + rank);
      int offset[4] = {0, 0, 0, 1 * param.x[3] / 2};
      quda::copyOffsetGauge(*collected_gauge, *buffer_gauge, offset);
    }
  }

  // Create a communicator with the split {1, 1, 1, 2} and push to the top.
  if (gridsize_from_cmdline[3] % 2 == 0) push_to_current({1, 1, 1, 2});

  plaquette_gauge(*collected_gauge);

  delete buffer_gauge;
  delete collected_gauge;

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  return 0;
}
