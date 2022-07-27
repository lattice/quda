#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <thread_array.h>
#include <kernel.h>

namespace quda {

  struct paths {
    const int num_paths;
    const int max_length;
    int *input_path[4];
    const int *length;
    const double *path_coeff;
    int *buffer;
    int count;

    paths(int ***input_path, int *length_h, double *path_coeff_h, int num_paths, int max_length) :
      num_paths(num_paths),
      max_length(max_length),
      count(0)
    {
      // create path struct in a single allocation
      size_t bytes = 4 * num_paths * max_length * sizeof(int) + num_paths * sizeof(int);
      int pad = ((sizeof(double) - bytes % sizeof(double)) % sizeof(double))/sizeof(int);
      bytes += pad*sizeof(int) + num_paths*sizeof(double);

      buffer = static_cast<int*>(pool_device_malloc(bytes));
      int *path_h = static_cast<int*>(safe_malloc(bytes));
      memset(path_h, 0, bytes);

      for (int dir=0; dir<4; dir++) {
        // flatten the input_path array for copying to the device
        for (int i=0; i < num_paths; i++) {
          for (int j=0; j < length_h[i]; j++) {
            path_h[dir*num_paths*max_length + i*max_length + j] = input_path[dir][i][j];
            if (dir==0) count++;
          }
        }
      }

      // length array
      memcpy(path_h + 4 * num_paths * max_length, length_h, num_paths*sizeof(int));

      // path_coeff array
      memcpy(path_h + 4 * num_paths * max_length + num_paths + pad, path_coeff_h, num_paths*sizeof(double));

      qudaMemcpy(buffer, path_h, bytes, qudaMemcpyHostToDevice);
      host_free(path_h);

      // finally set the pointers to the correct offsets in the buffer
      for (int d=0; d < 4; d++) this->input_path[d] = buffer + d*num_paths*max_length;
      length = buffer + 4*num_paths*max_length;
      path_coeff = reinterpret_cast<double*>(buffer + 4 * num_paths * max_length + num_paths + pad);
    }
    void free() {
      pool_device_free(buffer);
    }
  };


  template <typename Float_, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m, bool force_>
  struct GaugeForceArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr bool compute_force = force_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    typedef typename gauge_mapper<Float,recon_u>::type Gauge;
    typedef typename gauge_mapper<Float,recon_m>::type Mom;

    Mom mom;
    const Gauge u;

    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border

    Float epsilon; // stepsize and any other overall scaling factor
    const paths p;

    GaugeForceArg(GaugeField &mom, const GaugeField &u, double epsilon, const paths &p) :
      kernel_param(dim3(mom.VolumeCB(), 2, 4)),
      mom(mom),
      u(u),
      epsilon(epsilon),
      p(p)
    {
      for (int i=0; i<4; i++) {
	X[i] = mom.X()[i];
	E[i] = u.X()[i];
	border[i] = (E[i] - X[i])/2;
      }
    }
  };

  constexpr int flipDir(int dir) { return (7-dir); }
  constexpr bool isForwards(int dir) { return (dir <= 3); }

  template <typename Arg, int dir>
  __device__ __host__ inline void GaugeForceKernel(const Arg &arg, int idx, int parity)
  {
    using real = typename Arg::Float;
    typedef Matrix<complex<real>,Arg::nColor> Link;

    int x[4] = {0, 0, 0, 0};
    getCoords(x, idx, arg.X, parity);
    for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

    //linkA: current matrix
    //linkB: the loaded matrix in this round
    Link linkA, linkB, staple;
    thread_array<int, 4> dx{0};

    for (int i=0; i<arg.p.num_paths; i++) {
      real coeff = arg.p.path_coeff[i];
      if (coeff == 0) continue;

      const int* path = arg.p.input_path[dir] + i*arg.p.max_length;

      // start from end of link in direction dir
      int nbr_oddbit = (parity^1);
      dx[dir]++;

      int path0 = path[0];
      int lnkdir = isForwards(path0) ? path0 : flipDir(path0);

      if (isForwards(path0)) {
        linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
        linkA = linkB;
        dx[lnkdir]++; // now have to update location
	nbr_oddbit = nbr_oddbit^1;
      } else {
        dx[lnkdir]--; // if we are going backwards the link is on the adjacent site
        nbr_oddbit = nbr_oddbit^1;
	linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
        linkA = conj(linkB);
      }

      for (int j=1; j<arg.p.length[i]; j++) {

        int pathj = path[j];
        int lnkdir = isForwards(pathj) ? pathj : flipDir(pathj);

        if (isForwards(pathj)) {
          linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
          linkA = linkA * linkB;
          dx[lnkdir]++; // now have to update to new location
          nbr_oddbit = nbr_oddbit^1;
        } else {
          dx[lnkdir]--; // if we are going backwards the link is on the adjacent site
	  nbr_oddbit = nbr_oddbit^1;
          linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
          linkA = linkA * conj(linkB);
        }
      } //j
      staple = staple + coeff*linkA;
    } //i

    // multiply by U(x)
    linkA = arg.u(dir, linkIndex(x,arg.E), parity);
    linkA = linkA * staple;

    // update mom(x)
    Link mom = arg.mom(dir, idx, parity);
    if(arg.compute_force) {
      mom = mom - arg.epsilon * linkA;
      makeAntiHerm(mom);
    }
    else {
      mom = mom + arg.epsilon * linkA;
    }
    arg.mom(dir, idx, parity) = mom;
  }

  template <typename Arg> struct GaugeForce
  {
    const Arg &arg;
    constexpr GaugeForce(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }    

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      switch(dir) {
      case 0: GaugeForceKernel<Arg,0>(arg, x_cb, parity); break;
      case 1: GaugeForceKernel<Arg,1>(arg, x_cb, parity); break;
      case 2: GaugeForceKernel<Arg,2>(arg, x_cb, parity); break;
      case 3: GaugeForceKernel<Arg,3>(arg, x_cb, parity); break;
      }
    }
  };

}
