#include <cstdio>

#include <quda_internal.h>
#include <gauge_field.h>
#include <llfat_quda.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <fast_intdiv.h>
#include <tune_quda.h>
#include <instantiate.h>

#define MIN_COEFF 1e-7

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  struct LinkArg {
    typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type Link;
    typedef typename gauge_mapper<Float, recon, 18, QUDA_STAGGERED_PHASE_MILC>::type Gauge;

    Link link;
    Gauge u;
    Float coeff;

    unsigned int threads;

    int_fastdiv X[4];
    int_fastdiv E[4];
    int border[4];

    /** This keeps track of any parity changes that result in using a
    radius of 1 for the extended border (the staple computations use
    such an extension, and if an odd number of dimensions are
    partitioned then we have to correct for this when computing the local index */
    int odd_bit;

    LinkArg(GaugeField &link, const GaugeField &u, Float coeff) :
      threads(link.VolumeCB()),
      link(link),
      u(u),
      coeff(coeff)
    {
      if (u.StaggeredPhase() != QUDA_STAGGERED_PHASE_MILC && u.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
      for (int d=0; d<4; d++) {
        X[d] = link.X()[d];
        E[d] = u.X()[d];
        border[d] = (E[d] - X[d]) / 2;
      }
    }
  };

  template <typename Float, int dir, typename Arg>
  __device__ void longLinkDir(Arg &arg, int idx, int parity) {
    int x[4];
    int dx[4] = {0, 0, 0, 0};

    auto y = arg.u.coords;
    getCoords(x, idx, arg.X, parity);
    for (int d=0; d<4; d++) x[d] += arg.border[d];

    typedef Matrix<complex<Float>,3> Link;

    Link a = arg.u(dir, linkIndex(y, x, arg.E), parity);

    dx[dir]++;
    Link b = arg.u(dir, linkIndexShift(y, x, dx, arg.E), 1-parity);

    dx[dir]++;
    Link c = arg.u(dir, linkIndexShift(y, x, dx, arg.E), parity);
    dx[dir]-=2;

    arg.link(dir, idx, parity) = arg.coeff * a * b * c;
  }

  template <typename Float, typename Arg>
  __global__ void computeLongLink(Arg arg) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockIdx.y*blockDim.y + threadIdx.y;
    int dir = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx >= arg.threads) return;
    if (dir >= 4) return;

    switch(dir) {
    case 0: longLinkDir<Float, 0>(arg, idx, parity); break;
    case 1: longLinkDir<Float, 1>(arg, idx, parity); break;
    case 2: longLinkDir<Float, 2>(arg, idx, parity); break;
    case 3: longLinkDir<Float, 3>(arg, idx, parity); break;
    }
    return;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class LongLink : public TunableVectorYZ {
    LinkArg<Float, nColor, recon> arg;
    const GaugeField &meta;
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; }

  public:
    LongLink(const GaugeField &u, GaugeField &lng, double coeff) :
      TunableVectorYZ(2,4),
      arg(lng, u, coeff),
      meta(lng)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());

      apply(0);
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      computeLongLink<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 2*4*arg.threads*198; }
    long long bytes() const { return 2*4*arg.threads*(3*arg.u.Bytes()+arg.link.Bytes()); }
  };

  void computeLongLink(GaugeField &lng, const GaugeField &u, double coeff)
  {
    instantiate<LongLink, ReconstructNo12>(u, lng, coeff); // u first arg so we pick its recon
  }

  template <typename Float, typename Arg>
  __global__ void computeOneLink(Arg arg)
  {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    int dir =  blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= arg.threads) return;
    if (dir >= 4) return;

    auto x = arg.u.coords;
    getCoords(x, idx, arg.X, parity);
    for (int d=0; d<4; d++) x[d] += arg.border[d];

    typedef Matrix<complex<Float>,3> Link;

    Link a = arg.u(dir, linkIndex(x,arg.E), parity);

    arg.link(dir, idx, parity) = arg.coeff*a;

    return;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class OneLink : public TunableVectorYZ {
    LinkArg<Float, nColor, recon> arg;
    const GaugeField &meta;
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; }

  public:
    OneLink(const GaugeField &u, GaugeField &fat, double coeff) :
      TunableVectorYZ(2,4),
      arg(fat, u, coeff),
      meta(fat)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());

      apply(0);
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      computeOneLink<Float><<<tp.grid,tp.block>>>(arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 2*4*arg.threads*18; }
    long long bytes() const { return 2*4*arg.threads*(arg.u.Bytes()+arg.link.Bytes()); }
  };

  void computeOneLink(GaugeField &fat, const GaugeField &u, double coeff)
  {
    if (u.StaggeredPhase() != QUDA_STAGGERED_PHASE_MILC && u.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
    instantiate<OneLink, ReconstructNo12>(u, fat, coeff);
  }

  template <typename Float, typename Fat, typename Staple, typename Mulink, typename Gauge>
  struct StapleArg {
    unsigned int threads;

    int_fastdiv X[4];
    int_fastdiv E[4];
    int border[4];

    int_fastdiv inner_X[4];
    int inner_border[4];

    /** This keeps track of any parity changes that result in using a
    radius of 1 for the extended border (the staple computations use
    such an extension, and if an odd number of dimensions are
    partitioned then we have to correct for this when computing the local index */
    int odd_bit;

    Gauge u;
    Fat fat;
    Staple staple;
    Mulink mulink;
    Float coeff;

    int n_mu;
    int mu_map[4];

    StapleArg(Fat fat, Staple staple, Mulink mulink, Gauge u, Float coeff,
	      const GaugeField &fat_meta, const GaugeField &u_meta) :
      threads(1), fat(fat), staple(staple), mulink(mulink), u(u), coeff(coeff),
      odd_bit( (commDimPartitioned(0)+commDimPartitioned(1) +
                commDimPartitioned(2)+commDimPartitioned(3))%2 )
    {
      for (int d=0; d<4; d++) {
        X[d] = (fat_meta.X()[d] + u_meta.X()[d]) / 2;
        E[d] = u_meta.X()[d];
        border[d] = (E[d] - X[d]) / 2;
        threads *= X[d];

        inner_X[d] = fat_meta.X()[d];
        inner_border[d] = (E[d] - inner_X[d]) / 2;
      }
      threads /= 2; // account for parity in y dimension
    }
  };

  template<typename Float, int mu, int nu, typename Arg>
  __device__ inline void computeStaple(Matrix<complex<Float>,3> &staple, Arg &arg, int x[], int parity) {
    typedef Matrix<complex<Float>,3> Link;
    int *y = arg.u.coords, *y_mu = arg.mulink.coords, dx[4] = {0, 0, 0, 0};

    /* Computes the upper staple :
     *                 mu (B)
     *               +-------+
     *       nu	   |	   |
     *	     (A)   |	   |(C)
     *		   X	   X
     */
    {
      /* load matrix A*/
      Link a = arg.u(nu, linkIndex(y, x, arg.E), parity);

      /* load matrix B*/
      dx[nu]++;
      Link b = arg.mulink(mu, linkIndexShift(y_mu, x, dx, arg.E), 1-parity);
      dx[nu]--;

      /* load matrix C*/
      dx[mu]++;
      Link c = arg.u(nu, linkIndexShift(y, x, dx, arg.E), 1-parity);
      dx[mu]--;

      staple = a * b * conj(c);
    }

    /* Computes the lower staple :
     *                 X       X
     *           nu    |       |
     *	         (A)   |       | (C)
     *		       +-------+
     *                  mu (B)
     */
    {
      /* load matrix A*/
      dx[nu]--;
      Link a = arg.u(nu, linkIndexShift(y, x, dx, arg.E), 1-parity);

      /* load matrix B*/
      Link b = arg.mulink(mu, linkIndexShift(y_mu, x, dx, arg.E), 1-parity);

      /* load matrix C*/
      dx[mu]++;
      Link c = arg.u(nu, linkIndexShift(y, x, dx, arg.E), parity);
      dx[mu]--;
      dx[nu]++;

      staple = staple + conj(a)*b*c;
    }
  }

  template<typename Float, bool save_staple, typename Arg>
  __global__ void computeStaple(Arg arg, int nu)
  {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockIdx.y*blockDim.y + threadIdx.y;
    if (idx >= arg.threads) return;

    int mu_idx = blockIdx.z*blockDim.z + threadIdx.z;
    if (mu_idx >= arg.n_mu) return;
    int mu;
    switch(mu_idx) {
    case 0: mu = arg.mu_map[0]; break;
    case 1: mu = arg.mu_map[1]; break;
    case 2: mu = arg.mu_map[2]; break;
    }

    int x[4];
    getCoords(x, idx, arg.X, (parity+arg.odd_bit)%2);
    for (int d=0; d<4; d++) x[d] += arg.border[d];

    typedef Matrix<complex<Float>,3> Link;
    Link staple;
    switch(mu) {
    case 0:
      switch(nu) {
      case 1: computeStaple<Float,0,1>(staple, arg, x, parity); break;
      case 2: computeStaple<Float,0,2>(staple, arg, x, parity); break;
      case 3: computeStaple<Float,0,3>(staple, arg, x, parity); break;
      } break;
    case 1:
      switch(nu) {
      case 0: computeStaple<Float,1,0>(staple, arg, x, parity); break;
      case 2: computeStaple<Float,1,2>(staple, arg, x, parity); break;
      case 3: computeStaple<Float,1,3>(staple, arg, x, parity); break;
      } break;
    case 2:
      switch(nu) {
      case 0: computeStaple<Float,2,0>(staple, arg, x, parity); break;
      case 1: computeStaple<Float,2,1>(staple, arg, x, parity); break;
      case 3: computeStaple<Float,2,3>(staple, arg, x, parity); break;
      } break;
    case 3:
      switch(nu) {
      case 0: computeStaple<Float,3,0>(staple, arg, x, parity); break;
      case 1: computeStaple<Float,3,1>(staple, arg, x, parity); break;
      case 2: computeStaple<Float,3,2>(staple, arg, x, parity); break;
      } break;
    }

    // exclude inner halo
    if ( !(x[0] < arg.inner_border[0] || x[0] >= arg.inner_X[0] + arg.inner_border[0] ||
	   x[1] < arg.inner_border[1] || x[1] >= arg.inner_X[1] + arg.inner_border[1] ||
	   x[2] < arg.inner_border[2] || x[2] >= arg.inner_X[2] + arg.inner_border[2] ||
	   x[3] < arg.inner_border[3] || x[3] >= arg.inner_X[3] + arg.inner_border[3]) ) {
      // convert to inner coords
      int inner_x[] = {x[0]-arg.inner_border[0], x[1]-arg.inner_border[1], x[2]-arg.inner_border[2], x[3]-arg.inner_border[3]};
      Link fat = arg.fat(mu, linkIndex(inner_x, arg.inner_X), parity);
      fat += arg.coeff * staple;
      arg.fat(mu, linkIndex(inner_x, arg.inner_X), parity) = fat;
    }

    if (save_staple) arg.staple(mu, linkIndex(x, arg.E), parity) = staple;
    return;
  }

  template <typename Float, typename Arg>
  class Staple : public TunableVectorYZ {
    Arg &arg;
    const GaugeField &meta;
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; }
    int nu;
    int dir1;
    int dir2;
    bool save_staple;

  public:
    Staple(Arg &arg, int nu, int dir1, int dir2, bool save_staple, const GaugeField &meta)
      : TunableVectorYZ(2,(3 - ( (dir1 > -1) ? 1 : 0 ) - ( (dir2 > -1) ? 1 : 0 ))),
	arg(arg), meta(meta), nu(nu), dir1(dir1), dir2(dir2), save_staple(save_staple)
	{
	  // compute the map for z thread index to mu index in the kernel
	  // mu != nu 3 -> n_mu = 3
	  // mu != nu != rho 2 -> n_mu = 2
	  // mu != nu != rho != sig 1 -> n_mu = 1
	  arg.n_mu = 3 - ( (dir1 > -1) ? 1 : 0 ) - ( (dir2 > -1) ? 1 : 0 );
	  int j=0;
	  for (int i=0; i<4; i++) {
	    if (i==nu || i==dir1 || i==dir2) continue; // skip these dimensions
	    arg.mu_map[j++] = i;
	  }
	  assert(j == arg.n_mu);
	}

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (save_staple)
	computeStaple<Float,true><<<tp.grid,tp.block>>>(arg, nu);
      else
	computeStaple<Float,false><<<tp.grid,tp.block>>>(arg, nu);
    }

    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << meta.AuxString() << comm_dim_partitioned_string();
      aux << ",nu=" << nu << ",dir1=" << dir1 << ",dir2=" << dir2 << ",save=" << save_staple;
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    void preTune() { arg.fat.save(); arg.staple.save(); }
    void postTune() { arg.fat.load(); arg.staple.load(); }

    long long flops() const {
      return 2*arg.n_mu*arg.threads*( 4*198 + 18 + 36 );
    }
    long long bytes() const {
      return arg.n_mu*2*meta.VolumeCB()*arg.fat.Bytes()*2 // fat load/store is only done on interior
	+ arg.n_mu*2*arg.threads*(4*arg.u.Bytes() + 2*arg.mulink.Bytes() + (save_staple ? arg.staple.Bytes() : 0));
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct Staple_ {
    Staple_(const GaugeField &u, GaugeField &fat, GaugeField &staple, const GaugeField &mulink,
            int nu, int dir1, int dir2, double coeff, bool save_staple)
    { // FIXME - incorporate another level of reconstruct peel off in instantiate
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type L;
      typedef typename gauge_mapper<Float,recon,18,QUDA_STAGGERED_PHASE_MILC>::type G;
      if (mulink.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        StapleArg<Float,L,L,L,G> arg(L(fat), L(staple), L(mulink), G(u), coeff, fat, u);
        Staple<Float,decltype(arg)> stapler(arg, nu, dir1, dir2, save_staple, fat);
        stapler.apply(0);
      } else if (mulink.Reconstruct() == recon) {
        StapleArg<Float,L,L,G,G> arg(L(fat), L(staple), G(mulink), G(u), coeff, fat, u);
        Staple<Float,decltype(arg)> stapler(arg, nu, dir1, dir2, save_staple, fat);
        stapler.apply(0);
      } else {
        errorQuda("Reconstruct %d is not supported\n", u.Reconstruct());
      }
    }
  };

  // Compute the staple field for direction nu,excluding the directions dir1 and dir2.
  void computeStaple(GaugeField &fat, GaugeField &staple, const GaugeField &mulink, const GaugeField &u,
		     int nu, int dir1, int dir2, double coeff, bool save_staple)
  {
    instantiate<Staple_, ReconstructNo12>(u, fat, staple, mulink, nu, dir1, dir2, coeff, save_staple);
  }

  void fatLongKSLink(GaugeField *fat, GaugeField *lng, const GaugeField& u, const double *coeff)
  {
#ifdef GPU_FATLINK
    GaugeFieldParam gParam(u);
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.setPrecision(gParam.Precision());
    gParam.create = QUDA_NULL_FIELD_CREATE;
    auto staple = GaugeField::Create(gParam);
    auto staple1 = GaugeField::Create(gParam);

    if ( ((fat->X()[0] % 2 != 0) || (fat->X()[1] % 2 != 0) || (fat->X()[2] % 2 != 0) || (fat->X()[3] % 2 != 0))
	&& (u.Reconstruct()  != QUDA_RECONSTRUCT_NO)){
      errorQuda("Reconstruct %d and odd dimensionsize is not supported by link fattening code (yet)\n",
		u.Reconstruct());
    }

    computeOneLink(*fat, u, coeff[0]-6.0*coeff[5]);

    // if this pointer is not NULL, compute the long link
    if (lng) computeLongLink(*lng, u, coeff[1]);

    // Check the coefficients. If all of the following are zero, return.
    if (fabs(coeff[2]) >= MIN_COEFF || fabs(coeff[3]) >= MIN_COEFF ||
	fabs(coeff[4]) >= MIN_COEFF || fabs(coeff[5]) >= MIN_COEFF) {

      for (int nu = 0; nu < 4; nu++) {
        computeStaple(*fat, *staple, u, u, nu, -1, -1, coeff[2], 1);

        if (coeff[5] != 0.0) computeStaple(*fat, *staple, *staple, u, nu, -1, -1, coeff[5], 0);

        for (int rho = 0; rho < 4; rho++) {
          if (rho != nu) {

            computeStaple(*fat, *staple1, *staple, u, rho, nu, -1, coeff[3], 1);

            if (fabs(coeff[4]) > MIN_COEFF) {
              for (int sig = 0; sig < 4; sig++) {
                if (sig != nu && sig != rho) {
                  computeStaple(*fat, *staple, *staple1, u, sig, nu, rho, coeff[4], 0);
                }
              } //sig
            } // MIN_COEFF
          }
        } //rho
      } //nu
    }

    qudaDeviceSynchronize();
    checkCudaError();

    delete staple;
    delete staple1;
#else
    errorQuda("Fat-link computation not enabled");
#endif
  }

#undef MIN_COEFF

} // namespace quda
