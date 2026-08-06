#pragma once

#include <kernels/contraction.cuh>

namespace quda
{

  /**
     Argument structure for the nucleon two-point contraction.  The
     twelve spin-color source components of each flavor propagator are
     held as an array of accessors, since the baryon contraction is
     trilinear in the propagator and cannot be decomposed into the
     pairwise contractions used by the meson kernels.
   */
  template <typename Float, int nColor_> struct BaryonContractionSummedArg : public ReduceArg<spinor_array> {
    using reduce_t = spinor_array;
    static constexpr int reduction_dim = 3; // temporal correlators only

    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr int nProp = nSpin * nColor;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false;

    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    F u[nProp];
    F d[nProp];
    int mom_mode[4];
    QudaFFTSymmType fft_type[4];
    int source_position[4];
    int NxNyNzNt[4];
    int offsets[4];

    int_fastdiv X[4]; // grid dimensions

    BaryonContractionSummedArg(cvector_ref<const ColorSpinorField> &u, cvector_ref<const ColorSpinorField> &d,
                               const int *const source_position_in, const int *const mom_mode_in,
                               const QudaFFTSymmType *const fft_type_in) :
      ReduceArg<reduce_t>(dim3(u.Volume() / u.X(reduction_dim), 1, u.X(reduction_dim)), u.X(reduction_dim))
    {
      for (int i = 0; i < 4; i++) {
        X[i] = u.X(i);
        source_position[i] = source_position_in[i];
        mom_mode[i] = mom_mode_in[i];
        fft_type[i] = fft_type_in[i];
        offsets[i] = comm_coord(i) * u.X(i);
        NxNyNzNt[i] = comm_dim(i) * u.X(i);
      }
      for (int i = 0; i < nProp; i++) {
        this->u[i] = u[i];
        this->d[i] = d[i];
      }
    }
  };

  /**
     Momentum-projected nucleon two-point contraction with the
     standard C*gamma_5 diquark interpolator,

       chi(x) = eps_{abc} [u^{aT}(x) (C g5) d^b(x)] u^c_alpha(x).

     With Dt^{bb'} = (C g5) (S_d^{bb'})^T (C g5)^T (transpose in spin) the
     open-spin correlator at each site is

       C_{alpha alpha'}(x) = eps_{abc} eps_{a'b'c'} { S_u^{cc'}_{alpha alpha'} Tr_s[S_u^{aa'} Dt^{bb'}]
                                                      - [S_u^{ca'} Dt^{bb'} S_u^{ac'}]_{alpha alpha'} }.

     All 4x4 spin components are returned per timeslice so that any
     parity projector can be applied downstream.  Conventions are
     DeGrand-Rossi with C = gamma_4 gamma_2, for which C g5 is real with
     (C g5)_{s, cg5_idx[s]} = cg5_sign[s].
   */
  template <typename Arg> struct NucleonContractFT : plus<spinor_array> {

    using reduce_t = spinor_array;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 1;

    const Arg &arg;
    constexpr NucleonContractFT(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    // overload comm_reduce to defer until the entire "tile" is complete
    template <typename U> static inline void comm_reduce(U &) { }

    // Second int param is unused in the MultiReduce functor in this use case.
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int xyz, int, int t)
    {
      constexpr int nSpin = Arg::nSpin;
      constexpr int nColor = Arg::nColor;
      using real = typename Arg::real;
      using Vector = ColorSpinor<real, nColor, nSpin>;
      using Cplx = complex<real>;

      // The coordinate of the sink
      int sink[4];
      sink_from_t_xyz<Arg::reduction_dim>(sink, t, xyz, arg.X);

      // Fourier phase factor, computed as in StaggeredContractFT
      complex<double> phase(1.0, 0.0);
#pragma unroll
      for (int dir = 0; dir < 4; dir++) {
        auto dXi_dot_Pi
          = 2.0 * (sink[dir] + arg.offsets[dir] - arg.source_position[dir]) * arg.mom_mode[dir] / arg.NxNyNzNt[dir];
        complex<double> ph;
        if (arg.fft_type[dir] == QUDA_FFT_SYMM_EVEN) {
          ph = {cospi(dXi_dot_Pi), 0.0};
        } else if (arg.fft_type[dir] == QUDA_FFT_SYMM_ODD) {
          ph = {0.0, sinpi(dXi_dot_Pi)};
        } else {
          ph = {cospi(dXi_dot_Pi), sinpi(dXi_dot_Pi)};
        }
        phase *= ph;
      }

      int parity = 0;
      int idx = idx_from_t_xyz<Arg::reduction_dim>(t, xyz, arg.X);
      int idx_cb = getParityCBFromFull(parity, arg.X, idx);

      // Propagators at this site: S[spin_sink][color_sink][spin_source][color_source]
      Cplx Su[nSpin][nColor][nSpin][nColor];
      Cplx Dt[nSpin][nColor][nSpin][nColor];
      {
        Cplx Sd[nSpin][nColor][nSpin][nColor];
        for (int sj = 0; sj < nSpin; sj++) {
          for (int cj = 0; cj < nColor; cj++) {
            Vector u_vec = arg.u[sj * nColor + cj](idx_cb, parity);
            Vector d_vec = arg.d[sj * nColor + cj](idx_cb, parity);
            for (int si = 0; si < nSpin; si++) {
              for (int ci = 0; ci < nColor; ci++) {
                Su[si][ci][sj][cj] = u_vec(si, ci);
                Sd[si][ci][sj][cj] = d_vec(si, ci);
              }
            }
          }
        }

        // Dt^{bb'}_{s s'} = (C g5)_{s g} (S_d^T)_{g g'} (C g5)_{s' g'} = sign[s] sign[s'] Sd_{g[s'] b, g[s] b'}
        constexpr int cg5_idx[4] = {1, 0, 3, 2};
        constexpr int cg5_sign[4] = {1, -1, 1, -1};
        for (int s = 0; s < nSpin; s++) {
          for (int b = 0; b < nColor; b++) {
            for (int sp = 0; sp < nSpin; sp++) {
              for (int bp = 0; bp < nColor; bp++) {
                Dt[s][b][sp][bp] = static_cast<real>(cg5_sign[s] * cg5_sign[sp]) * Sd[cg5_idx[sp]][b][cg5_idx[s]][bp];
              }
            }
          }
        }
      }

      constexpr int eps[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
      constexpr int eps_sign[6] = {1, 1, 1, -1, -1, -1};

      Cplx C[nSpin][nSpin] = {};
      for (int ip = 0; ip < 6; ip++) {
        const int a = eps[ip][0], b = eps[ip][1], c = eps[ip][2];
        for (int jp = 0; jp < 6; jp++) {
          const int ap = eps[jp][0], bp = eps[jp][1], cp = eps[jp][2];
          const real sgn = static_cast<real>(eps_sign[ip] * eps_sign[jp]);

          // T1 = Tr_s[ S_u^{aa'} Dt^{bb'} ]
          Cplx T1(0.0, 0.0);
          for (int s = 0; s < nSpin; s++)
            for (int sp = 0; sp < nSpin; sp++) T1 += Su[s][a][sp][ap] * Dt[sp][b][s][bp];

          // P = Dt^{bb'} S_u^{ac'} (spin matrix product)
          Cplx P[nSpin][nSpin];
          for (int s = 0; s < nSpin; s++) {
            for (int sp = 0; sp < nSpin; sp++) {
              Cplx tmp(0.0, 0.0);
              for (int k = 0; k < nSpin; k++) tmp += Dt[s][b][k][bp] * Su[k][a][sp][cp];
              P[s][sp] = tmp;
            }
          }

          for (int al = 0; al < nSpin; al++) {
            for (int alp = 0; alp < nSpin; alp++) {
              // term2 = [ S_u^{ca'} P ]_{al alp}
              Cplx t2(0.0, 0.0);
              for (int k = 0; k < nSpin; k++) t2 += Su[al][c][k][ap] * P[k][alp];
              C[al][alp] += sgn * (Su[al][c][alp][cp] * T1 - t2);
            }
          }
        }
      }

      reduce_t result_all_channels = {};
      for (int al = 0; al < nSpin; al++) {
        for (int alp = 0; alp < nSpin; alp++) {
          complex<double> z(C[al][alp].real(), C[al][alp].imag());
          z *= phase;
          result_all_channels[al * nSpin + alp][0] = z.real();
          result_all_channels[al * nSpin + alp][1] = z.imag();
        }
      }

      return plus::operator()(result_all_channels, result);
    }
  };

} // namespace quda
