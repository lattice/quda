#pragma once

#include <array>
#include <enum_quda.h>
#include <util_quda.h>

namespace quda
{

  template <QudaReconstructType recon> constexpr bool is_enabled() { return true; }
#if !(QUDA_RECONSTRUCT & 4)
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_NO>() { return false; }
#endif
#if !(QUDA_RECONSTRUCT & 2)
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_13>() { return false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_12>() { return false; }
#endif
#if !(QUDA_RECONSTRUCT & 1)
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_9>() { return false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_8>() { return false; }
#endif

  struct ReconstructFull {
    static constexpr std::array<QudaReconstructType, 5> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_9, QUDA_RECONSTRUCT_8};
  };

  struct ReconstructWilson {
    static constexpr std::array<QudaReconstructType, 3> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8};
  };

  struct ReconstructStaggered {
    static constexpr std::array<QudaReconstructType, 3> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_9};
  };

  struct ReconstructNo12 {
    static constexpr std::array<QudaReconstructType, 2> recon = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12};
  };

  struct ReconstructNone {
    static constexpr std::array<QudaReconstructType, 1> recon = {QUDA_RECONSTRUCT_NO};
  };

  struct ReconstructMom {
    static constexpr std::array<QudaReconstructType, 2> recon = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_10};
  };

  struct Reconstruct10 {
    static constexpr std::array<QudaReconstructType, 1> recon = {QUDA_RECONSTRUCT_10};
  };

  /**
     @brief This class instantiates the Apply class based on the
     instantiated templates below.
  */
  template <bool enabled, template <typename, int, QudaReconstructType> class Apply, typename Float, int nColor,
            QudaReconstructType recon, typename G, typename... Args>
  struct instantiateApply {
    instantiateApply(G &U, Args &&... args) { Apply<Float, nColor, recon>(U, args...); }
  };

  /**
     @brief This class is a specialization which does not instantiate
     the Apply class if the is_enabled has evaluated to false.
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Float, int nColor,
            QudaReconstructType recon, typename G, typename... Args>
  struct instantiateApply<false, Apply, Float, nColor, recon, G, Args...> {
    instantiateApply(G &U, Args &&... args)
    {
      errorQuda("QUDA_RECONSTRUCT=%d does not enable %d", QUDA_RECONSTRUCT, recon);
    }
  };

  /**
     @brief Instantiate the reconstruction template at index i and
     recurse to prior element
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Float, int nColor, typename Recon,
            int i, typename G, typename... Args>
  struct instantiateReconstruct {
    instantiateReconstruct(G &U, Args &&... args)
    {
      if (U.Reconstruct() == Recon::recon[i]) {
        instantiateApply<is_enabled<Recon::recon[i]>(), Apply, Float, nColor, Recon::recon[i], G, Args...>(U, args...);
      } else {
        instantiateReconstruct<Apply, Float, nColor, Recon, i - 1, G, Args...>(U, args...);
      }
    }
  };

  /**
     @brief Termination specialization of instantiateReconstruct
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Float, int nColor, typename Recon,
            typename G, typename... Args>
  struct instantiateReconstruct<Apply, Float, nColor, Recon, 0, G, Args...> {
    instantiateReconstruct(G &U, Args &&... args)
    {
      if (U.Reconstruct() == Recon::recon[0]) {
        instantiateApply<is_enabled<Recon::recon[0]>(), Apply, Float, nColor, Recon::recon[0], G, Args...>(U, args...);
      } else {
        errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
      }
    }
  };

  /**
     @brief This instantiate function is used to instantiate the colors
     @param[in] U Gauge field
     @param[in,out] args Additional arguments for kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon, typename Float, typename G,
            typename... Args>
  constexpr void instantiate(G &U, Args &&... args)
  {
    if (U.Ncolor() == 3) {
      constexpr int i = Recon::recon.size() - 1;
      instantiateReconstruct<Apply, Float, 3, Recon, i, G, Args...>(U, args...);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the precisions
     @param[in] U Gauge field
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = ReconstructFull, typename G,
            typename... Args>
  constexpr void instantiate(G &U, Args &&... args)
  {
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8
      instantiate<Apply, Recon, double>(U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      instantiate<Apply, Recon, float>(U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }

  // these are used in dslash.h

  struct WilsonReconstruct {
    static constexpr std::array<QudaReconstructType, 3> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8};
  };

  struct StaggeredReconstruct {
    static constexpr std::array<QudaReconstructType, 3> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_9};
  };

} // namespace quda
