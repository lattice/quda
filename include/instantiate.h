#pragma once

#include <array>
#include <enum_quda.h>
#include <util_quda.h>

namespace quda
{

  /**
     @brief Helper function for returning if a given precision is enabled
     @tparam precision The precision requested
     @return True if enabled, false if not
  */
  template <QudaPrecision precision> constexpr bool is_enabled();
  template <> constexpr bool is_enabled<QUDA_DOUBLE_PRECISION>() { return (QUDA_PRECISION & 8) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_SINGLE_PRECISION>() { return (QUDA_PRECISION & 4) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_HALF_PRECISION>() { return (QUDA_PRECISION & 2) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_QUARTER_PRECISION>() { return (QUDA_PRECISION & 1) ? true : false; }

  /**
     @brief Helper function for returning if a given reconstruct is enabled
     @tparam reconstruct The reconstruct requested
     @return True if enabled, false if not
  */
  template <QudaReconstructType reconstruct> constexpr bool is_enabled();
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_NO>() { return (QUDA_RECONSTRUCT & 4) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_13>() { return (QUDA_RECONSTRUCT & 2) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_12>() { return (QUDA_RECONSTRUCT & 2) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_9>() { return (QUDA_RECONSTRUCT & 1) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_8>() { return (QUDA_RECONSTRUCT & 1) ? true : false; }
  template <> constexpr bool is_enabled<QUDA_RECONSTRUCT_10>() { return true; }

  struct ReconstructFull {
    static constexpr std::array<QudaReconstructType, 6> recon
      = {QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_9, QUDA_RECONSTRUCT_8, QUDA_RECONSTRUCT_10};
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
     @brief Instantiate the reconstruction template at index i and
     recurse to prior element
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Float, int nColor, typename Recon,
            int i, typename G, typename... Args>
  void instantiateReconstruct(G &U, Args &&... args)
  {
    if (U.Reconstruct() == Recon::recon[i]) {
      if constexpr (is_enabled<Recon::recon[i]>())
        Apply<Float, nColor, Recon::recon[i]>(U, args...);
      else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable %d", QUDA_RECONSTRUCT, Recon::recon[i]);
    } else if constexpr (i > 0) {
      instantiateReconstruct<Apply, Float, nColor, Recon, i - 1, G, Args...>(U, args...);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
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
  constexpr void instantiate(G &U, Args &&...args)
  {
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled<QUDA_DOUBLE_PRECISION>())
        instantiate<Apply, Recon, double>(U, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        instantiate<Apply, Recon, float>(U, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }

  /**
     @brief This instantiate2 function is used to instantiate the
     precisions, with double precision always enabled.  This is a
     temporary addition until we fuse this with the original function
     above when we enforce C++17
     @param[in] U Gauge field
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = ReconstructFull, typename G,
            typename... Args>
  constexpr void instantiate2(G &U, Args &&...args)
  {
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      instantiate<Apply, Recon, double>(U, args...);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        instantiate<Apply, Recon, float>(U, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the clover precision
     @param[in] c CloverField we wish to instantiate
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename> class Apply, typename C, typename... Args>
  constexpr void instantiate(C &c, Args &&... args)
  {
    if (c.Precision() == QUDA_DOUBLE_PRECISION) {
      Apply<double>(c, args...);
    } else if (c.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        Apply<float>(c, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", c.Precision());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the colors
     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Additional arguments for kernels
  */
  template <template <typename, int> class Apply, typename store_t, typename F, typename... Args>
  constexpr void instantiate(F &field, Args &&...args)
  {
    if (field.Ncolor() == 3) {
      Apply<store_t, 3>(field, args...);
    } else {
      errorQuda("Unsupported number of colors %d\n", field.Ncolor());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the
     precision and number of colors
     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename, int> class Apply, typename F, typename... Args>
  constexpr void instantiate(F &field, Args &&... args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled<QUDA_DOUBLE_PRECISION>())
        instantiate<Apply, double>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        instantiate<Apply, float>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled<QUDA_HALF_PRECISION>())
        instantiate<Apply, short>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled<QUDA_QUARTER_PRECISION>())
        instantiate<Apply, int8_t>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

  /**
     @brief The instantiatePrecision function is used to instantiate
     the precision.  Note unlike the "instantiate" functions above,
     this helper always instantiates double precision regardless of
     the QUDA_PRECISION value: this enables its use for copy interface
     routines which should always enable double precision support.

     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename> class Apply, typename F, typename... Args>
  constexpr void instantiatePrecision(F &field, Args &&... args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      // always instantiate double precision
      Apply<double>(field, args...);
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        Apply<float>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled<QUDA_HALF_PRECISION>())
        Apply<short>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled<QUDA_QUARTER_PRECISION>())
        Apply<int8_t>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

  /**
     @brief The instantiatePrecision2 function is used to instantiate
     the precision for a class that accepts 2 typename arguments, with
     the first typename corresponding to the precision being
     instantiated at hand.  This is useful for copy routines, where we
     need to instantiate a second, e.g., destination, precision after
     already instantiating the first, e.g., source, precision.
     Similar to the "instantiatePrecision" function above, this helper
     always instantiates double precision regardless of the
     QUDA_PRECISION value: this enables its use for copy interface
     routines which should always enable double precision support.

     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename, typename> class Apply, typename T, typename F, typename... Args>
  constexpr void instantiatePrecision2(F &field, Args &&... args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      // always instantiate double precision
      Apply<double, T>(field, args...);
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        Apply<float, T>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled<QUDA_HALF_PRECISION>())
        Apply<short, T>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled<QUDA_QUARTER_PRECISION>())
        Apply<int8_t, T>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

  /**
     @brief The instantiatePrecision function is used to instantiate
     the precision
     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename> class Apply, typename F, typename... Args>
  constexpr void instantiatePrecisionMG(F &field, Args &&... args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Apply<double>(field, args...);
#else
      errorQuda("Multigrid not supported in double precision");
#endif
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        Apply<float>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled<QUDA_HALF_PRECISION>())
        Apply<short>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled<QUDA_QUARTER_PRECISION>())
        Apply<int8_t>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
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
