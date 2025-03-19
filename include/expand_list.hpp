#include <tune_quda.h>
#include <int_factor_array.hpp>

namespace quda
{

  /**
      @brief This helper class instantiates the following mapping:
          tp.aux.x -> Bx in x_atom_size * [factors of (x + x_atom_size - 1) / x_atom_size];
          tp.aux.y -> By in y_atom_size * [factors of (y + y_atom_size - 1) / y_atom_size];
          tp.aux.z -> Bz in z_atom_size * [factors of (z + z_atom_size - 1) / z_atom_size];
          tp.aux.w -> Bw in w_atom_size * [factors of (w + w_atom_size - 1) / w_atom_size].
        See `void expand(TuneParam &tp, const qudaStream_t &stream)`
   */
  template <class Callable, int x, int x_atom_size, int y, int y_atom_size, int z, int z_atom_size, int w, int w_atom_size>
  class expand_aux_t
  {

    Callable &_callable;

    static constexpr IntFactorArray<(x + x_atom_size - 1) / x_atom_size, x_atom_size> x_factors {};
    static constexpr IntFactorArray<(y + y_atom_size - 1) / y_atom_size, y_atom_size> y_factors {};
    static constexpr IntFactorArray<(z + z_atom_size - 1) / z_atom_size, z_atom_size> z_factors {};
    static constexpr IntFactorArray<(w + w_atom_size - 1) / w_atom_size, w_atom_size> w_factors {};

    template <int Bx, int By, int Bz, size_t W, size_t... Ws>
    void span_w(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<W, Ws...>)
    {
      constexpr int Bw = w_factors[W];
      if (tp.aux.w == Bw) {
        _callable.template launch_mma<Bx, By, Bz, Bw>(tp, stream);
      } else {
        if constexpr (sizeof...(Ws) > 0) {
          span_w<Bx, By, Bz>(tp, stream, std::index_sequence<Ws...>());
        } else {
          errorQuda("Invalid tp.aux.w(=%d)", tp.aux.w);
        }
      }
    }

    template <int Bx, int By, size_t Z, size_t... Zs>
    void span_z(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<Z, Zs...>)
    {
      constexpr int Bz = z_factors[Z];
      if (tp.aux.z == Bz) {
        std::make_index_sequence<w_factors.size()> w_indices;
        span_w<Bx, By, Bz>(tp, stream, w_indices);
      } else {
        if constexpr (sizeof...(Zs) > 0) {
          span_z<Bx, By>(tp, stream, std::index_sequence<Zs...>());
        } else {
          errorQuda("Invalid tp.aux.z(=%d)", tp.aux.z);
        }
      }
    }

    template <int Bx, size_t Y, size_t... Ys>
    void span_y(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<Y, Ys...>)
    {
      constexpr int By = y_factors[Y];
      if (tp.aux.y == By) {
        std::make_index_sequence<z_factors.size()> z_indices;
        span_z<Bx, By>(tp, stream, z_indices);
      } else {
        if constexpr (sizeof...(Ys) > 0) {
          span_y<Bx>(tp, stream, std::index_sequence<Ys...>());
        } else {
          errorQuda("Invalid tp.aux.y(=%d)", tp.aux.y);
        }
      }
    }

    template <size_t X, size_t... Xs>
    void span_x(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<X, Xs...>)
    {
      constexpr int Bx = x_factors[X];
      if (tp.aux.x == Bx) {
        std::make_index_sequence<y_factors.size()> y_indices;
        span_y<Bx>(tp, stream, y_indices);
      } else {
        if constexpr (sizeof...(Xs) > 0) {
          span_x(tp, stream, std::index_sequence<Xs...>());
        } else {
          errorQuda("Invalid tp.aux.x(=%d)", tp.aux.x);
        }
      }
    }

  public:
    /**
        @brief invoke `_callable.template launch_mma<Bx, By, Bz, Bw>(tp, stream);` based on the tp.aux values
            tp.aux.x -> Bx in x_atom_size * [factors of (x + x_atom_size - 1) / x_atom_size];
            tp.aux.y -> By in y_atom_size * [factors of (y + y_atom_size - 1) / y_atom_size];
            tp.aux.z -> Bz in z_atom_size * [factors of (z + z_atom_size - 1) / z_atom_size];
            tp.aux.w -> Bw in w_atom_size * [factors of (w + w_atom_size - 1) / w_atom_size].
          For example, if x_atom_size = 8, x = 48, then Bx can take values in [8, 16, 24, 48]; when tp.aux.x == 0,
          Bx = 8; when tp.aux.x == 1, Bx = 16; when tp.aux.x == 2, Bx = 24; when tp.aux.x == 3, Bx = 48.
        @param tp The TuneParam parameter
        @param stream The stream parameter
     */
    void expand(TuneParam &tp, const qudaStream_t &stream)
    {
      std::make_index_sequence<x_factors.size()> x_indices;
      span_x(tp, stream, x_indices);
    }

    expand_aux_t(Callable &callable) : _callable(callable) { }

    /**
        @brief Get the Bx value
        @param tp The TuneParam parameter
     */
    int get_x(const TuneParam &tp) const
    {
      if (x_factors.get_index(tp.aux.x) >= x_factors.size()) { errorQuda("Invalid tp.aux.x = %d\n", tp.aux.x); }
      return tp.aux.x;
    }

    /**
        @brief Get the By value
        @param tp The TuneParam parameter
     */
    int get_y(const TuneParam &tp) const
    {
      if (y_factors.get_index(tp.aux.y) >= y_factors.size()) { errorQuda("Invalid tp.aux.y = %d\n", tp.aux.y); }
      return tp.aux.y;
    }

    /**
        @brief Get the Bz value
        @param tp The TuneParam parameter
     */
    int get_z(const TuneParam &tp) const
    {
      if (z_factors.get_index(tp.aux.z) >= z_factors.size()) { errorQuda("Invalid tp.aux.z = %d\n", tp.aux.z); }
      return tp.aux.z;
    }

    /**
        @brief Get the Bw value
        @param tp The TuneParam parameter
     */
    int get_w(const TuneParam &tp) const
    {
      if (w_factors.get_index(tp.aux.w) >= w_factors.size()) { errorQuda("Invalid tp.aux.w = %d\n", tp.aux.w); }
      return tp.aux.w;
    }

    template <unsigned int Int, unsigned int Multiple>
    bool advancer(int &aux, TuneParam &param, const IntFactorArray<Int, Multiple> &factors) const
    {
      if (factors.get_index(aux) < factors.size() - 1) {
        aux = factors[factors.get_index(aux) + 1];
        return _callable.set_mma_param(param);
      } else {
        return false;
      }
    }

    /**
        @brief Advance to the next possible aux value and return true; return false we have gone to the last
          possible value
        @return whether or not an advance is performed
        @param tp The TuneParam parameter
     */
    bool advance_aux(TuneParam &param) const
    {
      if (advancer(param.aux.x, param, x_factors)) {
        return true;
      } else {
        param.aux.x = x_atom_size;
        if (advancer(param.aux.y, param, y_factors)) {
          return true;
        } else {
          param.aux.y = y_atom_size;
          if (advancer(param.aux.z, param, z_factors)) {
            return true;
          } else {
            param.aux.z = z_atom_size;
            if (advancer(param.aux.w, param, w_factors)) {
              return true;
            } else {
              param.aux.w = w_atom_size;
              return false;
            }
          }
        }
      }
    }

    /**
        @brief Initialize aux
        @param tp The TuneParam parameter
     */
    void init_aux(TuneParam &param) const
    {
      param.aux.x = x_atom_size;
      param.aux.y = y_atom_size;
      param.aux.z = z_atom_size;
      param.aux.w = w_atom_size;
    }
  };

} // namespace quda
