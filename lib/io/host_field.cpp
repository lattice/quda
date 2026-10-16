#include <cstring>
#include <util_quda.h>
#include <io/host_field.h>

namespace quda
{

  namespace io
  {

    namespace
    {
      void check_precision(QudaPrecision prec, const char *what)
      {
        if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
          errorQuda("Unsupported %s precision %d", what, prec);
      }

      std::vector<void *> to_vector(const void *const arrays[], int n)
      {
        std::vector<void *> v(n);
        for (int i = 0; i < n; i++) v[i] = const_cast<void *>(arrays[i]);
        return v;
      }

      /**
         @brief The arrays holding a site, and the site's index in them
       */
      inline const std::vector<void *> &site_arrays(const std::vector<void *> &arrays,
                                                    const std::vector<void *> &odd_arrays, size_t half_volume,
                                                    size_t &index)
      {
        if (odd_arrays.empty() || index < half_volume) return arrays;
        index -= half_volume;
        return odd_arrays;
      }

      template <typename Host, typename Stored>
      void unpack_site(const std::vector<void *> &arrays, const std::vector<void *> &odd_arrays, size_t half_volume,
                       int reals, size_t index, const char *site)
      {
        const auto &a = site_arrays(arrays, odd_arrays, half_volume, index);
        for (auto i = 0u; i < a.size(); i++) {
          auto dst = static_cast<Host *>(a[i]) + index * reals;
          for (int j = 0; j < reals; j++) {
            Stored value;
            memcpy(&value, site + (i * reals + j) * sizeof(Stored), sizeof(Stored));
            dst[j] = value;
          }
        }
      }

      template <typename Host, typename Stored>
      void pack_site(const std::vector<void *> &arrays, const std::vector<void *> &odd_arrays, size_t half_volume,
                     int reals, size_t index, char *site)
      {
        const auto &a = site_arrays(arrays, odd_arrays, half_volume, index);
        for (auto i = 0u; i < a.size(); i++) {
          auto src = static_cast<const Host *>(a[i]) + index * reals;
          for (int j = 0; j < reals; j++) {
            Stored value = src[j];
            memcpy(site + (i * reals + j) * sizeof(Stored), &value, sizeof(Stored));
          }
        }
      }
    } // namespace

    HostField::HostField(void *const arrays[], int n, QudaPrecision prec, int reals) :
      arrays(arrays, arrays + n), prec(prec), reals(reals)
    {
      check_precision(prec, "host field");
    }

    HostField::HostField(const void *const arrays[], int n, QudaPrecision prec, int reals) :
      arrays(to_vector(arrays, n)), prec(prec), reals(reals)
    {
      check_precision(prec, "host field");
    }

    HostField::HostField(void *const even[], void *const odd[], int n, QudaPrecision prec, int reals, size_t half_volume) :
      arrays(even, even + n), odd_arrays(odd, odd + n), half_volume(half_volume), prec(prec), reals(reals)
    {
      check_precision(prec, "host field");
    }

    HostField::HostField(const void *const even[], const void *const odd[], int n, QudaPrecision prec, int reals,
                         size_t half_volume) :
      arrays(to_vector(even, n)), odd_arrays(to_vector(odd, n)), half_volume(half_volume), prec(prec), reals(reals)
    {
      check_precision(prec, "host field");
    }

    SiteUnpack HostField::unpacker(QudaPrecision file_prec) const
    {
      check_precision(file_prec, "file");
      auto f = prec == QUDA_DOUBLE_PRECISION ?
        (file_prec == QUDA_DOUBLE_PRECISION ? unpack_site<double, double> : unpack_site<double, float>) :
        (file_prec == QUDA_DOUBLE_PRECISION ? unpack_site<float, double> : unpack_site<float, float>);
      return [arrays = arrays, odd_arrays = odd_arrays, half_volume = half_volume, reals = reals,
              f](int, size_t index, const char *site) { f(arrays, odd_arrays, half_volume, reals, index, site); };
    }

    SitePack HostField::packer(QudaPrecision file_prec) const
    {
      check_precision(file_prec, "file");
      auto f = prec == QUDA_DOUBLE_PRECISION ?
        (file_prec == QUDA_DOUBLE_PRECISION ? pack_site<double, double> : pack_site<double, float>) :
        (file_prec == QUDA_DOUBLE_PRECISION ? pack_site<float, double> : pack_site<float, float>);
      return [arrays = arrays, odd_arrays = odd_arrays, half_volume = half_volume, reals = reals,
              f](int, size_t index, char *site) { f(arrays, odd_arrays, half_volume, reals, index, site); };
    }

  } // namespace io

} // namespace quda
