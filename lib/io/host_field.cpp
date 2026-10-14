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

      template <typename Host, typename Stored>
      void unpack_site(const std::vector<void *> &arrays, int reals, size_t index, const char *site)
      {
        for (auto i = 0u; i < arrays.size(); i++) {
          auto dst = static_cast<Host *>(arrays[i]) + index * reals;
          for (int j = 0; j < reals; j++) {
            Stored value;
            memcpy(&value, site + (i * reals + j) * sizeof(Stored), sizeof(Stored));
            dst[j] = value;
          }
        }
      }

      template <typename Host, typename Stored>
      void pack_site(const std::vector<void *> &arrays, int reals, size_t index, char *site)
      {
        for (auto i = 0u; i < arrays.size(); i++) {
          auto src = static_cast<const Host *>(arrays[i]) + index * reals;
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
      arrays(n), prec(prec), reals(reals)
    {
      check_precision(prec, "host field");
      for (int i = 0; i < n; i++) this->arrays[i] = const_cast<void *>(arrays[i]);
    }

    SiteUnpack HostField::unpacker(QudaPrecision file_prec) const
    {
      check_precision(file_prec, "file");
      auto f = prec == QUDA_DOUBLE_PRECISION ?
        (file_prec == QUDA_DOUBLE_PRECISION ? unpack_site<double, double> : unpack_site<double, float>) :
        (file_prec == QUDA_DOUBLE_PRECISION ? unpack_site<float, double> : unpack_site<float, float>);
      return [arrays = arrays, reals = reals, f](int, size_t index, const char *site) { f(arrays, reals, index, site); };
    }

    SitePack HostField::packer(QudaPrecision file_prec) const
    {
      check_precision(file_prec, "file");
      auto f = prec == QUDA_DOUBLE_PRECISION ?
        (file_prec == QUDA_DOUBLE_PRECISION ? pack_site<double, double> : pack_site<double, float>) :
        (file_prec == QUDA_DOUBLE_PRECISION ? pack_site<float, double> : pack_site<float, float>);
      return [arrays = arrays, reals = reals, f](int, size_t index, char *site) { f(arrays, reals, index, site); };
    }

  } // namespace io

} // namespace quda
