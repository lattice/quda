#pragma once

#include <vector>
#include <io/site_io.h>

namespace quda
{

  namespace io
  {

    /**
       @brief Host field data as the file formats see it: a list of
       items, each a 4-d field with a fixed number of reals per site at
       the QUDA host index (even sites first, then odd).  A site in a
       file holds the items one after another.  An item is one array,
       or two when its even and odd sites lie apart in memory, as for
       the 4-d slices of a full 5-d vector (see host_field() in
       vector_io.cpp for which fields need which).
     */
    class HostField
    {
      std::vector<void *> arrays;     // one per item; for two-array items, their even sites
      std::vector<void *> odd_arrays; // for two-array items, their odd sites; otherwise empty
      size_t half_volume = 0;         // for two-array items, the local 4-d half volume
      QudaPrecision prec;
      int reals;

    public:
      /**
         @param[in] arrays The arrays, one per item
         @param[in] n Number of items
         @param[in] prec Precision of the arrays (double or single)
         @param[in] reals Reals per item and site
       */
      HostField(void *const arrays[], int n, QudaPrecision prec, int reals);

      /**
         @brief A field that is only packed (saved), never unpacked
       */
      HostField(const void *const arrays[], int n, QudaPrecision prec, int reals);

      /**
         @brief Items given by their even and their odd sites: host
         index h < half_volume is at h in the even array, any other at
         h - half_volume in the odd array
         @param[in] even The slices' even sites, one array per item
         @param[in] odd The slices' odd sites, one array per item
         @param[in] n Number of items (vectors times Ls)
         @param[in] prec Precision of the arrays (double or single)
         @param[in] reals Reals per item and site
         @param[in] half_volume The local 4-d half volume
       */
      HostField(void *const even[], void *const odd[], int n, QudaPrecision prec, int reals, size_t half_volume);

      /**
         @brief Items given by their even and their odd sites that are only packed (saved), never unpacked
       */
      HostField(const void *const even[], const void *const odd[], int n, QudaPrecision prec, int reals,
                size_t half_volume);

      /**
         @brief Number of items
       */
      int items() const { return static_cast<int>(arrays.size()); }

      /**
         @brief Precision of the arrays
       */
      QudaPrecision precision() const { return prec; }

      /**
         @brief Size of an item at a site in a file of the given precision
       */
      size_t item_bytes(QudaPrecision file_prec) const { return reals * file_prec; }

      /**
         @brief Size of a site in a file of the given precision
       */
      size_t site_bytes(QudaPrecision file_prec) const { return arrays.size() * item_bytes(file_prec); }

      /**
         @brief The layout of this field's sites in a file
         @param[in] file_prec Precision of the file
         @param[in] offset Byte offset of the data
         @param[in] swap Whether the file's byte order is the opposite of the host's
         @param[in] checksum How the file's checksum is computed
       */
      DataLayout layout(QudaPrecision file_prec, size_t offset, bool swap, ChecksumType checksum) const
      {
        DataLayout l;
        l.offset = offset;
        l.site_bytes = site_bytes(file_prec);
        l.word_bytes = file_prec;
        l.swap = swap;
        l.checksum = checksum;
        return l;
      }

      /**
         @brief Return a callback that converts a site from the file
         precision (in native byte order) and stores it in the arrays
       */
      SiteUnpack unpacker(QudaPrecision file_prec) const;

      /**
         @brief Return a callback that fills in a site in the file precision from the arrays
       */
      SitePack packer(QudaPrecision file_prec) const;
    };

  } // namespace io

} // namespace quda
