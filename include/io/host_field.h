#pragma once

#include <vector>
#include <io/site_io.h>

namespace quda
{

  namespace io
  {

    /**
       @brief Host field data as the file formats see it: one array per
       item (a direction of a gauge field, or a vector), each holding a
       fixed number of reals per site at the QUDA host index.  A site in
       a file holds the items one after another.
     */
    class HostField
    {
      std::vector<void *> arrays;
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
         @brief Number of items
       */
      int items() const { return static_cast<int>(arrays.size()); }

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
